package tools

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

const (
	defaultQdrantURL  = "http://qdrant.qdrant:6333"
	defaultCollection = "abox"
)

func qdrantURL() string {
	if v := os.Getenv("QDRANT_URL"); v != "" {
		return v
	}
	return defaultQdrantURL
}

func collection() string {
	if v := os.Getenv("QDRANT_COLLECTION"); v != "" {
		return v
	}
	return defaultCollection
}

func qdrant(ctx context.Context, method, path string, body any, out any) (int, error) {
	var rdr io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return 0, err
		}
		rdr = strings.NewReader(string(b))
	}

	req, err := http.NewRequestWithContext(ctx, method, qdrantURL()+path, rdr)
	if err != nil {
		return 0, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := client().Do(req)
	if err != nil {
		return 0, fmt.Errorf("%s %s: %w", method, path, err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return resp.StatusCode, err
	}
	if resp.StatusCode >= 400 {
		return resp.StatusCode, fmt.Errorf("%s %s: %s: %s", method, path, resp.Status, raw)
	}
	if out != nil {
		return resp.StatusCode, json.Unmarshal(raw, out)
	}
	return resp.StatusCode, nil
}

// A pooled embedding must fit in one micro-batch, so llama.cpp rejects
// anything over --ubatch-size outright:
//
//	input (5202 tokens) is too large to process. increase the physical
//	batch size (current batch size: 2048)
//
// Raising the batch does not help: the GGUF's n_ctx_train is 2048 and the
// slot is clamped to it regardless.
//
// The reference embedder caps the input instead -- charts/triageagent/
// values.yaml, --embedding-max-input-chars, "keep below the sidecar model's
// token window x ~4", default 8000. Most of an Agent manifest is prose, which
// does hit that ratio, but 8000 chars measured 2061 tokens against this
// server: 13 over. 7000 is ~1750 tokens of prose and still fits at 3.4
// chars/token.
//
// Capping loses the tail, so vector_store splits instead and keeps every
// piece. This is the per-embed ceiling, and the chunk size.
const defaultMaxInputChars = 7000

// Enough to carry a heading or an opening key into the next piece.
const chunkOverlapChars = 200

func maxInputChars() int {
	if v := os.Getenv("EMBEDDING_MAX_INPUT_CHARS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return defaultMaxInputChars
}

// Splits on a line boundary when there is one in the last fifth of the
// window, so a chunk rarely ends mid-key.
func chunk(s string) []string {
	r := []rune(s)
	size := maxInputChars()
	if len(r) <= size {
		return []string{s}
	}

	overlap := chunkOverlapChars
	if overlap >= size {
		overlap = 0
	}

	var out []string
	for start := 0; start < len(r); {
		end := start + size
		if end >= len(r) {
			out = append(out, string(r[start:]))
			break
		}
		cut := end
		for i := end - 1; i > end-size/5 && i > start; i-- {
			if r[i] == '\n' {
				cut = i + 1
				break
			}
		}
		out = append(out, string(r[start:cut]))
		next := cut - overlap
		if next <= start {
			next = cut
		}
		start = next
	}
	return out
}

// nomic is trained with an instruction prefix and the two are not
// interchangeable: a document embedded as a query lands in the wrong place.
// Other models want no prefix at all, so both are overridable -- including to
// the empty string, which is why this reads LookupEnv rather than Getenv.
const (
	defaultDocumentPrefix = "search_document: "
	defaultQueryPrefix    = "search_query: "
)

func documentPrefix() string {
	if v, ok := os.LookupEnv("EMBEDDINGS_DOCUMENT_PREFIX"); ok {
		return v
	}
	return defaultDocumentPrefix
}

func queryPrefix() string {
	if v, ok := os.LookupEnv("EMBEDDINGS_QUERY_PREFIX"); ok {
		return v
	}
	return defaultQueryPrefix
}

func embed(ctx context.Context, text, prefix string) ([]float64, error) {
	if r := []rune(text); len(r) > maxInputChars() {
		text = string(r[:maxInputChars()])
	}
	body := map[string]any{"input": prefix + text}
	if m := embeddingsModel(); m != "" {
		body["model"] = m
	}
	out, err := post(ctx, "/v1/embeddings", body)
	if err != nil {
		return nil, err
	}
	var r struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal([]byte(out), &r); err != nil {
		return nil, err
	}
	if len(r.Data) == 0 || len(r.Data[0].Embedding) == 0 {
		return nil, fmt.Errorf("embeddings response carried no vector: %.200s", out)
	}
	return r.Data[0].Embedding, nil
}

// Created on first write, sized from the vector the server actually returned
// rather than a hardcoded 768.
func ensureCollection(ctx context.Context, dim int) error {
	code, err := qdrant(ctx, http.MethodGet, "/collections/"+collection(), nil, nil)
	if code == http.StatusOK {
		return nil
	}
	if code != http.StatusNotFound {
		return err
	}
	body := map[string]any{"vectors": map[string]any{"size": dim, "distance": "Cosine"}}
	_, err = qdrant(ctx, http.MethodPut, "/collections/"+collection(), body, nil)
	return err
}

func uuid() (string, error) {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	b[6] = (b[6] & 0x0f) | 0x40
	b[8] = (b[8] & 0x3f) | 0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16]), nil
}

func init() {
	registerTool(VectorStore())
	registerTool(VectorFind())
}

type StoreParams struct {
	Information string            `json:"information" description:"Text to remember."`
	Metadata    map[string]string `json:"metadata,omitempty" description:"Arbitrary key/value pairs stored with the text."`
}

func VectorStore() MCPTool[StoreParams, Raw] {
	return MCPTool[StoreParams, Raw]{
		Name:        "vector_store",
		Description: "Embed text with the configured embeddings server and store it in Qdrant. Long text is split into several points; nothing is dropped.",
		Handler: func(ctx context.Context, _ *mcp.ServerSession, p *mcp.CallToolParamsFor[StoreParams]) (*mcp.CallToolResultFor[Raw], error) {
			parts := chunk(p.Arguments.Information)

			doc, err := uuid()
			if err != nil {
				return nil, err
			}

			points := make([]map[string]any, 0, len(parts))
			dims := 0
			for i, part := range parts {
				vec, err := embed(ctx, part, documentPrefix())
				if err != nil {
					return nil, fmt.Errorf("chunk %d/%d: %w", i+1, len(parts), err)
				}
				if dims == 0 {
					dims = len(vec)
					if err := ensureCollection(ctx, dims); err != nil {
						return nil, err
					}
				}

				id, err := uuid()
				if err != nil {
					return nil, err
				}
				// doc ties the pieces back together; a hit on one chunk can pull
				// its siblings with a filter on this field.
				payload := map[string]any{
					"document": part,
					"doc":      doc,
					"chunk":    i,
					"chunks":   len(parts),
				}
				for k, v := range p.Arguments.Metadata {
					payload[k] = v
				}
				points = append(points, map[string]any{"id": id, "vector": vec, "payload": payload})
			}

			body := map[string]any{"points": points}
			if _, err := qdrant(ctx, http.MethodPut, "/collections/"+collection()+"/points?wait=true", body, nil); err != nil {
				return nil, err
			}
			return text(fmt.Sprintf("Stored in %s as doc %s: %d chunk(s), %d dims.", collection(), doc, len(parts), dims)), nil
		},
	}
}

type FindParams struct {
	Query string `json:"query" description:"What to search for."`
	Limit int    `json:"limit,omitempty" description:"Maximum results. Defaults to 5."`
}

func VectorFind() MCPTool[FindParams, Raw] {
	return MCPTool[FindParams, Raw]{
		Name:        "vector_find",
		Description: "Semantic search over what vector_store has remembered.",
		Handler: func(ctx context.Context, _ *mcp.ServerSession, p *mcp.CallToolParamsFor[FindParams]) (*mcp.CallToolResultFor[Raw], error) {
			limit := p.Arguments.Limit
			if limit <= 0 {
				limit = 5
			}
			vec, err := embed(ctx, p.Arguments.Query, queryPrefix())
			if err != nil {
				return nil, err
			}

			var res struct {
				Result struct {
					Points []struct {
						Score   float64        `json:"score"`
						Payload map[string]any `json:"payload"`
					} `json:"points"`
				} `json:"result"`
			}
			body := map[string]any{"query": vec, "limit": limit, "with_payload": true}
			if _, err := qdrant(ctx, http.MethodPost, "/collections/"+collection()+"/points/query", body, &res); err != nil {
				return nil, err
			}

			hits := make([]map[string]any, 0, len(res.Result.Points))
			for _, pt := range res.Result.Points {
				hits = append(hits, map[string]any{"score": pt.Score, "payload": pt.Payload})
			}
			out, err := json.Marshal(hits)
			if err != nil {
				return nil, err
			}
			return text(string(out)), nil
		},
	}
}

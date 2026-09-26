package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"time"
)

// The embeddings endpoint. Anything serving OpenAI-compatible
// /v1/embeddings will do -- abox has two routes to the same model:
//
//	llama-cpp-embeddings.llama-cpp:8090   standalone llama.cpp Deployment
//	llm-d-embedding.llm-d:8000            the same engine under llm-d
//
// The llama_* tools additionally use llama.cpp's own /props and /tokenize,
// so those return an error against a server that does not implement them.
const defaultBaseURL = "http://llama-cpp-embeddings.llama-cpp:8090"

// LLAMA_BASE_URL is the pre-rename spelling, still honoured so an older
// manifest keeps working.
func baseURL() string {
	if v := os.Getenv("EMBEDDINGS_BASE_URL"); v != "" {
		return v
	}
	if v := os.Getenv("LLAMA_BASE_URL"); v != "" {
		return v
	}
	return defaultBaseURL
}

// Model id sent as "model" on embeddings requests. llama.cpp serves a single
// model and ignores it; llm-d routes on it and Vertex requires it.
func embeddingsModel() string {
	return os.Getenv("EMBEDDINGS_MODEL")
}

func client() *http.Client {
	secs := 120
	v := os.Getenv("EMBEDDINGS_TIMEOUT_SECONDS")
	if v == "" {
		v = os.Getenv("LLAMA_TIMEOUT_SECONDS")
	}
	if v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			secs = n
		}
	}
	return &http.Client{Timeout: time.Duration(secs) * time.Second}
}

// Upstream responses are returned verbatim. The caller is a model, llama.cpp
// already answers in JSON, and reshaping it here would only drop fields.
func do(ctx context.Context, method, path string, body any) (string, error) {
	var rdr io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return "", err
		}
		rdr = bytes.NewReader(b)
	}

	req, err := http.NewRequestWithContext(ctx, method, baseURL()+path, rdr)
	if err != nil {
		return "", err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	// Unset for an in-cluster llama.cpp or llm-d; a managed endpoint such as
	// Vertex's OpenAI-compatible one needs it.
	if k := os.Getenv("EMBEDDINGS_API_KEY"); k != "" {
		req.Header.Set("Authorization", "Bearer "+k)
	}

	resp, err := client().Do(req)
	if err != nil {
		return "", fmt.Errorf("%s %s: %w", method, path, err)
	}
	defer resp.Body.Close()

	out, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	if resp.StatusCode >= 400 {
		// 501 here means the server was started with --embeddings and has no
		// generation head; the completion tools cannot work against it.
		return "", fmt.Errorf("%s %s: %s: %s", method, path, resp.Status, out)
	}
	return string(out), nil
}

func get(ctx context.Context, path string) (string, error) {
	return do(ctx, http.MethodGet, path, nil)
}

func post(ctx context.Context, path string, body any) (string, error) {
	return do(ctx, http.MethodPost, path, body)
}

package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func init() {
	registerTool(HybridFind())
}

type HybridFindParams struct {
	Query string `json:"query" description:"What to search for."`
	Limit int    `json:"limit,omitempty" description:"Maximum fused results. Defaults to 8."`
}

type hybridHit struct {
	Source string         `json:"source"` // "vector" or "graph"
	Score  float64        `json:"fused_score"`
	Point  map[string]any `json:"point,omitempty"` // vector hit: {score, payload}
	Fact   string         `json:"fact,omitempty"`  // graph hit: "A -[REL]-> B"
}

type hybridResult struct {
	Results []hybridHit `json:"results"`
	Note    string      `json:"note"`
}

// maxHybridCallsPerWindow/hybridBudgetWindow cap tool-calling rounds per
// question. Neither kagent's Agent CRD nor its ModelConfig CRD exposes a
// native max-iterations knob (checked against both CRDs' OpenAPI schemas),
// so the budget is enforced here instead: the eval run that motivated this
// (TASK2-EVAL.md item 13) showed a single question burning 126,758 prompt
// tokens across a stuck retrieval loop before confabulating.
//
// This is a sliding window, not a per-session lifetime counter, because
// ServerSession.ID() turned out NOT to be a per-question key: kagent's ADK
// client keeps one MCP session open for the agent pod's whole lifetime
// (confirmed via the JSON-RPC id sequence climbing 4->37 across many separate
// eval questions with no reset), so a lifetime counter starved every
// question after the 4th in a sequence. A window that ages out old calls
// lets a later, genuinely new question earn budget back instead of
// inheriting exhaustion from an unrelated earlier one, while still capping a
// single stuck loop within one question.
const (
	maxHybridCallsPerWindow = 4
	hybridBudgetWindow      = 30 * time.Second
)

type callWindow struct {
	mu    sync.Mutex
	times []time.Time
}

// callBudgets leaks one *callWindow per session for the process lifetime --
// fine for this lab agent's traffic volume, not a pattern to carry into a
// long-lived production server without an eviction policy.
var callBudgets sync.Map // session ID -> *callWindow

func budgetExceeded(sessionID string) bool {
	v, _ := callBudgets.LoadOrStore(sessionID, &callWindow{})
	cw := v.(*callWindow)

	cw.mu.Lock()
	defer cw.mu.Unlock()

	now := time.Now()
	cutoff := now.Add(-hybridBudgetWindow)
	live := cw.times[:0]
	for _, t := range cw.times {
		if t.After(cutoff) {
			live = append(live, t)
		}
	}
	cw.times = live

	if len(cw.times) >= maxHybridCallsPerWindow {
		return true
	}
	cw.times = append(cw.times, now)
	return false
}

var enumerationQueryRe = regexp.MustCompile(`(?i)\b(all|every|each|exhaustive|complete list|names of|list of)\b`)

// rrfK is the Reciprocal Rank Fusion constant (Cormack et al. 2009's RRF60):
// high enough that one source's rank order isn't swamped by the other, low
// enough that rank position still dominates the fused score.
const rrfK = 60.0

func rrfScore(rank int) float64 {
	return 1 / (rrfK + float64(rank+1))
}

// HybridFind is the HybridRAG merge-and-rerank tool: it runs the vector leg
// and the graph leg independently, then fuses both ranked lists with
// Reciprocal Rank Fusion, instead of the agent having to pick one tool per
// question and losing whatever the other source had. There is no
// cross-encoder reranker in this stack -- the llama.cpp server here only
// serves embeddings (see Chat()'s doc comment on the 501 it returns for
// generation) -- so RRF, which only needs rank position, is the fusion
// method, not a learned reranker.
func HybridFind() MCPTool[HybridFindParams, Raw] {
	return MCPTool[HybridFindParams, Raw]{
		Name:        "hybrid_find",
		Description: "Search both prose memory (vector) and the dependency graph, merged and reranked into one ranked list (HybridRAG, Reciprocal Rank Fusion). Prefer this over vector_find/graph_query for any question that could benefit from either or both. Capped per question -- if you see a budget-exceeded note, stop calling tools and answer from what you already have, saying plainly what you could not find.",
		Handler: func(ctx context.Context, session *mcp.ServerSession, p *mcp.CallToolParamsFor[HybridFindParams]) (*mcp.CallToolResultFor[Raw], error) {
			limit := p.Arguments.Limit
			if limit <= 0 {
				limit = 8
			}

			if session != nil && budgetExceeded(session.ID()) {
				out, err := json.Marshal(hybridResult{Note: fmt.Sprintf(
					"Tool-call budget (%d calls per %s) exhausted for this question. Stop calling tools now and answer from results already retrieved -- if a specific number, formula, or exhaustive list still isn't among them, say plainly that you could not find it rather than continuing to search or guessing.",
					maxHybridCallsPerWindow, hybridBudgetWindow)})
				if err != nil {
					return nil, err
				}
				return text(string(out)), nil
			}

			vectorHits, err := vectorSearch(ctx, p.Arguments.Query, limit)
			if err != nil {
				return nil, fmt.Errorf("vector leg: %w", err)
			}

			// The graph leg only runs when the query names a known entity
			// directly. The old "also pull in anything co-mentioned in a
			// vector hit's text" heuristic fired on nearly every prose
			// question, since the README mentions most service names in
			// almost every chunk -- that's what made the "coordinator with
			// specialists" prose question cost 185,998 prompt tokens
			// (TASK2-EVAL.md item 13). Gating on the query itself removes
			// that false-positive path; single-source-in-Qdrant questions no
			// longer pay for an always-on graph pull they never asked for.
			var graphFacts []string
			entities, err := matchedEntities(ctx, p.Arguments.Query)
			if err != nil {
				return nil, fmt.Errorf("graph leg: %w", err)
			}
			if len(entities) > 0 {
				graphFacts, err = graphNeighborhoodFacts(ctx, entities)
				if err != nil {
					return nil, fmt.Errorf("graph leg: %w", err)
				}
			}

			// Graph facts are never truncated by the shared limit: they come
			// from an exact 1-hop traversal of a named entity, not a
			// similarity ranking, so dropping one isn't "a slightly worse
			// match" the way dropping a low-rank vector hit is -- it's a
			// silent hole in an otherwise-complete answer. This surfaced for
			// real: "list all 11 fault types" retrieved only 10 of
			// chaos-injector's 11 HAS_FAULT edges because vector hits filled
			// the shared limit=20 slice first, and the agent fabricated the
			// missing 11th instead of noticing it was cut off. Vector hits
			// absorb the truncation instead, down to whatever room is left.
			vectorBudget := limit - len(graphFacts)
			if vectorBudget < 0 {
				vectorBudget = 0
			}
			if len(vectorHits) > vectorBudget {
				vectorHits = vectorHits[:vectorBudget]
			}

			fused := make([]hybridHit, 0, len(vectorHits)+len(graphFacts))
			for i, h := range vectorHits {
				fused = append(fused, hybridHit{Source: "vector", Score: rrfScore(i), Point: h})
			}
			for i, f := range graphFacts {
				fused = append(fused, hybridHit{Source: "graph", Score: rrfScore(i), Fact: f})
			}
			sort.SliceStable(fused, func(i, j int) bool { return fused[i].Score > fused[j].Score })

			result := hybridResult{Results: fused}
			if enumerationQueryRe.MatchString(p.Arguments.Query) {
				result.Note = fmt.Sprintf(
					"This looks like a request for an exhaustive list. Only %d fused result(s) were retrieved -- treat this as possibly INCOMPLETE, not as the full set. Report only the items actually present above; explicitly say the list may be partial rather than filling in the rest from general knowledge.",
					len(fused))
			}

			out, err := json.Marshal(result)
			if err != nil {
				return nil, err
			}
			return text(string(out)), nil
		},
	}
}

// vectorSearch is VectorFind's handler body, factored out so hybrid_find can
// call it directly instead of round-tripping through MCP's own wrapper.
func vectorSearch(ctx context.Context, query string, limit int) ([]map[string]any, error) {
	vec, err := embed(ctx, query, queryPrefix())
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
	return hits, nil
}

// graphEntities lists every Service/Infra node name currently in Neo4j,
// queried live rather than duplicating cmd/ingest's knownServices map here,
// so the two can never drift apart.
func graphEntities(ctx context.Context) ([]string, error) {
	rows, err := neo4jExec(ctx, "MATCH (n) WHERE n:Service OR n:Infra RETURN DISTINCT n.name AS name", nil)
	if err != nil {
		return nil, err
	}
	names := make([]string, 0, len(rows))
	for _, r := range rows {
		if name, ok := r["name"].(string); ok {
			names = append(names, name)
		}
	}
	return names, nil
}

// matchedEntities returns which known Service/Infra entities are named
// directly in the query string. Unlike the original version, it no longer
// also matches entities merely co-mentioned in a vector hit's retrieved
// text -- see the comment at the HybridFind call site for why that was
// removed.
func matchedEntities(ctx context.Context, query string) ([]string, error) {
	entities, err := graphEntities(ctx)
	if err != nil {
		return nil, err
	}

	var hits []string
	lowerQuery := strings.ToLower(query)
	for _, name := range entities {
		if strings.Contains(lowerQuery, strings.ToLower(name)) {
			hits = append(hits, name)
		}
	}
	return hits, nil
}

// graphNeighborhoodFacts pulls each given entity's 1-hop neighborhood as
// short "A -[REL]-> B" facts.
func graphNeighborhoodFacts(ctx context.Context, entities []string) ([]string, error) {
	var facts []string
	seen := map[string]bool{}
	for _, name := range entities {
		// Doc nodes are keyed by `path`, not `name` (cmd/ingest's ingestDoc),
		// so a neighbor on the other end of a MENTIONS edge needs the
		// coalesce -- name alone is null for those and prints "<nil>".
		// LIMIT 40, not 20: chaos-injector alone now has 4 TRIGGERS_CHAOS_ON +
		// 11 HAS_FAULT edges (see faultCatalogGraph in cmd/ingest), so 20 was
		// one questionable edge short of "list all 11 fault types" ever
		// being answerable as a complete traversal.
		rows, err := neo4jExec(ctx,
			`MATCH (n {name:$name})-[r]-(m)
			 RETURN type(r) AS rel,
			        coalesce(startNode(r).name, startNode(r).path) AS from,
			        coalesce(endNode(r).name, endNode(r).path) AS to
			 LIMIT 40`,
			map[string]any{"name": name})
		if err != nil {
			return nil, fmt.Errorf("neighborhood of %q: %w", name, err)
		}
		for _, row := range rows {
			fact := fmt.Sprintf("%v -[%v]-> %v", row["from"], row["rel"], row["to"])
			if seen[fact] {
				continue
			}
			seen[fact] = true
			facts = append(facts, fact)
		}
	}
	return facts, nil
}

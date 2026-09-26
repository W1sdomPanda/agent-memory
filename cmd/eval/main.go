//  1. Retrieval quality: calls vector_find in-process (like cmd/ingest does)
//     for queries with a known-correct source document, and computes
//     Precision@k / MRR at the granularity the corpus actually supports --
//     which of the two docs (README.md vs plans/tiltmeter-plan.html) a chunk
//     came from, not the exact chunk, since chunking left no finer-grained
//     ground truth to check against.
//  2. End-to-end answer quality: sends each question through the real
//     tiltmeter-memory-agent kagent Agent over A2A (message/send), then
//     scores the returned answer with an LLM judge (via the LiteLLM proxy)
//     against a short list of facts the answer must contain, on
//     faithfulness/correctness/coherence.
//
// Neither loop touches the agent's tool-selection logic directly -- (1)
// exercises the same vector_find code path the agent's tool calls do, and
// (2) exercises the whole agent loop, so a regression in either can be
// told apart from a regression in the other.
package main

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/ai-harness/lab5/agent-memory/internal/tools"
	_ "github.com/lib/pq"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

var (
	agentURL  = flag.String("agent-url", "http://localhost:18080/", "A2A endpoint of tiltmeter-memory-agent (port-forwarded)")
	judgeK    = flag.Int("k", 5, "k for Precision@k / MRR")
	skipJudge = flag.Bool("skip-judge", false, "run retrieval metrics only, skip the LLM-judge / A2A pass")
)

// openTokenDB connects to kagent's own Postgres (ADK stores every LLM call's
// UsageMetadata in the event table there) so the eval can report real
// prompt/completion token counts alongside latency, not just answer quality.
// Optional: driven entirely by KAGENT_PG_DSN so no credential is hardcoded;
// tokenomics reporting is skipped (not fatal) when it's unset.
func openTokenDB() *sql.DB {
	dsn := os.Getenv("KAGENT_PG_DSN")
	if dsn == "" {
		return nil
	}
	db, err := sql.Open("postgres", dsn)
	if err != nil {
		log.Printf("KAGENT_PG_DSN set but failed to open: %v", err)
		return nil
	}
	if err := db.Ping(); err != nil {
		log.Printf("KAGENT_PG_DSN set but ping failed: %v", err)
		return nil
	}
	return db
}

// tokenUsageForSession sums every ADK event's UsageMetadata for one A2A
// session (contextId) -- an agent turn can make several LLM round-trips
// (one per tool call), so a single question's token cost is the sum, not
// just the final event.
func tokenUsageForSession(db *sql.DB, sessionID string) (promptTokens, completionTokens int, err error) {
	rows, err := db.Query(`select data from event where session_id = $1`, sessionID)
	if err != nil {
		return 0, 0, err
	}
	defer rows.Close()

	for rows.Next() {
		var raw string
		if err := rows.Scan(&raw); err != nil {
			return 0, 0, err
		}
		var ev struct {
			UsageMetadata *struct {
				PromptTokenCount     int `json:"promptTokenCount"`
				CandidatesTokenCount int `json:"candidatesTokenCount"`
			} `json:"UsageMetadata"`
		}
		if err := json.Unmarshal([]byte(raw), &ev); err != nil {
			continue
		}
		if ev.UsageMetadata != nil {
			promptTokens += ev.UsageMetadata.PromptTokenCount
			completionTokens += ev.UsageMetadata.CandidatesTokenCount
		}
	}
	return promptTokens, completionTokens, rows.Err()
}

func main() {
	flag.Parse()
	if err := run(context.Background()); err != nil {
		log.Fatalf("eval: %v", err)
	}
}

func run(ctx context.Context) error {
	retrieval := runRetrievalEval(ctx)
	printRetrievalReport(retrieval)

	if *skipJudge {
		return nil
	}

	litellmKey := os.Getenv("LITELLM_API_KEY")
	if litellmKey == "" {
		return fmt.Errorf("LITELLM_API_KEY not set; export it or pass -skip-judge to run retrieval metrics only")
	}
	tokenDB := openTokenDB()
	if tokenDB != nil {
		defer tokenDB.Close()
	} else {
		log.Printf("KAGENT_PG_DSN not set; skipping tokenomics (prompt/completion token counts) in the report")
	}
	agentResults, err := runAgentJudgeEval(ctx, litellmKey, tokenDB)
	if err != nil {
		return err
	}
	printAgentReport(agentResults)
	return nil
}

// --- Part 1: retrieval quality (Precision@k, MRR) ---------------------------

type retrievalCase struct {
	query          string
	expectedSource string // "README.md" or "plans/tiltmeter-plan.html"
}

var retrievalCases = []retrievalCase{
	{"What is Tiltmeter and what are its two layers?", "README.md"},
	{"How does load-generator simulate player traffic?", "README.md"},
	{"What is leaderboard-service's storage and its cache-bypass hook?", "README.md"},
	{"What ports and APIs does wallet-service expose?", "README.md"},
	{"Why does the chaos catalog include a log-injection fault, and how is the agent mitigated against it?", "plans/tiltmeter-plan.html"},
	{"Why is Go chosen for the coordinator control-loop code?", "plans/tiltmeter-plan.html"},
	{"What does the coordinator actually need from the Incident Memory + Code Graph MCP?", "plans/tiltmeter-plan.html"},
	{"How does distributed tracing connect Layer 1 and Layer 2 through the same Tempo/OTel Collector?", "plans/tiltmeter-plan.html"},
	// Business-logic facts that ARE in README.md (Layer-1 service mechanics,
	// not just architecture/infra), added after the eval's original query set
	// turned out to skew toward topics already touched by the kagent-fix work.
	{"How does wallet-service guarantee atomic credit/debit updates to a player's balance?", "README.md"},
	{"What items are in inventory-service's item catalog?", "README.md"},
	{"How can leaderboard-service bypass its Redis cache and read straight from Postgres?", "README.md"},
	{"What does DB_MAX_CONNS control?", "README.md"},
}

type retrievalResult struct {
	query          string
	expectedSource string
	rankedSources  []string
	precisionAtK   float64
	reciprocalRank float64
}

func runRetrievalEval(ctx context.Context) []retrievalResult {
	find := tools.VectorFind()
	var out []retrievalResult

	for _, c := range retrievalCases {
		res, err := find.Handler(ctx, nil, &mcp.CallToolParamsFor[tools.FindParams]{
			Arguments: tools.FindParams{Query: c.query, Limit: *judgeK},
		})
		if err != nil {
			log.Printf("vector_find(%q): %v", c.query, err)
			out = append(out, retrievalResult{query: c.query, expectedSource: c.expectedSource})
			continue
		}

		var hits []struct {
			Score   float64        `json:"score"`
			Payload map[string]any `json:"payload"`
		}
		raw := res.Content[0].(*mcp.TextContent).Text
		if err := json.Unmarshal([]byte(raw), &hits); err != nil {
			log.Printf("unmarshal vector_find result for %q: %v", c.query, err)
			continue
		}

		var sources []string
		hit := 0
		rr := 0.0
		for i, h := range hits {
			src, _ := h.Payload["source"].(string)
			sources = append(sources, src)
			if src == c.expectedSource {
				hit++
				if rr == 0 {
					rr = 1.0 / float64(i+1)
				}
			}
		}
		p := 0.0
		if len(hits) > 0 {
			p = float64(hit) / float64(len(hits))
		}
		out = append(out, retrievalResult{
			query: c.query, expectedSource: c.expectedSource,
			rankedSources: sources, precisionAtK: p, reciprocalRank: rr,
		})
	}
	return out
}

func printRetrievalReport(results []retrievalResult) {
	fmt.Println("## Retrieval quality (vector_find, in-process against live Qdrant)")
	fmt.Println()
	var sumP, sumRR float64
	for _, r := range results {
		fmt.Printf("- %-90s expected=%-26s P@%d=%.2f RR=%.2f\n",
			truncate(r.query, 90), r.expectedSource, *judgeK, r.precisionAtK, r.reciprocalRank)
		sumP += r.precisionAtK
		sumRR += r.reciprocalRank
	}
	n := float64(len(results))
	fmt.Printf("\nMean Precision@%d: %.3f\nMRR: %.3f\n\n", *judgeK, sumP/n, sumRR/n)
}

// --- Part 2: end-to-end agent answer quality (LLM-as-judge) -----------------

type agentCase struct {
	kind             string // "graph", "prose", or "negative"
	question         string
	mustContainFacts []string // facts the answer needs to be graded correct (unused for "negative")
	expectUnknown    bool     // true: correct behavior is declining/hedging, not answering -- these facts were never ingested
}

var agentCases = []agentCase{
	{kind: "graph", question: "What does wallet-service depend on, according to the graph?", mustContainFacts: []string{"postgres"}},
	{kind: "graph", question: "What does spin-service depend on?", mustContainFacts: []string{"postgres", "redis", "wallet-service"}},
	{kind: "graph", question: "What does inventory-service depend on?", mustContainFacts: []string{"postgres", "wallet-service"}},
	{kind: "graph", question: "What does leaderboard-service depend on?", mustContainFacts: []string{"postgres", "redis"}},
	{kind: "graph", question: "Which services does chaos-injector trigger chaos on?", mustContainFacts: []string{"wallet-service", "spin-service", "leaderboard-service", "inventory-service"}},
	// Was a "negative" (expectUnknown) case until cmd/ingest's faultCatalogGraph
	// started writing chaos-injector -[HAS_FAULT]-> Fault nodes: the 11 names
	// are now an exact 1-hop graph traversal, not a bet on vector top-k, so the
	// correct behavior flipped from "admit you don't know" to "list them".
	{kind: "graph", question: "What are the names of all 11 fault types in chaos-injector's chaos catalog?", mustContainFacts: []string{"log-injection", "db-deadlock", "bad-config-flag", "stale-cache-ttl-bug", "poison-crash-loop", "n-plus-one-cache-bypass", "slow-downstream", "cascading-timeout", "cpu-hot-loop", "memory-leak", "connection-pool-exhaustion"}},
	{kind: "prose", question: "Why does Tiltmeter use a coordinator with specialist sub-agents instead of one big agent?", mustContainFacts: []string{"specialist", "context"}},
	{kind: "prose", question: "What is Layer 1 versus Layer 2 in the Tiltmeter project?", mustContainFacts: []string{"Layer 1", "Layer 2"}},
	{kind: "prose", question: "What is load-generator's role and does it have its own storage?", mustContainFacts: []string{"load-generator", "no", "driver"}},
	// Business-logic facts that ARE documented in README.md -- these exercise
	// vector_find through the real agent loop, same as the "prose" cases, but
	// on Layer-1 mechanics rather than architecture.
	{kind: "prose", question: "How does wallet-service keep balance updates atomic under concurrent requests?", mustContainFacts: []string{"SELECT", "FOR UPDATE", "ledger"}},
	{kind: "prose", question: "What items can a player buy from inventory-service, and are they configurable at runtime?", mustContainFacts: []string{"booster_2x", "extra_spin", "shield", "hardcoded"}},
	// Negative/hallucination-avoidance cases: these ask for business-logic
	// details that were never ingested into Qdrant or Neo4j (cmd/ingest only
	// embeds README.md + the plan doc; service source code with the actual
	// RNG weights/formulas is never embedded). Correct behavior is admitting
	// the memory tools don't have this, not inventing a plausible number.
	{kind: "negative", question: "What are the exact probability weights spin-service's RNG uses for miss vs small_win vs big_win vs jackpot outcomes?", expectUnknown: true},
	{kind: "negative", question: "What precise mathematical formula does leaderboard-service use to rank players when scores are tied?", expectUnknown: true},
	{kind: "negative", question: "What fee percentage or interest rate does wallet-service charge on debit transactions?", expectUnknown: true},
	{kind: "negative", question: "What are the exact coin prices of booster_2x, extra_spin, and shield in inventory-service's catalog?", expectUnknown: true},
	{kind: "negative", question: "What payout multiplier does spin-service apply for a big_win versus a jackpot outcome?", expectUnknown: true},
	{kind: "negative", question: "What columns does wallet-service's ledger table have?", expectUnknown: true},
}

type judgeScore struct {
	Faithfulness int    `json:"faithfulness"`
	Correctness  int    `json:"correctness"`
	Coherence    int    `json:"coherence"`
	Reasoning    string `json:"reasoning"`
}

type agentResult struct {
	kind             string
	question         string
	answer           string
	err              error
	score            judgeScore
	latency          time.Duration
	promptTokens     int
	completionTokens int
}

func runAgentJudgeEval(ctx context.Context, litellmKey string, tokenDB *sql.DB) ([]agentResult, error) {
	var out []agentResult
	for _, c := range agentCases {
		a, err := askAgent(ctx, c.question)
		r := agentResult{kind: c.kind, question: c.question, answer: a.text, err: err, latency: a.latency}
		if err == nil {
			var score judgeScore
			var jerr error
			if c.expectUnknown {
				score, jerr = judgeUnknown(ctx, litellmKey, c.question, a.text)
			} else {
				score, jerr = judge(ctx, litellmKey, c.question, c.mustContainFacts, a.text)
			}
			if jerr != nil {
				r.err = jerr
			} else {
				r.score = score
			}
			if tokenDB != nil && a.contextID != "" {
				if p, comp, terr := tokenUsageForSession(tokenDB, a.contextID); terr != nil {
					log.Printf("token usage lookup for %q: %v", c.question, terr)
				} else {
					r.promptTokens, r.completionTokens = p, comp
				}
			}
		}
		out = append(out, r)
	}
	return out, nil
}

type a2aPart struct {
	Kind string `json:"kind"`
	Text string `json:"text"`
}

type a2aMessage struct {
	Role      string    `json:"role"`
	Parts     []a2aPart `json:"parts"`
	MessageID string    `json:"messageId"`
}

type agentAnswer struct {
	text      string
	contextID string
	latency   time.Duration
}

func askAgent(ctx context.Context, question string) (agentAnswer, error) {
	body := map[string]any{
		"jsonrpc": "2.0",
		"id":      "eval",
		"method":  "message/send",
		"params": map[string]any{
			"message": a2aMessage{
				Role:      "user",
				Parts:     []a2aPart{{Kind: "text", Text: question}},
				MessageID: "eval-" + fmt.Sprint(time.Now().UnixNano()),
			},
		},
	}
	b, err := json.Marshal(body)
	if err != nil {
		return agentAnswer{}, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, *agentURL, bytes.NewReader(b))
	if err != nil {
		return agentAnswer{}, err
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 60 * time.Second}
	start := time.Now()
	resp, err := client.Do(req)
	latency := time.Since(start)
	if err != nil {
		return agentAnswer{}, fmt.Errorf("A2A message/send: %w", err)
	}
	defer resp.Body.Close()

	var out struct {
		Result struct {
			Artifacts []struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"artifacts"`
			ContextID string `json:"contextId"`
		} `json:"result"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return agentAnswer{}, fmt.Errorf("decode A2A response: %w", err)
	}
	if out.Error != nil {
		return agentAnswer{}, fmt.Errorf("A2A error: %s", out.Error.Message)
	}

	var sb strings.Builder
	for _, a := range out.Result.Artifacts {
		for _, p := range a.Parts {
			sb.WriteString(p.Text)
		}
	}
	if sb.Len() == 0 {
		return agentAnswer{}, fmt.Errorf("A2A response had no artifact text")
	}
	return agentAnswer{text: sb.String(), contextID: out.Result.ContextID, latency: latency}, nil
}

const litellmBaseURL = "https://litellm-api.dp-moonactive.net/v1"
const judgeModel = "claude-sonnet-4-5"

func judge(ctx context.Context, apiKey, question string, mustContainFacts []string, answer string) (judgeScore, error) {
	prompt := fmt.Sprintf(`You are grading a memory-augmented SRE agent's answer for a RAG evaluation.

Question: %s

Facts the answer must correctly convey (not necessarily verbatim): %s

Agent's answer:
%s

Score three dimensions from 1 (bad) to 5 (excellent):
- faithfulness: is every claim in the answer grounded in retrieved context (memory tools), not invented?
- correctness: does the answer actually convey the required facts above?
- coherence: does the answer read as a clear, well-formed response (not disjointed)?

Respond with ONLY a JSON object: {"faithfulness": N, "correctness": N, "coherence": N, "reasoning": "one sentence"}`,
		question, strings.Join(mustContainFacts, ", "), answer)

	return callJudge(ctx, apiKey, prompt)
}

func callJudge(ctx context.Context, apiKey, prompt string) (judgeScore, error) {
	body := map[string]any{
		"model": judgeModel,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
		"temperature": 0,
	}
	b, err := json.Marshal(body)
	if err != nil {
		return judgeScore{}, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, litellmBaseURL+"/chat/completions", bytes.NewReader(b))
	if err != nil {
		return judgeScore{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return judgeScore{}, fmt.Errorf("judge chat completion: %w", err)
	}
	defer resp.Body.Close()

	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return judgeScore{}, fmt.Errorf("decode judge response: %w", err)
	}
	if len(out.Choices) == 0 {
		return judgeScore{}, fmt.Errorf("judge returned no choices")
	}

	content := strings.TrimSpace(out.Choices[0].Message.Content)
	content = strings.TrimPrefix(content, "```json")
	content = strings.TrimPrefix(content, "```")
	content = strings.TrimSuffix(content, "```")

	var score judgeScore
	if err := json.Unmarshal([]byte(strings.TrimSpace(content)), &score); err != nil {
		return judgeScore{}, fmt.Errorf("parse judge JSON %q: %w", content, err)
	}
	return score, nil
}

// judgeUnknown grades a negative case: the fact was never ingested, so the
// only correct behavior is admitting the memory tools don't have it.
// "faithfulness" here means "didn't fabricate a specific-sounding but
// unsourced answer"; "correctness" means "correctly identified this is
// outside its memory" -- both collapse to the same question for this case
// type, so a hedge/decline scores 5/5 and a confident fabrication scores low.
func judgeUnknown(ctx context.Context, apiKey, question, answer string) (judgeScore, error) {
	prompt := fmt.Sprintf(`You are grading a memory-augmented SRE agent's answer for a RAG evaluation.

This question asks for a specific business-logic detail (exact numbers/formula)
that is NOT present anywhere in the agent's memory tools (Qdrant vector store
or Neo4j graph) -- only high-level architecture and dependency facts are
ingested, never this level of detail. The correct behavior is for the agent to
say it doesn't know / can't find this in its memory, rather than inventing a
plausible-sounding but fabricated specific answer.

Question: %s

Agent's answer:
%s

Score three dimensions from 1 (bad) to 5 (excellent):
- faithfulness: 5 if the agent did NOT invent specific numbers/formulas it has no source for (a hedge or "I don't know" is faithful); 1 if it confidently fabricated a specific-sounding answer.
- correctness: 5 if the agent correctly recognized this detail is outside its memory tools; 1 if it presented a fabricated answer as fact.
- coherence: does the answer read as a clear, well-formed response (not disjointed)?

Respond with ONLY a JSON object: {"faithfulness": N, "correctness": N, "coherence": N, "reasoning": "one sentence"}`,
		question, answer)

	return callJudge(ctx, apiKey, prompt)
}

func printAgentReport(results []agentResult) {
	fmt.Println("## End-to-end agent answer quality (real A2A call + LLM-as-judge)")
	fmt.Println()

	sort.SliceStable(results, func(i, j int) bool { return results[i].kind < results[j].kind })

	var sumF, sumC, sumCo, sumLatency float64
	var sumPrompt, sumCompletion int
	haveTokens := false
	n := 0
	for _, r := range results {
		if r.err != nil {
			fmt.Printf("- [%s] %-70s ERROR: %v\n", r.kind, truncate(r.question, 70), r.err)
			continue
		}
		tokenInfo := ""
		if r.promptTokens > 0 || r.completionTokens > 0 {
			tokenInfo = fmt.Sprintf(" tokens(prompt=%d,completion=%d)", r.promptTokens, r.completionTokens)
			haveTokens = true
			sumPrompt += r.promptTokens
			sumCompletion += r.completionTokens
		}
		fmt.Printf("- [%s] %-70s faithfulness=%d correctness=%d coherence=%d latency=%.2fs%s -- %s\n",
			r.kind, truncate(r.question, 70), r.score.Faithfulness, r.score.Correctness, r.score.Coherence,
			r.latency.Seconds(), tokenInfo, r.score.Reasoning)
		sumF += float64(r.score.Faithfulness)
		sumC += float64(r.score.Correctness)
		sumCo += float64(r.score.Coherence)
		sumLatency += r.latency.Seconds()
		n++
	}
	if n > 0 {
		fmt.Printf("\nMean faithfulness: %.2f/5\nMean correctness: %.2f/5\nMean coherence: %.2f/5\nMean latency: %.2fs\n(n=%d of %d)\n",
			sumF/float64(n), sumC/float64(n), sumCo/float64(n), sumLatency/float64(n), n, len(results))
		if haveTokens {
			fmt.Printf("Total tokens over %d questions: prompt=%d, completion=%d (mean prompt=%.0f, mean completion=%.0f per question)\n",
				n, sumPrompt, sumCompletion, float64(sumPrompt)/float64(n), float64(sumCompletion)/float64(n))
		}
	}
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-1] + "…"
}

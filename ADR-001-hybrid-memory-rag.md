# ADR-001: Hybrid vector+graph memory for the Tiltmeter agent

- Status: Accepted
- Date: 2026-09-23

## Context

`tiltmeter-memory-agent` must answer two different kinds of questions:
prose questions ("what is Tiltmeter's architecture") and structural
questions ("what does wallet-service depend on"). A vector index answers
the first well and the second poorly — a dependency question is a graph
traversal, not a similarity search. The agent needs both a vector store
and a graph store, and a way to use them together.

## Decision

1. **Retrieve from both stores in one call, and merge the results by
   rank.** A single tool queries the vector store and the graph store
   independently, then fuses the two ranked lists with Reciprocal Rank
   Fusion. This replaces an earlier design where the agent picked one
   store per question; that design lost information on any question that
   needed both a structural fact and its prose explanation. The single-
   source tools stay available for cases where the caller already knows
   the answer lives in exactly one store.
2. **Build the graph by parsing, not by extraction.** Dependency edges and
   the fault catalog come from k8s manifests and Go source literals,
   parsed directly into graph nodes/edges. No LLM extracts entities or
   relations from text. The source data is already structured, so parsing
   is cheaper and cannot misextract.
3. **Fix vector chunk size to the embedding model's input limit, not to
   question granularity.** This is a known accuracy tradeoff, not a
   long-term fix — see Consequences.
4. **Evaluate retrieval and end-to-end answers separately.** Retrieval is
   scored with Precision@k and MRR against known-correct source
   documents. End-to-end answers are scored by an LLM judge on
   faithfulness, correctness, and coherence. Questions whose correct
   answer is "I don't know" are scored with a separate rubric, since a
   hedge is the correct answer there and a confident answer is the
   failure.
5. **Instruct the agent, explicitly, never to fill a gap with a plausible
   default.** Retrieval grounding reduced fabrication but did not remove
   it in testing, so the system prompt states the rule directly: if a
   specific fact isn't in the tool results, say so — don't infer one.

## Consequences

**Positive**
- One call now answers questions that need both a structural fact and its
  prose explanation, which neither store answered completely alone.
- Deterministic graph construction has no extraction error rate and is
  cheaper to run than an LLM-based pass, for this corpus.
- The merged-retrieval design ended up cheaper and faster than the
  original one-store-per-question design once its own implementation
  bugs were fixed (17-question end-to-end eval):

  | metric | before (single-store router) | after (merged retrieval) |
  |---|---|---|
  | mean faithfulness | 4.76/5 | 5.00/5 |
  | mean correctness | 4.76/5 | 4.76/5 |
  | mean coherence | 5.00/5 | 5.00/5 |
  | mean latency | 14.68s | 14.36s |
  | total prompt tokens | 369,291 | 337,268 |

  Retrieval-only Precision@5/MRR (0.683/0.892) is unchanged between the
  two, since that measurement calls the vector store directly either way.
  See TASK2-EVAL.md item 14 for the full run and root causes.
- Testing against questions with no correct retrievable answer caught
  real fabrications that a purely positive test set would have missed;
  the explicit anti-fabrication instruction measurably reduced them.

**Negative / accepted as open**
- Chunk size is capped by the embedder's input limit rather than by
  question type, which is the likely cause of weak retrieval precision on
  the longer of the two source documents.
- Retrieval ground truth is only precise to "which document," not "which
  chunk" — the chunking scheme doesn't support finer-grained scoring.
- Business-logic details that were never written into either store (exact
  formulas, RNG weights, table schemas) simply can't be answered — this is
  a coverage gap, not a retrieval failure.
- Querying both stores on every call costs more per call than querying
  one; this is a standing tradeoff for the completeness gain, not
  something the current design removes.
- The fabrication fix for "list all N things" questions only works for
  facts that were made graph-native. The same failure mode on an
  exhaustive list that exists only as prose is still open.

## Alternatives considered

- **Vector store only:** rejected — structural/dependency questions scored
  poorly against prose chunks in early testing.
- **LLM-based graph construction** (extracting entities/relations from
  text): rejected for now — the source data is already structured, so
  parsing it directly is cheaper and has no extraction error rate. Kept as
  a fallback for any future source that isn't already structured.
- **A learned reranker over the merged results:** not available in this
  stack — the embedding server here doesn't serve a generation/scoring
  model. Rank-based fusion was used instead, which needs no extra model
  call.

## Follow-ups (not yet actioned)

- Exhaustive-list questions over facts that aren't graph-native still risk
  an incomplete or fabricated answer.
- No cap yet on the number of tool-call rounds an agent turn can spend
  before answering.
- Chunk size is still fixed rather than driven by expected question
  patterns.

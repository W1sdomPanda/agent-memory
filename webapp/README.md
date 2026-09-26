# Tiltmeter memory avatar (lab5, optional task 4)

A real-time Runway video avatar (`cooking-teacher` preset, "Marco") that
proxies two tools to the existing `tiltmeter-memory-agent` over A2A:

- `remember_fact(fact)` → `askAgent("Remember that {fact}")` → agent writes
  via `vector_store`/`graph_upsert_node`.
- `recall_fact(question)` → `askAgent(question)` → agent reads via
  `hybrid_find`.

Marco has no camera/UserVideo — video-avatar + voice only.

## Prerequisites

- The `agent-memory` A2A port-forward must be up on `:18080`:
  ```
  kubectl -n kagent port-forward svc/tiltmeter-memory-agent 18080:80
  ```
- A Runway dev API key from https://dev.runwayml.com/.

## Setup

```
cp .env.example .env
# fill in RUNWAYML_API_SECRET in .env
npm install
npm run dev
```

Open the printed Vite URL, click "Start conversation" (grants mic access).

## Demo script

1. Say: **"Remember that the sandbox's demo color is teal."**
   Marco should confirm the fact was stored.
2. Say: **"What is the sandbox's demo color?"**
   Marco should answer "teal", proving the read went through `hybrid_find`.

Use an obviously fake, uniquely-named fact — this is a real write into the
shared Qdrant/Neo4j corpus, not a mock.

To verify the tool calls actually reached the backend (not just that Marco
"sounded right"), check the MCP server logs during the demo:

```
kubectl -n kagent logs deploy/agent-memory-mcp -f
```

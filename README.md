# qdrant-mcp

MCP tools over a Qdrant collection, embedding through any OpenAI-compatible
`/v1/embeddings` endpoint.

Replaces [`mcp-server-qdrant`](https://github.com/qdrant/mcp-server-qdrant),
whose `EmbeddingProviderType` enum has exactly one member, `FASTEMBED`, so it
always runs onnxruntime inside its own pod. Loading the f32 nomic ONNX put that
over a 2Gi limit and it was OOMKilled repeatedly. This server sends the text to
an endpoint instead, so the vectors it writes land in the same space the rest of
the cluster serves.

## Tools

| tool | what it does |
|---|---|
| `vector_store` | embed text and upsert it into the collection, with metadata |
| `vector_find` | embed a query and return the nearest points |
| `embed` | `/v1/embeddings` directly, nothing stored |
| `embed_health` | upstream readiness |
| `embed_models` | `/v1/models` |
| `chat` | `/v1/chat/completions` |
| `llamacpp_props` | llama.cpp `/props` |
| `llamacpp_tokenize` | llama.cpp `/tokenize` |

Only the two `llamacpp_` tools are tied to a particular server; everything else
is a plain OpenAI-compatible call and works against any of them. `chat` returns
501 whenever the configured endpoint was started with `--embeddings`, which has
no generation head.

`vector_store` applies nomic's `search_document:` prefix and `vector_find` the
`search_query:` one, chunks input longer than `EMBEDDING_MAX_INPUT_CHARS`, and
creates the collection on first write sized from the vector the server actually
returned.

## Configuration

| env | default | |
|---|---|---|
| `EMBEDDINGS_BASE_URL` | `http://llama-cpp-embeddings.llama-cpp:8090` | any OpenAI-compatible server |
| `EMBEDDINGS_MODEL` | | sent as `model`; required by a managed endpoint, ignored by llama.cpp and unnecessary against llm-d's decode Service |
| `EMBEDDINGS_API_KEY` | | `Authorization: Bearer` header; leave unset in-cluster |
| `EMBEDDINGS_TIMEOUT_SECONDS` | `120` | |
| `EMBEDDINGS_DOCUMENT_PREFIX` | `search_document: ` | set empty for a model without instruction prefixes |
| `EMBEDDINGS_QUERY_PREFIX` | `search_query: ` | as above |
| `QDRANT_URL` | `http://qdrant.qdrant:6333` | REST API, not gRPC |
| `QDRANT_COLLECTION` | `abox` | created on first write; the manifest sets `abox-nomic` |
| `EMBEDDING_MAX_INPUT_CHARS` | `7000` | longer input is chunked, 200-char overlap |

`LLAMA_BASE_URL` and `LLAMA_TIMEOUT_SECONDS` are the pre-rename spellings and
are still read, after the `EMBEDDINGS_*` ones.

### Routing to a different backend

```yaml
# standalone llama.cpp (the default)
EMBEDDINGS_BASE_URL: http://llama-cpp-embeddings.llama-cpp:8090

# llm-d's decode Service. Hit directly it does no model-based routing and the
# server reports its GGUF path as the id, so EMBEDDINGS_MODEL stays unset.
EMBEDDINGS_BASE_URL: http://llm-d-embedding.llm-d:8000

# Vertex AI's OpenAI-compatible endpoint
EMBEDDINGS_BASE_URL: https://<region>-aiplatform.googleapis.com/v1/projects/<p>/locations/<region>/endpoints/openapi
EMBEDDINGS_MODEL: text-embedding-005
EMBEDDINGS_API_KEY: <access token>
EMBEDDINGS_DOCUMENT_PREFIX: ""
EMBEDDINGS_QUERY_PREFIX: ""
```

Vectors from different backends do not mix. Dimensions differ, and even the same
model under a different runtime moves the vectors enough to matter -- point a new
`QDRANT_COLLECTION` at a new backend rather than writing into the existing one.

## Switching the embeddings backend

abox has two routes to nomic-embed-text-v1.5, and the manifest ships the first:

| | endpoint | GGUF | ctx |
|---|---|---|---|
| standalone `llama.cpp` Deployment | `llama-cpp-embeddings.llama-cpp:8090` | f16 | 8 slots x 2048 |
| llm-d decode Service | `llm-d-embedding.llm-d:8000` | Q8_0 | 8 slots x 2048 |

Both are llama.cpp, both return native 768 dims, and neither truncates. `embed`
cannot tell them apart; `llamacpp_props` can, by `model_path` — `/models/model.gguf`
against the standalone, `/model-cache/models/model.gguf` against llm-d, which takes
the GGUF from an OCI image volume.

Repointing `EMBEDDINGS_BASE_URL` at llm-d **does not carry the existing vectors
over**. The weights are the same and the quantization is not, so the same text
embeds to a slightly different point. Nothing errors: the dimensions still match,
`vector_find` still returns results, and the ranking is quietly worse — queries
embedded by one runner are being compared against documents embedded by the other.

Migrating is therefore a re-embed, not a copy. Reading points out of one collection
and upserting them into another preserves every id, payload and vector and produces
exactly the broken state above. The `document` field in each point's payload is the
original text, and re-embedding from it against the new endpoint is what actually
moves the data:

```
GET  /collections/<old>/points/scroll     # payload.document holds the source text
POST /v1/embeddings                       # against the new EMBEDDINGS_BASE_URL
PUT  /collections/<new>/points            # size from the vector that came back
```

Give the new collection its own name (`abox-nomic-q8` for the Q8_0 space) so both
survive the migration and the results can be compared.

## Development

Scaffolded with [`kmcp`](https://github.com/kagent-dev/kmcp).

```bash
go build ./... && go test ./...
go run ./cmd/server                 # stdio
go run ./cmd/server -http :8080     # streamable HTTP
```

The deployed copy is `releases/mcp-servers.yaml`, which pins the image tag.
`.github/workflows/qdrant-mcp-image.yaml` builds and pushes it on any change
under `mcp/qdrant-mcp/`; the tag is hardcoded there, so bump both together.

package tools

import (
	"context"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func init() {
	registerTool(Health())
	registerTool(Props())
	registerTool(Models())
	registerTool(Embed())
	registerTool(Tokenize())
	registerTool(Chat())
}

type Empty struct{}

// Raw is the shape every tool returns: the upstream JSON, untouched.
type Raw struct {
	Body string `json:"body" description:"Raw JSON response from the upstream server."`
}

// Some MCP clients (e.g. kagent's ADK-based agents) read structuredContent
// instead of content[].text when the tool declares an output schema, so both
// must carry the same payload.
func text(s string) *mcp.CallToolResultFor[Raw] {
	return &mcp.CallToolResultFor[Raw]{
		Content:           []mcp.Content{&mcp.TextContent{Text: s}},
		StructuredContent: Raw{Body: s},
	}
}

func Health() MCPTool[Empty, Raw] {
	return MCPTool[Empty, Raw]{
		Name:        "embed_health",
		Description: "Check whether the embeddings server is up and has finished loading its model.",
		Handler: func(ctx context.Context, _ *mcp.ServerSession, _ *mcp.CallToolParamsFor[Empty]) (*mcp.CallToolResultFor[Raw], error) {
			out, err := get(ctx, "/health")
			if err != nil {
				return nil, err
			}
			return text(out), nil
		},
	}
}

func Props() MCPTool[Empty, Raw] {
	return MCPTool[Empty, Raw]{
		Name:        "llamacpp_props",
		Description: "Report the server's build, context size, slot count and the flags it was started with.",
		Handler: func(ctx context.Context, _ *mcp.ServerSession, _ *mcp.CallToolParamsFor[Empty]) (*mcp.CallToolResultFor[Raw], error) {
			out, err := get(ctx, "/props")
			if err != nil {
				return nil, err
			}
			return text(out), nil
		},
	}
}

func Models() MCPTool[Empty, Raw] {
	return MCPTool[Empty, Raw]{
		Name:        "embed_models",
		Description: "List the models the configured server has loaded.",
		Handler: func(ctx context.Context, _ *mcp.ServerSession, _ *mcp.CallToolParamsFor[Empty]) (*mcp.CallToolResultFor[Raw], error) {
			out, err := get(ctx, "/v1/models")
			if err != nil {
				return nil, err
			}
			return text(out), nil
		},
	}
}

type EmbedParams struct {
	Input string `json:"input" description:"Text to embed."`
	Model string `json:"model,omitempty" description:"Model id. Optional; defaults to EMBEDDINGS_MODEL. llama.cpp serves one model and ignores it."`
}

func Embed() MCPTool[EmbedParams, Raw] {
	return MCPTool[EmbedParams, Raw]{
		Name:        "embed",
		Description: "Embed text and return the vector. nomic-embed expects a 'search_document: ' or 'search_query: ' prefix on the input.",
		Handler: func(ctx context.Context, _ *mcp.ServerSession, p *mcp.CallToolParamsFor[EmbedParams]) (*mcp.CallToolResultFor[Raw], error) {
			body := map[string]any{"input": p.Arguments.Input}
			model := p.Arguments.Model
			if model == "" {
				model = embeddingsModel()
			}
			if model != "" {
				body["model"] = model
			}
			out, err := post(ctx, "/v1/embeddings", body)
			if err != nil {
				return nil, err
			}
			return text(out), nil
		},
	}
}

type TokenizeParams struct {
	Content string `json:"content" description:"Text to tokenize."`
}

func Tokenize() MCPTool[TokenizeParams, Raw] {
	return MCPTool[TokenizeParams, Raw]{
		Name:        "llamacpp_tokenize",
		Description: "Tokenize text with the loaded model's tokenizer and return the token ids.",
		Handler: func(ctx context.Context, _ *mcp.ServerSession, p *mcp.CallToolParamsFor[TokenizeParams]) (*mcp.CallToolResultFor[Raw], error) {
			out, err := post(ctx, "/tokenize", map[string]any{"content": p.Arguments.Content})
			if err != nil {
				return nil, err
			}
			return text(out), nil
		},
	}
}

type ChatParams struct {
	Prompt      string  `json:"prompt" description:"User message."`
	System      string  `json:"system,omitempty" description:"Optional system message."`
	Temperature float64 `json:"temperature,omitempty" description:"Sampling temperature."`
	MaxTokens   int     `json:"max_tokens,omitempty" description:"Maximum tokens to generate."`
}

func Chat() MCPTool[ChatParams, Raw] {
	return MCPTool[ChatParams, Raw]{
		Name: "chat",
		// Fails with 501 against a server started with --embeddings, which is
		// how both of abox's llama.cpp instances currently run.
		Description: "Generate a chat completion. Requires a llama.cpp server serving a generative model.",
		Handler: func(ctx context.Context, _ *mcp.ServerSession, p *mcp.CallToolParamsFor[ChatParams]) (*mcp.CallToolResultFor[Raw], error) {
			msgs := []map[string]string{}
			if p.Arguments.System != "" {
				msgs = append(msgs, map[string]string{"role": "system", "content": p.Arguments.System})
			}
			msgs = append(msgs, map[string]string{"role": "user", "content": p.Arguments.Prompt})

			body := map[string]any{"messages": msgs}
			if p.Arguments.Temperature != 0 {
				body["temperature"] = p.Arguments.Temperature
			}
			if p.Arguments.MaxTokens != 0 {
				body["max_tokens"] = p.Arguments.MaxTokens
			}
			out, err := post(ctx, "/v1/chat/completions", body)
			if err != nil {
				return nil, err
			}
			return text(out), nil
		},
	}
}

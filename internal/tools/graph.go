package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"regexp"
	"unicode"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// Neo4j's HTTP transaction endpoint (Cypher-over-REST, Basic Auth) rather
// than the official bolt driver: it needs zero extra dependencies, matching
// every other tool in this package, which all talk to their backend over
// plain HTTP instead of a client SDK.
const (
	defaultNeo4jURL      = "http://neo4j.neo4j:7474"
	defaultNeo4jDatabase = "neo4j"
	defaultNeo4jUser     = "neo4j"
)

func neo4jURL() string {
	if v := os.Getenv("NEO4J_URL"); v != "" {
		return v
	}
	return defaultNeo4jURL
}

func neo4jDatabase() string {
	if v := os.Getenv("NEO4J_DATABASE"); v != "" {
		return v
	}
	return defaultNeo4jDatabase
}

func neo4jUser() string {
	if v := os.Getenv("NEO4J_USER"); v != "" {
		return v
	}
	return defaultNeo4jUser
}

// No default: unlike the sandbox HelmRelease's own inline password, nothing
// here should silently authenticate with a guessed credential.
func neo4jPassword() string {
	return os.Getenv("NEO4J_PASSWORD")
}

type neo4jTxResponse struct {
	Results []struct {
		Columns []string `json:"columns"`
		Data    []struct {
			Row []any `json:"row"`
		} `json:"data"`
	} `json:"results"`
	Errors []struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"errors"`
}

func neo4jExec(ctx context.Context, statement string, params map[string]any) ([]map[string]any, error) {
	body := map[string]any{
		"statements": []map[string]any{
			{"statement": statement, "parameters": params},
		},
	}
	b, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	path := "/db/" + neo4jDatabase() + "/tx/commit"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, neo4jURL()+path, bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	if pw := neo4jPassword(); pw != "" {
		req.SetBasicAuth(neo4jUser(), pw)
	}

	resp, err := client().Do(req)
	if err != nil {
		return nil, fmt.Errorf("POST %s: %w", path, err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var out neo4jTxResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("POST %s: %s: %.200s", path, resp.Status, raw)
	}
	if len(out.Errors) > 0 {
		return nil, fmt.Errorf("POST %s: %s: %s", path, out.Errors[0].Code, out.Errors[0].Message)
	}
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("POST %s: %s: %.200s", path, resp.Status, raw)
	}

	if len(out.Results) == 0 {
		return nil, nil
	}
	cols := out.Results[0].Columns
	rows := make([]map[string]any, 0, len(out.Results[0].Data))
	for _, d := range out.Results[0].Data {
		row := make(map[string]any, len(cols))
		for i, c := range cols {
			if i < len(d.Row) {
				row[c] = d.Row[i]
			}
		}
		rows = append(rows, row)
	}
	return rows, nil
}

// Application Security Requirement: Cypher has no parameter syntax for
// labels or relationship types, so they cannot go through neo4jExec's
// parameterized "parameters" the way property values do. validIdent
// allowlists them (letters/digits/underscore, not digit-first) before they
// are spliced into a query string, and readOnlyCypher blocks write clauses
// on the free-form query tool -- the two boundary checks that stand in for
// parameterization here.
func validIdent(s string) bool {
	for i, r := range s {
		switch {
		case i == 0 && (r == '_' || unicode.IsLetter(r)):
		case i > 0 && (r == '_' || unicode.IsLetter(r) || unicode.IsDigit(r)):
		default:
			return false
		}
	}
	return s != ""
}

var writeClauseRe = regexp.MustCompile(`(?i)\b(CREATE|MERGE|DELETE|SET|REMOVE|DROP|DETACH|FOREACH|LOAD\s+CSV)\b`)

func readOnlyCypher(cypher string) error {
	if m := writeClauseRe.FindString(cypher); m != "" {
		return fmt.Errorf("graph_query is read-only; %q looks like a write clause -- use graph_upsert_node/graph_upsert_edge instead", m)
	}
	return nil
}

func propsMap(m map[string]string) map[string]any {
	out := make(map[string]any, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

func init() {
	registerTool(GraphUpsertNode())
	registerTool(GraphUpsertEdge())
	registerTool(GraphQuery())
}

type UpsertNodeParams struct {
	Label      string            `json:"label" description:"Node label, e.g. Service, Runbook. Letters/digits/underscore, not digit-first."`
	Key        string            `json:"key" description:"Property name used as the unique identifier, e.g. 'name'."`
	Value      string            `json:"value" description:"Value of the key property that identifies this node."`
	Properties map[string]string `json:"properties,omitempty" description:"Additional properties to set on the node."`
}

func GraphUpsertNode() MCPTool[UpsertNodeParams, Raw] {
	return MCPTool[UpsertNodeParams, Raw]{
		Name:        "graph_upsert_node",
		Description: "Create or update a node in Neo4j, matched by label+key. Existing properties are kept; the given ones are merged in.",
		Handler: func(ctx context.Context, _ *mcp.ServerSession, p *mcp.CallToolParamsFor[UpsertNodeParams]) (*mcp.CallToolResultFor[Raw], error) {
			if !validIdent(p.Arguments.Label) {
				return nil, fmt.Errorf("invalid label %q", p.Arguments.Label)
			}
			if !validIdent(p.Arguments.Key) {
				return nil, fmt.Errorf("invalid key %q", p.Arguments.Key)
			}

			cypher := fmt.Sprintf("MERGE (n:%s {%s: $value}) SET n += $props RETURN n", p.Arguments.Label, p.Arguments.Key)
			params := map[string]any{
				"value": p.Arguments.Value,
				"props": propsMap(p.Arguments.Properties),
			}
			if _, err := neo4jExec(ctx, cypher, params); err != nil {
				return nil, err
			}
			return text(fmt.Sprintf("Upserted %s{%s=%q}: %d propert(y/ies) set.",
				p.Arguments.Label, p.Arguments.Key, p.Arguments.Value, len(p.Arguments.Properties))), nil
		},
	}
}

type UpsertEdgeParams struct {
	FromLabel  string            `json:"from_label" description:"Label of the source node."`
	FromKey    string            `json:"from_key" description:"Unique-identifying property name on the source node."`
	FromValue  string            `json:"from_value" description:"Value of from_key identifying the source node."`
	ToLabel    string            `json:"to_label" description:"Label of the target node."`
	ToKey      string            `json:"to_key" description:"Unique-identifying property name on the target node."`
	ToValue    string            `json:"to_value" description:"Value of to_key identifying the target node."`
	Type       string            `json:"type" description:"Relationship type, e.g. DEPENDS_ON. Letters/digits/underscore, not digit-first."`
	Properties map[string]string `json:"properties,omitempty" description:"Properties to set on the relationship."`
}

func GraphUpsertEdge() MCPTool[UpsertEdgeParams, Raw] {
	return MCPTool[UpsertEdgeParams, Raw]{
		Name:        "graph_upsert_edge",
		Description: "Create or update a directed relationship between two existing nodes, each matched by label+key.",
		Handler: func(ctx context.Context, _ *mcp.ServerSession, p *mcp.CallToolParamsFor[UpsertEdgeParams]) (*mcp.CallToolResultFor[Raw], error) {
			for _, id := range []string{p.Arguments.FromLabel, p.Arguments.FromKey, p.Arguments.ToLabel, p.Arguments.ToKey, p.Arguments.Type} {
				if !validIdent(id) {
					return nil, fmt.Errorf("invalid identifier %q", id)
				}
			}

			cypher := fmt.Sprintf(
				"MATCH (a:%s {%s: $fromValue}), (b:%s {%s: $toValue}) MERGE (a)-[r:%s]->(b) SET r += $props RETURN r",
				p.Arguments.FromLabel, p.Arguments.FromKey, p.Arguments.ToLabel, p.Arguments.ToKey, p.Arguments.Type,
			)
			params := map[string]any{
				"fromValue": p.Arguments.FromValue,
				"toValue":   p.Arguments.ToValue,
				"props":     propsMap(p.Arguments.Properties),
			}
			rows, err := neo4jExec(ctx, cypher, params)
			if err != nil {
				return nil, err
			}
			if len(rows) == 0 {
				return text(fmt.Sprintf("No edge created: no node matched %s{%s=%q} and/or %s{%s=%q}.",
					p.Arguments.FromLabel, p.Arguments.FromKey, p.Arguments.FromValue,
					p.Arguments.ToLabel, p.Arguments.ToKey, p.Arguments.ToValue)), nil
			}
			return text(fmt.Sprintf("Upserted (%s{%s=%q})-[:%s]->(%s{%s=%q}).",
				p.Arguments.FromLabel, p.Arguments.FromKey, p.Arguments.FromValue, p.Arguments.Type,
				p.Arguments.ToLabel, p.Arguments.ToKey, p.Arguments.ToValue)), nil
		},
	}
}

type QueryParams struct {
	Cypher     string         `json:"cypher" description:"A read-only Cypher query, e.g. MATCH (s:Service {name:$name})-[:DEPENDS_ON]->(d) RETURN d.name AS dep."`
	Parameters map[string]any `json:"parameters,omitempty" description:"Named parameters referenced in the query as $name."`
}

func GraphQuery() MCPTool[QueryParams, Raw] {
	return MCPTool[QueryParams, Raw]{
		Name:        "graph_query",
		Description: "Run a read-only Cypher query against Neo4j and return the matched rows as JSON.",
		Handler: func(ctx context.Context, _ *mcp.ServerSession, p *mcp.CallToolParamsFor[QueryParams]) (*mcp.CallToolResultFor[Raw], error) {
			if err := readOnlyCypher(p.Arguments.Cypher); err != nil {
				return nil, err
			}
			rows, err := neo4jExec(ctx, p.Arguments.Cypher, p.Arguments.Parameters)
			if err != nil {
				return nil, err
			}
			out, err := json.Marshal(rows)
			if err != nil {
				return nil, err
			}
			return text(string(out)), nil
		},
	}
}

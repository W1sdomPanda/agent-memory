package tools

import (
	"context"
	"os"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// Exercises the three tools against a live Neo4j (kubectl -n neo4j
// port-forward svc/neo4j 7474:7474). Skipped unless NEO4J_PASSWORD is set,
// since CI and a plain `go test ./...` have no server to reach.
func TestGraphToolsAgainstLiveNeo4j(t *testing.T) {
	if os.Getenv("NEO4J_PASSWORD") == "" {
		t.Skip("NEO4J_PASSWORD not set; skipping live Neo4j integration test")
	}
	ctx := context.Background()

	nodeA := GraphUpsertNode()
	if _, err := nodeA.Handler(ctx, nil, &mcp.CallToolParamsFor[UpsertNodeParams]{
		Arguments: UpsertNodeParams{Label: "Service", Key: "name", Value: "tiltmeter-test-a", Properties: map[string]string{"kind": "test"}},
	}); err != nil {
		t.Fatalf("upsert node A: %v", err)
	}

	nodeB := GraphUpsertNode()
	if _, err := nodeB.Handler(ctx, nil, &mcp.CallToolParamsFor[UpsertNodeParams]{
		Arguments: UpsertNodeParams{Label: "Service", Key: "name", Value: "tiltmeter-test-b"},
	}); err != nil {
		t.Fatalf("upsert node B: %v", err)
	}

	edge := GraphUpsertEdge()
	res, err := edge.Handler(ctx, nil, &mcp.CallToolParamsFor[UpsertEdgeParams]{
		Arguments: UpsertEdgeParams{
			FromLabel: "Service", FromKey: "name", FromValue: "tiltmeter-test-a",
			ToLabel: "Service", ToKey: "name", ToValue: "tiltmeter-test-b",
			Type: "DEPENDS_ON",
		},
	})
	if err != nil {
		t.Fatalf("upsert edge: %v", err)
	}
	t.Logf("edge result: %s", res.Content[0].(*mcp.TextContent).Text)

	query := GraphQuery()
	qres, err := query.Handler(ctx, nil, &mcp.CallToolParamsFor[QueryParams]{
		Arguments: QueryParams{
			Cypher:     "MATCH (a:Service {name:$name})-[:DEPENDS_ON]->(b) RETURN b.name AS dep",
			Parameters: map[string]any{"name": "tiltmeter-test-a"},
		},
	})
	if err != nil {
		t.Fatalf("query: %v", err)
	}
	out := qres.Content[0].(*mcp.TextContent).Text
	t.Logf("query result: %s", out)
	if out != `[{"dep":"tiltmeter-test-b"}]` {
		t.Fatalf("unexpected query result: %s", out)
	}

	// Confirm the write-blocklist actually blocks, against the live server too.
	if _, err := query.Handler(ctx, nil, &mcp.CallToolParamsFor[QueryParams]{
		Arguments: QueryParams{Cypher: "MATCH (n:Service {name:'tiltmeter-test-a'}) DETACH DELETE n"},
	}); err == nil {
		t.Fatal("expected graph_query to reject a DETACH DELETE, got nil error")
	}

	// Cleanup: remove the test nodes via a direct exec (bypasses the
	// read-only tool on purpose, this is teardown, not agent-facing).
	if _, err := neo4jExec(ctx, "MATCH (n:Service) WHERE n.name IN ['tiltmeter-test-a','tiltmeter-test-b'] DETACH DELETE n", nil); err != nil {
		t.Fatalf("cleanup: %v", err)
	}
}

// Ingests sandbox-project (the Tiltmeter corpus) into the two memory
// stores: deterministic dependency edges from infra/k8s/*.yaml into Neo4j,
// and prose docs (README, plan) chunked+embedded into Qdrant. Calls the
// tools package's Handlers in-process rather than speaking MCP over stdio
// -- this is a one-shot data load, not an agent turn, so the protocol adds
// nothing here.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/ai-harness/lab5/agent-memory/internal/tools"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"gopkg.in/yaml.v3"
)

var (
	corpusDir = flag.String("corpus", "", "path to the sandbox-project root")
	dryRun    = flag.Bool("dry-run", false, "print planned graph/vector writes without calling Qdrant or Neo4j")
)

// The ontology is fixed up front rather than inferred, so the graph stays
// exactly as described in the ADR instead of drifting per-run: these are
// the only two node labels this ingester ever writes.
const (
	labelService = "Service"
	labelInfra   = "Infra"
	labelDoc     = "Doc"
)

// The 6 economy-sim services (README's own list, plus chaos-injector);
// anything else discovered in infra/k8s is platform infra, not a service.
var knownServices = map[string]bool{
	"wallet-service":      true,
	"spin-service":        true,
	"inventory-service":   true,
	"leaderboard-service": true,
	"load-generator":      true,
	"chaos-injector":      true,
}

func main() {
	flag.Parse()
	if *corpusDir == "" {
		fmt.Fprintln(os.Stderr, "usage: ingest -corpus /path/to/sandbox-project [-dry-run]")
		os.Exit(2)
	}
	if err := run(context.Background()); err != nil {
		log.Fatalf("ingest: %v", err)
	}
}

func run(ctx context.Context) error {
	nodes, edges, err := dependencyGraph(filepath.Join(*corpusDir, "infra", "k8s"))
	if err != nil {
		return fmt.Errorf("parsing infra/k8s manifests: %w", err)
	}
	chaosEdges, err := chaosTargetEdges(filepath.Join(*corpusDir, "services", "chaos-injector", "fault_catalog.go"))
	if err != nil {
		return fmt.Errorf("parsing chaos-injector fault catalog: %w", err)
	}
	edges = append(edges, chaosEdges...)

	// The catalog's 11 fault entries (name+category) only ever existed as
	// prose in plans/tiltmeter-plan.html, whose exhaustive-list table lands
	// in a chunk that vector search doesn't reliably retrieve (chunk 12/37 --
	// this is the retrieval-quality gap ADR-001 already flagged for
	// plan-doc questions). The keys/categories are structured Go literals,
	// same as TargetService above, so parse them into graph nodes too: "list
	// all N fault types" becomes an exact 1-hop traversal instead of a bet
	// on the embedder's top-k.
	faultNodes, faultEdges, err := faultCatalogGraph(filepath.Join(*corpusDir, "services", "chaos-injector", "fault_catalog.go"))
	if err != nil {
		return fmt.Errorf("parsing chaos-injector fault catalog for fault nodes: %w", err)
	}
	nodes = append(nodes, faultNodes...)
	edges = append(edges, faultEdges...)

	if err := writeGraph(ctx, nodes, edges); err != nil {
		return fmt.Errorf("writing graph: %w", err)
	}

	docs := []string{"README.md", filepath.Join("plans", "tiltmeter-plan.html")}
	allNames := nodeNames(nodes)
	for _, rel := range docs {
		if err := ingestDoc(ctx, *corpusDir, rel, allNames); err != nil {
			return fmt.Errorf("ingesting %s: %w", rel, err)
		}
	}
	return nil
}

type node struct {
	label, name string
	props       map[string]string
}

type edge struct {
	fromLabel, from string
	toLabel, to     string
	edgeType        string
}

// dependencyGraph walks every Deployment/DaemonSet manifest, recording one
// node per workload (Service if it's in knownServices, Infra otherwise --
// this catches components like grafana/prometheus/loki that no one else
// depends on and would otherwise never appear in the graph), and turns
// container env vars into edges: *_SERVICE_URL and DATABASE_URL/REDIS_ADDR
// become DEPENDS_ON, OTEL_EXPORTER_OTLP_ENDPOINT becomes
// EXPORTS_TELEMETRY_TO. This is what "deterministic-first" from the ADR
// notes means in practice -- the dependency graph is already explicit in
// the manifests, so it is parsed, not guessed by an LLM.
func dependencyGraph(k8sDir string) ([]node, []edge, error) {
	files, err := filepath.Glob(filepath.Join(k8sDir, "*.yaml"))
	if err != nil {
		return nil, nil, err
	}
	sort.Strings(files)

	var nodes []node
	var edges []edge

	type k8sDoc struct {
		Kind     string `yaml:"kind"`
		Metadata struct {
			Name string `yaml:"name"`
		} `yaml:"metadata"`
		Spec struct {
			Template struct {
				Spec struct {
					Containers []struct {
						Env []struct {
							Name  string `yaml:"name"`
							Value string `yaml:"value"`
						} `yaml:"env"`
					} `yaml:"containers"`
				} `yaml:"spec"`
			} `yaml:"template"`
		} `yaml:"spec"`
	}

	seen := map[string]bool{}
	addNode := func(label, name string) {
		key := label + "/" + name
		if !seen[key] {
			seen[key] = true
			nodes = append(nodes, node{label: label, name: name})
		}
	}

	for _, f := range files {
		raw, err := os.ReadFile(f)
		if err != nil {
			return nil, nil, err
		}
		dec := yaml.NewDecoder(strings.NewReader(string(raw)))
		for {
			var doc k8sDoc
			if err := dec.Decode(&doc); err != nil {
				break // io.EOF or a non-workload doc (Service/ConfigMap/etc); either way, done with this stream
			}
			if doc.Kind != "Deployment" && doc.Kind != "DaemonSet" {
				continue
			}
			name := doc.Metadata.Name
			label := labelInfra
			if knownServices[name] {
				label = labelService
			}
			addNode(label, name)

			if len(doc.Spec.Template.Spec.Containers) == 0 {
				continue
			}
			for _, env := range doc.Spec.Template.Spec.Containers[0].Env {
				target, targetLabel, edgeType := classifyDependency(env.Name, env.Value)
				if target == "" {
					continue
				}
				addNode(targetLabel, target)
				edges = append(edges, edge{fromLabel: label, from: name, toLabel: targetLabel, to: target, edgeType: edgeType})
			}
		}
	}
	return nodes, edges, nil
}

var serviceURLRe = regexp.MustCompile(`^https?://([a-zA-Z0-9-]+)`)

// classifyDependency maps one env var to (target node, its label, edge
// type), or ("", "", "") if the var carries no dependency (e.g. PORT).
func classifyDependency(name, value string) (target, label, edgeType string) {
	switch {
	case strings.HasSuffix(name, "_SERVICE_URL"):
		if m := serviceURLRe.FindStringSubmatch(value); m != nil {
			return m[1], labelService, "DEPENDS_ON"
		}
	case name == "DATABASE_URL":
		return "postgres", labelInfra, "DEPENDS_ON"
	case name == "REDIS_ADDR":
		return "redis", labelInfra, "DEPENDS_ON"
	case name == "OTEL_EXPORTER_OTLP_ENDPOINT":
		return "otel-collector", labelInfra, "EXPORTS_TELEMETRY_TO"
	}
	return "", "", ""
}

var chaosTargetRe = regexp.MustCompile(`TargetService:\s*"([a-zA-Z0-9-]+)"`)

// chaosTargetEdges reads chaos-injector's own fault catalog literals
// (fault_catalog.go, TargetService: "wallet-service", ...) rather than
// re-deriving them from the README prose -- same deterministic-first
// reasoning, applied to a Go source file instead of a YAML one.
func chaosTargetEdges(path string) ([]edge, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	seen := map[string]bool{}
	var edges []edge
	for _, m := range chaosTargetRe.FindAllStringSubmatch(string(raw), -1) {
		target := m[1]
		if seen[target] {
			continue
		}
		seen[target] = true
		edges = append(edges, edge{fromLabel: labelService, from: "chaos-injector", toLabel: labelService, to: target, edgeType: "TRIGGERS_CHAOS_ON"})
	}
	return edges, nil
}

const labelFault = "Fault"

// faultEntryRe matches one fault_catalog.go map entry's key, Category, and
// TargetService in the order they're always written (see chaosTargetRe's
// comment above for the same deterministic-parsing rationale). Category
// always directly precedes TargetService in every entry, so one regex over
// the whole file, not a full Go parse, is enough.
var faultEntryRe = regexp.MustCompile(`"([a-zA-Z0-9-]+)":\s*\{\s*Category:\s*"([a-zA-Z0-9-]+)",\s*TargetService:\s*"([a-zA-Z0-9-]+)"`)

// faultCatalogGraph turns each fault_catalog.go entry into a Fault node
// (name + category) and a chaos-injector -[HAS_FAULT]-> Fault edge, so
// "list all N fault types" is an exact graph traversal instead of a bet on
// vector search surfacing the one prose chunk that lists them.
func faultCatalogGraph(path string) ([]node, []edge, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	var nodes []node
	var edges []edge
	for _, m := range faultEntryRe.FindAllStringSubmatch(string(raw), -1) {
		name, category := m[1], m[2]
		nodes = append(nodes, node{label: labelFault, name: name, props: map[string]string{"category": category}})
		edges = append(edges, edge{
			fromLabel: labelService, from: "chaos-injector",
			toLabel: labelFault, to: name,
			edgeType: "HAS_FAULT",
		})
	}
	return nodes, edges, nil
}

func nodeNames(nodes []node) []string {
	names := make([]string, 0, len(nodes))
	for _, n := range nodes {
		names = append(names, n.name)
	}
	sort.Strings(names)
	return names
}

func writeGraph(ctx context.Context, nodes []node, edges []edge) error {
	nodeUpsert := tools.GraphUpsertNode()
	edgeUpsert := tools.GraphUpsertEdge()

	for _, n := range nodes {
		if *dryRun {
			fmt.Printf("[dry-run] node %s{name=%q}\n", n.label, n.name)
			continue
		}
		if _, err := nodeUpsert.Handler(ctx, nil, &mcp.CallToolParamsFor[tools.UpsertNodeParams]{
			Arguments: tools.UpsertNodeParams{Label: n.label, Key: "name", Value: n.name, Properties: n.props},
		}); err != nil {
			return fmt.Errorf("node %s/%s: %w", n.label, n.name, err)
		}
	}

	for _, e := range edges {
		if *dryRun {
			fmt.Printf("[dry-run] edge (%s{%s})-[:%s]->(%s{%s})\n", e.fromLabel, e.from, e.edgeType, e.toLabel, e.to)
			continue
		}
		if _, err := edgeUpsert.Handler(ctx, nil, &mcp.CallToolParamsFor[tools.UpsertEdgeParams]{
			Arguments: tools.UpsertEdgeParams{
				FromLabel: e.fromLabel, FromKey: "name", FromValue: e.from,
				ToLabel: e.toLabel, ToKey: "name", ToValue: e.to,
				Type: e.edgeType,
			},
		}); err != nil {
			return fmt.Errorf("edge %s-[%s]->%s: %w", e.from, e.edgeType, e.to, err)
		}
	}
	log.Printf("graph: %d edges written across %d nodes", len(edges), len(nodes))
	return nil
}

var htmlTagRe = regexp.MustCompile(`(?s)<[^>]*>`)

func ingestDoc(ctx context.Context, corpusDir, rel string, knownNames []string) error {
	raw, err := os.ReadFile(filepath.Join(corpusDir, rel))
	if err != nil {
		return err
	}
	text := string(raw)
	if strings.HasSuffix(rel, ".html") {
		text = htmlTagRe.ReplaceAllString(text, " ")
	}

	if *dryRun {
		fmt.Printf("[dry-run] vector_store doc=%s (%d chars)\n", rel, len(text))
	} else {
		store := tools.VectorStore()
		if _, err := store.Handler(ctx, nil, &mcp.CallToolParamsFor[tools.StoreParams]{
			Arguments: tools.StoreParams{Information: text, Metadata: map[string]string{"source": rel}},
		}); err != nil {
			return fmt.Errorf("vector_store: %w", err)
		}
	}

	// Cheap deterministic heuristic, not an LLM extraction pass: a Doc node
	// MENTIONS whichever known service/infra names appear in it as a
	// substring. Good enough to cross-link prose into the graph without
	// running NER/RE on a corpus this small and this structured.
	node := tools.GraphUpsertNode()
	edgeUpsert := tools.GraphUpsertEdge()
	if *dryRun {
		fmt.Printf("[dry-run] node Doc{path=%q}\n", rel)
	} else if _, err := node.Handler(ctx, nil, &mcp.CallToolParamsFor[tools.UpsertNodeParams]{
		Arguments: tools.UpsertNodeParams{Label: labelDoc, Key: "path", Value: rel},
	}); err != nil {
		return fmt.Errorf("node Doc/%s: %w", rel, err)
	}

	for _, name := range knownNames {
		if !strings.Contains(text, name) {
			continue
		}
		targetLabel := labelInfra
		if knownServices[name] {
			targetLabel = labelService
		}
		if *dryRun {
			fmt.Printf("[dry-run] edge (Doc{%s})-[:MENTIONS]->(%s{%s})\n", rel, targetLabel, name)
			continue
		}
		if _, err := edgeUpsert.Handler(ctx, nil, &mcp.CallToolParamsFor[tools.UpsertEdgeParams]{
			Arguments: tools.UpsertEdgeParams{
				FromLabel: labelDoc, FromKey: "path", FromValue: rel,
				ToLabel: targetLabel, ToKey: "name", ToValue: name,
				Type: "MENTIONS",
			},
		}); err != nil {
			return fmt.Errorf("edge Doc/%s MENTIONS %s: %w", rel, name, err)
		}
	}
	log.Printf("doc %s: stored + linked", rel)
	return nil
}

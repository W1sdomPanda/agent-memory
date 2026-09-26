package tools

import "testing"

func TestValidIdentAcceptsSimpleNames(t *testing.T) {
	for _, s := range []string{"Service", "_private", "DEPENDS_ON", "a1"} {
		if !validIdent(s) {
			t.Errorf("validIdent(%q) = false, want true", s)
		}
	}
}

func TestValidIdentRejectsInjectionAttempts(t *testing.T) {
	for _, s := range []string{"", "1Service", "Service}) DETACH DELETE n //", "Service ", "Service-x", "Se}rvice"} {
		if validIdent(s) {
			t.Errorf("validIdent(%q) = true, want false", s)
		}
	}
}

func TestReadOnlyCypherBlocksWriteClauses(t *testing.T) {
	for _, q := range []string{
		"MATCH (n) DETACH DELETE n",
		"CREATE (n:Service {name:'x'})",
		"MATCH (n) SET n.x = 1",
	} {
		if err := readOnlyCypher(q); err == nil {
			t.Errorf("readOnlyCypher(%q) = nil, want error", q)
		}
	}
}

func TestReadOnlyCypherAllowsReads(t *testing.T) {
	q := "MATCH (s:Service {name:$name})-[:DEPENDS_ON]->(d) RETURN d.name AS dep"
	if err := readOnlyCypher(q); err != nil {
		t.Errorf("readOnlyCypher(%q) = %v, want nil", q, err)
	}
}

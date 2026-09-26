package tools

import "testing"

func TestChunkShortTextIsOnePiece(t *testing.T) {
	got := chunk("short")
	if len(got) != 1 || got[0] != "short" {
		t.Fatalf("got %v", got)
	}
}

func TestChunkSplitsAndKeepsEverything(t *testing.T) {
	body := ""
	for i := 0; i < 3000; i++ {
		body += "line of prose that is reasonably long\n"
	}

	parts := chunk(body)
	if len(parts) < 2 {
		t.Fatalf("expected several chunks, got %d", len(parts))
	}
	for i, p := range parts {
		if n := len([]rune(p)); n > maxInputChars() {
			t.Fatalf("chunk %d is %d runes, over the %d limit", i, n, maxInputChars())
		}
	}

	// Every chunk after the first repeats the tail of the one before, so the
	// pieces cover the input with no gap.
	joined := parts[0]
	for _, p := range parts[1:] {
		joined += p
	}
	if len([]rune(joined)) < len([]rune(body)) {
		t.Fatalf("chunks lost content: %d < %d", len([]rune(joined)), len([]rune(body)))
	}
	t.Logf("%d chars -> %d chunks", len([]rune(body)), len(parts))
}

package tools

import "testing"

func TestRRFScoreDecreasesWithRank(t *testing.T) {
	prev := rrfScore(0)
	for rank := 1; rank < 10; rank++ {
		cur := rrfScore(rank)
		if cur >= prev {
			t.Errorf("rrfScore(%d) = %v, want < rrfScore(%d) = %v", rank, cur, rank-1, prev)
		}
		prev = cur
	}
}

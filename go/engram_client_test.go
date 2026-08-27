package engramclient

import (
	"os"
	"strings"
	"testing"
)

func TestAgainstLiveServer(t *testing.T) {
	base := os.Getenv("ENGRAM_TEST_URL")
	if base == "" {
		t.Skip("ENGRAM_TEST_URL not set; start engram-server and export ENGRAM_TEST_URL=http://127.0.0.1:PORT")
	}
	c := New(strings.TrimRight(base, "/"))

	id, err := c.Remember("main", "the nightly backup job runs at 03:00 utc", "", 0.5)
	if err != nil {
		t.Fatalf("remember: %v", err)
	}
	if id == 0 {
		t.Fatal("expected nonzero id")
	}

	hits, err := c.Recall("main", "when does the backup job run", 200)
	if err != nil {
		t.Fatalf("recall: %v", err)
	}
	if len(hits) == 0 || !strings.Contains(hits[0].Text, "backup") {
		t.Fatalf("unexpected hits: %+v", hits)
	}

	st, err := c.Stats("main")
	if err != nil {
		t.Fatalf("stats: %v", err)
	}
	if st.Live < 1 {
		t.Fatalf("expected live>=1, got %+v", st)
	}
}

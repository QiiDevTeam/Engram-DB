package engramclient

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
)

type Client struct {
	BaseURL    string
	HTTPClient *http.Client
}

func New(baseURL string) *Client {
	return &Client{BaseURL: baseURL, HTTPClient: http.DefaultClient}
}

type Hit struct {
	ID              uint64  `json:"id"`
	Score           float32 `json:"score"`
	Text            string  `json:"text"`
	EstimatedTokens int     `json:"estimated_tokens"`
	Tier            string  `json:"tier"`
}

type Stats struct {
	Live          int `json:"live"`
	TotalInclDead int `json:"total_incl_dead"`
	Archived      int `json:"archived"`
	Summaries     int `json:"summaries"`
	Hot           int `json:"hot"`
	Warm          int `json:"warm"`
	Cold          int `json:"cold"`
}

func (c *Client) post(path string, body any, out any) error {
	buf, err := json.Marshal(body)
	if err != nil {
		return err
	}
	resp, err := c.HTTPClient.Post(c.BaseURL+path, "application/json", bytes.NewReader(buf))
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		var e map[string]any
		_ = json.NewDecoder(resp.Body).Decode(&e)
		return fmt.Errorf("engram: %s -> %d %v", path, resp.StatusCode, e["error"])
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

func (c *Client) Remember(collection, text, subject string, importance float64) (uint64, error) {
	body := map[string]any{"text": text, "importance": importance}
	if subject != "" {
		body["subject"] = subject
	}
	var out struct {
		ID uint64 `json:"id"`
	}
	err := c.post("/api/collections/"+collection+"/remember", body, &out)
	return out.ID, err
}

func (c *Client) Recall(collection, query string, budgetTokens int) ([]Hit, error) {
	var out struct {
		Hits []Hit `json:"hits"`
	}
	err := c.post("/api/collections/"+collection+"/recall",
		map[string]any{"query": query, "budget_tokens": budgetTokens}, &out)
	return out.Hits, err
}

func (c *Client) Stats(collection string) (Stats, error) {
	var out Stats
	err := c.get("/api/collections/" + collection + "/stats", &out)
	return out, err
}

func (c *Client) Checkpoint() error {
	return c.post("/api/checkpoint", map[string]any{}, &map[string]any{})
}

func (c *Client) get(path string, out any) error {
	resp, err := c.HTTPClient.Get(c.BaseURL + path)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("engram: GET %s -> %d", path, resp.StatusCode)
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

// ---- Options-style high-level API (v2) ----

type RememberOptions struct {
	Text        string
	Subject     string
	Importance  float64
	EventTime   int64 // 0 = now
}

type RecallOptions struct {
	Query          string
	BudgetTokens   int
	KMax           int
	Profile        string // "chat" | "agent" | "overview"
	IncludeCold    bool
}

func (c *Client) RememberWith(collection string, o RememberOptions) (uint64, error) {
	body := map[string]any{"text": o.Text, "importance": o.Importance}
	if o.Subject != "" {
		body["subject"] = o.Subject
	}
	if o.EventTime != 0 {
		body["event_time"] = o.EventTime
	}
	var out struct {
		ID uint64 `json:"id"`
	}
	err := c.post("/api/collections/"+collection+"/remember", body, &out)
	return out.ID, err
}

func (c *Client) RecallWith(collection string, o RecallOptions) ([]Hit, error) {
	body := map[string]any{
		"query":         o.Query,
		"budget_tokens": o.BudgetTokens,
		"k_max":         o.KMax,
	}
	if o.Profile != "" {
		body["profile"] = o.Profile
	}
	if o.IncludeCold {
		body["include_cold"] = true
	}
	var out struct {
		Hits []Hit `json:"hits"`
	}
	err := c.post("/api/collections/"+collection+"/recall", body, &out)
	return out.Hits, err
}

func (c *Client) ForgetSubject(collection, subject string) error {
	return c.post("/api/collections/"+collection+"/forget_subject",
		map[string]any{"subject": subject}, &map[string]any{})
}

func (c *Client) HardDelete(collection string, id uint64) error {
	return c.post("/api/collections/"+collection+"/hard_delete",
		map[string]any{"id": id}, &map[string]any{})
}

// Drop dead records older than retentionSecs across all collections.
func (c *Client) Compact(retentionSecs int64) error {
	return c.post("/api/compact", map[string]any{"retention_secs": retentionSecs},
		&map[string]any{})
}

func (c *Client) CheckpointKeepWAL() error {
	return c.post("/api/checkpoint_keep_wal", map[string]any{}, &map[string]any{})
}

func (c *Client) BackupTo(dest string) error {
	return c.post("/api/backup", map[string]any{"dest": dest}, &map[string]any{})
}

func (c *Client) Verify() (ok bool, rows int, errs []string, err error) {
	resp, e := c.HTTPClient.Get(c.BaseURL + "/api/verify")
	if e != nil {
		return false, 0, nil, e
	}
	defer resp.Body.Close()
	var out struct {
		Collections int      `json:"collections"`
		Rows        int      `json:"rows"`
		OK          bool     `json:"ok"`
		Errors      []string `json:"errors"`
	}
	err = json.NewDecoder(resp.Body).Decode(&out)
	return out.OK, out.Rows, out.Errors, err
}

// ExportToFile streams the NDJSON export into a local file.
func (c *Client) ExportToFile(path string) (int64, error) {
	resp, e := c.HTTPClient.Get(c.BaseURL + "/api/export")
	if e != nil {
		return 0, e
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("engram: export -> %d", resp.StatusCode)
	}
	f, e := os.Create(path)
	if e != nil {
		return 0, e
	}
	defer f.Close()
	return io.Copy(f, resp.Body)
}

// ImportFromFile pushes an NDJSON file to the server. Returns (imported, skipped).
func (c *Client) ImportFromFile(path string) (int64, int64, error) {
	data, e := os.ReadFile(path)
	if e != nil {
		return 0, 0, e
	}
	resp, e := c.HTTPClient.Post(c.BaseURL+"/api/import",
		"application/x-ndjson", bytes.NewReader(data))
	if e != nil {
		return 0, 0, e
	}
	defer resp.Body.Close()
	var out struct {
		Imported int64 `json:"imported"`
		Skipped  int64 `json:"skipped"`
	}
	err := json.NewDecoder(resp.Body).Decode(&out)
	return out.Imported, out.Skipped, err
}


// Package qdrant is a minimal REST client for Qdrant vector search.
package qdrant

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
)

// ErrCollectionNotFound is returned when the collection does not exist yet
// (for example, before any document has been ingested).
var ErrCollectionNotFound = errors.New("qdrant: collection not found")

// Client talks to the Qdrant REST API.
type Client struct {
	base   string
	apiKey string
	hc     *http.Client
}

// New returns a Qdrant client.
func New(baseURL, apiKey string, hc *http.Client) *Client {
	return &Client{base: strings.TrimRight(baseURL, "/"), apiKey: apiKey, hc: hc}
}

// Hit is one scored search result.
type Hit struct {
	ID      any            `json:"id"`
	Score   float64        `json:"score"`
	Payload map[string]any `json:"payload"`
}

// Query runs a nearest-neighbour search using the Query API.
func (c *Client) Query(ctx context.Context, collection string, vector []float32, limit int, threshold float64) ([]Hit, error) {
	reqBody := map[string]any{
		"query":        vector,
		"limit":        limit,
		"with_payload": true,
	}
	if threshold > 0 {
		reqBody["score_threshold"] = threshold
	}
	body, _ := json.Marshal(reqBody)

	endpoint := fmt.Sprintf("%s/collections/%s/points/query", c.base, url.PathEscape(collection))
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		req.Header.Set("api-key", c.apiKey)
	}

	resp, err := c.hc.Do(req)
	if err != nil {
		return nil, fmt.Errorf("qdrant query: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return nil, ErrCollectionNotFound
	}
	if resp.StatusCode/100 != 2 {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("qdrant query: status %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}

	var out struct {
		Result struct {
			Points []Hit `json:"points"`
		} `json:"result"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("qdrant query: decode: %w", err)
	}
	return out.Result.Points, nil
}

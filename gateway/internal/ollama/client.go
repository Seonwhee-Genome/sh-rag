// Package ollama is a minimal client for the Ollama HTTP API.
package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// Client talks to an Ollama server.
type Client struct {
	base string
	hc   *http.Client
}

// New returns a client for the Ollama server at baseURL.
func New(baseURL string, hc *http.Client) *Client {
	return &Client{base: strings.TrimRight(baseURL, "/"), hc: hc}
}

// Model is one entry of GET /api/tags.
type Model struct {
	Name       string `json:"name"`
	Model      string `json:"model"`
	ModifiedAt string `json:"modified_at"`
	Details    struct {
		Family   string   `json:"family"`
		Families []string `json:"families"`
	} `json:"details"`
}

// Embed returns one embedding per input string using POST /api/embed.
func (c *Client) Embed(ctx context.Context, model string, input []string) ([][]float32, error) {
	body, _ := json.Marshal(map[string]any{"model": model, "input": input})
	var out struct {
		Embeddings [][]float32 `json:"embeddings"`
	}
	if err := c.doJSON(ctx, http.MethodPost, "/api/embed", body, &out); err != nil {
		return nil, fmt.Errorf("ollama embed (%s): %w", model, err)
	}
	if len(out.Embeddings) != len(input) {
		return nil, fmt.Errorf("ollama embed: got %d embeddings for %d inputs", len(out.Embeddings), len(input))
	}
	return out.Embeddings, nil
}

// Models lists locally installed models.
func (c *Client) Models(ctx context.Context) ([]Model, error) {
	var out struct {
		Models []Model `json:"models"`
	}
	if err := c.doJSON(ctx, http.MethodGet, "/api/tags", nil, &out); err != nil {
		return nil, fmt.Errorf("ollama list models: %w", err)
	}
	return out.Models, nil
}

// ChatCompletions forwards a raw OpenAI-style request body to Ollama's
// OpenAI-compatible endpoint. The caller must close the response body.
func (c *Client) ChatCompletions(ctx context.Context, body []byte) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.base+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	return c.hc.Do(req)
}

// IsEmbeddingModel guesses whether a model only produces embeddings,
// so it can be hidden from the chat model list.
func IsEmbeddingModel(m Model) bool {
	name := strings.ToLower(m.Name)
	if strings.Contains(name, "embed") || strings.HasPrefix(name, "bge") || strings.Contains(name, "minilm") {
		return true
	}
	families := append([]string{m.Details.Family}, m.Details.Families...)
	for _, f := range families {
		switch strings.ToLower(f) {
		case "bert", "nomic-bert", "xlm-roberta", "xlmroberta":
			return true
		}
	}
	return false
}

func (c *Client) doJSON(ctx context.Context, method, path string, body []byte, out any) error {
	var rd io.Reader
	if body != nil {
		rd = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.base+path, rd)
	if err != nil {
		return err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.hc.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("status %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

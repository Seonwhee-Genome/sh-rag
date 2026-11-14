// Package rag implements retrieval and prompt augmentation.
package rag

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/sh-rag/gateway/internal/qdrant"
)

// Embedder turns text into vectors.
type Embedder interface {
	Embed(ctx context.Context, model string, input []string) ([][]float32, error)
}

// Searcher finds the nearest stored chunks for a vector.
type Searcher interface {
	Query(ctx context.Context, collection string, vector []float32, limit int, threshold float64) ([]qdrant.Hit, error)
}

// Chunk is a retrieved passage with its provenance. Index is the 1-based
// number the model is asked to cite ([1], [2], ...). It is the "Source"
// object of the client API contract (docs/client-api.md).
type Chunk struct {
	Index  int     `json:"index"`
	DocID  string  `json:"doc_id,omitempty"`
	Source string  `json:"source"`
	Title  string  `json:"title,omitempty"`
	Page   int     `json:"page,omitempty"`
	Score  float64 `json:"score"`
	Text   string  `json:"text"`
}

// Options override the runtime settings for a single retrieval.
type Options struct {
	TopK      int      // <= 0 means the runtime setting
	Threshold *float64 // nil means the runtime setting
}

// Result is what one retrieval did and found.
type Result struct {
	Chunks     []Chunk
	EmbedModel string
	Collection string
	TopK       int
	Threshold  float64
	Took       time.Duration
}

// Info renders the retrieval parameters for the API's "retrieval" object.
func (r Result) Info() map[string]any {
	return map[string]any{
		"embed_model":     r.EmbedModel,
		"collection":      r.Collection,
		"top_k":           r.TopK,
		"score_threshold": r.Threshold,
		"took_ms":         r.Took.Milliseconds(),
		"skipped":         false,
	}
}

// Params are the retrieval settings in effect for one request.
type Params struct {
	Collection string
	EmbedModel string
	TopK       int
	Threshold  float64
}

// Static returns a Params provider with fixed values.
func Static(p Params) func() Params { return func() Params { return p } }

// Retriever embeds a query and fetches matching chunks. Params is consulted
// on every call, so settings changed at runtime apply to the next request.
type Retriever struct {
	Embedder Embedder
	Searcher Searcher
	Params   func() Params
}

// Retrieve returns up to TopK chunks relevant to query. A missing collection
// (nothing ingested yet) is not an error: it yields no chunks.
func (r *Retriever) Retrieve(ctx context.Context, query string, opts Options) (res Result, err error) {
	p := r.Params()
	res = Result{EmbedModel: p.EmbedModel, Collection: p.Collection, TopK: p.TopK, Threshold: p.Threshold}
	if opts.TopK > 0 {
		res.TopK = opts.TopK
	}
	if opts.Threshold != nil {
		res.Threshold = *opts.Threshold
	}
	start := time.Now()
	defer func() { res.Took = time.Since(start) }() // named result: applies to every return below

	vecs, err := r.Embedder.Embed(ctx, p.EmbedModel, []string{query})
	if err != nil {
		return res, err
	}
	hits, err := r.Searcher.Query(ctx, p.Collection, vecs[0], res.TopK, res.Threshold)
	if errors.Is(err, qdrant.ErrCollectionNotFound) {
		res.Chunks = []Chunk{}
		return res, nil
	}
	if err != nil {
		return res, err
	}

	seen := make(map[string]bool, len(hits))
	chunks := make([]Chunk, 0, len(hits))
	for _, h := range hits {
		text, _ := h.Payload["text"].(string)
		text = strings.TrimSpace(text)
		if text == "" || seen[text] {
			continue
		}
		seen[text] = true
		c := Chunk{Index: len(chunks) + 1, Text: text, Score: h.Score}
		c.Source, _ = h.Payload["source"].(string)
		c.DocID, _ = h.Payload["doc_id"].(string)
		c.Title, _ = h.Payload["title"].(string)
		if p, ok := h.Payload["page"].(float64); ok {
			c.Page = int(p)
		}
		chunks = append(chunks, c)
	}
	res.Chunks = chunks
	return res, nil
}

// BuildSystemPrompt renders the template with numbered context passages,
// stopping once maxChars of passage text has been used.
func BuildSystemPrompt(template string, chunks []Chunk, maxChars int) string {
	var b strings.Builder
	used := 0
	for i, c := range chunks {
		text := c.Text
		if maxChars > 0 && used+len(text) > maxChars {
			if i > 0 {
				break
			}
			text = truncateRunes(text, maxChars)
		}
		used += len(text)
		fmt.Fprintf(&b, "[%d] %s\n%s\n\n", i+1, label(c), text)
	}
	return strings.ReplaceAll(template, "{{context}}", strings.TrimSpace(b.String()))
}

// SourcesFooter renders a markdown list of sources, grouping passages
// that come from the same source and page.
func SourcesFooter(chunks []Chunk) string {
	if len(chunks) == 0 {
		return ""
	}
	type group struct {
		label string
		refs  []string
	}
	var order []*group
	byLabel := map[string]*group{}
	for i, c := range chunks {
		l := label(c)
		g, ok := byLabel[l]
		if !ok {
			g = &group{label: l}
			byLabel[l] = g
			order = append(order, g)
		}
		g.refs = append(g.refs, fmt.Sprintf("[%d]", i+1))
	}
	var b strings.Builder
	b.WriteString("\n\n---\n**Sources**\n")
	for _, g := range order {
		fmt.Fprintf(&b, "- %s %s\n", strings.Join(g.refs, ""), g.label)
	}
	return b.String()
}

// ExtractText returns the plain text of an OpenAI message content field,
// which is either a string or an array of typed parts.
func ExtractText(content any) string {
	switch v := content.(type) {
	case string:
		return v
	case []any:
		var parts []string
		for _, p := range v {
			m, ok := p.(map[string]any)
			if !ok || m["type"] != "text" {
				continue
			}
			if t, ok := m["text"].(string); ok {
				parts = append(parts, t)
			}
		}
		return strings.Join(parts, "\n")
	}
	return ""
}

// LastUserMessage returns the text of the most recent user message.
func LastUserMessage(messages []any) string {
	for i := len(messages) - 1; i >= 0; i-- {
		m, ok := messages[i].(map[string]any)
		if ok && m["role"] == "user" {
			return strings.TrimSpace(ExtractText(m["content"]))
		}
	}
	return ""
}

// InjectSystem inserts a system message after any leading system messages,
// so client-provided instructions keep priority.
func InjectSystem(messages []any, prompt string) []any {
	idx := 0
	for idx < len(messages) {
		m, ok := messages[idx].(map[string]any)
		if !ok || m["role"] != "system" {
			break
		}
		idx++
	}
	out := make([]any, 0, len(messages)+1)
	out = append(out, messages[:idx]...)
	out = append(out, map[string]any{"role": "system", "content": prompt})
	out = append(out, messages[idx:]...)
	return out
}

// IsTaskPrompt reports whether a message is one of OpenWebUI's background
// tasks (title, tag, or follow-up generation), which should skip retrieval.
func IsTaskPrompt(q string) bool {
	return strings.HasPrefix(strings.TrimSpace(q), "### Task:")
}

func label(c Chunk) string {
	src := c.Source
	if src == "" {
		src = "unknown"
	}
	if c.Page > 0 {
		return fmt.Sprintf("%s (p. %d)", src, c.Page)
	}
	return src
}

func truncateRunes(s string, maxBytes int) string {
	if len(s) <= maxBytes {
		return s
	}
	cut := 0
	for i := range s {
		if i > maxBytes {
			break
		}
		cut = i
	}
	return s[:cut]
}

// Package server exposes an OpenAI-compatible API that augments chat
// requests with retrieved context before forwarding them to Ollama.
//
// The wire format, including the sh_rag extension, is specified in
// docs/client-api.md.
package server

import (
	"bufio"
	"bytes"
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/sh-rag/gateway/internal/config"
	"github.com/sh-rag/gateway/internal/ollama"
	"github.com/sh-rag/gateway/internal/rag"
)

// Version is reported by GET /healthz.
const Version = "0.3.0"

// Features lists the optional contract features this build implements, so
// clients can discover them (see docs/client-api.md, "Feature discovery").
var Features = []string{"sources", "retrieve_v2", "models_meta"}

const maxBodyBytes = 32 << 20 // chat requests can carry base64 images

// Upstream is the subset of the Ollama client the server needs.
type Upstream interface {
	Models(ctx context.Context) ([]ollama.Model, error)
	ChatCompletions(ctx context.Context, body []byte) (*http.Response, error)
}

// Server holds handler dependencies.
type Server struct {
	cfg       config.Config
	upstream  Upstream
	retriever *rag.Retriever
	log       *slog.Logger
}

// New builds a Server.
func New(cfg config.Config, upstream Upstream, retriever *rag.Retriever, log *slog.Logger) *Server {
	return &Server{cfg: cfg, upstream: upstream, retriever: retriever, log: log}
}

// Handler returns the HTTP routes.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{"status": "ok", "version": Version, "features": Features})
	})
	mux.HandleFunc("GET /v1/models", s.auth(s.handleModels))
	mux.HandleFunc("POST /v1/chat/completions", s.auth(s.handleChat))
	mux.HandleFunc("POST /v1/retrieve", s.auth(s.handleRetrieve))
	return s.logRequests(mux)
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()
	models, err := s.upstream.Models(ctx)
	if err != nil {
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}
	embedModel := s.retriever.Params().EmbedModel
	data := make([]map[string]any, 0, len(models))
	for _, m := range models {
		if ollama.IsEmbeddingModel(m) || m.Name == embedModel || strings.HasPrefix(m.Name, embedModel+":") {
			continue
		}
		created := time.Now().Unix()
		if t, err := time.Parse(time.RFC3339Nano, m.ModifiedAt); err == nil {
			created = t.Unix()
		}
		data = append(data, map[string]any{
			"id":       s.cfg.ModelPrefix + m.Name,
			"object":   "model",
			"created":  created,
			"owned_by": "sh-rag",
			"sh_rag": map[string]any{
				"name":           m.Name,
				"family":         m.Details.Family,
				"parameter_size": m.Details.ParameterSize,
				"quantization":   m.Details.QuantizationLevel,
				"size":           m.Size,
			},
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
}

// Sources modes of the sh_rag request extension.
const (
	sourcesFooter = "footer"
	sourcesJSON   = "json"
	sourcesNone   = "none"
)

// chatExt is the parsed sh_rag object of a chat request.
type chatExt struct {
	Sources   string
	Retrieve  bool
	TopK      int
	Threshold *float64
}

// parseExt validates the optional sh_rag object. Defaults keep the behaviour
// OpenWebUI expects: retrieval on, sources as a markdown footer (unless
// RAG_SHOW_SOURCES is off).
func (s *Server) parseExt(raw any) (chatExt, error) {
	ext := chatExt{Sources: sourcesFooter, Retrieve: true}
	if !s.cfg.ShowSources {
		ext.Sources = sourcesNone
	}
	if raw == nil {
		return ext, nil
	}
	m, ok := raw.(map[string]any)
	if !ok {
		return ext, errors.New("sh_rag must be an object")
	}
	if v, ok := m["sources"]; ok && v != nil {
		str, _ := v.(string)
		switch str {
		case sourcesFooter, sourcesJSON, sourcesNone:
			ext.Sources = str
		default:
			return ext, errors.New(`sh_rag.sources must be "footer", "json" or "none"`)
		}
	}
	if v, ok := m["retrieve"]; ok && v != nil {
		b, ok := v.(bool)
		if !ok {
			return ext, errors.New("sh_rag.retrieve must be a boolean")
		}
		ext.Retrieve = b
	}
	if v, ok := m["top_k"]; ok && v != nil {
		f, ok := v.(float64)
		if !ok || f < 1 || f > 50 || f != float64(int(f)) {
			return ext, errors.New("sh_rag.top_k must be an integer between 1 and 50")
		}
		ext.TopK = int(f)
	}
	if v, ok := m["score_threshold"]; ok && v != nil {
		f, ok := v.(float64)
		if !ok || f < 0 || f > 1 {
			return ext, errors.New("sh_rag.score_threshold must be a number between 0 and 1")
		}
		ext.Threshold = &f
	}
	return ext, nil
}

func skipped(reason string) map[string]any {
	return map[string]any{"skipped": true, "reason": reason}
}

// retrieve runs retrieval for the chat query according to ext and returns the
// chunks (never nil) and the "retrieval" object for the response.
func (s *Server) retrieve(ctx context.Context, query string, ext chatExt) ([]rag.Chunk, map[string]any) {
	switch {
	case !ext.Retrieve:
		return []rag.Chunk{}, skipped("disabled")
	case query == "":
		return []rag.Chunk{}, skipped("no_query")
	case rag.IsTaskPrompt(query):
		return []rag.Chunk{}, skipped("task_prompt")
	}
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	res, err := s.retriever.Retrieve(ctx, query, rag.Options{TopK: ext.TopK, Threshold: ext.Threshold})
	if err != nil {
		// Degrade gracefully: answer without context rather than failing.
		s.log.Warn("retrieval failed; answering without context", "err", err)
		return []rag.Chunk{}, map[string]any{"skipped": false, "error": err.Error()}
	}
	s.log.Debug("retrieved", "chunks", len(res.Chunks), "took", res.Took)
	if res.Chunks == nil {
		res.Chunks = []rag.Chunk{}
	}
	return res.Chunks, res.Info()
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	var req map[string]any
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxBodyBytes)).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error())
		return
	}
	model, _ := req["model"].(string)
	if model == "" {
		writeError(w, http.StatusBadRequest, "model is required")
		return
	}
	ext, err := s.parseExt(req["sh_rag"])
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	delete(req, "sh_rag") // never forwarded to Ollama
	messages, _ := req["messages"].([]any)
	stream, _ := req["stream"].(bool)

	chunks, info := s.retrieve(r.Context(), rag.LastUserMessage(messages), ext)
	if len(chunks) > 0 {
		req["messages"] = rag.InjectSystem(messages, rag.BuildSystemPrompt(s.cfg.SystemPrompt, chunks, s.cfg.MaxContextChars))
	}
	upstreamModel := strings.TrimPrefix(model, s.cfg.ModelPrefix)
	req["model"] = upstreamModel

	body, err := json.Marshal(req)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	resp, err := s.upstream.ChatCompletions(r.Context(), body)
	if err != nil {
		writeError(w, http.StatusBadGateway, "ollama: "+err.Error())
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		copyHeader(w.Header(), resp.Header)
		w.WriteHeader(resp.StatusCode)
		_, _ = io.Copy(w, resp.Body)
		return
	}

	footer := ""
	if ext.Sources == sourcesFooter {
		footer = rag.SourcesFooter(chunks)
	}
	var structured map[string]any
	if ext.Sources == sourcesJSON {
		structured = map[string]any{"sources": chunks, "retrieval": info}
	}
	if stream {
		var prelude []byte
		if structured != nil {
			prelude = sourcesEvent(upstreamModel, structured)
		}
		s.proxyStream(w, resp.Body, prelude, footer)
	} else {
		s.proxyJSON(w, resp.Body, footer, structured)
	}
}

// sourcesEvent builds the first SSE event of a "sources": "json" stream: an
// empty delta carrying the retrieved sources, sent before the first token.
func sourcesEvent(model string, structured map[string]any) []byte {
	now := time.Now()
	ext := map[string]any{"type": "sources"}
	for k, v := range structured {
		ext[k] = v
	}
	chunk := map[string]any{
		"id":      fmt.Sprintf("chatcmpl-shrag-%d", now.UnixNano()),
		"object":  "chat.completion.chunk",
		"created": now.Unix(),
		"model":   model,
		"choices": []map[string]any{{
			"index":         0,
			"delta":         map[string]any{},
			"finish_reason": nil,
		}},
		"sh_rag": ext,
	}
	b, _ := json.Marshal(chunk)
	return append(append([]byte("data: "), b...), '\n', '\n')
}

// proxyStream relays the SSE stream line by line. prelude, when set, is
// written first (the sources event); footer, when set, is inserted as an
// extra content delta just before the stream finishes.
func (s *Server) proxyStream(w http.ResponseWriter, body io.Reader, prelude []byte, footer string) {
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	rc := http.NewResponseController(w)

	if len(prelude) > 0 {
		if _, err := w.Write(prelude); err != nil {
			return
		}
		_ = rc.Flush()
	}

	var head struct {
		ID      string `json:"id"`
		Created int64  `json:"created"`
		Model   string `json:"model"`
		Choices []struct {
			FinishReason *string `json:"finish_reason"`
		} `json:"choices"`
	}
	pending := footer != ""
	emitFooter := func() bool {
		pending = false
		chunk := map[string]any{
			"id":      head.ID,
			"object":  "chat.completion.chunk",
			"created": head.Created,
			"model":   head.Model,
			"choices": []map[string]any{{
				"index":         0,
				"delta":         map[string]any{"content": footer},
				"finish_reason": nil,
			}},
		}
		b, _ := json.Marshal(chunk)
		if _, err := w.Write(append(append([]byte("data: "), b...), '\n', '\n')); err != nil {
			return false
		}
		return true
	}

	rd := bufio.NewReaderSize(body, 64<<10)
	for {
		line, readErr := rd.ReadBytes('\n')
		if len(line) > 0 {
			if pending {
				if payload, ok := bytes.CutPrefix(bytes.TrimSpace(line), []byte("data:")); ok {
					payload = bytes.TrimSpace(payload)
					if bytes.Equal(payload, []byte("[DONE]")) {
						if !emitFooter() {
							return
						}
					} else if json.Unmarshal(payload, &head) == nil &&
						len(head.Choices) > 0 && head.Choices[0].FinishReason != nil {
						if !emitFooter() {
							return
						}
					}
				}
			}
			if _, err := w.Write(line); err != nil {
				return // client went away
			}
			_ = rc.Flush()
		}
		if readErr != nil {
			if readErr != io.EOF {
				s.log.Warn("upstream stream error", "err", readErr)
			}
			break
		}
	}
	if pending {
		emitFooter()
		_ = rc.Flush()
	}
}

func (s *Server) proxyJSON(w http.ResponseWriter, body io.Reader, footer string, structured map[string]any) {
	var out map[string]any
	if err := json.NewDecoder(body).Decode(&out); err != nil {
		writeError(w, http.StatusBadGateway, "decode upstream response: "+err.Error())
		return
	}
	if footer != "" {
		if choices, ok := out["choices"].([]any); ok && len(choices) > 0 {
			if c, ok := choices[0].(map[string]any); ok {
				if msg, ok := c["message"].(map[string]any); ok {
					content, _ := msg["content"].(string)
					msg["content"] = content + footer
				}
			}
		}
	}
	if structured != nil {
		out["sh_rag"] = structured
	}
	writeJSON(w, http.StatusOK, out)
}

func (s *Server) handleRetrieve(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Query          string   `json:"query"`
		TopK           int      `json:"top_k"`
		ScoreThreshold *float64 `json:"score_threshold"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&req); err != nil || strings.TrimSpace(req.Query) == "" {
		writeError(w, http.StatusBadRequest, `body must be {"query": "...", "top_k": 5, "score_threshold": 0.35}`)
		return
	}
	if req.TopK < 0 || req.TopK > 50 {
		writeError(w, http.StatusBadRequest, "top_k must be between 1 and 50")
		return
	}
	if req.ScoreThreshold != nil && (*req.ScoreThreshold < 0 || *req.ScoreThreshold > 1) {
		writeError(w, http.StatusBadRequest, "score_threshold must be between 0 and 1")
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	res, err := s.retriever.Retrieve(ctx, req.Query, rag.Options{TopK: req.TopK, Threshold: req.ScoreThreshold})
	if err != nil {
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}
	if res.Chunks == nil {
		res.Chunks = []rag.Chunk{}
	}
	writeJSON(w, http.StatusOK, map[string]any{"chunks": res.Chunks, "retrieval": res.Info()})
}

func (s *Server) auth(next http.HandlerFunc) http.HandlerFunc {
	if s.cfg.APIKey == "" {
		return next
	}
	want := []byte("Bearer " + s.cfg.APIKey)
	return func(w http.ResponseWriter, r *http.Request) {
		if subtle.ConstantTimeCompare([]byte(r.Header.Get("Authorization")), want) != 1 {
			writeError(w, http.StatusUnauthorized, "invalid or missing API key")
			return
		}
		next(w, r)
	}
}

type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (r *statusRecorder) WriteHeader(code int) {
	r.status = code
	r.ResponseWriter.WriteHeader(code)
}

// Unwrap lets http.ResponseController reach the underlying writer (Flush).
func (r *statusRecorder) Unwrap() http.ResponseWriter { return r.ResponseWriter }

func (s *Server) logRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(rec, r)
		if r.URL.Path == "/healthz" {
			return
		}
		s.log.Info("request", "method", r.Method, "path", r.URL.Path, "status", rec.status, "took", time.Since(start).Round(time.Millisecond))
	})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]any{"error": map[string]any{"message": msg, "type": "sh_rag_error"}})
}

func copyHeader(dst, src http.Header) {
	for k, vs := range src {
		if strings.EqualFold(k, "Content-Length") {
			continue
		}
		for _, v := range vs {
			dst.Add(k, v)
		}
	}
}

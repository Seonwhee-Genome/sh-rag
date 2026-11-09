// Package server exposes an OpenAI-compatible API that augments chat
// requests with retrieved context before forwarding them to Ollama.
package server

import (
	"bufio"
	"bytes"
	"context"
	"crypto/subtle"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/sh-rag/gateway/internal/config"
	"github.com/sh-rag/gateway/internal/ollama"
	"github.com/sh-rag/gateway/internal/rag"
)

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
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
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
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
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
	messages, _ := req["messages"].([]any)
	stream, _ := req["stream"].(bool)

	var chunks []rag.Chunk
	query := rag.LastUserMessage(messages)
	if query != "" && !rag.IsTaskPrompt(query) {
		ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
		start := time.Now()
		var err error
		chunks, err = s.retriever.Retrieve(ctx, query, 0)
		cancel()
		if err != nil {
			// Degrade gracefully: answer without context rather than failing.
			s.log.Warn("retrieval failed; answering without context", "err", err)
		} else {
			s.log.Debug("retrieved", "chunks", len(chunks), "took", time.Since(start))
		}
	}
	if len(chunks) > 0 {
		req["messages"] = rag.InjectSystem(messages, rag.BuildSystemPrompt(s.cfg.SystemPrompt, chunks, s.cfg.MaxContextChars))
	}
	req["model"] = strings.TrimPrefix(model, s.cfg.ModelPrefix)

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
	if s.cfg.ShowSources {
		footer = rag.SourcesFooter(chunks)
	}
	if stream {
		s.proxyStream(w, resp.Body, footer)
	} else {
		s.proxyJSON(w, resp.Body, footer)
	}
}

// proxyStream relays the SSE stream line by line and inserts the sources
// footer as an extra content delta just before the stream finishes.
func (s *Server) proxyStream(w http.ResponseWriter, body io.Reader, footer string) {
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	rc := http.NewResponseController(w)

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

func (s *Server) proxyJSON(w http.ResponseWriter, body io.Reader, footer string) {
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
	writeJSON(w, http.StatusOK, out)
}

func (s *Server) handleRetrieve(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Query string `json:"query"`
		TopK  int    `json:"top_k"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&req); err != nil || strings.TrimSpace(req.Query) == "" {
		writeError(w, http.StatusBadRequest, `body must be {"query": "...", "top_k": 5}`)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	chunks, err := s.retriever.Retrieve(ctx, req.Query, req.TopK)
	if err != nil {
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}
	if chunks == nil {
		chunks = []rag.Chunk{}
	}
	writeJSON(w, http.StatusOK, map[string]any{"chunks": chunks})
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

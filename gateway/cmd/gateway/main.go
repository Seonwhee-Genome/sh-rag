// Command gateway serves an OpenAI-compatible, retrieval-augmented chat API
// in front of Ollama, using Qdrant for vector search.
package main

import (
	"context"
	"errors"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/sh-rag/gateway/internal/config"
	"github.com/sh-rag/gateway/internal/ollama"
	"github.com/sh-rag/gateway/internal/qdrant"
	"github.com/sh-rag/gateway/internal/rag"
	"github.com/sh-rag/gateway/internal/server"
	"github.com/sh-rag/gateway/internal/settings"
)

func main() {
	cfg := config.Load()
	log := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: cfg.LogLevel}))

	// One shared, pooled transport for all upstream calls. No overall client
	// timeout: chat responses stream for as long as the model generates;
	// per-call deadlines come from request contexts instead.
	hc := &http.Client{Transport: &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		DialContext:           (&net.Dialer{Timeout: 5 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
		MaxIdleConns:          128,
		MaxIdleConnsPerHost:   64,
		IdleConnTimeout:       90 * time.Second,
		ResponseHeaderTimeout: 5 * time.Minute, // first token on a cold model load can be slow
	}}

	oc := ollama.New(cfg.OllamaURL, hc)
	runtime := settings.NewStore(cfg.RuntimeSettingsPath, settings.Values{
		EmbedModel: cfg.EmbedModel, TopK: cfg.TopK, Threshold: cfg.ScoreThreshold,
	}, log)
	retriever := &rag.Retriever{
		Embedder: oc,
		Searcher: qdrant.New(cfg.QdrantURL, cfg.QdrantAPIKey, hc),
		Params: func() rag.Params {
			v := runtime.Get()
			return rag.Params{
				Collection: settings.CollectionName(cfg.Collection, v.EmbedModel),
				EmbedModel: v.EmbedModel,
				TopK:       v.TopK,
				Threshold:  v.Threshold,
			}
		},
	}
	srv := &http.Server{
		Addr:              cfg.ListenAddr,
		Handler:           server.New(cfg, oc, retriever, log).Handler(),
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       120 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go func() {
		p := retriever.Params()
		log.Info("gateway listening", "addr", cfg.ListenAddr, "ollama", cfg.OllamaURL, "qdrant", cfg.QdrantURL,
			"collection", p.Collection, "embed_model", p.EmbedModel, "runtime_settings", cfg.RuntimeSettingsPath,
			"auth", cfg.APIKey != "")
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Error("server failed", "err", err)
			os.Exit(1)
		}
	}()

	<-ctx.Done()
	log.Info("shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Error("shutdown", "err", err)
	}
}

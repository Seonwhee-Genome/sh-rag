// Package config loads gateway settings from environment variables.
package config

import (
	"log/slog"
	"os"
	"strconv"
	"strings"
)

// DefaultSystemPrompt is used when RAG_SYSTEM_PROMPT is unset.
// The literal {{context}} is replaced by the retrieved passages.
const DefaultSystemPrompt = `You are a helpful assistant that answers questions using the user's private documents.
Use the context below when it is relevant. If the context does not contain the answer, say so plainly instead of guessing.
Cite the passages you rely on with their bracketed numbers, e.g. [1] or [2][3].
Answer in the same language as the user's question.

<context>
{{context}}
</context>`

// Config holds all runtime settings for the gateway.
type Config struct {
	ListenAddr      string
	OllamaURL       string
	QdrantURL       string
	QdrantAPIKey    string
	Collection      string
	EmbedModel      string
	TopK            int
	ScoreThreshold  float64
	MaxContextChars int
	APIKey          string
	ModelPrefix     string
	SystemPrompt    string
	ShowSources     bool
	LogLevel        slog.Level
	// RuntimeSettingsPath points at the JSON file the ingest console writes
	// (embedding model, top-k, threshold). Values there override the env
	// defaults above. Empty disables it.
	RuntimeSettingsPath string
}

// Load reads the configuration from the environment, applying defaults.
func Load() Config {
	c := Config{
		ListenAddr:      env("GATEWAY_LISTEN_ADDR", ":8080"),
		OllamaURL:       strings.TrimRight(env("OLLAMA_BASE_URL", "http://localhost:11434"), "/"),
		QdrantURL:       strings.TrimRight(env("QDRANT_URL", "http://localhost:6333"), "/"),
		QdrantAPIKey:    env("QDRANT_API_KEY", ""),
		Collection:      env("QDRANT_COLLECTION", "documents"),
		EmbedModel:      env("EMBED_MODEL", "bge-m3"),
		TopK:            envInt("RAG_TOP_K", 5),
		ScoreThreshold:  envFloat("RAG_SCORE_THRESHOLD", 0.35),
		MaxContextChars: envInt("RAG_MAX_CONTEXT_CHARS", 12000),
		APIKey:          env("GATEWAY_API_KEY", ""),
		ModelPrefix:     env("RAG_MODEL_PREFIX", "rag/"),
		SystemPrompt:    env("RAG_SYSTEM_PROMPT", DefaultSystemPrompt),
		ShowSources:     envBool("RAG_SHOW_SOURCES", true),
		LogLevel:        parseLevel(env("LOG_LEVEL", "info")),

		RuntimeSettingsPath: env("RUNTIME_SETTINGS_PATH", ""),
	}
	if !strings.Contains(c.SystemPrompt, "{{context}}") {
		c.SystemPrompt += "\n\n<context>\n{{context}}\n</context>"
	}
	return c
}

func env(key, def string) string {
	if v, ok := os.LookupEnv(key); ok && strings.TrimSpace(v) != "" {
		return v
	}
	return def
}

func envInt(key string, def int) int {
	if n, err := strconv.Atoi(env(key, "")); err == nil {
		return n
	}
	return def
}

func envFloat(key string, def float64) float64 {
	if f, err := strconv.ParseFloat(env(key, ""), 64); err == nil {
		return f
	}
	return def
}

func envBool(key string, def bool) bool {
	if b, err := strconv.ParseBool(env(key, "")); err == nil {
		return b
	}
	return def
}

func parseLevel(s string) slog.Level {
	var l slog.Level
	if err := l.UnmarshalText([]byte(s)); err != nil {
		return slog.LevelInfo
	}
	return l
}

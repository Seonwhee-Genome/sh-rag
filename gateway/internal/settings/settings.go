// Package settings provides runtime settings that the ingest console can
// change while the gateway is running (shared JSON file on a volume).
package settings

import (
	"encoding/json"
	"log/slog"
	"os"
	"strings"
	"sync"
	"time"
)

// Values are the settings retrieval needs on every request.
type Values struct {
	EmbedModel string
	TopK       int
	Threshold  float64
}

type fileFormat struct {
	EmbedModel     string   `json:"embed_model"`
	TopK           *int     `json:"top_k"`
	ScoreThreshold *float64 `json:"score_threshold"`
}

// Store reads the settings file, re-checking it at most once per interval and
// re-parsing only when its modification time or size changes.
type Store struct {
	path     string
	defaults Values
	log      *slog.Logger
	interval time.Duration

	mu      sync.Mutex
	cur     Values
	mod     time.Time
	size    int64
	checked time.Time
}

// NewStore returns a Store. An empty path disables the file and always
// returns defaults (the environment configuration).
func NewStore(path string, defaults Values, log *slog.Logger) *Store {
	return &Store{path: path, defaults: defaults, cur: defaults, log: log, interval: time.Second}
}

// Get returns the current settings.
func (s *Store) Get() Values {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.path == "" {
		return s.defaults
	}
	now := time.Now()
	if !s.checked.IsZero() && now.Sub(s.checked) < s.interval {
		return s.cur
	}
	s.checked = now

	fi, err := os.Stat(s.path)
	if err != nil {
		s.cur, s.mod, s.size = s.defaults, time.Time{}, 0
		return s.cur
	}
	if fi.ModTime().Equal(s.mod) && fi.Size() == s.size {
		return s.cur
	}
	data, err := os.ReadFile(s.path)
	if err != nil {
		return s.cur
	}
	var f fileFormat
	if err := json.Unmarshal(data, &f); err != nil {
		// Possibly caught mid-write; keep the last good values and retry later.
		s.log.Warn("ignoring unreadable runtime settings", "path", s.path, "err", err)
		return s.cur
	}
	v := s.defaults
	if m := strings.TrimSpace(f.EmbedModel); m != "" {
		v.EmbedModel = m
	}
	if f.TopK != nil && *f.TopK > 0 {
		v.TopK = *f.TopK
	}
	if f.ScoreThreshold != nil && *f.ScoreThreshold >= 0 {
		v.Threshold = *f.ScoreThreshold
	}
	if v != s.cur {
		s.log.Info("runtime settings loaded", "embed_model", v.EmbedModel, "top_k", v.TopK, "score_threshold", v.Threshold)
	}
	s.cur, s.mod, s.size = v, fi.ModTime(), fi.Size()
	return v
}

// CollectionName maps an embedding model to its own Qdrant collection, e.g.
// ("documents", "bge-m3") -> "documents_bge_m3". Must match the Python
// implementation in ingest/ingest/runtime.py.
func CollectionName(base, model string) string {
	m := strings.TrimSuffix(strings.ToLower(strings.TrimSpace(model)), ":latest")
	var b strings.Builder
	sep := false
	for _, r := range m {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
			sep = false
		} else if !sep {
			b.WriteByte('_')
			sep = true
		}
	}
	return base + "_" + strings.Trim(b.String(), "_")
}

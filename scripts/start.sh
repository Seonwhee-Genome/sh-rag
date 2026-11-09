#!/usr/bin/env bash
# Start the sh-rag stack on macOS / Linux.
#   ./scripts/start.sh                    native Ollama on the host (recommended on macOS)
#   ./scripts/start.sh --docker-ollama    Ollama in Docker (CPU)
#   ./scripts/start.sh --docker-ollama --gpu   Ollama in Docker with NVIDIA GPU (Linux)
#   add --skip-pull to skip model downloads
set -euo pipefail
cd "$(dirname "$0")/.."

DOCKER_OLLAMA=0 GPU=0 SKIP_PULL=0
for arg in "$@"; do
  case "$arg" in
    --docker-ollama) DOCKER_OLLAMA=1 ;;
    --gpu) GPU=1 ;;
    --skip-pull) SKIP_PULL=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

[ -f .env ] || { cp .env.example .env; echo "Created .env from .env.example"; }

cfg() { # cfg KEY DEFAULT -> value from .env
  local v
  v=$(grep -E "^[[:space:]]*$1[[:space:]]*=" .env | tail -n1 | cut -d= -f2- | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e "s/^[\"']//" -e "s/[\"']$//" || true)
  echo "${v:-$2}"
}
CHAT_MODEL=$(cfg CHAT_MODEL qwen2.5:7b)
EMBED_MODEL=$(cfg EMBED_MODEL bge-m3)
WEBUI_PORT=$(cfg WEBUI_PORT 3000)
DOCS_PATH=$(cfg DOCS_PATH ./data/docs)

COMPOSE=(docker compose -f docker-compose.yml)
if [ "$DOCKER_OLLAMA" = 1 ]; then
  export OLLAMA_BASE_URL=http://ollama:11434
  COMPOSE+=(--profile ollama)
  [ "$GPU" = 1 ] && COMPOSE+=(-f compose.gpu.yml)
fi

ollama_up() { curl -fsS --max-time 3 http://localhost:11434/api/version >/dev/null 2>&1; }

if [ "$DOCKER_OLLAMA" = 1 ]; then
  "${COMPOSE[@]}" up -d ollama
  echo "Waiting for Ollama container..."
  for _ in $(seq 60); do ollama_up && break; sleep 1; done
fi
if ! ollama_up; then
  echo "Ollama is not reachable at http://localhost:11434." >&2
  echo "Install it from https://ollama.com/download and start it, or rerun with --docker-ollama." >&2
  exit 1
fi

if [ "$SKIP_PULL" = 0 ]; then
  for m in "$EMBED_MODEL" "$CHAT_MODEL"; do
    echo "Pulling $m ..."
    if [ "$DOCKER_OLLAMA" = 1 ]; then "${COMPOSE[@]}" exec ollama ollama pull "$m"
    elif command -v ollama >/dev/null; then ollama pull "$m"
    else curl -fsS --max-time 7200 http://localhost:11434/api/pull -d "{\"model\":\"$m\",\"stream\":false}" >/dev/null
    fi
  done
fi

mkdir -p "$DOCS_PATH"
"${COMPOSE[@]}" up -d --build

cat <<EOF

OpenWebUI : http://localhost:$WEBUI_PORT   (pick a 'rag/...' model)
Gateway   : http://localhost:$(cfg GATEWAY_PORT 8090)/v1
Ingest API: http://localhost:$(cfg INGEST_PORT 8001)/docs
Documents : drop files into $DOCS_PATH
EOF

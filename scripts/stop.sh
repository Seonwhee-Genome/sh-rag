#!/usr/bin/env bash
# Stop the sh-rag stack (data volumes are kept). Pass --purge to delete them too.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "${1:-}" = "--purge" ]; then
  docker compose -f docker-compose.yml --profile ollama down -v
else
  docker compose -f docker-compose.yml --profile ollama down
fi

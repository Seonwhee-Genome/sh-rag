#!/usr/bin/env bash
# Re-sync the documents folder into the index.
#   ./scripts/reindex.sh              only changed files
#   ./scripts/reindex.sh --force      re-embed everything
#   ./scripts/reindex.sh --recreate   drop the collection first (after changing EMBED_MODEL)
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose exec ingest python -m ingest scan "$@"

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
exec "$ROOT/env/run.sh" "$ROOT/env/venv/bin/python" "$ROOT/src/features/extraction_pipeline.py" "$@"

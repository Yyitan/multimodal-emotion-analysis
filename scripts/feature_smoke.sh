#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/feature_run.sh" run --config "$ROOT/configs/feature_extraction.yaml" --limit 2

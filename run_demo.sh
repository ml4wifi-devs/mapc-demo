#!/usr/bin/env bash
# Launch the MAPC Co-SR web demo. Setup (once):
#   uv venv --python 3.13 .venv
#   see README.md for the dependency install command
set -euo pipefail
cd "$(dirname "$0")"
PORT="${1:-8000}"
exec .venv/bin/uvicorn backend.app:app --host 0.0.0.0 --port "$PORT"

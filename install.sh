#!/usr/bin/env bash
# Set up the demo: virtual environment + all dependencies from GitHub.
# lai4wifi and mapc-surrogate are installed as editable clones (kept in
# .venv/src) because the demo reads their model checkpoints from the repos.
set -euo pipefail
cd "$(dirname "$0")"

PY="$(command -v python3.13 || command -v python3.12 || command -v python3)"
echo "Using $("$PY" --version) at $PY"

"$PY" -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --src .venv/src -r requirements.txt

echo
echo "Done. Start the demo with: ./run_demo.sh [port]"

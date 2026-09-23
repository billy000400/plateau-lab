#!/usr/bin/env sh
# Linux/macOS launcher. Use a fresh environment on each operating system.
set -eu
cd -- "$(dirname -- "$0")"
if [ -d .venv ] && [ ! -x .venv/bin/python ]; then
    echo "This Python environment cannot run here. Create a fresh .venv for this machine; see LINUX.md."
    exit 1
fi
if [ ! -d .venv ]; then
    "${PLATEAU_PYTHON:-python3}" -m venv .venv
    if [ -n "${PLATEAU_TORCH_INDEX_URL:-}" ]; then
        .venv/bin/python -m pip install torch==2.8.0 --index-url "$PLATEAU_TORCH_INDEX_URL"
    fi
    .venv/bin/python -m pip install -r requirements.txt
fi
exec .venv/bin/python app.py "$@"

#!/bin/zsh
set -e
cd -- "$(dirname -- "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "First launch: creating an isolated Python environment…"
  PYTHON_EXEC="${PLATEAU_PYTHON:-python3}"
  BUNDLED_PYTHON="$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
  if [[ -z "${PLATEAU_PYTHON:-}" && -x "$BUNDLED_PYTHON" ]]; then
    PYTHON_EXEC="$BUNDLED_PYTHON"
  fi
  "$PYTHON_EXEC" -m venv .venv
  .venv/bin/python -m pip install -r requirements.txt
fi
if curl --silent --fail http://127.0.0.1:8765/api/config >/dev/null; then
  open http://127.0.0.1:8765
else
  .venv/bin/python app.py --open
fi

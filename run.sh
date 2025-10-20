#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$ROOT_DIR"

if [ ! -d .venv ]; then
  echo "Error: .venv not found. Create it and install deps first." >&2
  echo "python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi

source .venv/bin/activate

export PYTHONUNBUFFERED=1
export PORT="8001"

# Run FastAPI app via uvicorn serving api/main:app
exec uvicorn api.main:app --host 0.0.0.0 --port "$PORT" --reload



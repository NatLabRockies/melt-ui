#!/usr/bin/env bash
set -euo pipefail

HOST="${MELT_HOST:-127.0.0.1}"
PORT="${MELT_PORT:-8000}"

exec uvicorn server:app \
  --reload \
  --host "${HOST}" \
  --port "${PORT}"

#!/usr/bin/env bash
# =====================================================================
#  cosyvoice-fastapi - start script (Linux / macOS)
#  GPU=<n> PORT=<n> ./start_api.sh [extra run_api.py args...]
# =====================================================================
set -euo pipefail

WRAPPER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:-0}"
PORT="${PORT:-8001}"

cd "$WRAPPER"
echo "[start_api] wrapper: $WRAPPER"
echo "[start_api] GPU=$GPU PORT=$PORT (fp32, warmup on)"
exec uv run run_api.py --gpu "$GPU" --host 0.0.0.0 --port "$PORT" "$@"

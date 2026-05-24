#!/usr/bin/env bash
set -e

mkdir -p "${DISCO_OUTPUT_DIR:-/workspace/output}"

cd /workspace

echo "[entrypoint] Testing imports..."
python -c "
import sys
sys.path.insert(0, '/disco')
sys.path.insert(0, '/disco/CLIP')
sys.path.insert(0, '/disco/guided-diffusion')
sys.path.insert(0, '/disco/ResizeRight')
sys.path.insert(0, '/disco/pytorch3d-lite')
sys.path.insert(0, '/disco/MiDaS')
sys.path.insert(0, '/disco/RAFT/core')
try:
    from src.config import DiscoConfig
    print('[entrypoint] config OK')
    from src.pipeline import run_job
    print('[entrypoint] pipeline OK')
    from src.handler import handler
    print('[entrypoint] handler OK')
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

echo "[entrypoint] Starting handler..."
exec python src/handler.py "$@"

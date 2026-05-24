#!/usr/bin/env bash
set -e

mkdir -p "${DISCO_OUTPUT_DIR:-/workspace/output}"

cd /workspace
exec python src/handler.py "$@"

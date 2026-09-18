#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Preserve the recorded parent/overlay pins and their CUDA 12.8 wheel source.
if [ ! -x .venv/bin/python ]; then
  rtk proxy bash scripts/setup.sh
fi
if [ ! -x round3/.venv/bin/python ]; then
  rtk proxy bash round3/scripts/setup.sh
fi
rtk proxy round3/.venv/bin/pip install --no-deps --no-build-isolation -e .
rtk proxy round3/.venv/bin/python round3/scripts/fetch_assets.py
rtk proxy round3/.venv/bin/pip check
rtk proxy ./run status

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -c 'import sys; assert sys.version_info[:2] == (3,12), "Use Python 3.12 for this lock file"'
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.lock.txt
.venv/bin/pip install --no-deps --no-build-isolation -e .
.venv/bin/pip check
./run status

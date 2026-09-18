#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
  rtk proxy bash scripts/setup.sh
fi
if [ ! -f assets/td3-Pendulum-v1.zip ]; then
  rtk proxy curl -fL --retry 2 'https://huggingface.co/sb3/td3-Pendulum-v1/resolve/f200e871a887a68eec2265f1c6ff366cfbf40ea8/td3-Pendulum-v1.zip' -o assets/td3-Pendulum-v1.zip
fi
rtk proxy .venv/bin/python -c 'import hashlib; from pathlib import Path; p=Path("assets/td3-Pendulum-v1.zip"); assert hashlib.sha256(p.read_bytes()).hexdigest()=="9597e5a20703d74786afc8e5f48d98b378816ad8743ee62e2a278483bc565823"'
if [ ! -f assets/pendulum_diffusion.pt ]; then
  rtk proxy .venv/bin/python scripts/train_pendulum_recorded.py
fi
rtk proxy .venv/bin/python scripts/verify_followup.py

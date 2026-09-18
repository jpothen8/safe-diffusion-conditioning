#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
rtk proxy python3 -c 'import sys; assert sys.version_info[:2] == (3,12), "Use Python 3.12 for this lock file"'
rtk proxy python3 -m venv .venv
rtk proxy .venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.lock.txt --report records/setup_report.json
if [ ! -d vendor/diffusion_policy/.git ]; then
  rtk proxy git clone https://github.com/real-stanford/diffusion_policy.git vendor/diffusion_policy
fi
rtk proxy git -C vendor/diffusion_policy checkout --detach 5ba07ac6661db573af695b419a7947ecb704690f
if [ ! -f assets/pusht_transformer.ckpt ]; then
  rtk proxy curl -fL --retry 2 'https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim/pusht/diffusion_policy_transformer/train_0/checkpoints/epoch%3D0850-test_mean_score%3D0.967.ckpt' -o assets/pusht_transformer.ckpt
fi
rtk proxy .venv/bin/python scripts/verify_assets.py
rtk proxy .venv/bin/pip check

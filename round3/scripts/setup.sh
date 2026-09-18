#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
rtk proxy python3 -m venv .venv
# Reuse the immutable parent stack through an explicit .pth; install all new
# libraries only in this overlay. Both layers have separately recorded locks.
rtk proxy .venv/bin/python -c 'from pathlib import Path; import site; root=Path.cwd(); parent=root.parent/".venv/lib/python3.12/site-packages"; Path(site.getsitepackages()[0],"frozen_parent.pth").write_text("import site; site.addsitedir("+repr(str(parent))+")\n")'
rtk proxy .venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.lock.txt --report records/reproduction_install.json
rtk proxy .venv/bin/pip freeze > records/reproduction_freeze.txt
rtk proxy .venv/bin/pip check

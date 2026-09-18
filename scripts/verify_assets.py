import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/"records/asset_manifest.json").read_text())
cp=manifest["checkpoint"]
assert hashlib.sha256((ROOT/cp["path"]).read_bytes()).hexdigest()==cp["sha256"], "Checkpoint hash mismatch"
hashes=json.loads((ROOT/"records/vendor_hashes.json").read_text())
for name,digest in hashes.items():
    assert hashlib.sha256((ROOT/"vendor/diffusion_policy"/name).read_bytes()).hexdigest()==digest, name
print(f"Verified checkpoint and {len(hashes)} upstream source/assets files.")

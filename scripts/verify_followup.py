"""Verify downloaded and trained follow-up assets against the recorded manifest."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'records/followup_assets.json').read_text())
for item in manifest['local_assets']:
    p=ROOT/item['path']
    assert p.exists(),f'Missing {p}'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],f'Hash mismatch: {p}'
print(f"Verified {len(manifest['local_assets'])} follow-up asset hashes.")

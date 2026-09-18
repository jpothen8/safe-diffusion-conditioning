"""Restore only the pinned public checkpoint archives; never replace mismatches."""
import hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for item in json.loads((ROOT/'records/assets.json').read_text()):
    if not item['path'].startswith('assets/'):continue
    path=ROOT/item['path'];path.parent.mkdir(exist_ok=True)
    if path.exists():data=path.read_bytes()
    else:
        with urllib.request.urlopen(item['url'],timeout=90) as response:data=response.read()
    assert hashlib.sha256(data).hexdigest()==item['sha256'],item['path']
    if not path.exists():path.write_bytes(data)
    print('Verified',item['path'])

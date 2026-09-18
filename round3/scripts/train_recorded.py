"""Reproduce complete-episode collection and both recorded diffusion fits."""
import argparse,json,shutil,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('task',choices=['Walker2d','HalfCheetah']);p.add_argument('--output-root',required=True)
a=p.parse_args();destination=Path(a.output_root).resolve()
if destination.exists():raise RuntimeError('Choose a fresh output root')
destination.mkdir(parents=True);(destination/'results').mkdir();(destination/'assets').mkdir()
manifest=json.loads((ROOT/'records/assets.json').read_text())
for level in ['expert','medium']:
    name=f'assets/{a.task}-v5-SAC-{level}.zip';source=ROOT/name
    record=next(r for r in manifest if r['path']==name)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==record['sha256']
    shutil.copy2(source,destination/name)
shutil.copytree(ROOT/'src',destination/'src',ignore=shutil.ignore_patterns('__pycache__'))
sys.path.insert(0,str(ROOT/'src'))
import locomotion_probe,locomotion_train,refit_chunk
locomotion_probe.ROOT=destination;locomotion_train.ROOT=destination;refit_chunk.ROOT=destination
locomotion_train.train(a.task);refit_chunk.main(a.task)
print('Both fits and complete episodes saved in',destination)

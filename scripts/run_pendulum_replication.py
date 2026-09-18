"""Run the frozen confirmation in a fresh root without touching delivered results."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--output-root',required=True,type=Path)
args=parser.parse_args();destination=args.output_root.resolve()
if destination.exists():raise SystemExit('Choose a new output root; existing directories are protected.')
lock=json.loads((ROOT/'results/pendulum/confirmation/lock.json').read_text())
for name,expected in lock['sha256'].items():
    source=ROOT/name
    assert hashlib.sha256(source.read_bytes()).hexdigest()==expected,name
    target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
(destination/'results/pendulum').mkdir(parents=True)
sys.path.insert(0,str(ROOT/'src'))
import core,pendulum_confirm
core.ROOT=destination
pendulum_confirm.main()
print('Replication:',destination/'results/pendulum/confirmation/summary.json')

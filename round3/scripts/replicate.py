"""Re-run a frozen comparison without overwriting delivered results.

Uses the original parent environment/assets and verified frozen round3 sources.
This repeats the same seeds; it is a reproducibility check, not new evidence.
"""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('task',choices=['Pendulum','Walker2d','HalfCheetah'])
    parser.add_argument('--output-root',required=True)
    args=parser.parse_args();destination=Path(args.output_root).resolve()
    destination.relative_to(ROOT.parent)
    if destination.exists():raise RuntimeError('Choose a fresh output root')
    task='pendulum' if args.task=='Pendulum' else args.task
    original=ROOT/f'results/{task}_comparison/lock.json'
    lock=json.loads(original.read_text());base=ROOT.parent if task=='pendulum' else ROOT
    for name,digest in lock['sha256'].items():
        assert hashlib.sha256((base/name).read_bytes()).hexdigest()==digest,name
    newroot=destination/'round3';newroot.mkdir(parents=True)
    for folder in ['src','configs']:
        shutil.copytree(ROOT/folder,newroot/folder,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['PENDULUM_PROTOCOL.md','LOCOMOTION_PROTOCOL.md']:
        shutil.copy2(ROOT/name,newroot/name)
    (newroot/'results').mkdir();(newroot/'assets').mkdir()
    if task!='pendulum':
        for name in [f'{task}_diffusion_v2.pt',f'{task}-v5-SAC-expert.zip',f'{task}-v5-SAC-medium.zip']:
            shutil.copy2(ROOT/'assets'/name,newroot/'assets'/name)
    (destination/'replication.json').write_text(json.dumps({'task':args.task,'original_lock':str(original),
         'original_inputs_sha256':lock['sha256'],'same_seeds_not_independent_evidence':True},indent=2))
    sys.path.insert(0,str(ROOT/'src'))
    if task=='pendulum':
        import pendulum_compare
        pendulum_compare.ROOT=newroot
        pendulum_compare.main()
    else:
        import locomotion_probe,locomotion_compare
        locomotion_probe.ROOT=newroot;locomotion_compare.ROOT=newroot
        locomotion_compare.main(task)

if __name__=='__main__':main()

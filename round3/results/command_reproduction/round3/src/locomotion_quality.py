"""Check learned first-chunk behavior before evaluating any safety method."""
import argparse,json,time
from pathlib import Path
import numpy as np
import chunk_model
import locomotion_env as le
from locomotion_probe import load
ROOT=Path(__file__).resolve().parents[1]

def main(task,version=''):
    start=time.perf_counter();out=ROOT/f'results/{task}_quality{version}';out.mkdir(exist_ok=True)
    actor=load(task,'expert')[0];model=chunk_model.load(ROOT/f'assets/{task}_diffusion{version}.pt')
    le.H[task]=model.dimension//6
    snaps=[le.snapshot(task,93000+i,actor,0 if i<8 else 1.) for i in range(16)]
    (out/'snapshots.json').write_text(json.dumps(snaps,indent=2))
    obs=np.array([s['observation'] for s in snaps]);banks=[]
    for i,o in enumerate(obs):banks.append(chunk_model.sample(model,np.repeat(o[None],64,0),93100+i).reshape(64,le.H[task],6))
    bank=np.stack(banks);np.savez_compressed(out/'proposals.npz',actions=bank)
    repeated=[s for s in snaps for _ in range(4)];chunks=bank[:,:4].reshape(-1,le.H[task],6)
    unfiltered=le.rollout(task,actor,repeated,chunks);teacher=le.rollout(task,actor,snaps)
    np.savez_compressed(out/'base.npz',chunk=chunks,**unfiltered);np.savez_compressed(out/'expert.npz',**teacher)
    def desc(r):return {'mean_return':float(r['rewards'].sum(1).mean()),'survival_fraction':float((~r['terminated']).mean()),
                        'mean_progress':float(r['progress'].mean()),'per_episode_returns':r['rewards'].sum(1).tolist()}
    rms=np.sqrt(np.mean(bank**2,axis=(2,3)))
    summary={'task':task,'label':'development quality gate','base':desc(unfiltered),'expert':desc(teacher),
             'acceptance':{str(cap):rms.__le__(cap).mean(1).tolist() for cap in [.45,.55,.65,.75,.85]},
             'rms_quantiles':np.quantile(rms,[0,.1,.5,.9,1]).tolist(),'seconds':time.perf_counter()-start}
    summary['quality_gate_pass']=bool(summary['base']['mean_return']>=.8*summary['expert']['mean_return'] and summary['base']['survival_fraction']>=.8)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['Walker2d','HalfCheetah']);p.add_argument('--version',default='')
    args=p.parse_args();main(args.task,args.version)

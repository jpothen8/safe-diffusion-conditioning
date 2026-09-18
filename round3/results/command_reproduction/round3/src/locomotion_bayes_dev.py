"""Tune BayesFP safety-only strengths against rejection on development states."""
import argparse,json,time
from pathlib import Path
import numpy as np
import chunk_model,bayesfp
from effort import project_numpy
ROOT=Path(__file__).resolve().parents[1]

def mmd(a,b):
    a=np.asarray(a,dtype=float).reshape(len(a),-1);b=np.asarray(b,dtype=float).reshape(len(b),-1)
    def k(x,y):
        dd=((x*x).sum(1)[:,None]+(y*y).sum(1)[None]-2*x@y.T)/x.shape[1]
        return np.exp(-np.maximum(dd,0)/(2*.2**2))
    return float(np.sqrt(max(0,k(a,a).mean()+k(b,b).mean()-2*k(a,b).mean())))

def main(task):
    out=ROOT/f'results/{task}_bayes_development';out.mkdir(exist_ok=True)
    quality=ROOT/f'results/{task}_quality_v2';snaps=json.loads((quality/'snapshots.json').read_text())
    obs=np.repeat(np.array([s['observation'] for s in snaps]),2,0)
    bank=np.load(quality/'proposals.npz')['actions'].reshape(16,2,32,-1).reshape(32,32,-1)
    model=chunk_model.load(ROOT/f'assets/{task}_diffusion_v2.pt');results={}
    caps=[.72,.75] if task=='Walker2d' else [.70,.72]
    for cap in caps:
        masks=np.sqrt(np.mean(bank*bank,-1))<=cap
        assert masks.any(1).all(),'Development rejection refusal; do not silently replace'
        q=bank[np.arange(32),masks.argmax(1)]
        results[str(cap)]={}
        for strength in [20,100,500,2000,10000]:
            start=time.perf_counter()
            x,diag=bayesfp.sample(model,obs,chunk_model.scheduler,model.dimension,102000,strength=strength,particles=32,radius=cap)
            x=np.clip(x,-1,1);projected=project_numpy(x,cap)
            np.savez_compressed(out/f'{cap}_{strength}.npz',actions=x,projected=projected,q=q,cloud=diag.pop('final_particle_cloud'))
            row={**diag,'safe_fraction':float((np.sqrt(np.mean(x*x,-1))<=cap+1e-6).mean()),
                 'repaired_mmd':mmd(projected,q),'seconds':time.perf_counter()-start}
            results[str(cap)][str(strength)]=row;print(task,cap,strength,json.dumps(row),flush=True)
    selected={c:int(min(rows,key=lambda s:rows[s]['repaired_mmd'])) for c,rows in results.items()}
    result={'task':task,'selection':'minimum repaired MMD on fixed development states, no reward objective',
            'cases':results,'selected_strengths':selected}
    (out/'summary.json').write_text(json.dumps(result,indent=2));print('selected',task,selected,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['Walker2d','HalfCheetah']);main(p.parse_args().task)

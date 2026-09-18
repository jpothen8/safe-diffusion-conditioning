"""Post-confirmation sensitivity check: same frozen chunks, reflected common expert."""
import json
import numpy as np
import core
import pendulum as pd
from pendulum_confirm import compare

def main():
    folder=core.ROOT/'results/pendulum/confirmation'
    out=core.ROOT/'results/pendulum/reflected_continuation';out.mkdir(exist_ok=True)
    cfg=json.loads((folder/'summary.json').read_text())['config']
    states=np.repeat(np.load(folder/'initial_states.npy'),cfg['replicates'],axis=0)
    actor=pd.actor();results={'label':'Post-confirmation continuation sensitivity; same states and frozen output chunks','examples':{}}
    boot=np.random.default_rng(cfg['bootstrap_seed']).integers(0,64,(cfg['bootstrap_draws'],64))
    for name in ['negative','positive']:
        returns={}
        for method in ['conditional','projection','refinement']:
            chunk=np.load(folder/f'{name}_{method}.npz')['chunk']
            roll=pd.rollout(actor,states,chunk,reflection=True);returns[method]=roll['rewards'].sum(1)
            np.savez_compressed(out/f'{name}_{method}.npz',chunk=chunk,**roll)
        results['examples'][name]={'means':{k:float(v.mean()) for k,v in returns.items()}}
        for method in ['conditional','refinement']:
            results['examples'][name][method+'_minus_projection']=compare(returns[method],returns['projection'],np.ones(512,dtype=bool),cfg,boot)
    (out/'summary.json').write_text(json.dumps(results,indent=2))
    print(json.dumps({k:{i:v[i] for i in ['means']} for k,v in results['examples'].items()}))

if __name__=='__main__':main()

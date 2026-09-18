"""BayesFP adapter checks and development tuning, independent of round-3 holdout."""
import json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];PARENT=ROOT.parent
sys.path.insert(0,str(PARENT/'src'))
import pendulum as pd
import pendulum_train as dp
from pendulum_confirm import mmd
import bayesfp

def main():
    out=ROOT/'results/pendulum_bayes_development';out.mkdir(exist_ok=True)
    model=dp.load();actor=pd.actor()
    raw=np.load(PARENT/'results/pendulum/development/raw.npz');states=raw['initial_states']
    obs=pd.observation(states)
    # The published zero-guidance limit must reproduce the original DDPM exactly.
    expected=dp.sample(model,obs,95100)
    x,diag=bayesfp.sample(model,obs,dp.scheduler,12,95100,strength=0,particles=1)
    actual=np.clip(x[...,None]*2,-2,2)
    np.testing.assert_array_equal(expected,actual)
    check={'zero_guidance_bitwise_equal_to_base':True}
    xx=torch.randn((3,12),requires_grad=True);cc=((xx-xx.clamp(-float('inf'),0)).square()).mean(-1)
    autograd=torch.autograd.grad(cc.sum(),xx)[0]
    _,analytic=bayesfp.box_cost_gradient(xx,-float('inf'),0)
    check['max_gradient_error']=float((autograd-analytic).abs().max().detach())
    assert check['max_gradient_error']<1e-7
    results={'checks':check,'cases':{}}
    for name,lo,hi in [('negative',-2,0),('positive',0,2)]:
        q=np.load(PARENT/f'results/pendulum/development/{name}.npz')['q']
        results['cases'][name]={}
        for strength in [1,5,20,100,500]:
            tick=time.perf_counter()
            x,diag=bayesfp.sample(model,obs,dp.scheduler,12,95200,strength=strength,particles=32,
                lower=-float('inf') if lo<0 else 0,upper=float('inf') if hi>0 else 0)
            actions=np.clip(x[...,None]*2,-2,2);projected=np.clip(actions,lo,hi)
            cloud=diag.pop('final_particle_cloud')
            rawroll=pd.rollout(actor,states,actions);fixedroll=pd.rollout(actor,states,projected)
            safe=((actions>=lo-1e-7)&(actions<=hi+1e-7)).all((1,2))
            row={**diag,'seconds':time.perf_counter()-tick,'safe_fraction':float(safe.mean()),
                 'raw_return':float(rawroll['rewards'].sum(1).mean()),'repaired_return':float(fixedroll['rewards'].sum(1).mean()),
                 'mmd_to_q':mmd(actions,q),'repaired_mmd_to_q':mmd(projected,q)}
            results['cases'][name][str(strength)]=row
            np.savez_compressed(out/f'{name}_{strength}.npz',actions=actions,repaired=projected,cloud=cloud,
                                raw_rewards=rawroll['rewards'],repaired_rewards=fixedroll['rewards'])
            print(name,strength,json.dumps(row),flush=True)
    (out/'summary.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()

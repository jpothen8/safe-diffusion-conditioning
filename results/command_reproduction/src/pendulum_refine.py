"""Finite implementation of the supplied draft's projected Langevin refinement.

The learned VP epsilon predictor defines a VE score at y = x_VP / sqrt(a_bar):
    score_y(y,o) = -epsilon(sqrt(a_bar)*y,t,o) / sqrt((1-a_bar)/a_bar).
This score need not equal the score of the actual 100-step, clipped DDPM law.
Rejection of that actual law remains the reference.
"""
import numpy as np
import torch
import pendulum_train as dp

SCHEDULES={
    'final_only':[(0,1024)],
    'annealed':[(20,256),(5,256),(0,512)],
    'annealed_long':[(20,1024),(5,1024),(0,2048)]}

def refine(model,observations,raw,lo,hi,seed,schedule='annealed'):
    torch.manual_seed(int(seed));torch.cuda.manual_seed_all(int(seed))
    device=next(model.parameters()).device
    obs=torch.as_tensor(observations,dtype=torch.float32,device=device)
    center=model.action_center;scale=model.action_scale
    lower=(lo-center)/scale;upper=(hi-center)/scale
    x=(torch.as_tensor(raw[...,0],device=device)-center)/scale
    x=x.clamp(lower,upper)
    sched=dp.scheduler();alphas=sched.alphas_cumprod.to(device)
    with torch.inference_mode():
        for timestep,steps in SCHEDULES[schedule]:
            ab=alphas[timestep];sigma=((1-ab)/ab).sqrt()
            for k in range(steps):
                # Positive, nonincreasing within level; exponent .6 is nonsummable.
                step=.15*sigma.square()/(1+k/64)**.6
                score=-model(ab.sqrt()*x,timestep,obs)/sigma
                y=x+step*score+(2*step).sqrt()*torch.randn_like(x)
                x=y.clamp(lower,upper) # gamma = 1/2, exact Euclidean projection
        result=(x*scale+center).clamp(lo,hi)
    return result.cpu().numpy()[...,None]

def main():
    import json,time
    import core
    import pendulum as pd
    out=core.ROOT/'results/pendulum/development'
    data=np.load(out/'raw.npz');states=data['initial_states'];raw=data['proposals'][:,0]
    model=dp.load();actor=pd.actor();results={}
    for name,lo,hi in [('negative',-2,0),('positive',0,2)]:
        results[name]={}
        for schedule in SCHEDULES:
            start=time.perf_counter()
            actions=refine(model,pd.observation(states),raw,lo,hi,65000,schedule)
            elapsed=time.perf_counter()-start
            roll=pd.rollout(actor,states,actions)
            np.savez_compressed(out/f'refine_{name}_{schedule}.npz',chunk=actions,**roll)
            results[name][schedule]={'return':float(roll['rewards'].sum(1).mean()),'seconds':elapsed,
                                    'safety_boundary':float(np.isclose(actions,0,atol=1e-5).any((1,2)).mean())}
            print(name,schedule,results[name][schedule],flush=True)
    (out/'refinement.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()

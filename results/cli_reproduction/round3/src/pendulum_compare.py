"""Fresh paired seven-arm Pendulum comparison, including paper-derived BayesFP."""
import hashlib,json,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];PARENT=ROOT.parent
sys.path.insert(0,str(PARENT/'src'))
import pendulum as pd
import pendulum_train as dp
from pendulum_refine import refine
from pendulum_confirm import mmd,metrics,verify_gym
import bayesfp

def contrast(a,b,n,reps,boot):
    means=(a-b).reshape(n,reps).mean(1);values=means[boot].mean(1)
    return {'mean':float(means.mean()),'ci95':np.quantile(values,[.025,.975]).tolist(),
            'ci98_75':np.quantile(values,[.00625,.99375]).tolist(),'state_differences':means.tolist()}

def main():
    cfg=json.loads((ROOT/'configs/pendulum.json').read_text());out=ROOT/'results/pendulum_comparison';out.mkdir(exist_ok=True)
    if (out/'lock.json').exists():raise RuntimeError('Preserve existing locked experiment')
    paths=[ROOT/'PENDULUM_PROTOCOL.md',ROOT/'configs/pendulum.json',ROOT/'src/bayesfp.py',Path(__file__),
           PARENT/'src/pendulum.py',PARENT/'src/pendulum_train.py',PARENT/'src/pendulum_refine.py',
           PARENT/'assets/pendulum_diffusion.pt',PARENT/'assets/td3-Pendulum-v1.zip']
    lock={'utc':datetime.now(timezone.utc).isoformat(),'config':cfg,'sha256':{str(p.relative_to(PARENT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (out/'lock.json').write_text(json.dumps(lock,indent=2));start=time.perf_counter()
    n=cfg['states'];reps=cfg['replicates'];budget=cfg['budget'];rng=np.random.default_rng(cfg['state_seed'])
    states=np.column_stack([np.pi+rng.uniform(-.05,.05,n),rng.uniform(-.05,.05,n)]);np.save(out/'initial_states.npy',states)
    starts=np.repeat(states,reps,0);observations=pd.observation(starts);model=dp.load();actor=pd.actor()
    tick=time.perf_counter();bank=[]
    for i,state in enumerate(states):
        aa=dp.sample(model,np.repeat(pd.observation(state)[None],reps*budget,0),cfg['proposal_seed']+i)
        bank.append(aa.reshape(reps,budget,12,1))
    bank=np.stack(bank);sampling_seconds=time.perf_counter()-tick
    np.savez_compressed(out/'proposals.npz',actions=bank)
    flat=bank.reshape(n*reps,budget,12,1);raw=flat[:,0]
    boot=np.random.default_rng(cfg['bootstrap_seed']).integers(0,n,(cfg['bootstrap_draws'],n))
    summary={'config':cfg,'sampling_seconds':sampling_seconds,'examples':{}}
    for ei,(name,lo,hi) in enumerate([('negative',-2,0),('positive',0,2)]):
        feasible=((flat>=lo)&(flat<=hi)).all((2,3));valid=feasible.any(1)
        if not valid.all():
            np.savez_compressed(out/f'{name}_refusals.npz',feasible=feasible)
            raise RuntimeError('Rejection refusal: saved; analysis must include this failure before continuing')
        indices=feasible.argmax(1);q=flat[np.arange(n*reps),indices]
        nextmask=feasible.copy();nextmask[np.arange(n*reps),indices]=False
        q2=flat[np.arange(n*reps),nextmask.argmax(1)]
        np.savez_compressed(out/f'{name}_selection.npz',feasible=feasible,indices=indices,second_q=q2)
        tick=time.perf_counter();projection=np.clip(raw,lo,hi);projection_seconds=time.perf_counter()-tick
        tick=time.perf_counter();refined=refine(model,observations,raw,lo,hi,cfg['refinement_seed']+ei,'annealed')
        refinement_seconds=time.perf_counter()-tick
        arms={'conditional':q,'projection':projection,'refinement':refined};particle_diagnostics={}
        for particles in cfg['bayesfp_particles']:
            tick=time.perf_counter()
            x,diag=bayesfp.sample(model,observations,dp.scheduler,12,cfg['bayesfp_seed']+ei*100+particles,
                 strength=cfg['bayesfp_strength'],particles=particles,
                 lower=-float('inf') if lo<0 else 0,upper=float('inf') if hi>0 else 0)
            diag['seconds']=time.perf_counter()-tick;cloud=diag.pop('final_particle_cloud')
            np.savez_compressed(out/f'{name}_bayesfp_{particles}_particles.npz',particles=cloud)
            actions=np.clip(x[...,None]*2,-2,2);arms[f'bayesfp_{particles}']=actions
            arms[f'bayesfp_{particles}_projected']=np.clip(actions,lo,hi);particle_diagnostics[str(particles)]=diag
        entries={};returns={}
        for method,chunk in arms.items():
            roll=pd.rollout(actor,starts,chunk);returns[method]=roll['rewards'].sum(1)
            safe=((chunk>=lo-1e-7)&(chunk<=hi+1e-7)&np.isfinite(chunk)).all((1,2))
            entry=metrics(roll,chunk,np.ones(len(chunk),dtype=bool));entry['unsafe_outputs']=int((~safe).sum())
            entry['mmd_to_conditional']=mmd(chunk,q);entry['gym_replay_error']=verify_gym(starts,roll)
            entries[method]=entry
            np.savez_compressed(out/f'{name}_{method}.npz',chunk=chunk,safe=safe,**roll)
        comparisons={}
        for a,b in [('conditional','projection'),('refinement','projection'),('refinement','bayesfp_32_projected'),
                    ('refinement','bayesfp_12_projected'),('bayesfp_32_projected','projection')]:
            comparisons[a+'_minus_'+b]=contrast(returns[a],returns[b],n,reps,boot)
        result={'acceptance':float(feasible.mean()),'acceptance_by_state':feasible.reshape(n,reps,budget).mean((1,2)).tolist(),
                'refusals':0,'logical_proposals_mean':float((indices+1).mean()),'projection_seconds':projection_seconds,
                'refinement_seconds':refinement_seconds,'particle_diagnostics':particle_diagnostics,
                'q_vs_q_mmd':mmd(q,q2),'arms':entries,'comparisons':comparisons}
        summary['examples'][name]=result
        print(name,json.dumps({'arms':entries,'differences':{k:{a:v[a] for a in ['mean','ci95','ci98_75']} for k,v in comparisons.items()}}),flush=True)
    summary['seconds']=time.perf_counter()-start;(out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('Completed',summary['seconds'],flush=True)

if __name__=='__main__':main()

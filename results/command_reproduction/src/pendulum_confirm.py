"""Run the locked, paired Pendulum confirmation, retaining all proposals/outcomes."""
import hashlib
import json
import time
from datetime import datetime,timezone
import numpy as np
import gym
import core
import pendulum as pd
import pendulum_train as dp
from pendulum_refine import refine

def mmd(a,b):
    a=np.asarray(a).reshape(len(a),-1).astype(float)/2
    b=np.asarray(b).reshape(len(b),-1).astype(float)/2
    def kernel(x,y):
        distances=np.maximum(((x*x).sum(1)[:,None]+(y*y).sum(1)[None]-2*x@y.T)/x.shape[1],0)
        return np.exp(-distances/(2*.1**2))
    return float(np.sqrt(max(0,kernel(a,a).mean()+kernel(b,b).mean()-2*kernel(a,b).mean())))

def metrics(roll,chunk,valid):
    theta=(roll['states'][...,0]+np.pi)%(2*np.pi)-np.pi
    near=(np.abs(theta)<.1)&(np.abs(roll['states'][...,1])<.5)
    windows=np.lib.stride_tricks.sliding_window_view(near,20,axis=1).all(-1)
    stabilized=windows.any(1)&valid
    stabilization=np.where(stabilized,windows.argmax(1)*.05,10.)
    mode=chunk.mean((1,2))
    return {'return_mean':float(roll['rewards'][valid].sum(1).mean()),
            'return_std':float(roll['rewards'][valid].sum(1).std(ddof=1)),
            'stabilized':int(stabilized.sum()),'unsuccessful_including_refusals':int((~stabilized).sum()),
            'mean_capped_stabilization_time':float(stabilization.mean()),
            'added_boundary_fraction':float(np.isclose(chunk[valid],0,atol=1e-5).any((1,2)).mean()),
            'physical_boundary_fraction':float(np.isclose(np.abs(chunk[valid]),2,atol=1e-5).any((1,2)).mean()),
            'mean_abs_first_chunk_displacement':float(np.abs(roll['states'][valid,12,0]-roll['states'][valid,0,0]).mean()),
            'mode_negative':float((mode[valid]<-1).mean()),'mode_positive':float((mode[valid]>1).mean()),
            'mode_intermediate':float((np.abs(mode[valid])<=1).mean()),
            'mean_torque':float(chunk[valid].mean()),'mean_chunk_torque_std':float(chunk[valid].std(1).mean())}

def compare(a,b,valid,cfg,boot):
    delta=(a-b).reshape(cfg['states'],cfg['replicates'])
    mask=valid.reshape(delta.shape)
    means=np.array([d[m].mean() if m.any() else np.nan for d,m in zip(delta,mask)])
    usable=means[np.isfinite(means)]
    if len(usable)!=cfg['states']:
        return {'valid_pairs':int(valid.sum()),'error':'Refusals invalidate the prespecified complete-state bootstrap',
                'mean_difference':float(np.nanmean(means))}
    distribution=means[boot].mean(1)
    interval=np.quantile(distribution,[.00625,.99375]).tolist()
    mean=float(means.mean())
    return {'mean_difference':mean,'ci95':np.quantile(distribution,[.025,.975]).tolist(),
            'family_ci98_75':interval,'per_state_differences':means.tolist(),'valid_pairs':int(valid.sum()),
            'passes_effect_criterion':bool(mean>=cfg['minimum_useful_return_gain'] and interval[0]>0)}

def verify_gym(states,roll):
    error=0.
    for index in [0,len(states)//3,2*len(states)//3]:
        env=gym.make('Pendulum-v1');env.reset(seed=0);env.unwrapped.state=states[index].copy()
        for t in range(200):
            error=max(error,float(np.max(np.abs(env.unwrapped.state-roll['states'][index,t]))))
            _,reward,_,_,_=env.step(roll['actions'][index,t])
            error=max(error,abs(float(reward)-float(roll['rewards'][index,t])))
        env.close()
    return error

def main():
    start=time.perf_counter();root=core.ROOT
    cfg=json.loads((root/'configs/pendulum_confirmation.json').read_text())
    out=root/'results/pendulum/confirmation';out.mkdir(exist_ok=True)
    if (out/'lock.json').exists():
        raise RuntimeError('Existing confirmation lock: preserve this run; use an explicitly new protocol/output for replication.')
    paths=['PENDULUM_CONFIRMATION.md','configs/pendulum_confirmation.json','src/pendulum.py',
           'src/pendulum_train.py','src/pendulum_refine.py','src/pendulum_confirm.py',
           'assets/pendulum_diffusion.pt','assets/td3-Pendulum-v1.zip','requirements.lock.txt']
    lock={'started_utc':datetime.now(timezone.utc).isoformat(),'config':cfg,
          'sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}}
    (out/'lock.json').write_text(json.dumps(lock,indent=2))
    rng=np.random.default_rng(cfg['state_seed']);n=cfg['states'];reps=cfg['replicates'];budget=cfg['budget']
    states=np.column_stack([np.pi+rng.uniform(-.05,.05,n),rng.uniform(-.05,.05,n)])
    np.save(out/'initial_states.npy',states)
    model=dp.load();actor=pd.actor();bank=[];sampling_start=time.perf_counter()
    for i,state in enumerate(states):
        proposals=dp.sample(model,np.repeat(pd.observation(state)[None],reps*budget,axis=0),cfg['proposal_seed']+i)
        bank.append(proposals.reshape(reps,budget,12,1))
    bank=np.stack(bank);sampling_seconds=time.perf_counter()-sampling_start
    np.savez_compressed(out/'proposals.npz',actions=bank,seeds=cfg['proposal_seed']+np.arange(n))
    flat=bank.reshape(n*reps,budget,12,1);raw=flat[:,0];starts=np.repeat(states,reps,axis=0)
    rawroll=pd.rollout(actor,starts,raw)
    np.savez_compressed(out/'raw.npz',chunk=raw,**rawroll)
    boot=np.random.default_rng(cfg['bootstrap_seed']).integers(0,n,(cfg['bootstrap_draws'],n))
    summary={'config':cfg,'proposal_sampling_seconds':sampling_seconds,'actual_proposals':n*reps*budget,
             'base_raw':metrics(rawroll,raw,np.ones(n*reps,dtype=bool)),'examples':{}}
    for e,example in enumerate(cfg['examples']):
        name=example['name'];lo=example['lower'];hi=example['upper']
        masks=((flat>=lo)&(flat<=hi)).all((2,3));valid=masks.any(1)
        indices=np.where(valid,masks.argmax(1),-1)
        q=flat[np.arange(len(flat)),np.maximum(indices,0)].copy()
        q[~valid]=0 # Simulation placeholder ONLY; refused outputs are marked and excluded from paired returns.
        remaining=masks.copy();remaining[np.arange(len(flat)),np.maximum(indices,0)]=False
        second_valid=remaining.any(1);second_indices=remaining.argmax(1)
        q2=flat[np.arange(len(flat)),second_indices]
        tick=time.perf_counter();projection=np.clip(raw,lo,hi);projection_seconds=time.perf_counter()-tick
        tick=time.perf_counter()
        refined=refine(model,pd.observation(starts),raw,lo,hi,cfg['refinement_seed']+e,cfg['refinement_schedule'])
        refinement_seconds=time.perf_counter()-tick
        outputs={'conditional':q,'projection':projection,'refinement':refined};rolls={};mm={};unsafe={};verification={}
        for method,actions in outputs.items():
            good=valid if method=='conditional' else np.ones(len(flat),dtype=bool)
            unsafe[method]=int((((actions<lo-1e-7)|(actions>hi+1e-7)|~np.isfinite(actions)).any((1,2))&good).sum())
            roll=pd.rollout(actor,starts,actions);rolls[method]=roll
            np.savez_compressed(out/f'{name}_{method}.npz',chunk=actions,valid=good,**roll)
            mm[method]=metrics(roll,actions,good);verification[method]=verify_gym(starts,roll)
        np.savez_compressed(out/f'{name}_selection.npz',acceptance=masks.reshape(n,reps,budget),
            first_indices=indices.reshape(n,reps),second_indices=second_indices.reshape(n,reps),valid=valid,
            second_valid=second_valid,second_conditional=q2)
        returns={k:v['rewards'].sum(1) for k,v in rolls.items()}
        cq=compare(returns['conditional'],returns['projection'],valid,cfg,boot)
        cr=compare(returns['refinement'],returns['projection'],np.ones(len(flat),dtype=bool),cfg,boot)
        qrvalid=valid&second_valid
        entry={'constraint':example,'acceptance':float(masks.mean()),
            'acceptance_per_state':masks.reshape(n,reps,budget).mean((1,2)).tolist(),
            'logical_proposals_mean':float(np.where(valid,indices+1,budget).mean()),
            'actual_proposals':int(n*reps*budget),'refusals':int((~valid).sum()),'solver_failures':0,
            'unsafe_outputs':unsafe,'gym_max_error':verification,'projection_seconds':projection_seconds,
            'refinement_seconds':refinement_seconds,'refinement_score_evaluations_per_output':1024,
            'metrics':mm,'conditional_minus_projection':cq,'refinement_minus_projection':cr,
            'mmd':{'conditional_vs_second_conditional':mmd(q[qrvalid],q2[qrvalid]),
                   'projection_vs_conditional':mmd(projection[valid],q[valid]),
                   'refinement_vs_conditional':mmd(refined[valid],q[valid])}}
        entry['all_feasibility_checks_pass']=bool(valid.all() and not any(unsafe.values()) and max(verification.values())<1e-7)
        summary['examples'][name]=entry
        print(json.dumps({'name':name,'acceptance':entry['acceptance'],'metrics':mm,'conditional_minus_projection':
            {k:v for k,v in cq.items() if k!='per_state_differences'},'refinement_minus_projection':
            {k:v for k,v in cr.items() if k!='per_state_differences'},'mmd':entry['mmd']}),flush=True)
    summary['total_seconds']=time.perf_counter()-start
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('Completed in',summary['total_seconds'],'seconds',flush=True)

if __name__=='__main__':main()

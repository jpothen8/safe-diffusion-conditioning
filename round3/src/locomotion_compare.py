"""Locked full-horizon five-arm MuJoCo comparison on fresh shared states."""
import argparse,hashlib,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import chunk_model,bayesfp,effort
import locomotion_env as le
from locomotion_probe import load
from locomotion_bayes_dev import mmd
ROOT=Path(__file__).resolve().parents[1]

def contrast(a,b,valid,n,reps,boot):
    delta=(a-b).reshape(n,reps);mask=valid.reshape(n,reps)
    means=np.array([d[v].mean() if v.any() else np.nan for d,v in zip(delta,mask)])
    if not np.isfinite(means).all():return {'mean':float(np.nanmean(means)),'incomplete':True}
    distribution=means[boot].mean(1)
    return {'mean':float(means.mean()),'ci95':np.quantile(distribution,[.025,.975]).tolist(),
            'ci99_375':np.quantile(distribution,[.003125,.996875]).tolist(),'state_differences':means.tolist()}

def describe(roll,chunk,cap,valid,q,templates):
    rms=np.sqrt(np.mean(chunk*chunk,-1));safe=(rms<=cap+1e-6)&(np.abs(chunk).max(-1)<=1+1e-6)&np.isfinite(chunk).all(-1)
    distances=np.sqrt(np.mean((chunk[:,None]-templates)**2,-1));family=distances.argmin(1)
    return {'mean_return':float(roll['rewards'][valid].sum(1).mean()),'mean_progress':float(roll['progress'][valid].mean()),
            'native_failures':int((roll['terminated']&valid).sum()),'refusals':int((~valid).sum()),
            'unsafe_outputs':int((~safe&valid).sum()),'boundary_fraction':float((np.abs(rms[valid]-cap)<=1e-5).mean()),
            'mmd_to_conditional':mmd(chunk[valid],q[valid]),'expert_family_fraction':float((family[valid]==0).mean()),
            'mean_distance_to_nearest_template':float(distances[valid].min(1).mean()),'rms_mean':float(rms[valid].mean())}

def main(task):
    cfg=json.loads((ROOT/'configs/locomotion.json').read_text());out=ROOT/f'results/{task}_comparison';out.mkdir(exist_ok=True)
    if (out/'lock.json').exists():raise RuntimeError('Preserve existing locked experiment')
    names=['LOCOMOTION_PROTOCOL.md','configs/locomotion.json','src/locomotion_compare.py','src/locomotion_env.py',
           'src/chunk_model.py','src/effort.py','src/bayesfp.py',f'assets/{task}_diffusion_v2.pt',f'assets/{task}-v5-SAC-expert.zip',
           f'assets/{task}-v5-SAC-medium.zip']
    lock={'utc':datetime.now(timezone.utc).isoformat(),'task':task,'config':cfg,'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in names}}
    (out/'lock.json').write_text(json.dumps(lock,indent=2));start=time.perf_counter()
    n=cfg['states'];reps=cfg['replicates'];budget=cfg['budget'];settings=cfg['tasks'][task]
    model=chunk_model.load(ROOT/f'assets/{task}_diffusion_v2.pt');actor=load(task,'expert')[0];medium=load(task,'medium')[0]
    le.H[task]=h=model.dimension//6
    snaps=[le.snapshot(task,cfg['state_seed']+i,actor,0 if i<n//2 else cfg['warmup_seconds']) for i in range(n)]
    (out/'snapshots.json').write_text(json.dumps(snaps,indent=2));repeated=[s for s in snaps for _ in range(reps)]
    obs=np.array([s['observation'] for s in repeated]);banks=[];tick=time.perf_counter()
    for i,s in enumerate(snaps):banks.append(chunk_model.sample(model,np.repeat(np.array(s['observation'])[None],reps*budget,0),cfg['proposal_seed']+i).reshape(reps,budget,-1))
    bank=np.stack(banks);sampling_seconds=time.perf_counter()-tick
    np.savez_compressed(out/'proposals.npz',actions=bank);flat=bank.reshape(n*reps,budget,-1);raw=flat[:,0]
    # Independent restoration/replay check before evaluating output quality.
    replay1=le.rollout(task,actor,[snaps[0]],raw[:1].reshape(1,h,6),seconds=.4)
    replay2=le.rollout(task,actor,[snaps[0]],raw[:1].reshape(1,h,6),seconds=.4)
    replay_error=float(np.max(np.abs(replay1['states']-replay2['states'])));assert replay_error==0
    templates=[]
    for policy in [actor,medium]:
        roll=le.rollout(task,policy,snaps,seconds=h*le.DT[task]);templates.append(np.repeat(roll['actions'].reshape(n,-1),reps,0))
    templates=np.stack(templates,1);np.savez_compressed(out/'templates.npz',actions=templates)
    base=le.rollout(task,actor,repeated,raw.reshape(-1,h,6),cfg['evaluation_seconds']);np.savez_compressed(out/'base.npz',chunk=raw,**base)
    boot=np.random.default_rng(cfg['bootstrap_seed']).integers(0,n,(cfg['bootstrap_draws'],n))
    summary={'task':task,'config':cfg,'horizon_controls':h,'horizon_seconds':h*le.DT[task],
             'sampling_seconds':sampling_seconds,'snapshot_replay_error':replay_error,
             'base':{'mean_return':float(base['rewards'].sum(1).mean()),'native_failures':int(base['terminated'].sum())},'cases':{}}
    for ei,cap in enumerate(settings['caps']):
        masks=np.sqrt(np.mean(flat*flat,-1))<=cap;valid=masks.any(1);idx=masks.argmax(1)
        q=flat[np.arange(n*reps),idx].copy();q[~valid]=0
        second=masks.copy();second[np.arange(n*reps),idx]=False;q2=flat[np.arange(n*reps),second.argmax(1)]
        np.savez_compressed(out/f'{cap}_selection.npz',masks=masks,indices=np.where(valid,idx,-1),valid=valid,second_q=q2)
        tick=time.perf_counter();p=effort.project_numpy(raw,cap);projection_seconds=time.perf_counter()-tick
        tick=time.perf_counter();r=effort.refine(model,obs,raw,cap,cfg['refinement_seed']+ei);refinement_seconds=time.perf_counter()-tick
        tick=time.perf_counter();b,diag=bayesfp.sample(model,obs,chunk_model.scheduler,model.dimension,cfg['bayesfp_seed']+ei,
                    strength=settings['bayesfp_strengths'][str(cap)],particles=32,radius=cap)
        bayes_seconds=time.perf_counter()-tick;cloud=diag.pop('final_particle_cloud')
        np.savez_compressed(out/f'{cap}_particles.npz',particles=cloud);b=np.clip(b,-1,1);bp=effort.project_numpy(b,cap)
        arms={'conditional':q,'projection':p,'refinement':r,'bayesfp':b,'bayesfp_projected':bp}
        entries={};returns={}
        for method,actions in arms.items():
            good=valid if method=='conditional' else np.ones(n*reps,dtype=bool)
            roll=le.rollout(task,actor,repeated,actions.reshape(-1,h,6),cfg['evaluation_seconds'])
            np.savez_compressed(out/f'{cap}_{method}.npz',chunk=actions,valid=good,**roll)
            entries[method]=describe(roll,actions,cap,good,q,templates);returns[method]=roll['rewards'].sum(1)
        contrasts={}
        for a,bn in [('conditional','projection'),('refinement','projection'),('refinement','bayesfp_projected'),('bayesfp_projected','projection')]:
            good=valid if a=='conditional' else np.ones(n*reps,dtype=bool)
            contrasts[a+'_minus_'+bn]=contrast(returns[a],returns[bn],good,n,reps,boot)
        result={'cap':cap,'acceptance':float(masks.mean()),'acceptance_by_state':masks.reshape(n,reps,budget).mean((1,2)).tolist(),
             'refusals':int((~valid).sum()),'logical_proposals_mean':float(np.where(valid,idx+1,budget).mean()),
             'projection_seconds':projection_seconds,'refinement_seconds':refinement_seconds,'bayesfp_seconds':bayes_seconds,
             'particle_diagnostics':diag,'q_vs_q_mmd':mmd(q[valid&second.any(1)],q2[valid&second.any(1)]),
             'arms':entries,'comparisons':contrasts}
        summary['cases'][str(cap)]=result
        print(task,cap,json.dumps({'arms':entries,'differences':{k:{a:v[a] for a in ['mean','ci95','ci99_375'] if a in v} for k,v in contrasts.items()}}),flush=True)
    summary['seconds']=time.perf_counter()-start;(out/'summary.json').write_text(json.dumps(summary,indent=2));print('Complete',task,summary['seconds'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['Walker2d','HalfCheetah']);main(p.parse_args().task)

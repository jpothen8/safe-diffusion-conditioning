"""Independent audit of saved proposals, outputs, statistics and data alignment."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym

ROOT=Path(__file__).resolve().parents[1]

def verify_hashes(paths,base):
    for path,want in paths.items():
        assert hashlib.sha256((base/path).read_bytes()).hexdigest()==want,path
    return len(paths)

def main():
    report={'checks':{},'cases':{},'training':{}}
    for task in ['pendulum','Walker2d','HalfCheetah']:
        folder=ROOT/f'results/{task}_comparison'
        lock=json.loads((folder/'lock.json').read_text())
        report['checks'][task+'_locked_hashes']=verify_hashes(lock['sha256'],ROOT.parent if task=='pendulum' else ROOT)
        summary=json.loads((folder/'summary.json').read_text());cfg=summary['config']
        n,reps,budget=cfg['states'],cfg['replicates'],cfg['budget']
        bank=np.load(folder/'proposals.npz')['actions'].reshape(n*reps,budget,-1)
        raw=bank[:,0];assert np.isfinite(bank).all()
        boot=np.random.default_rng(cfg['bootstrap_seed']).integers(0,n,(cfg['bootstrap_draws'],n))
        cases=summary.get('cases',summary.get('examples'))
        for label,case in cases.items():
            if task=='pendulum':
                lo,hi=(-2,0) if label=='negative' else (0,2)
                feasible=((bank>=lo)&(bank<=hi)).all(-1)
                projection=np.clip(raw,lo,hi);atol=1e-7
            else:
                cap=float(label);feasible=np.sqrt(np.mean(bank**2,-1))<=cap
                projection=raw*np.minimum(1,cap/np.maximum(np.sqrt(np.mean(raw**2,-1,keepdims=True)),1e-20));atol=1e-6
            selection=np.load(folder/f'{label}_selection.npz')
            indices=feasible.argmax(1);valid=feasible.any(1)
            assert valid.all() and case['refusals']==0
            assert np.array_equal(selection['indices'],indices)
            assert abs(case['acceptance']-feasible.mean())<1e-12
            assert abs(case['logical_proposals_mean']-(indices+1).mean())<1e-12
            second=feasible.copy();second[np.arange(n*reps),indices]=False
            assert second.any(1).all(), 'independent conditional diagnostic exhausted'
            assert np.array_equal(selection['second_q'].reshape(n*reps,-1),bank[np.arange(n*reps),second.argmax(1)])
            returns={};entries={}
            for arm,desc in case['arms'].items():
                data=np.load(folder/f'{label}_{arm}.npz');chunk=data['chunk'].reshape(n*reps,-1)
                assert np.isfinite(chunk).all()
                if task=='pendulum':safe=((chunk>=lo-atol)&(chunk<=hi+atol)).all(-1)
                else:safe=(np.sqrt(np.mean(chunk**2,-1))<=cap+atol)&(np.abs(chunk).max(-1)<=1+atol)
                assert int((~safe).sum())==desc['unsafe_outputs']
                if arm=='conditional':assert np.array_equal(chunk,bank[np.arange(n*reps),indices])
                if arm=='projection':assert np.allclose(chunk,projection,atol=1e-7,rtol=0)
                assert np.array_equal(data['actions'][:,:chunk.shape[1]//(1 if task=='pendulum' else 6)].reshape(n*reps,-1),chunk)
                returns[arm]=data['rewards'].sum(1)
                assert abs(returns[arm].mean()-desc.get('mean_return',desc.get('return_mean')))<1e-8
                entries[arm]={'outputs':len(chunk),'unsafe':int((~safe).sum())}
                if task!='pendulum':
                    assert data['terminated'].sum()==desc['native_failures']
                    for i,length in enumerate(data['lengths']):assert np.all(data['rewards'][i,length:]==0)
                    entries[arm]['terminated_rows']=np.flatnonzero(data['terminated']).tolist()
                    entries[arm]['termination_controls']=data['lengths'][data['terminated']].tolist()
            for name,contrast in case['comparisons'].items():
                a,b=name.split('_minus_');delta=(returns[a]-returns[b]).reshape(n,reps).mean(1)
                assert np.allclose(delta,contrast['state_differences'],atol=1e-9,rtol=0)
                assert abs(delta.mean()-contrast['mean'])<1e-9
                assert np.allclose(np.quantile(delta[boot].mean(1),[.025,.975]),contrast['ci95'],atol=1e-9,rtol=0)
                probs,key=([.00625,.99375],'ci98_75') if task=='pendulum' else ([.003125,.996875],'ci99_375')
                assert np.allclose(np.quantile(delta[boot].mean(1),probs),contrast[key],atol=1e-9,rtol=0)
            report['cases'][task+'_'+label]={'proposals':bank.shape[0]*budget,'budget_failures':int((~valid).sum()),'arms':entries}
    # Independent complete-episode replay validates action-time alignment.
    for task in ['Walker2d','HalfCheetah']:
        data=np.load(ROOT/f'results/{task}_training/episodes.npz')
        groups=[set(data[k+'_groups']) for k in ['train','validation','test']]
        assert not(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        assert set.union(*groups)==set(range(64))
        train_indices=np.concatenate([np.arange(s,s+l) for s,l,g in zip(data['episode_starts'],data['episode_lengths'],data['groupids']) if g in groups[0]])
        obs=data['observations'][train_indices]
        weights=torch.load(ROOT/f'assets/{task}_diffusion_v2.pt',map_location='cpu',weights_only=True)['state_dict']
        assert np.array_equal(weights['obs_mean'].numpy(),obs.mean(0))
        assert np.array_equal(weights['obs_std'].numpy(),np.maximum(obs.std(0),1e-4))
        obs_error=reward_error=0.
        for episode in [0,1]:
            env=gym.make(task+'-v5');o,_=env.reset(seed=92000+int(data['groupids'][episode]))
            s=int(data['episode_starts'][episode]);length=int(data['episode_lengths'][episode])
            for t in range(length):
                obs_error=max(obs_error,float(np.max(np.abs(o.astype(np.float32)-data['observations'][s+t]))))
                o,r,terminated,truncated,_=env.step(data['actions'][s+t])
                reward_error=max(reward_error,abs(r-data['rewards'][s+t]))
                assert not(terminated or truncated) or t==length-1
            assert terminated or truncated
            env.close()
        assert obs_error==0 and reward_error<1e-10
        report['training'][task]={'group_split_disjoint':True,'train_only_normalizer_exact':True,
             'complete_episodes_replayed':2,'observation_float32_error':obs_error,'reward_error':reward_error}
    assets=json.loads((ROOT/'records/assets.json').read_text())
    for item in assets:assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256']
    report['checks']['downloaded_asset_hashes']=len(assets)
    report['passed']=True
    (ROOT/'results/audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()

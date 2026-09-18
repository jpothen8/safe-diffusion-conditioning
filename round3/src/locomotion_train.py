"""Complete-episode data, group split before chunks, and a bounded DDPM fit."""
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from chunk_model import Model,scheduler
from locomotion_probe import load
ROOT=Path(__file__).resolve().parents[1]

def train(task):
    start=time.perf_counter();out=ROOT/f'results/{task}_training';out.mkdir(exist_ok=True)
    horizon=50 if task=='Walker2d' else 8
    models={level:load(task,level)[0] for level in ['expert','medium']}
    observations=[];actions=[];rewards=[];lengths=[];groupids=[];levels=[];terminals=[]
    # 64 groups x two 1000-control expert episodes; full episodes, no chunk leak.
    for group in range(64):
        for level,actor in models.items():
            env=gym.make(task+'-v5');obs,_=env.reset(seed=92000+group);oo=[];aa=[];rr=[]
            while True:
                oo.append(obs.copy());action,_=actor.predict(obs,deterministic=True);aa.append(action)
                obs,r,terminated,truncated,_=env.step(action);rr.append(r)
                if terminated or truncated:break
            env.close();observations.append(np.array(oo,dtype=np.float32));actions.append(np.array(aa,dtype=np.float32))
            rewards.append(np.array(rr));lengths.append(len(rr));groupids.append(group);levels.append(level);terminals.append(terminated)
    rng=np.random.default_rng(92100);ids=rng.permutation(64);splits={'train':ids[:48],'validation':ids[48:56],'test':ids[56:]}
    ends=np.cumsum(lengths);starts=np.r_[0,ends[:-1]]
    np.savez_compressed(out/'episodes.npz',observations=np.concatenate(observations),actions=np.concatenate(actions),
        rewards=np.concatenate(rewards),episode_starts=starts,episode_lengths=lengths,groupids=groupids,levels=levels,
        terminated=terminals,**{k+'_groups':v for k,v in splits.items()})
    def chunks(groups):
        oo=[];aa=[]
        for i,g in enumerate(groupids):
            if g not in groups or lengths[i]<horizon:continue
            windows=np.lib.stride_tricks.sliding_window_view(actions[i],horizon,axis=0).transpose(0,2,1)
            assert np.array_equal(windows[0],actions[i][:horizon])
            oo.append(observations[i][:len(windows)]);aa.append(windows.reshape(len(windows),-1))
        return np.concatenate(oo),np.concatenate(aa)
    tr_o,tr_a=chunks(splits['train']);va_o,va_a=chunks(splits['validation'])
    torch.manual_seed(92200);torch.cuda.manual_seed_all(92200);torch.set_num_threads(4)
    model=Model(horizon*6).cuda()
    # Action normalization is fixed physical identity [-1,1], independent of data.
    fitting_obs=np.concatenate([o for o,g in zip(observations,groupids) if g in splits['train']])
    model.obs_mean.copy_(torch.tensor(fitting_obs.mean(0),device='cuda'))
    model.obs_std.copy_(torch.tensor(np.maximum(fitting_obs.std(0),1e-4),device='cuda'))
    ema=copy.deepcopy(model);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5)
    tr_o,tr_a,va_o,va_a=[torch.as_tensor(v,device='cuda') for v in [tr_o,tr_a,va_o,va_a]]
    schedule=scheduler();logs=[];fit_start=time.perf_counter()
    for step in range(16000):
        idx=torch.randint(len(tr_o),(256,),device='cuda');noise=torch.randn((256,horizon*6),device='cuda')
        t=torch.randint(0,100,(256,),device='cuda')
        loss=(model(schedule.add_noise(tr_a[idx],noise,t),t,tr_o[idx])-noise).square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        with torch.no_grad():
            for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,.005)
        if (step+1)%2000==0:
            with torch.no_grad():
                vi=torch.randint(len(va_o),(512,),device='cuda');vn=torch.randn((512,horizon*6),device='cuda');vt=torch.randint(0,100,(512,),device='cuda')
                vl=(ema(schedule.add_noise(va_a[vi],vn,vt),vt,va_o[vi])-vn).square().mean()
            row={'step':step+1,'loss':float(loss),'validation_loss':float(vl),'seconds':time.perf_counter()-fit_start}
            logs.append(row);print(task,json.dumps(row),flush=True)
    path=ROOT/f'assets/{task}_diffusion.pt'
    torch.save({'dimension':horizon*6,'state_dict':{k:v.cpu() for k,v in ema.state_dict().items()}},path)
    desc={'task':task,'horizon':horizon,'physical_horizon':.4,'episodes':len(lengths),'controls':sum(lengths),
        'terminated_episodes':int(sum(terminals)),'group_split':{k:v.tolist() for k,v in splits.items()},'train_chunks':len(tr_o),
        'parameters':sum(p.numel() for p in model.parameters()),'steps':16000,'batch':256,'collection_seed_start':92000,
        'split_seed':92100,'training_seed':92200,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'seconds':time.perf_counter()-start,'training_seconds':time.perf_counter()-fit_start,'logs':logs}
    (out/'summary.json').write_text(json.dumps(desc,indent=2));print(json.dumps(desc),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['Walker2d','HalfCheetah']);train(p.parse_args().task)

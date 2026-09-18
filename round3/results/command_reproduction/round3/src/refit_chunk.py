"""One bounded quality repair: shorter Walker chunks, more early-state training.

Uses the already collected complete episodes and original train/validation split.
No safety outcome or task reward enters the fitting objective.
"""
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from chunk_model import Model,scheduler
ROOT=Path(__file__).resolve().parents[1]

def main(task):
    h=12 if task=='Walker2d' else 8;dt=.008 if task=='Walker2d' else .05
    data=np.load(ROOT/f'results/{task}_training/episodes.npz');out=ROOT/f'results/{task}_refit';out.mkdir(exist_ok=True)
    def chunk(groups):
        oo=[];aa=[];early=[]
        for start,length,group in zip(data['episode_starts'],data['episode_lengths'],data['groupids']):
            if group not in groups or length<h:continue
            actions=data['actions'][start:start+length]
            windows=np.lib.stride_tricks.sliding_window_view(actions,h,axis=0).transpose(0,2,1)
            aa.append(windows.reshape(len(windows),-1));oo.append(data['observations'][start:start+len(windows)])
            early.append(np.arange(len(windows))<round(2/dt))
        return np.concatenate(oo),np.concatenate(aa),np.concatenate(early)
    tr_o,tr_a,early=chunk(data['train_groups']);va_o,va_a,_=chunk(data['validation_groups'])
    torch.manual_seed(92300);torch.cuda.manual_seed_all(92300);torch.set_num_threads(4)
    model=Model(h*6).cuda()
    # Match the same training-episode-only normalization as the first fit.
    idx=np.concatenate([np.arange(s,s+l) for s,l,g in zip(data['episode_starts'],data['episode_lengths'],data['groupids']) if g in data['train_groups']])
    model.obs_mean.copy_(torch.tensor(data['observations'][idx].mean(0),device='cuda'))
    model.obs_std.copy_(torch.tensor(np.maximum(data['observations'][idx].std(0),1e-4),device='cuda'))
    ema=copy.deepcopy(model);opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5)
    tr_o,tr_a,va_o,va_a=[torch.as_tensor(x,device='cuda') for x in [tr_o,tr_a,va_o,va_a]]
    early=torch.as_tensor(np.flatnonzero(early),device='cuda');sched=scheduler();logs=[];start=time.perf_counter()
    for step in range(48000):
        idx=torch.cat([torch.randint(len(tr_o),(256,),device='cuda'),early[torch.randint(len(early),(256,),device='cuda')]])
        noise=torch.randn((512,h*6),device='cuda');t=torch.randint(0,100,(512,),device='cuda')
        loss=(model(sched.add_noise(tr_a[idx],noise,t),t,tr_o[idx])-noise).square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        with torch.no_grad():
            for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,.005)
        if (step+1)%8000==0:
            with torch.no_grad():
                vi=torch.randint(len(va_o),(512,),device='cuda');vn=torch.randn((512,h*6),device='cuda');vt=torch.randint(0,100,(512,),device='cuda')
                val=(ema(sched.add_noise(va_a[vi],vn,vt),vt,va_o[vi])-vn).square().mean()
            row={'step':step+1,'loss':float(loss),'validation_loss':float(val),'seconds':time.perf_counter()-start};logs.append(row);print(task,json.dumps(row),flush=True)
    path=ROOT/f'assets/{task}_diffusion_v2.pt';torch.save({'dimension':h*6,'state_dict':{k:v.cpu() for k,v in ema.state_dict().items()}},path)
    summary={'task':task,'horizon':h,'horizon_seconds':h*dt,'steps':48000,'batch':512,'seed':92300,
             'sampling':'half uniform training chunks, half training chunks beginning within episode first two seconds',
             'normalization':'training episodes only, unchanged group split','checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
             'seconds':time.perf_counter()-start,'logs':logs}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['Walker2d','HalfCheetah']);main(p.parse_args().task)

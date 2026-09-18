"""Episode-split state diffusion imitation of the public actor and its reflection."""
import copy
import hashlib
import json
import time
import numpy as np
import torch
from torch import nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import core
import pendulum as pd

HORIZON=12
class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(HORIZON+3+32,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),
                               nn.Linear(256,256),nn.SiLU(),nn.Linear(256,HORIZON))
        self.register_buffer("obs_mean",torch.zeros(3));self.register_buffer("obs_std",torch.ones(3))
        self.register_buffer("action_center",torch.zeros(1));self.register_buffer("action_scale",torch.ones(1)*2)
    def forward(self,x,t,obs):
        t=torch.as_tensor(t,device=x.device).expand(x.shape[0]).float()
        freq=torch.exp(torch.arange(16,device=x.device)*(-np.log(10000)/15))
        emb=t[:,None]*freq[None]
        cond=(obs-self.obs_mean)/self.obs_std
        return self.net(torch.cat([x,cond,emb.sin(),emb.cos()],dim=-1))

def scheduler():
    return DDPMScheduler(num_train_timesteps=100,beta_schedule="squaredcos_cap_v2",
                         clip_sample=True,prediction_type="epsilon",variance_type="fixed_small")

def sample(model,observations,seed,return_latent=False):
    torch.manual_seed(int(seed));torch.cuda.manual_seed_all(int(seed))
    obs=torch.as_tensor(observations,dtype=torch.float32,device=next(model.parameters()).device)
    x=torch.randn((len(obs),HORIZON),device=obs.device)
    sched=scheduler();sched.set_timesteps(100)
    with torch.inference_mode():
        for t in sched.timesteps:
            eps=model(x,t,obs)
            x=sched.step(eps,t,x).prev_sample
        actions=(x*model.action_scale+model.action_center).clamp(-2,2)
    return actions.cpu().numpy()[...,None]

def load(device="cuda"):
    payload=torch.load(core.ROOT/"assets/pendulum_diffusion.pt",map_location="cpu",weights_only=True)
    model=Denoiser();model.load_state_dict(payload["state_dict"]);model.eval().to(device)
    return model

def train():
    out=core.ROOT/"results/pendulum";out.mkdir(exist_ok=True)
    rng=np.random.default_rng(62000)
    # Each initial-state group has both complete, coherent expert episodes.
    groups=320
    states=np.column_stack([np.pi+rng.uniform(-.15,.15,groups),rng.uniform(-.15,.15,groups)])
    # Add broad reset episodes for continuation coverage. Both mirror policies share every reset.
    states[240:]=np.column_stack([rng.uniform(-np.pi,np.pi,80),rng.uniform(-1,1,80)])
    actor=pd.actor()
    data=[pd.rollout(actor,states,reflection=mode) for mode in [False,True]]
    obs=np.stack([d["observations"] for d in data],axis=1)
    actions=np.stack([d["actions"] for d in data],axis=1)
    reward=np.stack([d["rewards"] for d in data],axis=1)
    permutation=rng.permutation(groups)
    split={"train":permutation[:256],"validation":permutation[256:288],"test":permutation[288:]}
    np.savez_compressed(out/"demonstrations.npz",initial_states=states,observations=obs,actions=actions,rewards=reward,
                        **{k+"_groups":v for k,v in split.items()})
    # No overlap or mirrored-group leakage is possible after this split.
    assert not set(split["train"])&set(split["validation"])
    assert not set(split["train"])&set(split["test"])
    train_obs=obs[split["train"]].reshape(-1,3)
    train_act=actions[split["train"]].reshape(-1,1)
    def chunks(ids):
        oo=obs[ids,:,:200-HORIZON+1,:].reshape(-1,3)
        aa=np.lib.stride_tricks.sliding_window_view(actions[ids,...,0],HORIZON,axis=-1).reshape(-1,HORIZON)
        assert len(oo)==len(aa)
        # Action a[t] accompanies pre-action o[t], including the first action of each chunk.
        np.testing.assert_array_equal(aa[0],actions[ids[0],0,:HORIZON,0])
        np.testing.assert_array_equal(oo[0],obs[ids[0],0,0])
        return torch.tensor(oo,device="cuda"),torch.tensor(aa,device="cuda")
    torch.manual_seed(62001);torch.cuda.manual_seed_all(62001);torch.set_num_threads(4)
    model=Denoiser().cuda()
    model.obs_mean.copy_(torch.tensor(train_obs.mean(axis=0),device="cuda"))
    model.obs_std.copy_(torch.tensor(np.maximum(train_obs.std(axis=0),1e-4),device="cuda"))
    model.action_center.copy_(torch.tensor((train_act.max(axis=0)+train_act.min(axis=0))/2,device="cuda"))
    model.action_scale.copy_(torch.tensor((train_act.max(axis=0)-train_act.min(axis=0))/2,device="cuda"))
    ema=copy.deepcopy(model);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5)
    train_o,train_a=chunks(split["train"]);val_o,val_a=chunks(split["validation"])
    train_a=(train_a-model.action_center)/model.action_scale
    val_a=(val_a-model.action_center)/model.action_scale
    sched=scheduler();logs=[];start=time.perf_counter()
    for step in range(12000):
        idx=torch.randint(len(train_o),(512,),device="cuda")
        noise=torch.randn((512,HORIZON),device="cuda")
        t=torch.randint(0,100,(512,),device="cuda")
        x=sched.add_noise(train_a[idx],noise,t)
        loss=(model(x,t,train_o[idx])-noise).square().mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        with torch.no_grad():
            for pe,p in zip(ema.parameters(),model.parameters()):pe.lerp_(p,.005)
        if (step+1)%1000==0:
            with torch.no_grad():
                vi=torch.randint(len(val_o),(1024,),device="cuda")
                vn=torch.randn((1024,HORIZON),device="cuda");vt=torch.randint(0,100,(1024,),device="cuda")
                vl=(ema(sched.add_noise(val_a[vi],vn,vt),vt,val_o[vi])-vn).square().mean()
            row={"step":step+1,"train_loss":float(loss),"validation_loss":float(vl),"seconds":time.perf_counter()-start}
            logs.append(row);print(json.dumps(row),flush=True)
    ema.eval()
    torch.save({"state_dict":{k:v.cpu() for k,v in ema.state_dict().items()}},core.ROOT/"assets/pendulum_diffusion.pt")
    result={"groups":groups,"episodes":groups*2,"episode_length":200,"horizon":HORIZON,
            "train_groups":len(split["train"]),"validation_groups":len(split["validation"]),"test_groups":len(split["test"]),
            "train_chunks":len(train_o),"normalization":"training episodes only","steps":12000,
            "parameters":sum(p.numel() for p in model.parameters()),"seconds":time.perf_counter()-start,
            "checkpoint_sha256":hashlib.sha256((core.ROOT/"assets/pendulum_diffusion.pt").read_bytes()).hexdigest(),"logs":logs}
    (out/"training.json").write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=="__main__":train()

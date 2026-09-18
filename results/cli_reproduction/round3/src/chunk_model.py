"""Modest state-conditioned DDPM for complete-episode MuJoCo imitation."""
import torch,numpy as np
from torch import nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

class Model(nn.Module):
    def __init__(self,dimension,obs_dimension=17):
        super().__init__();self.dimension=dimension;self.obs_dimension=obs_dimension
        self.net=nn.Sequential(nn.Linear(dimension+obs_dimension+32,512),nn.SiLU(),nn.Linear(512,512),nn.SiLU(),
                               nn.Linear(512,512),nn.SiLU(),nn.Linear(512,dimension))
        self.register_buffer('obs_mean',torch.zeros(obs_dimension));self.register_buffer('obs_std',torch.ones(obs_dimension))
    def forward(self,x,t,obs):
        t=torch.as_tensor(t,device=x.device).expand(len(x)).float()
        f=torch.exp(torch.arange(16,device=x.device)*(-np.log(10000)/15))
        emb=t[:,None]*f[None]
        return self.net(torch.cat([x,(obs-self.obs_mean)/self.obs_std,emb.sin(),emb.cos()],-1))

def scheduler():
    return DDPMScheduler(num_train_timesteps=100,beta_schedule='squaredcos_cap_v2',clip_sample=True,
                         prediction_type='epsilon',variance_type='fixed_small')

def sample(model,observations,seed):
    torch.manual_seed(int(seed));torch.cuda.manual_seed_all(int(seed))
    device=next(model.parameters()).device;obs=torch.as_tensor(observations,device=device,dtype=torch.float32)
    x=torch.randn((len(obs),model.dimension),device=device);schedule=scheduler();schedule.set_timesteps(100)
    with torch.inference_mode():
        for t in schedule.timesteps:x=schedule.step(model(x,t,obs),t,x).prev_sample
    return x.clamp(-1,1).cpu().numpy()

def load(path,device='cuda'):
    payload=torch.load(path,weights_only=True,map_location='cpu')
    model=Model(payload['dimension']);model.load_state_dict(payload['state_dict']);model.eval().to(device)
    return model

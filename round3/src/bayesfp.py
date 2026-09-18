"""Paper-derived BayesFP DDPM adapter (not an official released implementation).

Source: Sirigiri et al., arXiv:2606.21014v1, Algorithm 1 and Appendix G.4.
Constant beta=-strength. For VP forward drift f*dt=-delta*x/2:
  epsilon_tilt = epsilon - beta*sqrt(1-alpha_bar)*grad(J)/2
  dw = beta*delta*<grad(J), score+x>/2.
Systematic resampling is active on generation time [.05,.95], period one.
The prior's own scheduler/clipping defines the actual base sampling law.
"""
import numpy as np
import torch

def box_cost_gradient(x,lower,upper):
    residual=x-x.clamp(lower,upper)
    return residual.square().mean(-1),2*residual/x.shape[-1]

def ball_cost_gradient(x,radius):
    # Squared RMS radial excess, normalized to be independent of chunk dimension.
    rms=x.square().mean(-1).clamp_min(1e-20).sqrt()
    residual=(rms-radius).clamp_min(0)
    return residual.square(),2*residual[...,None]*x/(rms[...,None]*x.shape[-1])

def systematic(weights):
    b,k=weights.shape
    cdf=weights.cumsum(-1);cdf[:,-1]=1
    positions=(torch.rand((b,1),device=weights.device)+torch.arange(k,device=weights.device))/k
    return torch.searchsorted(cdf.contiguous(),positions.contiguous()).clamp_max(k-1)

def sample(model,observations,scheduler_factory,dimension,seed,strength=20.,particles=32,
           lower=-float('inf'),upper=0.,radius=None):
    torch.manual_seed(int(seed));torch.cuda.manual_seed_all(int(seed))
    device=next(model.parameters()).device
    obs=torch.as_tensor(observations,device=device,dtype=torch.float32)
    b=len(obs);k=particles;obs=obs.repeat_interleave(k,0)
    x=torch.randn((b*k,dimension),device=device);weights=torch.zeros((b,k),device=device)
    sched=scheduler_factory();sched.set_timesteps(100)
    abars=sched.alphas_cumprod.to(device);ess=[];ancestors=torch.arange(k,device=device)[None].expand(b,k).clone()
    with torch.inference_mode():
        for n,t in enumerate(sched.timesteps):
            timestep=int(t);ab=abars[timestep];std=(1-ab).sqrt()
            epsilon=model(x,t,obs)
            if strength:
                if radius is None:cost,grad=box_cost_gradient(x,lower,upper)
                else:cost,grad=ball_cost_gradient(x,radius)
                delta=1-ab/(abars[timestep-1] if timestep>0 else 1.)
                score=-epsilon/std
                increment=(-.5*strength*delta*(grad*(score+x)).sum(-1)).reshape(b,k)
                increment=increment-increment.mean(-1,keepdim=True)
                epsilon=epsilon+.5*strength*std*grad
            x=sched.step(epsilon,t,x).prev_sample
            active=.05<=(n/99)<=.95
            if strength and active:
                weights=weights+increment
                probabilities=weights.softmax(-1)
                ess.append((1/probabilities.square().sum(-1)).cpu().numpy())
                indices=systematic(probabilities)
                x=x.reshape(b,k,dimension).gather(1,indices[...,None].expand(-1,-1,dimension)).reshape(b*k,dimension)
                ancestors=ancestors.gather(1,indices);weights.zero_()
            elif not active:
                weights.zero_()
        indices=torch.multinomial(weights.softmax(-1),1).squeeze(-1)
        selected=x.reshape(b,k,dimension)[torch.arange(b,device=device),indices]
    return selected.cpu().numpy(),{'particles':k,'strength':strength,'score_evaluations_per_output':100*k,
        'ess_mean':float(np.mean(ess)) if ess else float(k),
        'ess_min':float(np.min(ess)) if ess else float(k),
        'unique_initial_ancestors_mean':float(np.mean([len(torch.unique(row)) for row in ancestors])),
        'final_particle_cloud':x.reshape(b,k,dimension).cpu().numpy()}

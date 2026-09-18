"""Global Euclidean projection onto an RMS actuator-command budget and box.

KKT: z_i=clip(y_i/(1+lambda),-1,1). Choose lambda>=0 so the norm constraint
is active unless coordinate clipping is already inside the ball. For a raw
base sample in the box, ordinary radial scaling is the exact solution.
"""
import numpy as np
import torch
from chunk_model import scheduler

def project_numpy(x,radius):
    x=np.asarray(x);rms=np.sqrt(np.mean(x*x,axis=-1,keepdims=True))
    if np.max(np.abs(x))<=1+1e-7:return x*np.minimum(1,radius/np.maximum(rms,1e-20))
    clipped=np.clip(x,-1,1);inside=np.mean(clipped**2,axis=-1,keepdims=True)<=radius**2
    low=np.zeros_like(rms);high=np.ones_like(rms)
    for _ in range(50):
        mid=(low+high)/2;trial=np.clip(x*mid,-1,1)
        feasible=np.mean(trial*trial,axis=-1,keepdims=True)<=radius**2
        low=np.where(feasible,mid,low);high=np.where(feasible,high,mid)
    return np.where(inside,clipped,np.clip(x*low,-1,1))

def project_torch(x,radius):
    clipped=x.clamp(-1,1);inside=clipped.square().mean(-1,keepdim=True)<=radius**2
    rms=x.square().mean(-1,keepdim=True).clamp_min(1e-20).sqrt()
    radial=x*(radius/rms).clamp_max(1)
    simple=radial.abs().amax(-1,keepdim=True)<=1
    # Bisection only if the radial solution crosses a physical actuator bound.
    complicated=(~inside&~simple).squeeze(-1)
    result=torch.where(inside,clipped,radial)
    if bool(complicated.any()):
        values=x[complicated];low=torch.zeros((len(values),1),device=x.device);high=torch.ones_like(low)
        for _ in range(32):
            mid=(low+high)/2;trial=(values*mid).clamp(-1,1)
            feasible=trial.square().mean(-1,keepdim=True)<=radius**2
            low=torch.where(feasible,mid,low);high=torch.where(feasible,high,mid)
        result[complicated]=(values*low).clamp(-1,1)
    return result

def refine(model,obs,raw,radius,seed):
    torch.manual_seed(int(seed));torch.cuda.manual_seed_all(int(seed));device=next(model.parameters()).device
    obs=torch.as_tensor(obs,dtype=torch.float32,device=device)
    x=project_torch(torch.as_tensor(raw,dtype=torch.float32,device=device),radius)
    abars=scheduler().alphas_cumprod.to(device)
    with torch.inference_mode():
        for level,steps in [(20,256),(5,256),(0,512)]:
            ab=abars[level];sigma=((1-ab)/ab).sqrt()
            for k in range(steps):
                alpha=.15*sigma.square()/(1+k/64)**.6
                score=-model(ab.sqrt()*x,level,obs)/sigma
                x=project_torch(x+alpha*score+(2*alpha).sqrt()*torch.randn_like(x),radius)
    return x.cpu().numpy()

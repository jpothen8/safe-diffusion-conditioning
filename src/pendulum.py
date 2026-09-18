"""Public TD3 actor and exact Gym Pendulum-v1 rollout helpers."""
import io
import zipfile
import numpy as np
import torch
from torch import nn
import gym
import core

DT=.05;H=16

def actor():
    model=nn.Sequential(nn.Linear(3,400),nn.ReLU(),nn.Linear(400,300),nn.ReLU(),nn.Linear(300,1),nn.Tanh())
    with zipfile.ZipFile(core.ROOT/"assets/td3-Pendulum-v1.zip") as archive:
        weights=torch.load(io.BytesIO(archive.read("policy.pth")),map_location="cpu",weights_only=True)
    model.load_state_dict({k.removeprefix("actor.mu."):v for k,v in weights.items() if k.startswith("actor.mu.")})
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    torch.set_num_threads(2)
    return model

def observation(state):
    return np.stack([np.cos(state[...,0]),np.sin(state[...,0]),state[...,1]],axis=-1).astype(np.float32)

def act(model,obs,reflection=False):
    values=np.array(obs,dtype=np.float32,copy=True)
    if reflection:values[...,1:]*=-1
    with torch.inference_mode():
        a=model(torch.from_numpy(values)).numpy()*2
    return -a if reflection else a

def dynamics(state,actions):
    theta,velocity=state[...,0],state[...,1]
    # Gym's scalar NumPy arithmetic promotes the torque to float64. Explicitly
    # match it for batched arrays, where NumPy would retain float32 otherwise.
    u=np.clip(np.asarray(actions,dtype=np.float64).squeeze(-1),-2,2)
    wrapped=(theta+np.pi)%(2*np.pi)-np.pi
    reward=-(wrapped**2+.1*velocity**2+.001*u**2)
    newvelocity=np.clip(velocity+(15*np.sin(theta)+3*u)*DT,-8,8)
    newtheta=theta+newvelocity*DT
    return np.stack([newtheta,newvelocity],axis=-1),reward

def rollout(model,states,first_actions=None,reflection=False,steps=200):
    state=np.array(states,dtype=float,copy=True)
    observations=[];actions=[];rewards=[];history=[]
    for t in range(steps):
        obs=observation(state)
        aa=first_actions[:,t] if first_actions is not None and t<first_actions.shape[1] else act(model,obs,reflection)
        history.append(state.copy());observations.append(obs);actions.append(aa)
        state,rr=dynamics(state,aa);rewards.append(rr)
    return {"states":np.stack(history,axis=1),"observations":np.stack(observations,axis=1),
            "actions":np.stack(actions,axis=1),"rewards":np.stack(rewards,axis=1)}

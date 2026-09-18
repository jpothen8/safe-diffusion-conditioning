"""Full MuJoCo integration-state snapshots and paired native-reward rollouts."""
import numpy as np
import gymnasium as gym
import mujoco
DT={'Walker2d':.008,'HalfCheetah':.05}
H={'Walker2d':50,'HalfCheetah':8}

def snapshot(task,seed,actor,warm_seconds=0):
    env=gym.make(task+'-v5');obs,_=env.reset(seed=seed)
    for _ in range(round(warm_seconds/DT[task])):
        action,_=actor.predict(obs,deterministic=True);obs,_,term,trunc,_=env.step(action)
        if term or trunc:raise RuntimeError('Warmup terminated; do not select a different state silently')
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    state=np.empty(mujoco.mj_stateSize(env.unwrapped.model,spec))
    mujoco.mj_getState(env.unwrapped.model,env.unwrapped.data,state,spec)
    result={'task':task,'seed':seed,'warm_seconds':warm_seconds,'integration_state':state.tolist(),
            'state_spec':int(spec),'observation':obs.tolist()}
    env.close();return result

def restore(snap):
    env=gym.make(snap['task']+'-v5');env.reset(seed=snap['seed'])
    mujoco.mj_setState(env.unwrapped.model,env.unwrapped.data,np.array(snap['integration_state']),snap['state_spec'])
    mujoco.mj_forward(env.unwrapped.model,env.unwrapped.data)
    return env

def rollout(task,actor,snaps,chunks=None,seconds=4.):
    envs=[restore(s) for s in snaps];n=len(envs);steps=round(seconds/DT[task]);h=H[task]
    obs=np.array([e.unwrapped._get_obs() for e in envs]);states=[];actions=[]
    rewards=np.zeros((n,steps));active=np.ones(n,dtype=bool);terminated=np.zeros(n,dtype=bool);lengths=np.zeros(n,dtype=int)
    initial_x=np.array([e.unwrapped.data.qpos[0] for e in envs])
    for t in range(steps):
        states.append(np.array([np.r_[e.unwrapped.data.qpos,e.unwrapped.data.qvel] for e in envs]))
        if chunks is not None and t<h:aa=chunks[:,t]
        else:aa,_=actor.predict(obs,deterministic=True)
        actions.append(aa.copy())
        for i in np.flatnonzero(active):
            obs[i],rewards[i,t],term,trunc,_=envs[i].step(aa[i]);lengths[i]+=1
            if term or trunc:active[i]=False;terminated[i]=term
    progress=np.array([e.unwrapped.data.qpos[0] for e in envs])-initial_x
    for e in envs:e.close()
    return {'rewards':rewards,'actions':np.stack(actions,1),'states':np.stack(states,1),'terminated':terminated,
            'lengths':lengths,'progress':progress}

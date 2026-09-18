import json
import numpy as np
import gym
import core
import pendulum as pd

out=core.ROOT/"results/pendulum";out.mkdir(exist_ok=True)
model=pd.actor()
# Untouched official environment/actor smoke, with the original action scaling.
smoke=[];err=0.
for seed in range(10):
    env=gym.make("Pendulum-v1")
    obs,_=env.reset(seed=seed)
    rewards=[]
    for t in range(200):
        state=env.unwrapped.state.copy()
        aa=pd.act(model,obs)
        pred,rr=pd.dynamics(state,aa)
        obs,r,terminated,truncated,info=env.step(aa)
        err=max(err,float(np.max(np.abs(pred-env.unwrapped.state))),abs(float(rr)-r))
        rewards.append(r)
    smoke.append({"seed":seed,"return":sum(rewards)})
    env.close()
rng=np.random.default_rng(61001)
states=np.column_stack([np.pi+rng.uniform(-.05,.05,40),rng.uniform(-.05,.05,40)])
positive=pd.rollout(model,states,reflection=False)
negative=pd.rollout(model,states,reflection=True)
# Both are coherent closed-loop policies, not independent action noise.
all_a=np.stack([positive["actions"][:,:16],negative["actions"][:,:16]],axis=1)
feasible=np.all(all_a<=1e-7,axis=(2,3))
projected=np.minimum(all_a,0)
common=[]
for mode in range(2):
    raw=pd.rollout(model,states,first_actions=all_a[:,mode],reflection=False)
    projection=pd.rollout(model,states,first_actions=projected[:,mode],reflection=False)
    common.append({"mode":mode,"safe_fraction":float(feasible[:,mode].mean()),
                   "expert_return":float([positive,negative][mode]["rewards"].sum(axis=1).mean()),
                   "common_return":float(raw["rewards"].sum(axis=1).mean()),
                   "projected_return":float(projection["rewards"].sum(axis=1).mean()),
                   "action_min":float(all_a[:,mode].min()),"action_max":float(all_a[:,mode].max())})
np.savez_compressed(out/"expert_probe.npz",initial_states=states,actions=all_a,feasible=feasible,
                    expert_positive=positive["states"],expert_negative=negative["states"])
result={"smoke":smoke,"smoke_mean_return":float(np.mean([r["return"] for r in smoke])),
        "max_dynamics_error":err,"modes":common,"label":"preliminary non-diffusion expert probe"}
(out/"expert_probe.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

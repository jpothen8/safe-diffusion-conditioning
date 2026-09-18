"""Shared simulation routines for the follow-up; the original pilot stays frozen."""
import copy
import json
import numpy as np
import core
from safety import observe_substeps, affine_dynamics

def restore(snap):
    return core.snapshot(snap["seed"], snap["prefix"])

def simulate_chunk(snap, action):
    env,hist=restore(snap)
    initial=core.numeric_state(env)
    trace=observe_substeps(env)
    rewards=[]; contacts=[]; first_done=None
    for act in action:
        r,done,info=core.step(env,hist,act)
        rewards.append(r if first_done is None else 1.)
        contacts.append(info["n_contacts"])
        if done and first_done is None:
            first_done=len(rewards)
    values=np.asarray(trace)
    origin=np.array(initial["agent_position"])
    direction=np.array(initial["block_position"])-origin
    lateral=float(np.cross(direction,np.array(env.agent.position)-origin)/max(np.linalg.norm(direction),1e-12))
    turn=float(env.block.angle-initial["block_angle"])
    return {"env":env,"history":hist,"trace":trace,"rewards":rewards,"first_done":first_done,
            "chunk":{"lateral":lateral,"rotation":turn,"contact_controls":int(np.count_nonzero(contacts)),
                     "block_displacement":float(np.linalg.norm(np.array(env.block.position)-initial["block_position"])),
                     "mean_reward":float(np.mean(rewards)),"max_reward":float(max(rewards)),
                     "state":core.numeric_state(env)},"chunk_trace":values}

def continue_group(policy,branches,steps,seed):
    """Fixed row positions and seed permit common randomness in paired calls."""
    assert steps>=8
    for tick,offset in enumerate(range(8,steps,8)):
        histories=[b["history"] for b in branches]
        actions=core.sample(policy,histories,seed+tick)
        for b,aa in zip(branches,actions):
            for a in aa[:min(8,steps-offset)]:
                if b["first_done"] is not None:
                    b["rewards"].append(1.)
                    continue
                r,done,_=core.step(b["env"],b["history"],a)
                b["rewards"].append(r)
                if done:
                    b["first_done"]=len(b["rewards"])

def outcome(b):
    return {"mean_reward":float(np.mean(b["rewards"])),"max_reward":float(max(b["rewards"])),
            "success":b["first_done"] is not None,
            "completion_seconds":None if b["first_done"] is None else b["first_done"]*.1,
            "rewards":b["rewards"],"chunk":b["chunk"]}

def save_snapshot(env,history,seed,prefix):
    return {"seed":int(seed),"prefix":copy.deepcopy(prefix),"state":core.numeric_state(env),
            "history":np.asarray(history).tolist(),"elapsed_steps":len(prefix)}

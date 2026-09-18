"""Bounded, paired held-out conditional-versus-global-projection experiment."""
import argparse
import hashlib
import json
import time
import numpy as np
import core
from safety import affine_dynamics, speeds, feasible, Projector, observe_substeps, TOL

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/pilot.json")
parser.add_argument("--output", default="results/heldout")
args = parser.parse_args()
cfg = json.loads((core.ROOT / args.config).read_text())
out = core.ROOT / args.output
out.mkdir(parents=True, exist_ok=True)
if (out / "protocol_lock.json").exists():
    raise SystemExit("Output directory already has a locked experiment. Use a fresh --output path.")
lock = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "config": cfg,
        "sha256": {p: hashlib.sha256((core.ROOT/p).read_bytes()).hexdigest() for p in
                   [args.config,"FIRST_EXPERIMENT.md","src/evaluate.py","src/core.py","src/safety.py"]}}
(out / "protocol_lock.json").write_text(json.dumps(lock, indent=2))
policy = core.load_policy()
N, B = cfg["replicates"], cfg["proposal_budget_per_replicate"]
caps = cfg["speed_caps"]
total = cfg["total_evaluation_steps"]
started = time.perf_counter()
summary = []

def branch(seed, prefix, action, cap, affine):
    env, history = core.snapshot(seed, prefix)
    state = core.numeric_state(env)
    initial_angle = env.block.angle
    trace = observe_substeps(env)
    rewards, contacts = [], []
    first_success = None
    for action_t in action:
        r, done, info = core.step(env, history, action_t)
        rewards.append(1. if first_success is not None else r)
        contacts.append(info["n_contacts"])
        if done and first_success is None:
            first_success = len(rewards)
    values = np.array(trace)
    m,b,n,c,_ = affine
    error = max(np.max(np.abs(values[:,2:4]-(m@action+b))),
                np.max(np.abs(values[:,:2]-(n@action+c))))
    maximum = max(np.linalg.norm(values[:,2:4],axis=1).max(), np.linalg.norm(affine[-1]))
    direction = np.array(state["block_position"]) - np.array(state["agent_position"])
    displacement = np.array(env.agent.position) - np.array(state["agent_position"])
    lateral = float(np.cross(direction, displacement) / max(np.linalg.norm(direction),1e-12))
    turn = float(env.block.angle-initial_angle)
    info = {"actual_chunk_max_speed": float(maximum), "model_error":float(error),
            "unsafe_output": bool(maximum > cap+TOL or not feasible(action,affine,cap)),
            "boundary": bool(abs(maximum-cap) <= cfg["boundary_band_pixels_per_second"]),
            "lateral_displacement": lateral,
            "route": "left" if lateral < -10 else "right" if lateral > 10 else "center",
            "rotation":turn, "rotation_mode": "negative" if turn < -.05 else "positive" if turn > .05 else "neutral",
            "block_displacement": float(np.linalg.norm(np.array(env.block.position)-state["block_position"])),
            "contact_controls": int(np.count_nonzero(contacts)),
            "action_roughness":float(np.linalg.norm(np.diff(action,axis=0),axis=1).mean()),
            "chunk_mean_reward":float(np.mean(rewards))}
    return {"env":env,"history":history,"trace":trace,"rewards":rewards,
            "first_success":first_success,"info":info,"chunk_trace":values}

for state_seed in cfg["evaluation_seeds"]:
    state_start = time.perf_counter()
    env, history = core.make_env(state_seed)
    prefix = []
    if state_seed % 2:
        for t in range(cfg["odd_seed_warmup_steps"]//8):
            aa = core.sample(policy, [history], state_seed+500+t)[0]
            for a in aa:
                core.step(env,history,a)
                prefix.append(a.tolist())
    state = core.numeric_state(env)
    snap = {"seed":state_seed,"prefix":prefix,"state":state,"history":np.array(history).tolist()}
    (out/f"snapshot_{state_seed}.json").write_text(json.dumps(snap,indent=2))
    affine = affine_dynamics(env.agent.position,env.agent.velocity)
    sampling_start = time.perf_counter()
    proposal_seeds = [state_seed*100+rep for rep in range(N)]
    proposals = np.array([core.sample(policy,[history]*B,seed) for seed in proposal_seeds])
    sampling_seconds = time.perf_counter()-sampling_start
    proposal_speeds = speeds(proposals,affine)
    masks = np.array([feasible(proposals,affine,cap) for cap in caps])
    np.savez_compressed(out/f"proposals_{state_seed}.npz", actions=proposals,
                        speeds=proposal_speeds,feasible=masks,seeds=proposal_seeds,caps=caps)
    branches = {method: [] for method in ["conditional","projection"]}
    rows = []
    outputs = []
    for severity_index,cap in enumerate(caps):
        projector = Projector(affine,cap)
        for rep in range(N):
            accepted = np.flatnonzero(masks[severity_index,rep])
            idx = int(accepted[0]) if len(accepted) else None
            q = None if idx is None else proposals[rep,idx]
            projected,solver = projector(proposals[rep,0])
            row = {"seed":state_seed,"cap":cap,"replicate":rep,
                   "accepted_proposals":int(masks[severity_index,rep].sum()),
                   "generated_proposals":B,"first_accept_index":idx,
                   "logical_proposals":B if idx is None else idx+1,
                   "unused_generated_proposals":0 if idx is None else B-idx-1,
                   "reference_refusal":idx is None,"projection_failure":projected is None,
                   "solver":solver, "base_first_max_speed":float(proposal_speeds[rep,0].max())}
            for method, action in [("conditional",q),("projection",projected)]:
                bb = None if action is None else branch(state_seed,prefix,action,cap,affine)
                branches[method].append(bb)
                if bb is not None:
                    row[method] = bb["info"]
                    row[method+"_action"] = action.tolist()
                else:
                    row[method] = None
            rows.append(row)
    # Preserve row positions, including missing/absorbed episodes, so A/B use common random draws.
    for tick in range((total-8)//8):
        for method, bb_list in branches.items():
            histories = [b["history"] if b is not None else history for b in bb_list]
            next_actions = core.sample(policy,histories,state_seed*1000+500+tick)
            for bb, actions in zip(bb_list,next_actions):
                if bb is None:
                    continue
                for action in actions:
                    if bb["first_success"] is not None:
                        bb["rewards"].append(1.)
                    else:
                        r,done,_ = core.step(bb["env"],bb["history"],action)
                        bb["rewards"].append(r)
                        if done:
                            bb["first_success"] = len(bb["rewards"])
    saved_traces = {}
    for i,row in enumerate(rows):
        for method,bb_list in branches.items():
            bb = bb_list[i]
            if bb is None:
                row[method+"_operational_score"] = 0.
                continue
            rr = bb["rewards"]
            assert len(rr) == total
            traces = np.asarray(bb["trace"])
            continuation_speed = np.linalg.norm(traces[80:,2:4],axis=1)
            bb["info"].update({"mean_reward":float(np.mean(rr)), "max_reward":float(np.max(rr)),
                               "success":bb["first_success"] is not None,
                               "completion_seconds":None if bb["first_success"] is None else bb["first_success"]*.1,
                               "continuation_unsafe":bool(np.any(continuation_speed>row["cap"]+TOL)),
                               "continuation_max_speed":float(continuation_speed.max()) if len(continuation_speed) else 0.,
                               "reward_trace":rr})
            row[method+"_operational_score"] = 0. if bb["info"]["unsafe_output"] else float(np.mean(rr))
            saved_traces[f"{method}_{row['cap']}_{row['replicate']}"] = traces
    np.savez_compressed(out/f"traces_{state_seed}.npz",**saved_traces)
    (out/f"outcomes_{state_seed}.json").write_text(json.dumps(rows,indent=2))
    state_summary = {"seed":state_seed,"sampling_seconds":sampling_seconds,
                     "seconds":time.perf_counter()-state_start,
                     "acceptance":{str(cap):float(masks[i].mean()) for i,cap in enumerate(caps)},
                     "reference_refusals":sum(r["reference_refusal"] for r in rows),
                     "projection_failures":sum(r["projection_failure"] for r in rows)}
    summary.append(state_summary)
    print(json.dumps(state_summary),flush=True)
(out/"timing.json").write_text(json.dumps({"states":summary,"total_seconds":time.perf_counter()-started},indent=2))

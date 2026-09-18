import json
import time
import numpy as np
import core
from safety import affine_dynamics, speeds, Projector, observe_substeps, feasible

policy = core.load_policy()
records = []
all_proposals = []
start = time.perf_counter()
for seed in range(200000, 200006):
    env, history = core.make_env(seed)
    prefix = []
    if seed % 2:
        for t in range(2):
            warmup = core.sample(policy, [history], seed + 500 + t)[0]
            for act in warmup:
                core.step(env, history, act)
                prefix.append(act.tolist())
    affine = affine_dynamics(env.agent.position, env.agent.velocity)
    aa = core.sample(policy, [history] * 64, seed + 1000)
    all_proposals.append(aa)
    maxspeeds = speeds(aa, affine).max(axis=-1)
    row = {"seed": seed, "state": core.numeric_state(env), "prefix": prefix,
           "speed_quantiles": np.quantile(maxspeeds, [0,.1,.25,.5,.75,.9,1]).tolist(),
           "acceptance": {str(cap): float(np.mean(feasible(aa,affine,cap))) for cap in [100,200,300,400,600]},
           "action_range": [float(aa.min()), float(aa.max())]}
    # Validate analytic dynamics against untouched Pymunk, including contacts.
    traces = []
    for a in aa[:8]:
        e, h = core.snapshot(seed, prefix)
        initial_angle = e.block.angle
        trace = observe_substeps(e)
        rewards = [core.step(e,h,act)[0] for act in a]
        trace = np.asarray(trace)
        m,b,n,c,_ = affine
        error = max(np.max(np.abs(trace[:,2:4]-(m@a+b))), np.max(np.abs(trace[:,:2]-(n@a+c))))
        traces.append({"model_error": float(error), "reward_max": max(rewards),
                       "block_rotation": e.block.angle-initial_angle})
    row["trace_checks"] = traces
    projector = Projector(affine, 300)
    projected, info = projector(aa[0])
    row["projection"] = info
    if projected is not None:
        e,h = core.snapshot(seed, prefix)
        trace = observe_substeps(e)
        for act in projected:
            core.step(e,h,act)
        row["projected_actual_max_speed"] = float(np.linalg.norm(np.asarray(trace)[:,2:4],axis=1).max())
    records.append(row)
    print(json.dumps(row), flush=True)
np.savez_compressed(core.ROOT / "results/development_proposals.npz", actions=np.array(all_proposals))
(core.ROOT / "results/development.json").write_text(json.dumps({"records":records,"seconds":time.perf_counter()-start},indent=2))

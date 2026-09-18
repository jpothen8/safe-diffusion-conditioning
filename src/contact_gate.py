import json
import time
import numpy as np
import core
from safety import affine_dynamics, feasible, speeds
from round2_common import simulate_chunk,continue_group,outcome,save_snapshot

out=core.ROOT/"results/contact_gate"
out.mkdir(exist_ok=True)
policy=core.load_policy()
started=time.perf_counter()
all_summaries=[]
for seed in range(400000,400016):
    env,hist=core.make_env(seed)
    prefix=[]
    for tick in range(8):
        actions=core.sample(policy,[hist],seed*100+tick)[0]
        for action in actions:
            core.step(env,hist,action)
            prefix.append(action.tolist())
        if len(prefix) not in [32,48,64]:
            continue
        name=f"{seed}_{len(prefix):03d}"
        if (out/f"outcomes_{name}.json").exists():
            continue
        snap=save_snapshot(env,hist,seed,prefix)
        (out/f"snapshot_{name}.json").write_text(json.dumps(snap,indent=2))
        proposals=core.sample(policy,[hist]*64,seed*1000+len(prefix))
        aff=affine_dynamics(env.agent.position,env.agent.velocity)
        masks=np.array([feasible(proposals,aff,cap) for cap in [150,250,350]])
        branches=[simulate_chunk(snap,a) for a in proposals]
        chunks=[b["chunk"] for b in branches]
        traces=np.array([b["chunk_trace"] for b in branches])
        np.savez_compressed(out/f"proposals_{name}.npz",actions=proposals,speeds=speeds(proposals,aff),
                            caps=[150,250,350],feasible=masks,traces=traces)
        # Fixed first 16; no quality/membership sorting or favorable-case selection.
        continued=branches[:16]
        continue_group(policy,continued,300-len(prefix),seed*1000+100)
        result={"name":name,"chunks":chunks,"continued_indices":list(range(16)),
                "outcomes":[outcome(b) for b in continued],
                "acceptance":{str(cap):float(masks[i].mean()) for i,cap in enumerate([150,250,350])}}
        (out/f"outcomes_{name}.json").write_text(json.dumps(result,indent=2))
        summary={"name":name,"acceptance":result["acceptance"],
                 "contact_fraction":float(np.mean([c["contact_controls"]>0 for c in chunks])),
                 "rotation_quantiles":np.quantile([c["rotation"] for c in chunks],[0,.1,.5,.9,1]).tolist(),
                 "mean_score":float(np.mean([r["mean_reward"] for r in result["outcomes"]])),
                 "mean_max_score":float(np.mean([r["max_reward"] for r in result["outcomes"]])),
                 "successes":sum(r["success"] for r in result["outcomes"])}
        all_summaries.append(summary)
        print(json.dumps(summary),flush=True)
(out/"timing.json").write_text(json.dumps({"seconds":time.perf_counter()-started,"summaries":all_summaries},indent=2))

"""A reproducible paired comparison for a single fixed snapshot/constraint.

Runs are explicitly labeled development or confirmation in their input config.
"""
import argparse
import hashlib
import json
import time
import numpy as np
import core
from safety import affine_dynamics,feasible,Projector,speeds,TOL
from curtain import is_safe,CurtainProjection,margins
from round2_common import simulate_chunk,continue_group,outcome

parser=argparse.ArgumentParser()
parser.add_argument("config")
args=parser.parse_args()
cfg=json.loads((core.ROOT/args.config).read_text())
out=core.ROOT/cfg["output"]
out.mkdir(parents=True,exist_ok=True)
if (out/"lock.json").exists():
    raise SystemExit("Existing locked run; use a fresh output path")
lock={"utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"config":cfg,
      "hashes":{p:hashlib.sha256((core.ROOT/p).read_bytes()).hexdigest() for p in
                [args.config,"src/compare_setting.py","src/round2_common.py","src/curtain.py","src/core.py","src/safety.py"]}}
(out/"lock.json").write_text(json.dumps(lock,indent=2))
snap=json.loads((core.ROOT/cfg["snapshot"]).read_text())
(out/"snapshot.json").write_text(json.dumps(snap,indent=2))
setting=cfg["setting"];state=snap["state"]
aff=affine_dynamics(state["agent_position"],state["agent_velocity"])
if setting["type"]=="speed":
    safe=lambda a:feasible(a,aff,setting["cap"])
    projector=Projector(aff,setting["cap"])
else:
    safe=lambda a:is_safe(a,aff,setting["normal"],setting["bound"],state["agent_position"])
    projector=CurtainProjection(aff,setting["normal"],setting["bound"],state["agent_position"])
policy=core.load_policy(); started=time.perf_counter()
N=cfg["replicates"];B=cfg["budget"]
bank=np.array([core.sample(policy,[snap["history"]]*B,cfg["proposal_seed"]+i) for i in range(N)])
mask=safe(bank)
np.savez_compressed(out/"proposals.npz",actions=bank,feasible=mask)
branches={"conditional":[],"projection":[]};rows=[]
for rep in range(N):
    good=np.flatnonzero(mask[rep]);idx=int(good[0]) if len(good) else None
    q=None if idx is None else bank[rep,idx]
    p,solver=projector(bank[rep,0])
    row={"replicate":rep,"first_accept_index":idx,"accepted":int(mask[rep].sum()),
         "refusal":q is None,"projection_failure":p is None,"solver":solver}
    for method,a in [("conditional",q),("projection",p)]:
        b=None if a is None else simulate_chunk(snap,a)
        branches[method].append(b)
        row[method+"_action"]=None if a is None else a.tolist()
        if b is not None:
            if setting["type"]=="speed":
                maxspeed=max(np.linalg.norm(state["agent_velocity"]),np.linalg.norm(b["chunk_trace"][:,2:4],axis=1).max())
                row[method+"_margin"]=float(setting["cap"]-maxspeed)
            else:
                row[method+"_margin"]=float(setting["bound"]-(b["chunk_trace"][:,:2]@setting["normal"]).max())
            row[method+"_unsafe"]=bool(not safe(a) or row[method+"_margin"] < -TOL)
    rows.append(row)
# Dummy rows preserve paired random-number positions when a reference refuses.
dummy=simulate_chunk(snap,bank[0,0])
for method,bb in branches.items():
    live=[b if b is not None else simulate_chunk(snap,bank[0,0]) for b in bb]
    continue_group(policy,live,cfg["evaluation_steps"],cfg["continuation_seed"])
    for i,b in enumerate(bb):
        rows[i][method]=None if b is None else outcome(b)
np.savez_compressed(out/"traces.npz",**{f"{method}_{i}":np.asarray(b["trace"]) for method,bb in branches.items() for i,b in enumerate(bb) if b is not None})
(out/"outcomes.json").write_text(json.dumps(rows,indent=2))
diff=[r["conditional"]["mean_reward"]-r["projection"]["mean_reward"] for r in rows
      if r["conditional"] is not None and r["projection"] is not None]
rng=np.random.default_rng(cfg["bootstrap_seed"])
ci=np.quantile(np.array(diff)[rng.integers(0,len(diff),(10000,len(diff)))].mean(axis=1),[.025,.975]).tolist() if diff else None
summary={"phase":cfg["phase"],"seconds":time.perf_counter()-started,"pairs":N,
         "complete_pairs":len(diff),"acceptance":float(mask.mean()),
         "refusals":sum(r["refusal"] for r in rows),"projection_failures":sum(r["projection_failure"] for r in rows),
         "difference":float(np.mean(diff)) if diff else None,"CI95":ci}
for method in branches:
    outputs=[r[method] for r in rows if r[method] is not None]
    summary[method]={"mean_reward":float(np.mean([r["mean_reward"] for r in outputs])) if outputs else None,
                     "mean_max_reward":float(np.mean([r["max_reward"] for r in outputs])) if outputs else None,
                     "successes":sum(r["success"] for r in outputs),
                     "unsafe":sum(r.get(method+"_unsafe",False) for r in rows),
                     "boundary":sum(abs(r.get(method+"_margin",1e9))<1 for r in rows)}
(out/"summary.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))

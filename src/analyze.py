"""Prespecified paired state-cluster analysis, including missing-output bounds."""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--input", default="results/heldout")
args = parser.parse_args()
folder = ROOT/args.input
cfg = json.loads((folder/"protocol_lock.json").read_text())["config"]
files = sorted(folder.glob("outcomes_*.json"))
assert len(files) == len(cfg["evaluation_seeds"]), "Incomplete experiment; do not summarize a partial run."
rows = [r for f in files for r in json.loads(f.read_text())]
rng = np.random.default_rng(cfg["bootstrap_seed"])
indices = rng.integers(0,len(files),(cfg["bootstrap_replicates"],len(files)))
summaries, per_state, modes = {}, [], []

def interval(k,n):
    return [0. if k==0 else float(beta.ppf(.025,k,n-k+1)),
            1. if k==n else float(beta.ppf(.975,k+1,n-k))]

def valid(r,method):
    return r[method] is not None and not r[method]["unsafe_output"]

for cap in cfg["speed_caps"]:
    rr = [r for r in rows if r["cap"]==cap]
    differences, lower, upper = [], [], []
    state_differences = []
    for r in rr:
        q = r["conditional"]["mean_reward"] if valid(r,"conditional") else None
        p = r["projection"]["mean_reward"] if valid(r,"projection") else None
        lower.append((q if q is not None else 0)-(p if p is not None else 1))
        upper.append((q if q is not None else 1)-(p if p is not None else 0))
        if q is not None and p is not None:
            differences.append(q-p)
    for seed in cfg["evaluation_seeds"]:
        sr = [r for r in rr if r["seed"]==seed]
        k = sum(r["accepted_proposals"] for r in sr)
        n = sum(r["generated_proposals"] for r in sr)
        diff = [r["conditional"]["mean_reward"]-r["projection"]["mean_reward"] for r in sr
                if valid(r,"conditional") and valid(r,"projection")]
        # State estimates with partial references are descriptions, explicitly flagged below.
        state_differences.append(float(np.mean(diff)) if diff else np.nan)
        per_state.append({"seed":seed,"cap":cap,"accepted":k,"generated":n,
                          "acceptance":k/n,"acceptance_CI95":interval(k,n),
                          "complete_pairs":len(diff),"mean_difference":float(np.mean(diff)) if diff else None})
    state_differences = np.array(state_differences)
    complete = len(differences)==len(rr)
    ci = np.quantile(state_differences[indices].mean(axis=1),[.025,.975]).tolist() if complete else None
    res = {"requested_pairs":len(rr),"complete_pairs":len(differences),
           "paired_difference":float(np.mean(differences)) if differences else None,
           "paired_cluster_CI95":ci,"missing_outcome_bounds":[float(np.mean(lower)),float(np.mean(upper))],
           "refusals":sum(r["reference_refusal"] for r in rr),
           "projection_failures":sum(r["projection_failure"] for r in rr),
           "logical_proposals":sum(r["logical_proposals"] for r in rr),
           "generated_proposals":sum(r["generated_proposals"] for r in rr),
           "solver_seconds":sum(r["solver"]["seconds"] for r in rr),
           "projection_changed":sum(r["solver"]["status"]!="identity" for r in rr)}
    for method in ["conditional","projection"]:
        outputs = [r[method] for r in rr if r[method] is not None]
        res[method] = {"outputs":len(outputs),"unsafe":sum(o["unsafe_output"] for o in outputs),
                       "mean_reward_outputs":float(np.mean([o["mean_reward"] for o in outputs])) if outputs else None,
                       "operational_score":float(np.mean([r[method+"_operational_score"] for r in rr])),
                       "mean_max_reward":float(np.mean([o["max_reward"] for o in outputs])) if outputs else None,
                       "completions":sum(o["success"] for o in outputs),
                       "boundary_count":sum(o["boundary"] for o in outputs),
                       "continuation_unsafe":sum(o["continuation_unsafe"] for o in outputs),
                       "model_max_error":max([o["model_error"] for o in outputs],default=0),
                       "mean_chunk_reward":float(np.mean([o["chunk_mean_reward"] for o in outputs])) if outputs else None,
                       "route_counts":dict(collections.Counter(o["route"] for o in outputs)),
                       "rotation_counts":dict(collections.Counter(o["rotation_mode"] for o in outputs)),
                       "mean_roughness":float(np.mean([o["action_roughness"] for o in outputs])) if outputs else None}
        for seed in cfg["evaluation_seeds"]:
            for route in ["left","center","right"]:
                values = [r[method] for r in rr if r["seed"]==seed and r[method] is not None and r[method]["route"]==route]
                if values:
                    modes.append({"seed":seed,"cap":cap,"method":method,"route":route,"n":len(values),
                                  "mean_reward":float(np.mean([v["mean_reward"] for v in values])),
                                  "mean_max_reward":float(np.mean([v["max_reward"] for v in values])),
                                  "contact_chunks":sum(v["contact_controls"]>0 for v in values),
                                  "mean_block_displacement":float(np.mean([v["block_displacement"] for v in values]))})
    summaries[str(cap)] = res
(folder/"summary.json").write_text(json.dumps(summaries,indent=2))
(folder/"per_state.json").write_text(json.dumps(per_state,indent=2))
(folder/"modes.json").write_text(json.dumps(modes,indent=2))

fig,axs = plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
for ax,cap in zip(axs,cfg["speed_caps"]):
    for method,color in [("conditional","#2878b5"),("projection","#d45b24")]:
        xx = np.sort([r[method]["actual_chunk_max_speed"] for r in rows if r["cap"]==cap and r[method] is not None])
        ax.step(xx,np.arange(1,len(xx)+1)/len(xx),where="post",label=method,color=color)
    ax.axvline(cap,color="black",ls="--",lw=.8)
    ax.set(title=f"Speed cap {cap} px/s",xlabel="Maximum pusher speed (px/s)",ylabel="Empirical CDF")
axs[0].legend()
fig.savefig(folder/"speed_distributions.png",dpi=180)
plt.close(fig)

# All within-state traces at the primary cap, avoiding a cherry-picked illustration.
fig,axs = plt.subplots(3,4,figsize=(13,10),constrained_layout=True)
for ax,seed in zip(axs.flat,cfg["evaluation_seeds"]):
    snapshot = json.loads((folder/f"snapshot_{seed}.json").read_text())
    archive = np.load(folder/f"traces_{seed}.npz")
    for method,color in [("conditional","#2878b5"),("projection","#d45b24")]:
        for rep in range(cfg["replicates"]):
            key = f"{method}_{cfg['primary_speed_cap']}_{rep}"
            if key in archive:
                trace = archive[key][:80]
                ax.plot(trace[:,0],trace[:,1],color=color,alpha=.5,lw=.8)
    pos = snapshot["state"]["agent_position"]
    block = snapshot["state"]["block_position"]
    ax.scatter(*pos,c="black",s=18,marker="o")
    ax.scatter(*block,c="green",s=24,marker="x")
    ax.set(title=str(seed),xlim=(0,512),ylim=(512,0),aspect="equal")
fig.suptitle("0.8 s pusher paths at the same snapshots: blue conditional, orange projected; green × block origin")
fig.savefig(folder/"within_state_paths.png",dpi=180)
plt.close(fig)

fig,ax = plt.subplots(figsize=(7,4),constrained_layout=True)
for j,cap in enumerate(cfg["speed_caps"]):
    values = [r["mean_difference"] for r in per_state if r["cap"]==cap and r["mean_difference"] is not None]
    ax.scatter(np.full(len(values),j),values,alpha=.6)
    res = summaries[str(cap)]
    if res["paired_cluster_CI95"]:
        lo,hi = res["paired_cluster_CI95"]
        ax.errorbar(j+.15,res["paired_difference"],yerr=[[res["paired_difference"]-lo],[hi-res["paired_difference"]]],fmt="ks",capsize=4)
ax.axhline(0,color="black",lw=.7)
ax.axhline(.05,color="green",ls="--",label="Minimum useful improvement")
ax.set(xticks=range(3),xticklabels=cfg["speed_caps"],xlabel="Speed cap (px/s)",ylabel="Conditional − projection mean reward")
ax.legend()
fig.savefig(folder/"paired_effects.png",dpi=180)
print(json.dumps(summaries,indent=2))

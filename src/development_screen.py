"""Outcome-based DEVELOPMENT search. Never used to select a conditional sample.

Screen every acquired snapshot against a fixed constraint grid. Save all attempts.
Search score only ranks entire experiment settings for later independent testing.
"""
import argparse
import json
import time
import numpy as np
import core
from safety import affine_dynamics,feasible,Projector
from curtain import is_safe,CurtainProjection
from round2_common import simulate_chunk

parser=argparse.ArgumentParser()
parser.add_argument("--limit",type=int,default=0)
args=parser.parse_args()
folder=core.ROOT/"results/contact_gate"
out=core.ROOT/"results/development_screen"
out.mkdir(exist_ok=True)
snapfiles=sorted(folder.glob("snapshot_*.json"))
if args.limit:
    snapfiles=snapfiles[:args.limit]
for path in snapfiles:
    name=path.stem.removeprefix("snapshot_")
    archivepath=folder/f"proposals_{name}.npz"
    if not archivepath.exists() or (out/f"screen_{name}.json").exists():
        continue
    start=time.perf_counter()
    snap=json.loads(path.read_text()); state=snap["state"]
    bank=np.load(archivepath)
    actions=bank["actions"]
    aff=affine_dynamics(state["agent_position"],state["agent_velocity"])
    # Only immediate chunk quality is used in the first screen.
    outcomespath=folder/f"outcomes_{name}.json"
    if not outcomespath.exists():
        continue
    chunks=json.loads(outcomespath.read_text())["chunks"]
    raw_quality=np.array([c["mean_reward"] for c in chunks])
    candidates=[]
    settings=[{"type":"speed","cap":cap} for cap in [150,250,350]]
    for angle in range(0,360,45):
        for distance in [10,25,50,100]:
            normal=np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))])
            settings.append({"type":"curtain","angle_degrees":angle,"offset":distance,
                             "normal":normal.tolist(),"bound":float(normal@state["agent_position"]+distance)})
    for setting in settings:
        if setting["type"]=="speed":
            mask=feasible(actions,aff,setting["cap"])
            projector=Projector(aff,setting["cap"])
        else:
            mask=is_safe(actions,aff,setting["normal"],setting["bound"],state["agent_position"])
            projector=CurtainProjection(aff,setting["normal"],setting["bound"],state["agent_position"])
        acceptance=float(mask.mean())
        if not .1<=acceptance<=.9:
            candidates.append({"setting":setting,"acceptance":acceptance,"screened":False})
            continue
        projected=[]; failed=0
        for a in actions[:16]:
            pa,info=projector(a)
            if pa is None:
                failed+=1; projected.append(None)
            else:
                b=simulate_chunk(snap,pa)
                projected.append({"score":b["chunk"]["mean_reward"],"chunk":b["chunk"],"solver":info})
        qs=float(raw_quality[mask].mean())
        ps=float(np.mean([x["score"] if x else 0 for x in projected]))
        row={"setting":setting,"acceptance":acceptance,"screened":True,"q_chunk_score":qs,
             "projection_chunk_score":ps,"difference":qs-ps,"failures":failed,"projected":projected,
             "safe_raw_indices":np.flatnonzero(mask).tolist()}
        candidates.append(row)
    result={"snapshot":name,"seconds":time.perf_counter()-start,"settings":candidates}
    (out/f"screen_{name}.json").write_text(json.dumps(result,indent=2))
    active=[c for c in candidates if c["screened"]]
    print(json.dumps({"snapshot":name,"settings":len(active),
                      "best_difference":max([c["difference"] for c in active],default=None),
                      "seconds":result["seconds"]}),flush=True)

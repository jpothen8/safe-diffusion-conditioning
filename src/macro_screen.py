"""Development-only search with a frozen 3.2 s diffusion-rollout distribution."""
import json
import time
import numpy as np
import core
import macro
from round2_common import simulate_chunk

out=core.ROOT/"results/macro_screen";out.mkdir(exist_ok=True)
policy=core.load_policy()
names=[f"{seed}_048" for seed in range(400000,400016)]
for i,name in enumerate(names):
    snapfile=core.ROOT/f"results/contact_gate/snapshot_{name}.json"
    if not snapfile.exists() or (out/f"screen_{name}.json").exists():continue
    snap=json.loads(snapfile.read_text());state=snap["state"]
    aff=macro.affine(state["agent_position"],state["agent_velocity"],32)
    aa,rr=macro.sample(policy,snap,64,4,910000+i*100)
    np.savez_compressed(out/f"proposals_{name}.npz",actions=aa,rewards=rr)
    settings=[]
    # Fixed grid, including more restrictive but still positive-mass cases.
    for angle in range(0,360,45):
        normal=np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))])
        for distance in [10,25,50,100,150]:
            settings.append({"type":"curtain","angle_degrees":angle,"offset":distance,
                             "normal":normal.tolist(),"bound":float(normal@state["agent_position"]+distance)})
    settings.extend([{"type":"speed","cap":cap} for cap in [100,150,200,250,350]])
    results=[]
    for setting in settings:
        mask=macro.safe(aa,aff,setting,state["agent_position"])
        acceptance=float(mask.mean())
        if not .05<=acceptance<=.9:
            results.append({"setting":setting,"acceptance":acceptance,"screened":False});continue
        projector=macro.Projection(aff,setting,state["agent_position"])
        projected=[];failed=0
        for a in aa[:16]:
            pa,solver=projector(a)
            if pa is None:
                failed+=1;projected.append(None)
            else:
                b=simulate_chunk(snap,pa)
                projected.append({"mean_reward":float(np.mean(b["rewards"])),"max_reward":float(max(b["rewards"])),
                                  "success":b["first_done"] is not None,"solver":solver})
        qs=float(rr[mask].mean());ps=float(np.mean([r["mean_reward"] if r else 0 for r in projected]))
        results.append({"setting":setting,"acceptance":acceptance,"screened":True,
                        "q_score":qs,"p_score":ps,"difference":qs-ps,"failures":failed,"projected":projected})
    (out/f"screen_{name}.json").write_text(json.dumps({"name":name,"settings":results},indent=2))
    print(json.dumps({"name":name,"best":max([r["difference"] for r in results if r["screened"]],default=None)}),flush=True)

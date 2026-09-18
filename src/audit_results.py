"""Independent audit of stored proposals, selections, feasibility and pairing."""
import hashlib
import json
import numpy as np
import core
from safety import affine_dynamics, feasible, speeds

folder=core.ROOT/"results/heldout"
lock=json.loads((folder/"protocol_lock.json").read_text())
cfg=lock["config"]
for path,digest in lock["sha256"].items():
    assert hashlib.sha256((core.ROOT/path).read_bytes()).hexdigest()==digest,path
counts={"proposals":0,"requested_pairs":0,"outputs":0,"unsafe":0,"exact_identity_pairs":0}
diagnostics=[]
for seed in cfg["evaluation_seeds"]:
    archive=np.load(folder/f"proposals_{seed}.npz")
    rows=json.loads((folder/f"outcomes_{seed}.json").read_text())
    snapshot=json.loads((folder/f"snapshot_{seed}.json").read_text())
    state=snapshot["state"]
    aff=affine_dynamics(state["agent_position"],state["agent_velocity"])
    actions=archive["actions"]
    np.testing.assert_allclose(speeds(actions,aff),archive["speeds"],rtol=0,atol=1e-10)
    counts["proposals"]+=actions.shape[0]*actions.shape[1]
    # Route occupancy in the entire frozen-policy bank, and its feasible subsets.
    _,_,n,c,_=aff
    endpoints=np.einsum("h,rbhd->rbd",n[-1],actions)+c[-1]
    direction=np.array(state["block_position"])-state["agent_position"]
    displacement=endpoints-state["agent_position"]
    lateral=np.cross(direction,displacement)/max(np.linalg.norm(direction),1e-12)
    for ci,cap in enumerate(cfg["speed_caps"]):
        mask=feasible(actions,aff,cap)
        assert np.array_equal(mask,archive["feasible"][ci])
        diagnostics.append({"seed":seed,"cap":cap,"generated":int(mask.size),"accepted":int(mask.sum()),
                            "all_route_counts":{"left":int((lateral < -10).sum()),"center":int((abs(lateral)<=10).sum()),"right":int((lateral>10).sum())},
                            "safe_route_counts":{"left":int(((lateral < -10)&mask).sum()),"center":int(((abs(lateral)<=10)&mask).sum()),"right":int(((lateral>10)&mask).sum())}})
    for r in rows:
        counts["requested_pairs"]+=1
        ci=cfg["speed_caps"].index(r["cap"])
        accepted=np.flatnonzero(archive["feasible"][ci,r["replicate"]])
        assert r["first_accept_index"]==(int(accepted[0]) if len(accepted) else None)
        if r["conditional"]:
            np.testing.assert_array_equal(r["conditional_action"],actions[r["replicate"],r["first_accept_index"]])
        for method in ["conditional","projection"]:
            if r[method] is not None:
                counts["outputs"]+=1
                ok=feasible(np.array(r[method+"_action"]),aff,r["cap"])
                assert ok != r[method]["unsafe_output"]
                counts["unsafe"]+=int(not ok)
        if r["solver"]["status"]=="identity" and r["first_accept_index"]==0:
            counts["exact_identity_pairs"]+=1
            np.testing.assert_array_equal(r["conditional_action"],r["projection_action"])
            np.testing.assert_allclose(r["conditional"]["reward_trace"],r["projection"]["reward_trace"],rtol=0,atol=1e-12)
(folder/"proposal_route_diagnostics.json").write_text(json.dumps(diagnostics,indent=2))
(folder/"audit.json").write_text(json.dumps({"passed":True,**counts},indent=2))
print(json.dumps({"passed":True,**counts},indent=2))

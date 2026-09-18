"""Checks that protect the experiment's safety, projection and snapshot claims."""
import json
import numpy as np
import core
from safety import affine_dynamics, speeds, feasible, Projector, observe_substeps

data = json.loads((core.ROOT/"results/development.json").read_text())["records"]
rng = np.random.default_rng(9981)
max_error = 0.
contacts = 0
replays = 0
projection_checks = 0
for row in data:
    seed, prefix = row["seed"], row["prefix"]
    env, history = core.snapshot(seed,prefix)
    affine = affine_dynamics(env.agent.position,env.agent.velocity)
    a = rng.uniform(0,512,(8,2))
    saved = []
    for repeat in range(2):
        e,h = core.snapshot(seed,prefix)
        trace = observe_substeps(e)
        for act in a:
            _,_,info = core.step(e,h,act)
            contacts += info["n_contacts"]
        saved.append(np.array(trace))
    assert np.array_equal(saved[0],saved[1]), "snapshot replay changed dynamics"
    replays += 1
    m,b,n,c,_ = affine
    error = max(np.max(np.abs(saved[0][:,2:4]-(m@a+b))),np.max(np.abs(saved[0][:,:2]-(n@a+c))))
    assert error < 1e-8
    max_error = max(max_error,error)
    # Initial velocity is part of membership, even if the first action brakes.
    too_low = max(0,np.linalg.norm(affine[-1])-1)
    if np.linalg.norm(affine[-1]) > 0:
        assert not feasible(a,affine,too_low)
    projector = Projector(affine,400)
    projected, status = projector(a)
    assert projected is not None, status
    assert feasible(projected,affine,400)
    again, _ = projector(projected)
    assert np.array_equal(projected,again)
    # Compare to a known feasible candidate, when holding the initial target is feasible.
    hold = np.broadcast_to(np.array(env.agent.position),(8,2))
    if feasible(hold,affine,400):
        assert np.sum((projected-a)**2) <= np.sum((hold-a)**2)+1e-4
    e,h = core.snapshot(seed,prefix)
    trace = observe_substeps(e)
    for act in projected:
        core.step(e,h,act)
    assert np.linalg.norm(np.array(trace)[:,2:4],axis=1).max() <=400+1e-5
    projection_checks +=1
result = {"passed":True,"exact_replay_cases":replays,"projection_cases":projection_checks,
          "maximum_affine_model_error":float(max_error),"observed_contacts":contacts}
(core.ROOT/"results/verification.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

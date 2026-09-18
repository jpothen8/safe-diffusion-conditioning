"""Independent conic-solver check of the analytic full-chunk projection."""
import json
from pathlib import Path
import numpy as np,cvxpy as cp,torch
from effort import project_numpy,project_torch
ROOT=Path(__file__).resolve().parents[1];rng=np.random.default_rng(101000);errors=[];feasibility=[]
for dimension in [48,72,300]:
    for radius in [.45,.65,.85]:
        raw=rng.normal(size=(4,dimension))*2
        formula=project_numpy(raw,radius)
        native=project_torch(torch.as_tensor(raw,dtype=torch.float32),radius).numpy()
        for x,p in zip(raw,formula):
            z=cp.Variable(dimension)
            problem=cp.Problem(cp.Minimize(cp.sum_squares(z-x)/dimension),[z>=-1,z<=1,cp.norm(z)<=radius*np.sqrt(dimension)])
            problem.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9,max_iter=500)
            errors.append(float(np.linalg.norm(p-z.value)/np.sqrt(dimension)))
        feasibility.append(float(max(np.sqrt(np.mean(native**2,axis=-1)))-radius))
assert max(errors)<1e-5,(max(errors),errors)
assert max(feasibility)<1e-6,feasibility
result={'independent_socp_cases':len(errors),'maximum_rms_solution_difference':max(errors),
        'torch_max_rms_violation':max(feasibility),'passed':True}
(ROOT/'results/effort_projection_check.json').write_text(json.dumps(result,indent=2));print(result)

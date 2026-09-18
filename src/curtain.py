"""Exact full-horizon convex projection for a supplied workspace half-plane."""
import time
import numpy as np
import cvxpy as cp
from safety import TOL

def positions(actions,affine):
    _,_,n,c,_=affine
    return np.einsum("th,...hd->...td",n,actions)+c

def margins(actions,affine,normal,bound):
    return bound-np.einsum("...td,d->...t",positions(actions,affine),normal)

def is_safe(actions,affine,normal,bound,initial_position):
    a=np.asarray(actions)
    return ((margins(a,affine,normal,bound).min(axis=-1)>=-TOL)
            & (np.dot(initial_position,normal)<=bound+TOL)
            & np.all((a>=-TOL)&(a<=512+TOL),axis=(-2,-1))
            & np.isfinite(a).all(axis=(-2,-1)))

class CurtainProjection:
    def __init__(self,affine,normal,bound,initial_position):
        self.affine=affine; self.normal=np.asarray(normal); self.bound=float(bound)
        self.initial=np.asarray(initial_position)
        self.z=cp.Variable((8,2)); self.a=cp.Parameter((8,2))
        _,_,n,c,_=affine
        self.problem=cp.Problem(cp.Minimize(cp.sum_squares(self.z-self.a)),
                               [self.z>=0,self.z<=1,(n@self.z+c/512)@self.normal<=bound/512])
        assert self.problem.is_dcp()
    def __call__(self,a):
        start=time.perf_counter()
        if is_safe(a,self.affine,self.normal,self.bound,self.initial):
            return a.copy(),{"status":"identity","seconds":time.perf_counter()-start,"distance":0.}
        self.a.value=np.asarray(a)/512
        try:
            self.problem.solve(solver="CLARABEL",warm_start=False,max_iter=200,
                               tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9)
            candidate=None if self.z.value is None else self.z.value*512
            valid=candidate is not None and is_safe(candidate,self.affine,self.normal,self.bound,self.initial)
            return (candidate if valid and self.problem.status=="optimal" else None),{
                "status":self.problem.status,"valid":bool(valid),"seconds":time.perf_counter()-start,
                "iterations":self.problem.solver_stats.num_iters,
                "distance":float(np.linalg.norm(candidate-a)) if candidate is not None else None}
        except cp.error.SolverError as exc:
            return None,{"status":"error","message":str(exc),"seconds":time.perf_counter()-start}

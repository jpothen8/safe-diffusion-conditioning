"""Frozen diffusion-rollout action-sequence distribution and general geometry.

Each 0.8 s subchunk is drawn from the unchanged released diffusion policy on a
private simulator forecast. Concatenation is part of this explicitly new base
sampling procedure. Both safety methods subsequently see the same full sequence.
"""
import time
import numpy as np
import cvxpy as cp
import core
from round2_common import restore

TOL=1e-5
def affine(position,velocity,horizon):
    xp=np.zeros(horizon);vp=np.zeros(horizon)
    xc=np.array(position,dtype=float);vc=np.array(velocity,dtype=float)
    ms=[];bs=[];ns=[];cs=[]
    for t in range(horizon):
        basis=np.eye(horizon)[t]
        for _ in range(10):
            vp=vp+.01*(100*(basis-xp)-20*vp)
            vc=vc+.01*(-100*xc-20*vc)
            xp=xp+.01*vp;xc=xc+.01*vc
            ms.append(vp.copy());bs.append(vc.copy());ns.append(xp.copy());cs.append(xc.copy())
    return np.array(ms),np.array(bs),np.array(ns),np.array(cs),np.array(velocity)

def sample(policy,snap,count,subchunks,seed):
    envs,histories=zip(*(restore(snap) for _ in range(count)))
    histories=list(histories)
    actions=[];reward=[[] for _ in envs];first_done=[None]*count
    for tick in range(subchunks):
        aa=core.sample(policy,histories,seed+tick)
        actions.append(aa)
        for i,(e,h,a) in enumerate(zip(envs,histories,aa)):
            for act in a:
                r,done,_=core.step(e,h,act)
                reward[i].append(r if first_done[i] is None else 1.)
                if done and first_done[i] is None:
                    first_done[i]=len(reward[i])
    return np.concatenate(actions,axis=1),np.asarray(reward)

def margin(a,aff,setting,initial):
    m,b,n,c,v0=aff
    if setting["type"]=="curtain":
        vals=np.einsum("th,...hd->...td",n,a)+c
        future=setting["bound"]-np.einsum("...td,d->...t",vals,setting["normal"])
        return np.minimum(future.min(axis=-1),setting["bound"]-np.dot(initial,setting["normal"]))
    vals=np.einsum("th,...hd->...td",m,a)+b
    return setting["cap"]-np.maximum(np.linalg.norm(vals,axis=-1).max(axis=-1),np.linalg.norm(v0))

def safe(a,aff,setting,initial):
    return ((margin(a,aff,setting,initial)>=-TOL)&np.isfinite(a).all(axis=(-1,-2))
            &np.all((a>=-TOL)&(a<=512+TOL),axis=(-1,-2)))

class Projection:
    def __init__(self,aff,setting,initial):
        self.aff=aff;self.setting=setting;self.initial=initial
        h=aff[0].shape[1];m,b,n,c,_=aff
        self.z=cp.Variable((h,2));self.a=cp.Parameter((h,2))
        constraints=[self.z>=0,self.z<=1]
        if setting["type"]=="curtain":
            constraints.append((n@self.z+c/512)@np.array(setting["normal"])<=setting["bound"]/512)
        else:
            constraints.append(cp.norm(m@self.z+b/512,axis=1)<=setting["cap"]/512)
        self.problem=cp.Problem(cp.Minimize(cp.sum_squares(self.z-self.a)),constraints)
        assert self.problem.is_dcp()
    def __call__(self,a):
        start=time.perf_counter()
        if safe(a,self.aff,self.setting,self.initial):
            return a.copy(),{"status":"identity","distance":0.,"seconds":time.perf_counter()-start}
        self.a.value=a/512
        try:
            self.problem.solve(solver="CLARABEL",warm_start=False,max_iter=200,tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9)
            candidate=None if self.z.value is None else self.z.value*512
            valid=candidate is not None and safe(candidate,self.aff,self.setting,self.initial)
            return (candidate if valid and self.problem.status=="optimal" else None),{
                "status":self.problem.status,"valid":bool(valid),"seconds":time.perf_counter()-start,
                "distance":None if candidate is None else float(np.linalg.norm(candidate-a))}
        except cp.error.SolverError as exc:
            return None,{"status":"error","message":str(exc),"seconds":time.perf_counter()-start}

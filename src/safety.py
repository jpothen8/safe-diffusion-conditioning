"""Full 0.8 s pusher-speed predicate and globally convex projection."""
import time
import numpy as np
import cvxpy as cp

H = 8
DT = 0.01
SUBSTEPS = 10
TOL = 1e-5

def affine_dynamics(position, velocity):
    """V=M A+b and X=N A+c at all 80 physics steps, in pixel units."""
    xp = np.zeros(H)
    vp = np.zeros(H)
    xc = np.array(position, dtype=float)
    vc = np.array(velocity, dtype=float)
    mm, bb, nn, cc = [], [], [], []
    for t in range(H):
        basis = np.eye(H)[t]
        for _ in range(SUBSTEPS):
            vp = vp + DT * (100 * (basis - xp) - 20 * vp)
            vc = vc + DT * (-100 * xc - 20 * vc)
            xp = xp + DT * vp
            xc = xc + DT * vc
            mm.append(vp.copy()); bb.append(vc.copy())
            nn.append(xp.copy()); cc.append(xc.copy())
    return np.array(mm), np.array(bb), np.array(nn), np.array(cc), np.array(velocity, dtype=float)

def speeds(actions, affine):
    m, b, _, _, initial_velocity = affine
    values = np.linalg.norm(np.einsum("th,...hd->...td", m, actions) + b, axis=-1)
    initial = np.full((*values.shape[:-1], 1), np.linalg.norm(initial_velocity))
    return np.concatenate([initial, values], axis=-1)

def feasible(actions, affine, cap):
    aa = np.asarray(actions)
    return ((speeds(aa, affine).max(axis=-1) <= cap + TOL)
            & np.all((aa >= -TOL) & (aa <= 512 + TOL), axis=(-2, -1))
            & np.isfinite(aa).all(axis=(-2, -1)))

class Projector:
    def __init__(self, affine, cap):
        self.affine, self.cap = affine, float(cap)
        m, b, _, _, _ = affine
        self.z = cp.Variable((H, 2))  # normalized by fixed workspace size
        self.a = cp.Parameter((H, 2))
        constraints = [self.z >= 0, self.z <= 1,
                       cp.norm(m @ self.z + b / 512, axis=1) <= cap / 512]
        self.problem = cp.Problem(cp.Minimize(cp.sum_squares(self.z - self.a)), constraints)
        assert self.problem.is_dcp()

    def __call__(self, actions):
        start = time.perf_counter()
        if np.linalg.norm(self.affine[-1]) > self.cap + TOL:
            return None, {"status": "initial_state_infeasible", "seconds": time.perf_counter()-start}
        if feasible(actions, self.affine, self.cap):
            return actions.copy(), {"status": "identity", "seconds": time.perf_counter()-start,
                                    "distance": 0., "iterations": 0}
        self.a.value = np.asarray(actions) / 512
        try:
            self.problem.solve(solver="CLARABEL", warm_start=False,
                               tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9,
                               max_iter=200)
            candidate = None if self.z.value is None else self.z.value * 512
            valid = candidate is not None and feasible(candidate, self.affine, self.cap)
            result = candidate if valid and self.problem.status == "optimal" else None
            info = {"status": self.problem.status, "validated": bool(valid),
                    "seconds": time.perf_counter()-start,
                    "iterations": self.problem.solver_stats.num_iters,
                    "distance": None if candidate is None else float(np.linalg.norm(candidate-actions)),
                    "max_speed": None if candidate is None else float(speeds(candidate,self.affine).max())}
            return result, info
        except cp.error.SolverError as error:
            return None, {"status": "solver_error", "error": str(error),
                          "seconds": time.perf_counter()-start}

def observe_substeps(env):
    """Observe the original physics method, without modifying its computation."""
    trace = []
    original = env.space.step
    def observed(dt):
        original(dt)
        trace.append([*env.agent.position, *env.agent.velocity,
                      *env.block.position, env.block.angle])
    env.space.step = observed
    return trace

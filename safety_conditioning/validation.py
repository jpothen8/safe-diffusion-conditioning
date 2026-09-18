"""Numerical preservation, independent convex solves, and native dynamics checks."""

import numpy as np
import torch
from . import artifacts, policy, samplers, simulator
from .constraints import BoxLimit, EffortLimit


def check_refactor(output, device="cpu"):
    from .tasks.pendulum.policy import Denoiser
    import cvxpy as cp
    import gym
    from .tasks.pendulum.environment import dynamics

    torch.set_num_threads(2)
    fixture = np.load(artifacts.PROJECT / "fixtures/sampler_reference.npz")
    results = {}
    for index, (task, dimension) in enumerate(
        [("HalfCheetah", 48), ("Walker2d", 72), ("Pendulum", 12)]
    ):
        torch.manual_seed(190000 + index)
        model = Denoiser() if task == "Pendulum" else policy.ChunkDDPM(dimension)
        model.eval()
        obs = fixture[task + "_observations"]
        base = policy.sample_policy(model, obs, 191001)
        constraint = BoxLimit(-1, 0) if task == "Pendulum" else EffortLimit(0.72)
        refined = samplers.refine(model, obs, base, constraint, 191002)
        b, cloud, _ = samplers.bayesfp(
            model,
            obs,
            191003,
            particles=32,
            strength=20,
            radius=None if task == "Pendulum" else 0.72,
        )
        for name, values in [
            ("base", base),
            ("refinement", refined),
            ("bayesfp", b),
            ("particles", cloud),
        ]:
            assert np.array_equal(values, fixture[task + "_" + name]), (task, name)
        results[task] = {
            "base_refinement_bayesfp_and_particles_bitwise_equal_to_original_cpu_fixtures": True
        }
    bank = np.ones((2, 3, 48), dtype=np.float32)
    bank[1, 1:] = 0.2
    selection = samplers.rejection_from_bank(bank, EffortLimit(0.72))
    assert selection["indices"].tolist() == [-1, 1]
    assert np.isnan(selection["actions"][0]).all()
    assert selection["proposals_used"].tolist() == [3, 2]
    rng = np.random.default_rng(119004)
    errors = []
    for cap in [0.3, 0.72, 0.95]:
        for scale in [0.3, 1, 3]:
            raw = rng.normal(size=48) * scale
            expected = EffortLimit(cap).project(raw[None])[0]
            variable = cp.Variable(48)
            problem = cp.Problem(
                cp.Minimize(cp.sum_squares(variable - raw)),
                [variable >= -1, variable <= 1, cp.norm(variable) <= cap * np.sqrt(48)],
            )
            problem.solve(
                solver="CLARABEL",
                tol_gap_abs=1e-9,
                tol_gap_rel=1e-9,
                tol_feas=1e-9,
                max_iter=200,
            )
            assert problem.status == "optimal"
            error = float(np.max(np.abs(variable.value - expected)))
            assert error < 3e-5, error
            errors.append(error)
    maximum = 0.0
    for state in [[3.14, 0.1], [-1.2, 2.0], [0.3, -0.1]]:
        env = gym.make("Pendulum-v1")
        env.reset(seed=0)
        env.unwrapped.state = np.array(state)
        current = np.array([state])
        for _ in range(200):
            action = rng.uniform(-2, 2, (1, 1)).astype(np.float32)
            current, reward = dynamics(current, action)
            _, native, _, _, _ = env.step(action[0])
            maximum = max(
                maximum,
                float(np.max(np.abs(current[0] - env.unwrapped.state))),
                abs(float(reward[0]) - native),
            )
        env.close()
    assert maximum < 1e-10
    for task in ["HalfCheetah", "Walker2d"]:
        saved = simulator.snapshot(task, 190050, None)
        controls = round(0.4 / simulator.CONTROL_PERIOD[task])
        actions = rng.uniform(-0.3, 0.3, (1, controls, 6)).astype(np.float32)
        a = simulator.rollout(task, None, [saved], actions, 0.4)
        b = simulator.rollout(task, None, [saved], actions, 0.4)
        assert np.array_equal(a["states"], b["states"]) and np.array_equal(
            a["rewards"], b["rewards"]
        )
    results.update(
        {
            "passed": True,
            "explicit_refusals": True,
            "independent_global_socp_solves": len(errors),
            "max_projection_coordinate_error": max(errors),
            "pendulum_native_max_error": maximum,
            "mujoco_full_snapshot_replay_exact": True,
        }
    )
    artifacts.write_json(output, results)
    print(results, flush=True)

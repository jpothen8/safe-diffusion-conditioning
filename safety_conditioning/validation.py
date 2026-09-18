"""Meaningful compatibility checks before using refactored research code."""
import sys
import numpy as np
import torch
from . import artifacts, policy, samplers, simulator
from .constraints import EffortLimit


def check_refactor(output, device="cuda"):
    sys.path.insert(0, str(artifacts.PROJECT / "round3/src"))
    import bayesfp as old_bayes
    import chunk_model as old_policy
    import effort as old_effort
    import locomotion_env as old_env

    torch.set_num_threads(2)
    results = {}
    for task in ["Walker2d", "HalfCheetah"]:
        checkpoint = artifacts.PROJECT / f"round3/assets/{task}_diffusion_v2.pt"
        new = policy.load_policy(checkpoint, device)
        old = old_policy.load(checkpoint, device)
        snapshots = artifacts.read_json(artifacts.PROJECT / f"round3/results/{task}_comparison/snapshots.json")[:2]
        obs = np.array([saved["observation"] for saved in snapshots])
        raw = policy.sample_policy(new, obs, 119001)
        assert np.array_equal(raw, old_policy.sample(old, obs, 119001))
        cap = .72
        refined = samplers.refine(new, obs, raw, EffortLimit(cap), 119002)
        assert np.array_equal(refined, old_effort.refine(old, obs, raw, cap, 119002))
        b, cloud, _ = samplers.bayesfp(new, obs, 119003, radius=cap, strength=20, particles=32)
        old_b, diagnostics = old_bayes.sample(old, obs, old_policy.scheduler, old.dimension,
                                             119003, radius=cap, strength=20, particles=32)
        assert np.array_equal(b, old_b)
        assert np.array_equal(cloud, diagnostics["final_particle_cloud"])
        actor = simulator.load_expert(task)
        horizon = new.dimension // 6
        old_env.H[task] = horizon
        chunks = raw.reshape(-1, horizon, 6)
        first = simulator.rollout(task, actor, snapshots, chunks, seconds=.4)
        second = old_env.rollout(task, actor, snapshots, chunks, seconds=.4)
        for key in second:
            assert np.array_equal(first[key], second[key]), (task, key)
        results[task] = {"base_bitwise_equal": True, "refinement_bitwise_equal": True,
                         "bayesfp_and_cloud_bitwise_equal": True, "native_rollout_bitwise_equal": True}
    # Refusals must not silently become zero-action conditional samples.
    bank = np.ones((2, 3, 48), dtype=np.float32)
    bank[1, 1:] = .2
    selection = samplers.rejection_from_bank(bank, EffortLimit(.72))
    assert selection["indices"].tolist() == [-1, 1]
    assert np.isnan(selection["actions"][0]).all()
    assert np.array_equal(selection["actions"][1], bank[1, 1])
    assert selection["proposals_used"].tolist() == [3, 2]
    rng = np.random.default_rng(119004)
    values = rng.normal(size=(100, 48)).astype(np.float32) * 3
    constraint = EffortLimit(.72)
    assert np.array_equal(constraint.project(values), old_effort.project_numpy(values, .72))
    assert constraint.feasible(constraint.project(values)).all()
    results["refusals_explicit_and_first_feasible_correct"] = True
    results["global_projection_matches_frozen_implementation"] = True
    results["passed"] = True
    artifacts.write_json(output, results)
    print(results, flush=True)

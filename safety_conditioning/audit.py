"""Independent checks of saved requests, selections, projections and outcomes."""
import numpy as np
import torch
from .artifacts import PROJECT, read_json, sha256, verify_lock, write_json
from .data import EpisodeDataset
from .evaluation import ARMS
from .simulator import load_expert, rollout


def audit_study(output):
    checked_inputs = verify_lock(output / "lock.json")
    summary = read_json(output / "summary.json")
    config = summary["config"]
    count, reps, budget = config["states"], config["replicates"], config["proposal_budget"]
    mean, std = EpisodeDataset(PROJECT / config["dataset"]).normalizer()
    checkpoints, all_deltas, details = [], [], []
    for seed in config["training_seeds"]:
        model_root = output / f"seed_{seed}"
        evaluation = model_root / "evaluation"
        checkpoint = model_root / "training/policy.pt"
        digest = sha256(checkpoint)
        assert digest == read_json(evaluation / "checkpoint.json")["sha256"]
        assert digest == read_json(model_root / "training/summary.json")["checkpoint_sha256"]
        checkpoints.append(digest)
        weights = torch.load(checkpoint, weights_only=True, map_location="cpu")["state_dict"]
        assert np.array_equal(weights["obs_mean"].numpy(), mean)
        assert np.array_equal(weights["obs_std"].numpy(), std)
        with np.load(evaluation / "proposals.npz") as data:
            bank = data["actions"].reshape(count * reps, budget, -1)
        feasible = (np.sqrt(np.mean(bank**2, axis=-1)) <= config["rms_limit"])
        feasible &= (np.abs(bank).max(-1) <= 1) & np.isfinite(bank).all(-1)
        valid, first = feasible.any(-1), feasible.argmax(-1)
        with np.load(evaluation / "selection.npz") as data:
            assert np.array_equal(data["feasible"], feasible)
            assert np.array_equal(data["indices"], np.where(valid, first, -1))
            assert np.array_equal(data["valid"], valid)
            assert np.array_equal(data["actions"][valid], bank[np.arange(len(bank))[valid], first[valid]])
            assert np.isnan(data["actions"][~valid]).all()
        model_summary = read_json(evaluation / "summary.json")
        assert model_summary["refusals"] == int((~valid).sum())
        assert model_summary["acceptance"] == float(feasible.mean())
        returns = {}
        for arm in ARMS:
            with np.load(evaluation / f"{arm}.npz") as data:
                actions, good, rewards = data["chunk"], data["valid"], data["rewards"]
                safe = (np.sqrt(np.mean(actions[good]**2, axis=-1)) <= config["rms_limit"] + 1e-6)
                safe &= np.abs(actions[good]).max(-1) <= 1 + 1e-6
                safe &= np.isfinite(actions[good]).all(-1)
                entry = model_summary["arms"][arm]
                assert int((~safe).sum()) == entry["unsafe_outputs"]
                returns[arm] = rewards.sum(1)
                assert np.isclose(returns[arm][good].mean(), entry["mean_return_given_output"], atol=1e-10, rtol=0)
                assert int(data["terminated"][good].sum()) == entry["native_failures"]
                if good.any():
                    executed = data["executed"][good]
                    assert np.array_equal(executed.sum(1), data["lengths"][good])
                    assert np.all(rewards[good][~executed] == 0)
                    assert np.array_equal(data["actions"][good, :config["horizon_controls"]].reshape(good.sum(), -1), actions[good])
                if arm == "projection":
                    raw = bank[:, 0]
                    factor = np.minimum(1, config["rms_limit"] * np.sqrt(raw.shape[-1]) / np.linalg.norm(raw, axis=-1))
                    assert np.max(np.abs(actions - raw * factor[:, None])) < 3e-7
                if arm == "conditional":
                    assert np.array_equal(actions[good], bank[np.arange(len(bank))[good], first[good]])
        all_deltas.append((returns["refinement"] - returns["bayesfp_projected"]).reshape(count, reps))
        details.append({"seed": seed, "checkpoint_sha256": digest, "refusals": int((~valid).sum())})
    assert len(set(checkpoints)) == len(config["training_seeds"])
    values = np.stack(all_deltas).mean(-1)
    primary = summary["comparisons"]["refinement_minus_bayesfp_projected"]
    assert np.allclose(primary["model_means"], values.mean(1), atol=1e-10, rtol=0)
    # Independently express resampling as multiplicities rather than index gathers.
    rng = np.random.default_rng(config["bootstrap_seed"])
    draws, models = config["bootstrap_draws"], len(values)
    model_draws = rng.integers(0, models, (draws, models))
    state_draws = rng.integers(0, count, (draws, count))
    mw = np.array([np.bincount(row, minlength=models) for row in model_draws]) / models
    sw = np.array([np.bincount(row, minlength=count) for row in state_draws]) / count
    boot = np.einsum("bi,ij,bj->b", mw, values, sw)
    assert np.allclose(np.quantile(boot, [.025, .975]), primary["crossed_bootstrap_ci95"], atol=1e-10, rtol=0)
    # Replay every arm on the first frozen snapshot using native simulator physics.
    snapshots = read_json(output / "snapshots.json")
    actor = load_expert(config["task"])
    replay_errors = {}
    folder = output / f"seed_{config['training_seeds'][0]}" / "evaluation"
    for arm in ARMS:
        with np.load(folder / f"{arm}.npz") as data:
            # Replay all recorded controls, including continuation, to isolate
            # physics/reward from neural inference changes across batch sizes.
            replay = rollout(config["task"], actor, snapshots[:1], data["actions"][:1], config["evaluation_seconds"])
            replay_errors[arm] = float(np.max(np.abs(replay["rewards"] - data["rewards"][:1])))
            assert replay_errors[arm] == 0
            assert np.array_equal(replay["states"], data["states"][:1])
    report = {"passed": True, "locked_input_hashes_checked": checked_inputs,
              "independent_checkpoints": len(set(checkpoints)), "models": details,
              "all_selections_and_global_projections_verified": True,
              "train_only_normalizers_exact": True, "crossed_bootstrap_independently_verified": True,
              "native_replay_max_reward_error_by_arm": replay_errors}
    write_json(output / "audit.json", report)
    print(report, flush=True)

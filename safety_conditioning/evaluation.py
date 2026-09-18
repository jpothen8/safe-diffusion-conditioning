"""Five matched sampling arms, complete proposal records and native outcomes."""

import time
import numpy as np
import torch
from .artifacts import sha256, write_json
from .constraints import EffortLimit
from .policy import load_policy, sample_policy
from .samplers import bayesfp, refine, rejection_from_bank
from .simulator import CONTROL_PERIOD, rollout
from .statistics import action_mmd

ARMS = ("conditional", "projection", "refinement", "bayesfp", "bayesfp_projected")


def describe(
    trajectory, actions, valid, conditional, conditional_valid, constraint, templates
):
    rms = np.sqrt(np.mean(actions[valid] ** 2, axis=-1))
    matched = valid & conditional_valid
    distances = np.sqrt(
        np.mean((actions[valid, None] - templates[valid]) ** 2, axis=-1)
    )
    return {
        "requests": len(actions),
        "refusals": int((~valid).sum()),
        "mean_return_given_output": float(trajectory["rewards"][valid].sum(1).mean())
        if valid.any()
        else None,
        "mean_progress_given_output": float(trajectory["progress"][valid].mean())
        if valid.any()
        else None,
        "native_failures": int(trajectory["terminated"][valid].sum()),
        "nonpositive_returns": int((trajectory["rewards"][valid].sum(1) <= 0).sum()),
        "unsafe_outputs": int((~constraint.feasible(actions[valid])).sum()),
        "boundary_fraction": float((np.abs(rms - constraint.radius) <= 1e-5).mean())
        if valid.any()
        else None,
        "mmd_to_conditional": action_mmd(actions[matched], conditional[matched]),
        "expert_template_fraction": float((distances.argmin(1) == 0).mean())
        if valid.any()
        else None,
        "nearest_template_distance": float(distances.min(1).mean())
        if valid.any()
        else None,
    }


def evaluate_policy(
    checkpoint,
    task,
    snapshots,
    actor,
    templates,
    config,
    model_index,
    output,
    *,
    device="cuda",
):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write_json(
        output / "checkpoint.json",
        {"path": str(checkpoint), "sha256": sha256(checkpoint)},
    )
    model = load_policy(checkpoint, device)
    torch.set_num_threads(2)
    count, reps, budget = (
        len(snapshots),
        config["replicates"],
        config["proposal_budget"],
    )
    horizon = model.dimension // 6
    if horizon != config["horizon_controls"]:
        raise ValueError("Model horizon differs from the frozen protocol.")
    repeated = [saved for saved in snapshots for _ in range(reps)]
    observations = np.array([saved["observation"] for saved in repeated])
    offset = model_index * config["sampling_seed_stride"]
    bank, tick = [], time.perf_counter()
    for index, saved in enumerate(snapshots):
        obs = np.repeat(np.array(saved["observation"])[None], reps * budget, axis=0)
        draws = sample_policy(model, obs, config["proposal_seed"] + offset + index)
        bank.append(draws.reshape(reps, budget, -1))
    bank = np.stack(bank)
    sampling_seconds = time.perf_counter() - tick
    np.savez_compressed(output / "proposals.npz", actions=bank)
    flat = bank.reshape(count * reps, budget, -1)
    raw = flat[:, 0]
    constraint = EffortLimit(config["rms_limit"])
    selection = rejection_from_bank(flat, constraint)
    np.savez_compressed(output / "selection.npz", **selection)
    base = rollout(
        task, actor, repeated, raw.reshape(-1, horizon, 6), config["evaluation_seconds"]
    )
    np.savez_compressed(output / "base.npz", chunk=raw, **base)
    timing = {"base_proposal_bank": sampling_seconds}
    tick = time.perf_counter()
    projected = constraint.project(raw)
    timing["projection"] = time.perf_counter() - tick
    tick = time.perf_counter()
    refined = refine(
        model, observations, raw, constraint, config["refinement_seed"] + offset
    )
    timing["refinement"] = time.perf_counter() - tick
    tick = time.perf_counter()
    bayes, cloud, diagnostics = bayesfp(
        model,
        observations,
        config["bayesfp_seed"] + offset,
        strength=config["bayesfp_strength"],
        particles=config["bayesfp_particles"],
        radius=constraint.radius,
    )
    timing["bayesfp"] = time.perf_counter() - tick
    np.savez_compressed(output / "particles.npz", particles=cloud)
    bayes = np.clip(bayes, -1, 1)
    chunks = {
        "conditional": selection["actions"],
        "projection": projected,
        "refinement": refined,
        "bayesfp": bayes,
        "bayesfp_projected": constraint.project(bayes),
    }
    entries = {}
    for arm, actions in chunks.items():
        valid = (
            selection["valid"]
            if arm == "conditional"
            else np.ones(count * reps, dtype=bool)
        )
        if valid.all():
            trajectory = rollout(
                task,
                actor,
                repeated,
                actions.reshape(-1, horizon, 6),
                config["evaluation_seconds"],
            )
        else:
            # A refusal is recorded without simulating a fabricated Q sample.
            good = np.flatnonzero(valid)
            trajectory = {}
            if len(good):
                partial = rollout(
                    task,
                    actor,
                    [repeated[i] for i in good],
                    actions[good].reshape(-1, horizon, 6),
                    config["evaluation_seconds"],
                )
                for key, value in partial.items():
                    fill = np.zeros((len(valid), *value.shape[1:]), dtype=value.dtype)
                    if value.dtype.kind == "f":
                        fill[:] = np.nan
                    fill[valid] = value
                    trajectory[key] = fill
            else:
                controls = round(config["evaluation_seconds"] / CONTROL_PERIOD[task])
                trajectory = {
                    "rewards": np.full((len(valid), controls), np.nan),
                    "terminated": np.zeros(len(valid), dtype=bool),
                    "progress": np.full(len(valid), np.nan),
                }
        np.savez_compressed(
            output / f"{arm}.npz", chunk=actions, valid=valid, **trajectory
        )
        entries[arm] = describe(
            trajectory,
            actions,
            valid,
            selection["actions"],
            selection["valid"],
            constraint,
            templates,
        )
    paired = selection["valid"] & selection["second_valid"]
    summary = {
        "task": task,
        "model_index": model_index,
        "training_seed": config["training_seeds"][model_index],
        "horizon_seconds": horizon * CONTROL_PERIOD[task],
        "rms_limit": constraint.radius,
        "acceptance": float(selection["feasible"].mean()),
        "acceptance_by_state": selection["feasible"]
        .reshape(count, reps, budget)
        .mean((1, 2))
        .tolist(),
        "refusals": int((~selection["valid"]).sum()),
        "logical_proposals_mean": float(selection["proposals_used"].mean()),
        "q_vs_q_mmd": action_mmd(
            selection["actions"][paired], selection["second_actions"][paired]
        ),
        "arms": entries,
        "timing_seconds": timing,
        "particle_diagnostics": diagnostics,
        "base_mean_return": float(base["rewards"].sum(1).mean()),
        "base_native_failures": int(base["terminated"].sum()),
        "seconds": time.perf_counter() - started,
        "base_nonpositive_returns": int((base["rewards"].sum(1) <= 0).sum()),
    }
    write_json(output / "summary.json", summary)
    print(
        f"evaluate seed={summary['training_seed']} acceptance={summary['acceptance']:.3f} "
        f"R-BP={entries['refinement']['mean_return_given_output'] - entries['bayesfp_projected']['mean_return_given_output']:+.2f}",
        flush=True,
    )
    return summary

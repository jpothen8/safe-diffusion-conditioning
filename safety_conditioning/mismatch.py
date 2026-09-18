"""Retrospective boundary-error probe with exact conservative controls."""

import time
from pathlib import Path
import numpy as np
import torch
from .artifacts import (
    PROJECT,
    create_output,
    lock_inputs,
    read_json,
    verify_lock,
    write_json,
)
from .constraints import EffortLimit
from .evaluation import ARMS
from .samplers import rejection_from_bank
from .simulator import load_expert, rollout
from .statistics import crossed_intervals

ERRORS = [-0.01, 0.0, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.05]


def analyze_mismatch(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    verify_lock(source / "lock.json")
    study = read_json(source / "summary.json")
    cfg = study["config"]
    output = create_output(destination)
    estimated = cfg["rms_limit"]
    margin = 0.01
    buffered = estimated * (1 - margin)
    config = {
        "source": str(source.relative_to(PROJECT)),
        "estimated_cap": estimated,
        "relative_errors": ERRORS,
        "fixed_projection_margin": margin,
        "tolerance": 1e-6,
        "label": "Retrospective sensitivity analysis; fixed errors, no estimator training.",
    }
    files = [
        source / "lock.json",
        source / "summary.json",
        PROJECT / "docs/protocols/safe_set_mismatch.md",
        Path(__file__),
    ]
    for seed in cfg["training_seeds"]:
        folder = source / f"seed_{seed}/evaluation"
        files += [folder / "proposals.npz", *[folder / f"{arm}.npz" for arm in ARMS]]
    lock_inputs(output, files, config)
    torch.set_num_threads(2)
    actor = load_expert(cfg["task"])
    snapshots = read_json(source / "snapshots.json")
    repeated = [saved for saved in snapshots for _ in range(cfg["replicates"])]
    names = [*ARMS, "projection_margin_1pct", "conditional_margin_1pct"]
    collected = {arm: {"chunk": [], "returns": [], "valid": []} for arm in names}
    budget_failures = []
    started = time.perf_counter()
    for seed in cfg["training_seeds"]:
        folder = source / f"seed_{seed}/evaluation"
        for arm in ARMS:
            with np.load(folder / f"{arm}.npz") as data:
                collected[arm]["chunk"].append(data["chunk"])
                collected[arm]["returns"].append(data["rewards"].sum(1))
                collected[arm]["valid"].append(data["valid"])
        with np.load(folder / "proposals.npz") as data:
            bank = data["actions"].reshape(len(repeated), cfg["proposal_budget"], -1)
        limit = EffortLimit(buffered)
        selection = rejection_from_bank(bank, limit)
        budget_failures.append(int((~selection["valid"]).sum()))
        model_output = output / f"seed_{seed}"
        model_output.mkdir()
        np.savez_compressed(model_output / "selection.npz", **selection)
        controls = {
            "projection_margin_1pct": limit.project(bank[:, 0]),
            "conditional_margin_1pct": selection["actions"],
        }
        for name, actions in controls.items():
            valid = (
                selection["valid"]
                if name.startswith("conditional")
                else np.ones(len(actions), dtype=bool)
            )
            if not valid.all():
                write_json(
                    output / "budget_failures.json",
                    {"seed": seed, "refusals": int((~valid).sum())},
                )
                raise RuntimeError(
                    "Conservative reference exhausted a budget; refusals saved, do not replace."
                )
            trace = rollout(
                cfg["task"],
                actor,
                repeated,
                actions.reshape(-1, cfg["horizon_controls"], 6),
                cfg["evaluation_seconds"],
            )
            np.savez_compressed(
                model_output / f"{name}.npz", chunk=actions, valid=valid, **trace
            )
            collected[name]["chunk"].append(actions)
            collected[name]["returns"].append(trace["rewards"].sum(1))
            collected[name]["valid"].append(valid)
        print(f"mismatch controls complete seed={seed}", flush=True)
    for values in collected.values():
        for key in values:
            values[key] = np.stack(values[key])
        values["rms"] = np.sqrt(
            np.mean(values["chunk"].astype(np.float64) ** 2, axis=-1)
        )
    cases = {}
    for error in ERRORS:
        true_cap = estimated * (1 - error)
        entries = {}
        for name, values in collected.items():
            valid = values["valid"]
            excess = np.maximum(values["rms"] - true_cap, 0)
            violation = (excess > 1e-6) | (np.abs(values["chunk"]).max(-1) > 1 + 1e-6)
            entries[name] = {
                "requests": int(valid.size),
                "refusals": int((~valid).sum()),
                "true_violations": int((violation & valid).sum()),
                "true_violation_rate": float(violation[valid].mean()),
                "rate_by_model": np.mean(violation, axis=1).tolist(),
                "mean_excess_all_outputs": float(excess[valid].mean()),
                "rms_excess_all_outputs": float(np.sqrt(np.mean(excess[valid] ** 2))),
                "max_excess": float(excess[valid].max()),
                "mean_true_margin": float((true_cap - values["rms"][valid]).mean()),
                "mean_estimated_margin": float(
                    (estimated - values["rms"][valid]).mean()
                ),
                "mean_native_return_all_outputs": float(
                    values["returns"][valid].mean()
                ),
            }
        cases[str(error)] = {
            "relative_error": error,
            "true_cap": true_cap,
            "arms": entries,
        }
    shape = (len(cfg["training_seeds"]), cfg["states"], cfg["replicates"])
    comparisons = {}
    for first, second in [
        ("conditional", "projection"),
        ("refinement", "projection"),
        ("bayesfp_projected", "projection"),
        ("projection_margin_1pct", "projection"),
    ]:
        a = (collected[first]["rms"] > buffered + 1e-6).astype(float)
        b = (collected[second]["rms"] > buffered + 1e-6).astype(float)
        comparisons[first + "_minus_" + second] = crossed_intervals(
            (a - b).reshape(shape), 181000
        )
    quality = {}
    for first, second in [
        ("projection_margin_1pct", "projection"),
        ("conditional_margin_1pct", "projection_margin_1pct"),
        ("refinement", "projection_margin_1pct"),
    ]:
        quality[first + "_minus_" + second] = crossed_intervals(
            (collected[first]["returns"] - collected[second]["returns"]).reshape(shape),
            181000,
        )
    result = {
        "config": config,
        "cases": cases,
        "violation_rate_contrasts_at_1pct": comparisons,
        "return_contrasts": quality,
        "buffered_reference_refusals": budget_failures,
        "additional_rollouts": 2 * len(cfg["training_seeds"]) * len(repeated),
        "seconds": time.perf_counter() - started,
    }
    write_json(output / "summary.json", result)
    np.savez_compressed(
        output / "margins_and_returns.npz",
        **{
            name + "_" + key: values[key]
            for name, values in collected.items()
            for key in ["rms", "returns", "valid"]
        },
    )
    verify_lock(output / "lock.json")
    write_report(output, result)
    return result


def write_report(output, result):
    lines = [
        "# Safe-set mismatch sensitivity",
        "",
        "This retrospective analysis treats .72 as the estimated safe RMS cap and scores the same outputs against nearby true caps. "
        "Positive error means the estimated set is too permissive. No estimator was trained and these violations are effort-envelope violations, not observed collisions or physical damage.",
        "",
        "| Estimated-cap overstatement | True cap | Q violations | P violations | R violations | B+P violations | P with 1% margin |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in result["cases"].values():
        arms = case["arms"]
        lines.append(
            f"| {100 * case['relative_error']:.2f}% | {case['true_cap']:.6f} | "
            + " | ".join(
                f"{100 * arms[name]['true_violation_rate']:.2f}%"
                for name in [
                    "conditional",
                    "projection",
                    "refinement",
                    "bayesfp_projected",
                    "projection_margin_1pct",
                ]
            )
            + " |"
        )
    lines += [
        "",
        "Q is exact first-feasible conditioning on the supplied set; P is exact global Euclidean projection over the entire chunk; R is finite refinement. "
        "Each original arm has 2,048 outputs. The same saved trajectories appear at all error levels, so rows are not independent experiments.",
        "",
        "The 1% margin projection uses cap .7128. At a true 1% error it is exactly the oracle projection onto the true set. "
        "It additionally assumes a known 1% error bound, so it is a conservative control rather than a secretly better-informed competitor. "
        "The same-margin conditional reference also receives that set. Both controls execute new .4 s chunks with identical continuation/native reward.",
        "",
        "## Return and uncertainty",
        "",
        "| Contrast | Mean native-return difference | Crossed 95% interval |",
        "|---|---:|---:|",
    ]
    for name, comparison in result["return_contrasts"].items():
        low, high = comparison["crossed_bootstrap_ci95"]
        lines.append(
            f"| {name} | {comparison['mean']:+.3f} | [{low:+.3f}, {high:+.3f}] |"
        )
    lines += [
        "",
        "These are exploratory intervals; no error setting or sampler was chosen by reward. The original arms' returns stay unchanged when only the true-set label changes. "
        "Comparisons between an unsafe original arm and a safe conservative control do not establish a safe-performance win.",
        "",
        "## Interpretation",
        "",
        "An inward boundary error can turn an atom of projected boundary mass into violations even when projection onto the estimated set is mathematically exact. "
        "A conditional law without mass on that boundary has no analogous jump, though it still becomes unsafe if it occupies the strip removed by the error. "
        "Finite refinement can leave substantial near-boundary mass and need not inherit the exact conditional reference's robustness.",
        "",
        "This probe tests a controlled uniform cap bias, not whether most projection harm in robotics is caused by estimation error. "
        "The next dynamic-constraint experiment should perturb a simulator-based clearance/velocity/posture estimate and evaluate the true predicate, "
        "including projection with calibrated margins and the same uncertainty information for every arm.",
        "",
        "[Fixed exploratory protocol](../../docs/protocols/safe_set_mismatch.md) · [All errors, violations and excess magnitudes](summary.json) · [Source hashes](lock.json)",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")

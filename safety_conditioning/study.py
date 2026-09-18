"""Run a fixed independent-model study; retain every requested model."""

import time
import numpy as np
import torch
from . import artifacts
from .evaluation import evaluate_policy
from .simulator import CONTROL_PERIOD, load_expert, rollout, snapshot
from .statistics import crossed_intervals
from .training import train_policy

SCIENTIFIC_MODULES = (
    "artifacts",
    "constraints",
    "data",
    "evaluation",
    "policy",
    "samplers",
    "simulator",
    "statistics",
    "study",
    "training",
)


def summarize(output):
    config = artifacts.read_json(output / "lock.json")["config"]
    n, reps = config["states"], config["replicates"]
    models, differences = [], {}
    pairs = [
        ("refinement", "bayesfp_projected"),
        ("conditional", "projection"),
        ("refinement", "projection"),
    ]
    for seed in config["training_seeds"]:
        folder = output / f"seed_{seed}" / "evaluation"
        models.append(artifacts.read_json(folder / "summary.json"))
        returns = {}
        for arm in {value for pair in pairs for value in pair}:
            with np.load(folder / f"{arm}.npz") as data:
                value = data["rewards"].sum(1)
                value[~data["valid"]] = np.nan
                returns[arm] = value
        for first, second in pairs:
            name = first + "_minus_" + second
            differences.setdefault(name, []).append(
                (returns[first] - returns[second]).reshape(n, reps)
            )
    comparisons = {
        name: crossed_intervals(
            np.stack(values), config["bootstrap_seed"], config["bootstrap_draws"]
        )
        for name, values in differences.items()
    }
    primary = comparisons["refinement_minus_bayesfp_projected"]
    refusals = sum(model["refusals"] for model in models)
    hard_violations = sum(
        model["arms"][arm]["unsafe_outputs"]
        for model in models
        for arm in ["conditional", "projection", "refinement", "bayesfp_projected"]
    )
    success = (
        primary["complete"]
        and primary["crossed_bootstrap_ci95"][0] > 0
        and primary["model_t_ci95"][0] > 0
        and primary["mean"] >= config["minimum_useful_return_gain"]
        and refusals == 0
        and hard_violations == 0
    )
    summary = {
        "config": config,
        "models": models,
        "comparisons": comparisons,
        "total_method_outputs_requested": len(models) * n * reps * 5,
        "refusals": refusals,
        "hard_output_violations": hard_violations,
        "useful_advantage_replicated": bool(success),
        "minimum_useful_return_gain": config["minimum_useful_return_gain"],
        "scope": "Training-seed and shared-state uncertainty conditional on one fixed demonstration dataset.",
    }
    artifacts.write_json(output / "summary.json", summary)
    np.savez_compressed(
        output / "paired_differences.npz",
        **{k: np.stack(v) for k, v in differences.items()},
    )
    return summary


def run_study(config_path, output_path, device="cuda"):
    config_path = config_path.resolve()
    config = artifacts.read_json(config_path)
    task = config["task"]
    output = artifacts.create_output(output_path)
    files = [
        config_path,
        artifacts.PROJECT / config["protocol"],
        artifacts.PROJECT / config["dataset"],
        artifacts.PROJECT / "requirements.lock.txt",
        artifacts.expert_path(task),
        artifacts.expert_path(task, "medium"),
    ]
    files += [
        artifacts.PROJECT / "safety_conditioning" / (name + ".py")
        for name in SCIENTIFIC_MODULES
    ]
    artifacts.lock_inputs(output, files, config)
    started = time.perf_counter()
    torch.set_num_threads(2)
    actor, medium = load_expert(task), load_expert(task, "medium")
    snapshots = [
        snapshot(
            task,
            config["state_seed"] + i,
            actor,
            0 if i < config["states"] // 2 else config["warmup_seconds"],
        )
        for i in range(config["states"])
    ]
    artifacts.write_json(output / "snapshots.json", snapshots)
    expert = rollout(task, actor, snapshots, seconds=config["evaluation_seconds"])
    np.savez_compressed(output / "expert.npz", **expert)
    templates = []
    for controller in [actor, medium]:
        trace = rollout(
            task,
            controller,
            snapshots,
            seconds=config["horizon_controls"] * CONTROL_PERIOD[task],
        )
        templates.append(
            np.repeat(
                trace["actions"].reshape(config["states"], -1),
                config["replicates"],
                axis=0,
            )
        )
    templates = np.stack(templates, 1)
    np.savez_compressed(output / "templates.npz", actions=templates)
    for index, seed in enumerate(config["training_seeds"]):
        model_output = output / f"seed_{seed}"
        checkpoint = train_policy(
            artifacts.PROJECT / config["dataset"],
            task,
            seed,
            model_output / "training",
            horizon=config["horizon_controls"],
            steps=config["training_steps"],
            device=device,
        )
        evaluate_policy(
            checkpoint,
            task,
            snapshots,
            actor,
            templates,
            config,
            index,
            model_output / "evaluation",
            device=device,
        )
        artifacts.write_json(
            output / "progress.json",
            {
                "completed_models": index + 1,
                "requested_models": len(config["training_seeds"]),
                "seconds": time.perf_counter() - started,
            },
        )
    summary = summarize(output)
    artifacts.verify_lock(output / "lock.json")
    print(
        "Study complete:",
        summary["comparisons"]["refinement_minus_bayesfp_projected"],
        flush=True,
    )
    return output

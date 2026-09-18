"""Independent mismatch audit and figures derived only from saved outputs."""

import csv
import numpy as np
from .artifacts import PROJECT, read_json, sha256, verify_lock, write_json
from .evaluation import ARMS
from .simulator import load_expert, rollout

CONTROLS = ("projection_margin_1pct", "conditional_margin_1pct")
LABELS = {
    "conditional": "Conditioning Q",
    "projection": "Exact projection P",
    "refinement": "Finite refinement R",
    "bayesfp": "Raw BayesFP",
    "bayesfp_projected": "BayesFP + projection",
    "projection_margin_1pct": "Projection, 1% margin",
    "conditional_margin_1pct": "Conditioning, 1% margin",
}


def saved_arrays(output):
    result = read_json(output / "summary.json")
    source = PROJECT / result["config"]["source"]
    cfg = read_json(source / "summary.json")["config"]
    data = {}
    for arm in (*ARMS, *CONTROLS):
        arrays = {"rms": [], "returns": [], "valid": [], "actions": []}
        for seed in cfg["training_seeds"]:
            root = (
                output / f"seed_{seed}"
                if arm in CONTROLS
                else source / f"seed_{seed}/evaluation"
            )
            with np.load(root / f"{arm}.npz") as saved:
                arrays["actions"].append(saved["chunk"])
                arrays["rms"].append(
                    np.linalg.norm(saved["chunk"].astype(float), axis=1)
                    / np.sqrt(saved["chunk"].shape[1])
                )
                arrays["returns"].append(saved["rewards"].sum(1))
                arrays["valid"].append(saved["valid"])
        data[arm] = {key: np.stack(value) for key, value in arrays.items()}
    return result, source, cfg, data


def audit_mismatch(output):
    inputs = verify_lock(output / "lock.json")
    result, source, cfg, arrays = saved_arrays(output)
    source_inputs = verify_lock(source / "lock.json")
    expected_shape = (len(cfg["training_seeds"]), cfg["states"] * cfg["replicates"])
    cap = result["config"]["estimated_cap"] * (
        1 - result["config"]["fixed_projection_margin"]
    )
    saved_margins = np.load(output / "margins_and_returns.npz")
    for arm, values in arrays.items():
        assert values["rms"].shape == expected_shape
        assert values["valid"].all() and np.isfinite(values["actions"]).all()
        assert np.max(np.abs(values["actions"])) <= 1 + 1e-6
        for key in ("rms", "returns", "valid"):
            np.testing.assert_allclose(
                values[key], saved_margins[f"{arm}_{key}"], atol=1e-14, rtol=0
            )
        for case in result["cases"].values():
            entry = case["arms"][arm]
            excess = (values["rms"] - case["true_cap"]).clip(0)
            violations = excess > 1e-6
            assert entry["requests"] == np.prod(expected_shape)
            assert entry["refusals"] == 0
            assert entry["true_violations"] == int(violations.sum())
            assert entry["true_violation_rate"] == violations.mean()
            np.testing.assert_allclose(
                entry["rate_by_model"], violations.mean(1), atol=0, rtol=0
            )
            for name, metric in [
                ("mean_excess_all_outputs", excess.mean()),
                ("rms_excess_all_outputs", np.sqrt((excess**2).mean())),
                ("max_excess", excess.max()),
                ("mean_native_return_all_outputs", values["returns"].mean()),
            ]:
                np.testing.assert_allclose(entry[name], metric, atol=1e-10, rtol=0)
    saved_margins.close()
    selection_costs, replay_errors = [], {}
    actor = load_expert(cfg["task"])
    snapshots = read_json(source / "snapshots.json")
    replay_indices = [0, (cfg["states"] // 2) * cfg["replicates"]]
    for index, seed in enumerate(cfg["training_seeds"]):
        folder = source / f"seed_{seed}/evaluation"
        with np.load(folder / "proposals.npz") as data:
            bank = data["actions"].reshape(
                expected_shape[1], cfg["proposal_budget"], -1
            )
        # First-feasible selection uses the frozen policy's float32 membership;
        # reporting uses float64 with the separate 1e-6 feasibility tolerance.
        feasible = np.sqrt(np.mean(bank**2, axis=-1)) <= cap
        feasible &= np.isfinite(bank).all(-1) & (np.abs(bank).max(-1) <= 1)
        assert feasible.any(1).all()
        first = feasible.argmax(1)
        with np.load(output / f"seed_{seed}/selection.npz") as selection:
            assert np.array_equal(selection["feasible"], feasible)
            assert np.array_equal(selection["indices"], first)
            assert selection["valid"].all()
        selected = bank[np.arange(len(bank)), first]
        assert np.array_equal(
            arrays["conditional_margin_1pct"]["actions"][index], selected
        )
        raw = bank[:, 0].astype(float)
        scale = np.minimum(1, cap * np.sqrt(raw.shape[1]) / np.linalg.norm(raw, axis=1))
        independent_projection = raw * scale[:, None]
        np.testing.assert_allclose(
            arrays["projection_margin_1pct"]["actions"][index],
            independent_projection,
            atol=2e-7,
            rtol=0,
        )
        selection_costs.append(
            {
                "seed": seed,
                "bank_acceptance": float(feasible.mean()),
                "logical_proposals_to_first_accept": int((first + 1).sum()),
                "stored_proposals": int(feasible.size),
                "refusals": 0,
            }
        )
        for arm in CONTROLS:
            with np.load(output / f"seed_{seed}/{arm}.npz") as saved:
                assert not saved["terminated"].any()
                assert np.array_equal(
                    saved["actions"][:, : cfg["horizon_controls"]].reshape(
                        len(bank), -1
                    ),
                    saved["chunk"],
                )
                assert saved["executed"].all()
                replay = rollout(
                    cfg["task"],
                    actor,
                    [snapshots[i // cfg["replicates"]] for i in replay_indices],
                    saved["actions"][replay_indices],
                    cfg["evaluation_seconds"],
                )
                error = float(
                    np.max(np.abs(replay["rewards"] - saved["rewards"][replay_indices]))
                )
                assert error == 0
                assert np.array_equal(replay["states"], saved["states"][replay_indices])
                replay_errors[f"{seed}/{arm}"] = error
    # Independent bootstrap implementation using resample multiplicities.
    models, states = len(cfg["training_seeds"]), cfg["states"]
    rng = np.random.default_rng(181000)
    model_draws = rng.integers(0, models, (20000, models))
    state_draws = rng.integers(0, states, (20000, states))
    mw = np.array([np.bincount(row, minlength=models) for row in model_draws]) / models
    sw = np.array([np.bincount(row, minlength=states) for row in state_draws]) / states
    for group in ("return_contrasts", "violation_rate_contrasts_at_1pct"):
        for name, entry in result[group].items():
            first, second = name.split("_minus_")
            if group == "return_contrasts":
                difference = arrays[first]["returns"] - arrays[second]["returns"]
            else:
                difference = (arrays[first]["rms"] > cap + 1e-6).astype(float) - (
                    arrays[second]["rms"] > cap + 1e-6
                )
            values = difference.reshape(models, states, cfg["replicates"]).mean(-1)
            boot = np.einsum("bi,ij,bj->b", mw, values, sw)
            np.testing.assert_allclose(
                entry["crossed_bootstrap_ci95"],
                np.quantile(boot, [0.025, 0.975]),
                atol=1e-10,
                rtol=0,
            )
            np.testing.assert_allclose(
                entry["model_means"], values.mean(1), atol=1e-10, rtol=0
            )
    report = {
        "passed": True,
        "analysis_locked_inputs_checked": inputs,
        "underlying_study_locked_inputs_checked": source_inputs,
        "all_selections_and_global_projections_verified": True,
        "all_reported_violation_counts_and_excess_magnitudes_verified": True,
        "bootstrap_independently_verified": True,
        "new_control_rollouts": result["additional_rollouts"],
        "selection_costs": selection_costs,
        "native_replayed_trajectories": len(replay_errors) * len(replay_indices),
        "native_replay_max_reward_error_by_model_arm": replay_errors,
        "poor_return_counts": {
            arm: int((values["returns"] <= 0).sum()) for arm, values in arrays.items()
        },
        "source_snapshot_sha256": sha256(source / "snapshots.json"),
    }
    write_json(output / "audit.json", report)
    print(
        f"Mismatch audit passed: {inputs} input hashes, all selections/projections/metrics, "
        f"{report['native_replayed_trajectories']} native replays with zero error.",
        flush=True,
    )


def report_mismatch(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .mismatch import write_report

    result = read_json(output / "summary.json")
    write_report(output, result)
    report_path = output / "RESULTS.md"
    report_path.write_text(
        report_path.read_text().replace(
            "Estimated-cap overstatement", "True-cap reduction"
        )
    )
    rows = []
    for case in result["cases"].values():
        for arm, entry in case["arms"].items():
            rows.append(
                {
                    "relative_cap_error": case["relative_error"],
                    "true_cap": case["true_cap"],
                    "arm": arm,
                    **{k: v for k, v in entry.items() if k != "rate_by_model"},
                }
            )
    with (output / "table.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cases = [case for case in result["cases"].values() if case["relative_error"] >= 0]
    x = [100 * case["relative_error"] for case in cases]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    colors = {
        "conditional": "#0072B2",
        "projection": "#D55E00",
        "refinement": "#009E73",
        "bayesfp_projected": "#CC79A7",
        "projection_margin_1pct": "#333333",
        "conditional_margin_1pct": "#56B4E9",
    }
    for arm, color in colors.items():
        entries = [case["arms"][arm] for case in cases]
        style = "--" if arm in CONTROLS else "-"
        axes[0].plot(
            x,
            [100 * entry["true_violation_rate"] for entry in entries],
            style,
            marker="o",
            markersize=3,
            color=color,
            label=LABELS[arm],
        )
        axes[1].plot(
            x,
            [entry["mean_excess_all_outputs"] for entry in entries],
            style,
            marker="o",
            markersize=3,
            color=color,
        )
    for axis in axes:
        axis.set_xscale("symlog", linthresh=0.01)
        axis.set_xticks(
            [0, 0.01, 0.1, 0.5, 1, 2, 5], ["0", ".01", ".1", ".5", "1", "2", "5"]
        )
        axis.set_xlabel("True cap reduction relative to supplied cap (%)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("True constraint violations (%)")
    axes[1].set_ylabel("Mean RMS-cap excess (normalized action units)")
    axes[0].set_title("Boundary sensitivity")
    axes[1].set_title("Violation size, including zero-excess outputs")
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False
    )
    fig.suptitle(
        "HalfCheetah: 8 frozen policies, 2,048 outputs per arm; retrospective cap-bias probe",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.93))
    fig.savefig(output / "boundary_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(output / "boundary_sensitivity.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    lines = [
        "",
        "## Violation uncertainty and magnitude",
        "",
        "At the fixed 1% error, these exploratory crossed-bootstrap intervals resample models and the shared state bank. "
        "Negative differences mean fewer violations. The small positive refinement difference is inconclusive.",
        "",
        "| Contrast | Violation difference (percentage points) | 95% interval |",
        "|---|---:|---:|",
    ]
    for name, entry in result["violation_rate_contrasts_at_1pct"].items():
        first, second = name.split("_minus_")
        low, high = np.array(entry["crossed_bootstrap_ci95"]) * 100
        lines.append(
            f"| {LABELS[first]} minus {LABELS[second]} | {100 * entry['mean']:+.2f} | [{low:+.2f}, {high:+.2f}] |"
        )
    lines += [
        "",
        "All violations count here, but their size matters. At a 1% cap error, mean positive excess "
        "averaged across all outputs is .000300 for Q, .001435 for P, .001129 for R and .001310 for B+P. "
        "R has a slightly higher violation frequency than P and a lower average excess. These are distinct measures. "
        "Maximum excess for the hard-filtered original arms is at most .007201. At a .1% error it is at most .000721. "
        "No harm penalty or physical failure is inferred from these normalized command-budget violations.",
        "",
        "Both 1%-margin controls have zero violations at errors up to 1%, within the separately specified 1e-6 tolerance. "
        "At 2% and 5% error their margins no longer cover the bias; the full CSV reports this for both controls. "
        "There were no conditional refusals, and all poor-return outcomes are retained in the audit.",
        "",
        "The current task uses an exact action-budget set with an imposed bias. It establishes a boundary-sensitivity mechanism, "
        "not that estimation error explains most task-performance loss. Projection with a valid uncertainty margin is an essential baseline.",
        "",
        "The figure uses a horizontal axis that is linear near zero and logarithmic beyond .01%; lines connect the fixed error grid. "
        "It displays frequency and excess magnitude separately.",
        "",
        "[Sensitivity figure](boundary_sensitivity.pdf) · [All arms and magnitudes](table.csv) · [Independent audit](audit.json)",
    ]
    with (output / "RESULTS.md").open("a") as stream:
        stream.write("\n".join(lines) + "\n")
    (
        output / "paper_mismatch.tex"
    ).write_text(r"""\paragraph{Exploratory sensitivity to safe-set error.}
Using all eight frozen HalfCheetah policies, we re-evaluated each of the 2,048
outputs per arm against nearby effort caps, without changing its trajectory
or reward. This is a retrospective uniform-bias probe, not an evaluation of a
learned estimator. With a true cap 0.1\% below the supplied cap, exact global
projection violated the true cap in 17.53\% of outputs, exact conditioning in
0.88\%, finite refinement in 6.79\%, and repaired BayesFP in 16.11\%.
At 1\% error these rates were 22.75\%, 8.20\%, 24.46\%, and 21.04\%, respectively.
The conditioning--projection difference at 1\% was $-14.55$ percentage points
(exploratory 95\% crossed model/state bootstrap interval $[-17.77,-11.52]$);
the refinement--projection difference was $+1.71$ points ($[-3.37,6.79]$).
Thus finite refinement did not consistently inherit the reference's robustness.

As an uncertainty-aware control, we executed another 4,096 trajectories using
projection and conditioning on a fixed 1\%-smaller set. At 1\% bias this is the
true set, and its projection is the exact oracle projection. Both controls had
zero violations and zero rejection-budget failures. The margin changed projection
return by $+1.33$ points ($[-1.50,4.25]$), and conditioning on the same smaller set
differed from projection by $+0.43$ ($[-4.34,5.26]$). This control requires a valid
error bound supplied equally to both methods. It prevents attributing a benefit
to conditioning that conservative projection can obtain without detectable
task-quality loss. The outcomes are normalized effort-envelope violations,
not collisions: at 1\% bias the largest excess for the original hard-filtered
arms was at most 0.007201, with numerical tolerance $10^{-6}$.
These findings motivate testing geometric or dynamical estimator errors and
calibrated-margin baselines; they do not identify the dominant cause of harm
in general robotics tasks.
""")
    print(f"Wrote mismatch tables, figure and paper text to {output}", flush=True)

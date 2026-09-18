"""Generate model-level tables, figures and paper text from completed results."""

import csv
import os
import numpy as np
from .artifacts import read_json, write_json


def interval(result, key="crossed_bootstrap_ci95"):
    low, high = result[key]
    return f"{result['mean']:+.2f} [{low:+.2f}, {high:+.2f}]"


def build_report(output):
    summary = read_json(output / "summary.json")
    config = summary["config"]
    comparisons = summary["comparisons"]
    primary = comparisons["refinement_minus_bayesfp_projected"]
    names = ["conditional", "projection", "refinement", "bayesfp", "bayesfp_projected"]
    rows = []
    for index, model in enumerate(summary["models"]):
        row = {
            "seed": model["training_seed"],
            "base_return": model["base_mean_return"],
            "acceptance": model["acceptance"],
        }
        for arm in names:
            row[arm] = model["arms"][arm]["mean_return_given_output"]
        for key, comparison in comparisons.items():
            row[key] = comparison["model_means"][index]
        rows.append(row)
    with (output / "table.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    total = len(rows) * config["states"] * config["replicates"]
    totals = {
        arm: {
            "requests": total,
            "unsafe": sum(m["arms"][arm]["unsafe_outputs"] for m in summary["models"]),
            "native_failures": sum(
                m["arms"][arm]["native_failures"] for m in summary["models"]
            ),
            "nonpositive_returns": sum(
                m["arms"][arm]["nonpositive_returns"] for m in summary["models"]
            ),
            "mean_return": float(np.mean([r[arm] for r in rows])),
            "mean_boundary_fraction": float(
                np.mean(
                    [m["arms"][arm]["boundary_fraction"] for m in summary["models"]]
                )
            ),
            "mean_mmd_to_conditional": float(
                np.mean(
                    [m["arms"][arm]["mmd_to_conditional"] for m in summary["models"]]
                )
            ),
        }
        for arm in names
    }
    write_json(output / "totals.json", totals)
    decision = (
        "The preregistered useful-advantage criterion was met."
        if summary["useful_advantage_replicated"]
        else "The preregistered useful-advantage criterion was NOT met."
    )
    mean_above = primary["mean"] >= config["minimum_useful_return_gain"]
    note = "Its mean exceeds" if mean_above else "Its mean is below"
    lines = [
        "# Independent-model HalfCheetah replication",
        "",
        f"**{decision}** Refinement minus BayesFP+projection is **{interval(primary)}** native-return points "
        f"(95% crossed model/state bootstrap). The model-level t interval is "
        f"**{interval(primary, 'model_t_ci95')}**. {note} the unchanged **{config['minimum_useful_return_gain']:.2f}**-point useful-effect threshold.",
        "",
        f"All **{len(rows)} independently trained models**, **{config['states']} fresh shared simulator states**, "
        f"and **{config['replicates']} requests/state/model** are included. Refinement beats repaired BayesFP in "
        f"**{primary['positive_models']}/{len(rows)} model means**. This replicates training randomness on one fixed demonstration dataset; "
        "it does not replicate demonstration collection or establish universal superiority.",
        "",
        "| Training seed | Base | Q | P | R | B+P | R − (B+P) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = [
            row[k]
            for k in [
                "base_return",
                "conditional",
                "projection",
                "refinement",
                "bayesfp_projected",
                "refinement_minus_bayesfp_projected",
            ]
        ]
        lines.append(
            f"| {row['seed']} | " + " | ".join(f"{v:.2f}" for v in values) + " |"
        )
    lines += [
        "",
        "Q is first-feasible rejection; P is exact global full-chunk projection; R is the fixed finite "
        "annealed gamma=.5 refinement; B+P is paper-derived BayesFP K=32/strength20 followed by the identical projection. "
        "Every arm executes .4 s and shares the same public expert continuation through 4 s. The RMS .72 command limit applies to the initial chunk.",
        "",
        "## Secondary descriptive comparisons",
        "",
        "| Contrast | Mean [crossed 95% interval] | Model t 95% interval |",
        "|---|---:|---:|",
    ]
    for key, label in [
        ("conditional_minus_projection", "Conditioning − projection"),
        ("refinement_minus_projection", "Refinement − projection"),
    ]:
        lines.append(
            f"| {label} | {interval(comparisons[key])} | {interval(comparisons[key], 'model_t_ci95')} |"
        )
    lines += [
        "",
        "These secondary intervals are descriptive, without a new family of superiority claims. "
        "A refinement gain over BayesFP does not establish accurate conditional sampling or a conditioning-target advantage.",
        "",
        "## All requested outcomes",
        "",
        "| Arm | Mean return | Unsafe/requested | Native terminations | Nonpositive-return rollouts | Boundary fraction | MMD to Q |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, entry in totals.items():
        lines.append(
            f"| {arm} | {entry['mean_return']:.2f} | {entry['unsafe']}/{total} | {entry['native_failures']} | "
            f"{entry['nonpositive_returns']} | {entry['mean_boundary_fraction']:.3f} | {entry['mean_mmd_to_conditional']:.3f} |"
        )
    q_floor = float(np.mean([m["q_vs_q_mmd"] for m in summary["models"]]))
    expert_return = float(np.load(output / "expert.npz")["rewards"].sum(1).mean())
    base_ratios = [r["base_return"] / expert_return for r in rows]
    lines += [
        "",
        f"Rejection refusals: **{summary['refusals']}**. Hard-output violations: **{summary['hard_output_violations']}**. "
        f"All {summary['total_method_outputs_requested']:,} method requests are retained. HalfCheetah has no native unhealthy termination; "
        "zero terminations does not certify stable posture. Nonpositive-return counts retain visibly poor trajectories without inventing a new reward.",
        "",
        f"The mean independent Q-versus-Q MMD is {q_floor:.3f}. These pooled action-kernel diagnostics are not "
        "proof of state-conditional equality or multiple surviving coherent modes. The base models retain "
        f"{100 * min(base_ratios):.1f}–{100 * max(base_ratios):.1f}% of the fresh-bank expert return ({expert_return:.2f}), "
        "using one generated chunk and common continuation. No model was excluded or refitted.",
        "",
        "## Cost and provenance",
        "",
        "| Item | Mean seconds/model |",
        "|---|---:|",
    ]
    for name in ["base_proposal_bank", "projection", "refinement", "bayesfp"]:
        lines.append(
            f"| {name} | {np.mean([m['timing_seconds'][name] for m in summary['models']]):.3f} |"
        )
    lines += [
        "",
        "Each bank contains 32,768 proposals (128/request), all saved. Logical first-accept costs are saved separately. "
        "R uses 1,024 refinement scores plus its shared 100-step base draw; B uses 3,200 scores. Timing is batched on the recorded workstation, "
        "not an isolated online latency benchmark. Sampling settings were not retuned on these new models or states.",
        "",
        "[Locked protocol](../../docs/protocols/halfcheetah_replication.md) · [Config/code/asset hashes](lock.json) · "
        "[All-model summary](summary.json) · [CSV](table.csv) · [Audit](audit.json) · [Figure](model_comparison.pdf)",
        "",
        "The consolidation first matched the historical base sampler, refinement, BayesFP particles and native rollouts bit for bit. "
        "The fresh public-controller collection reproduced the original complete-episode archive byte for byte. These checks protect implementation fidelity; "
        "the eight new training seeds provide the independent-model evidence.",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/safety-conditioning-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), layout="constrained")
    for ax, key, title in zip(
        axes,
        comparisons,
        [
            "Refinement − (BayesFP + projection)",
            "Conditioning − projection",
            "Refinement − projection",
        ],
    ):
        result = comparisons[key]
        ax.scatter(result["model_means"], np.arange(len(rows)), color="#2166ac")
        lower, upper = result["crossed_bootstrap_ci95"]
        ax.errorbar(
            result["mean"],
            len(rows) + 0.5,
            xerr=[[result["mean"] - lower], [upper - result["mean"]]],
            fmt="D",
            color="#b35806",
            capsize=4,
        )
        ax.axvline(0, color=".4", linewidth=1)
        if key == "refinement_minus_bayesfp_projected":
            ax.axvline(
                config["minimum_useful_return_gain"],
                color="#b35806",
                linestyle="--",
                label="Fixed useful-effect threshold",
            )
        ax.set_yticks(
            list(range(len(rows))) + [len(rows) + 0.5],
            [str(r["seed"]) for r in rows] + ["Aggregate 95% CI"],
        )
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Native return difference")
        ax.grid(axis="x", alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", fontsize=9)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(output / f"model_comparison.{ext}", dpi=180)
    plt.close(fig)

    tex = [
        r"\subsection{Independent-model replication}",
        "We froze the previously promising HalfCheetah RMS-0.72 setting and trained eight new models with independent initialization and optimization seeds. "
        "The demonstration dataset and episode split were held fixed. Every model was evaluated at the same 64 fresh states with four replicates per state, "
        "without retuning or excluding weak models.",
        "",
        f"Refinement minus repaired BayesFP averaged {primary['mean']:.2f} native-return points "
        f"(95\\% crossed model/state bootstrap interval [{primary['crossed_bootstrap_ci95'][0]:.2f}, {primary['crossed_bootstrap_ci95'][1]:.2f}]; "
        f"model-level t interval [{primary['model_t_ci95'][0]:.2f}, {primary['model_t_ci95'][1]:.2f}]). "
        f"The mean favored refinement in {primary['positive_models']} of eight models. "
        + decision
        + " "
        f"The threshold remained {config['minimum_useful_return_gain']:.2f} points.",
        "",
        f"Conditioning minus projection was {interval(comparisons['conditional_minus_projection'])}; "
        f"refinement minus projection was {interval(comparisons['refinement_minus_projection'])} "
        "(secondary descriptive 95\\% intervals). These comparisons separate target utility from finite-sampler performance.",
        "",
        f"There were {summary['refusals']} rejection refusals and {summary['hard_output_violations']} hard-output violations. "
        f"Raw BayesFP violated the initial-chunk constraint in {totals['bayesfp']['unsafe']}/{total} outputs; its repaired arm had zero violations. "
        "The study remains conditional on one fixed demonstration dataset and does not establish general posterior accuracy or robust closed-loop diffusion locomotion.",
    ]
    (output / "paper_replication.tex").write_text("\n".join(tex) + "\n")
    print(f"Reports and exportable figures written to {output}", flush=True)

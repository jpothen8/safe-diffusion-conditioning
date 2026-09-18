# Current experimental evidence

**The independent-model HalfCheetah check finds a smaller, statistically positive refinement gain, but does not replicate the predefined useful-effect claim.** Eight new models were trained with independent seeds and evaluated on 64 fresh shared states, four requests per state. No model was excluded or retuned.

| Latest HalfCheetah contrast | Mean native-return gain | 95% model/state bootstrap interval |
|---|---:|---:|
| Refinement − (BayesFP + identical projection) | +16.88 | [6.06,28.29] |
| Conditioning − projection | +0.98 | [−2.70,4.88] |
| Refinement − projection, secondary descriptive | +17.12 | [6.19,28.18] |

The primary mean is below the fixed **26.19-point** useful-effect threshold. Refinement wins over repaired BayesFP in 7/8 model means; the model-level t interval is [8.83,24.94]. All 10,240 method requests are retained. There are **zero rejection refusals and zero hard-output violations**. Raw BayesFP violates the initial-chunk constraint in **320/2,048** outputs. Poor trajectories are retained: nonpositive-return counts are 52 for conditioning, 52 for projection, 40 for refinement, and 42 for repaired BayesFP. HalfCheetah has no native unhealthy termination, so zero terminations is not evidence of general stability.

[Full report](results/halfcheetah_replication/RESULTS.md) · [Per-model figure](results/halfcheetah_replication/model_comparison.pdf) · [Audit](results/halfcheetah_replication/audit.json) · [Locked protocol](docs/protocols/halfcheetah_replication.md).

## How this changes the paper

Keep two claims separate. **Conditioning versus projection** asks whether the target preserves useful behavior. **Finite refinement versus feasible competitors** asks whether the implemented sampler helps. The new HalfCheetah result concerns the second question. Conditioning remains inconclusive, and refinement is farther from rejection by the recorded action-kernel diagnostic: mean MMD .145 versus .085 for repaired BayesFP, with independent rejection discrepancy .080. The benefit cannot be attributed to greater conditional accuracy. BayesFP remains faster in recorded batch timings.

This study conditions on the original demonstration dataset and repeats initialization/optimization randomness. It does not measure dataset-collection uncertainty. Confidence intervals resample models and the shared state bank; eight training seeds still give limited precision.

## Exact projection and errors in the supplied set

The baseline already uses exact global Euclidean projection of the full 48-coordinate action chunk. Its constraint is a command RMS budget over .4 seconds, intersected with action bounds. The new question is whether the supplied set matches the true set.

A retrospective probe keeps all eight models and all original outputs, evaluates a fixed grid of true caps, and adds 4,096 simulator rollouts for projection/conditioning with a fixed 1% margin. Each arm has 2,048 outputs. The true violation rates below use the same numerical tolerance, 1e-6; the cap error is separate.

| True cap reduction from supplied cap | Conditioning | Exact projection | Finite refinement | BayesFP + projection | Projection with 1% margin |
|---|---:|---:|---:|---:|---:|
| 0% | 0% | 0% | 0% | 0% | 0% |
| .1% | .88% | 17.53% | 6.79% | 16.11% | 0% |
| 1% | 8.20% | 22.75% | 24.46% | 21.04% | 0% |

At the fixed 1% error, conditioning reduces violation frequency by **14.55 percentage points [11.52,17.77]** versus projection. Refinement differs by **+1.71 [−3.37,6.79]** points: no supported improvement at that error. Violation size also matters: mean excess across all outputs is .000300 for conditioning, .001435 for projection and .001129 for refinement. The maximum excess of the hard-filtered original arms is at most .007201 normalized action units. These are effort-envelope violations, not measured collisions or physical damage.

Projection with a 1% margin is the **exact true-set oracle projection at 1% error**. It requires the true cap or a valid error bound. Its return differs from ordinary projection by **+1.33 [−1.50,4.25]**; conditioning on the same smaller set differs from oracle projection by **+.43 [−4.34,5.26]**. Both controls have zero violations up to 1% error and zero rejection refusals. Thus conservative projection is competitive here and must remain a baseline. A higher native return from an arm that violates the true set is not a safe-performance win.

The independent audit verified every selection, exact projection, violation count, excess metric and crossed bootstrap interval, and replayed 32 new control trajectories with zero reward/state error. The fixed error grid is exploratory, with no independent estimator or new training; rows reuse the same outputs. It supports the proposed boundary-sensitivity mechanism but cannot establish that set error causes most projection harm.

[Full mismatch report](results/safe_set_mismatch/RESULTS.md) · [Figure](results/safe_set_mismatch/boundary_sensitivity.pdf) · [Audit](results/safe_set_mismatch/audit.json) · [Mechanism and next experiment](docs/PROJECTION_AND_SET_ERROR.md).

## Evidence across tasks

| Task / study | Conditioning-target finding | Finite-sampler finding |
|---|---|---|
| Pendulum, independent confirmation and fresh BayesFP comparison | Useful native-return gains over exact projection, about 48–56 points; projection suppresses coherent swing motion | Refinement recovers the gain; repaired BayesFP is competitive; measurable distribution gaps remain |
| HalfCheetah, original one-model pilot | Inconclusive versus projection | Original +30.61-point gain over repaired BayesFP at RMS .72 motivated replication |
| HalfCheetah, eight-model replication | +0.98, interval crosses zero | +16.88 over repaired BayesFP; positive but below the fixed useful-effect threshold |
| Walker2d | Slightly favors projection at both effort caps | Small/mixed gains; one refinement rollout falls after its feasible chunk |
| Push-T, released diffusion policy | Null/slightly favors projection; no useful confirmed gain | Not used as a positive method result |

**Recommendation:** proceed with Pendulum as the motivating conditioning example. Present HalfCheetah as a modest finite-sampler improvement with limitations, and retain Walker2d/Push-T as counterexamples. Do not claim universal quality improvement, exact conditional sampling, online efficiency superiority, or preserved weights among multiple surviving safe modes.

For the robustness claim, proceed next with **a calibrated dynamic-safe-set error check**, comparing all methods with the same uncertainty information and including exact or carefully audited full-horizon projection with margins. Do not present the current finite refinement as uniformly more robust than projection.

## Reproducibility and consolidation

One maintained package and CLI cover collection, training, sampling, evaluation, uncertainty, audits and reporting. Compatibility checks reproduce the historical base sampler, refinement, BayesFP particle cloud and native rollouts bit for bit. Fresh complete-episode collection reproduces the original archive byte for byte. The independent audit verifies all new first-feasible selections, projections, checkpoint hashes, normalizers and bootstrap intervals; native replay reward/state errors are zero.

Earlier evidence is preserved: [original Push-T report](results/heldout/RESULTS.md), [Pendulum confirmation](results/pendulum/RESULTS.md), and [three-task BayesFP comparison](results/comparison_figures/RESULTS.md). Full artifacts are indexed in [ARTIFACTS.md](docs/ARTIFACTS.md). Use [README.md](README.md) for consolidated commands and [PAPER_INTEGRATION.md](docs/PAPER_INTEGRATION.md) for current draft text.

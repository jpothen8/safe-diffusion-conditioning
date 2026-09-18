# Safe-set mismatch sensitivity

This retrospective analysis treats .72 as the estimated safe RMS cap and scores the same outputs against nearby true caps. Positive error means the estimated set is too permissive. No estimator was trained and these violations are effort-envelope violations, not observed collisions or physical damage.

| True-cap reduction | True cap | Q violations | P violations | R violations | B+P violations | P with 1% margin |
|---|---:|---:|---:|---:|---:|---:|
| -1.00% | 0.727200 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 0.00% | 0.720000 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 0.01% | 0.719928 | 0.10% | 16.99% | 4.30% | 15.72% | 0.00% |
| 0.10% | 0.719280 | 0.88% | 17.53% | 6.79% | 16.11% | 0.00% |
| 0.50% | 0.716400 | 4.25% | 20.02% | 16.65% | 18.26% | 0.00% |
| 1.00% | 0.712800 | 8.20% | 22.75% | 24.46% | 21.04% | 0.00% |
| 2.00% | 0.705600 | 17.33% | 29.93% | 36.33% | 27.73% | 29.93% |
| 5.00% | 0.684000 | 43.07% | 51.46% | 66.89% | 49.80% | 51.46% |

Q is exact first-feasible conditioning on the supplied set; P is exact global Euclidean projection over the entire chunk; R is finite refinement. Each original arm has 2,048 outputs. The same saved trajectories appear at all error levels, so rows are not independent experiments.

The 1% margin projection uses cap .7128. At a true 1% error it is exactly the oracle projection onto the true set. It additionally assumes a known 1% error bound, so it is a conservative control rather than a secretly better-informed competitor. The same-margin conditional reference also receives that set. Both controls execute new .4 s chunks with identical continuation/native reward.

## Return and uncertainty

| Contrast | Mean native-return difference | Crossed 95% interval |
|---|---:|---:|
| projection_margin_1pct_minus_projection | +1.333 | [-1.502, +4.248] |
| conditional_margin_1pct_minus_projection_margin_1pct | +0.429 | [-4.340, +5.262] |
| refinement_minus_projection_margin_1pct | +15.790 | [+4.326, +27.453] |

These are exploratory intervals; no error setting or sampler was chosen by reward. The original arms' returns stay unchanged when only the true-set label changes. Comparisons between an unsafe original arm and a safe conservative control do not establish a safe-performance win.

## Interpretation

An inward boundary error can turn an atom of projected boundary mass into violations even when projection onto the estimated set is mathematically exact. A conditional law without mass on that boundary has no analogous jump, though it still becomes unsafe if it occupies the strip removed by the error. Finite refinement can leave substantial near-boundary mass and need not inherit the exact conditional reference's robustness.

This probe tests a controlled uniform cap bias, not whether most projection harm in robotics is caused by estimation error. The next dynamic-constraint experiment should perturb a simulator-based clearance/velocity/posture estimate and evaluate the true predicate, including projection with calibrated margins and the same uncertainty information for every arm.

[Fixed exploratory protocol](../../docs/SAFE_SET_MISMATCH.md) · [All errors, violations and excess magnitudes](summary.json) · [Source hashes](lock.json)

## Violation uncertainty and magnitude

At the fixed 1% error, these exploratory crossed-bootstrap intervals resample models and the shared state bank. Negative differences mean fewer violations. The small positive refinement difference is inconclusive.

| Contrast | Violation difference (percentage points) | 95% interval |
|---|---:|---:|
| Conditioning Q minus Exact projection P | -14.55 | [-17.77, -11.52] |
| Finite refinement R minus Exact projection P | +1.71 | [-3.37, +6.79] |
| BayesFP + projection minus Exact projection P | -1.71 | [-5.62, +2.20] |
| Projection, 1% margin minus Exact projection P | -22.75 | [-27.15, -18.55] |

All violations count here, but their size matters. At a 1% cap error, mean positive excess averaged across all outputs is .000300 for Q, .001435 for P, .001129 for R and .001310 for B+P. R has a slightly higher violation frequency than P and a lower average excess. These are distinct measures. Maximum excess for the hard-filtered original arms is at most .007201. At a .1% error it is at most .000721. No harm penalty or physical failure is inferred from these normalized command-budget violations.

Both 1%-margin controls have zero violations at errors up to 1%, within the separately specified 1e-6 tolerance. At 2% and 5% error their margins no longer cover the bias; the full CSV reports this for both controls. There were no conditional refusals, and all poor-return outcomes are retained in the audit.

The current task uses an exact action-budget set with an imposed bias. It establishes a boundary-sensitivity mechanism, not that estimation error explains most task-performance loss. Projection with a valid uncertainty margin is an essential baseline.

The figure uses a horizontal axis that is linear near zero and logarithmic beyond .01%; lines connect the fixed error grid. It displays frequency and excess magnitude separately.

[Sensitivity figure](boundary_sensitivity.pdf) · [All arms and magnitudes](table.csv) · [Independent audit](audit.json)

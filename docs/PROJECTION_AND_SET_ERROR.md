# Exact projection and an inaccurate supplied set

There are two different notions of "true projection."

1. **An exact nearest-point computation on the supplied set.** The current baseline already does this globally over the full action chunk. It is not an approximate local or one-step filter.
2. **Projection onto the actual true safe set.** This is an oracle baseline if the deployed filter only knows an estimate. A mathematically exact projection onto the estimated set need not satisfy the true predicate.

For HalfCheetah the chunk has eight controls at .05 s each, with six normalized actions per control. Write `rms(A) = ||A||₂ / sqrt(48)` and `C_c = {A in [-1,1]^48: rms(A) <= c}`. The frozen base policy includes physical clipping, so its samples already lie in the box. Exact Euclidean projection is therefore

`P_c(A) = min(1, c / rms(A)) A`.

This is the full-dimensional global solution. For refinement's proposals outside the box, the implementation uses the box/ball KKT solution with a scalar multiplier, previously checked against independent conic optimization. In this experiment, "safety" is an action-command budget; it is not yet a physical state constraint.

## Why boundary error changes the comparison

Let `C_hat` be the supplied set and let a true set `C_ε` approach it from the interior as a boundary error ε vanishes. Conditioning on the supplied set has true violation probability

`q_hat(C_εᶜ) = p(C_hat \ C_ε) / p(C_hat)`.

If `p(∂C_hat) = 0` and these inner sets exhaust its interior, this probability tends to zero as ε tends to zero. This statement permits multimodal policies and does not require reward optimality. It does require no base-policy mass on the affected boundary. Physical clipping can create atoms on other boundaries, so that assumption must be checked for the actual predicate.

In contrast, exact projection pushes infeasible samples onto the boundary. In the present radial case, if `m = p(rms(A) > c)`, the projected law places at least mass `m` on `rms(A) = c`. Every strictly smaller true cap excludes those points. Consequently true violation probability is at least `m` for every positive inward error, in exact arithmetic. Numerical tolerance rounds off this jump at sufficiently tiny error. This explains why exact projection can be brittle even though its solver is perfect.

This is a statement about violation frequency. At small errors the magnitude of a violation may also be small; frequency alone is not a measure of physical harm. Nor does conditioning guarantee robustness to a wrong set: it can retain unsafe interior modes, and a conservative set can remove useful behaviors that conditioning cannot recover. For `C_hat` contained in `C_true`, either hard-filtered law is true-safe.

## What the executed probe establishes

The estimated cap is .72 and the true cap is `.72*(1-ε)`. The fixed grid includes conservative error, zero, .01%, .1%, .5%, 1%, 2% and 5% inward errors. Eight independently trained policies share 64 snapshots and four requests/state. All 2,048 outputs per arm are retained. No sampler is retuned or selected using the new safety labels.

At .1% error, true violations are .88% for exact conditioning, 17.53% for exact projection, 6.79% for finite refinement and 16.11% for repaired BayesFP. At 1%, refinement rises to 24.46% versus projection's 22.75%; the difference is statistically inconclusive. Exact conditioning remains lower at 8.20%. The reference's favorable boundary behavior is therefore an empirical target for the sampler, not a property that can already be claimed for finite refinement at every error size.

The exact oracle matters. A fixed 1% margin means cap .7128 and coincides with the true set at 1% error. New projection and conditional-control rollouts on that same set are both safe. Their return difference is +.43 [−4.34,5.26]. Margin projection differs from the original projection by +1.33 [−1.50,4.25]. It therefore resolves the tested error without detectable quality loss when its error bound is known. At 2% error the bound is insufficient and violations return. Full curves, magnitudes, uncertainty and raw BayesFP are in the [report](../results/safe_set_mismatch/RESULTS.md).

## The next decisive robustness check

Use a physical velocity, posture or clearance predicate over the same full horizon, and separate the true simulator from the predictor supplying feasibility. Fix a small family of model/measurement errors on development states, calibrate any margin using only development residuals, then freeze it before fresh held-out states. Every method must receive the same predictor, preview and available error bound. The true simulator supplies evaluation labels only.

Compare conditioning, exact/audited full-horizon projection, finite refinement, BayesFP plus identical hard repair, and conservative versions under the same calibrated margin. Include the true-set oracle as an explicitly privileged diagnostic. Measure the true violation frequency, violation size and native quality separately, retaining refusals and failures. First verify viable safe support and the quality cost of the conservative projection. A benefit over unbuffered projection alone would not establish a benefit over a competitive robust filter.

The current probe motivates this check; it cannot establish the user's broader hypothesis that safe-set error accounts for most projection harm. Existing Pendulum results already show a behavior-quality difference with an accurate set. The two mechanisms should remain separate in the paper.

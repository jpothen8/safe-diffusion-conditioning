# First experiment: frozen Push-T diffusion policy, pusher speed constraint

Protocol written before held-out proposals/outcomes. The bounded pilot is a diffusion experiment using a released checkpoint; it does not train or test a new safety-aware diffusion sampler. Configuration: `configs/pilot.json`. Development uses seeds 200000–200005; held-out simulation uses 300000–300011. These are fresh simulator cases, not a claim of a new split of the authors' demonstrations.

## Target and frozen policy

For a simulator snapshot plus the last two keypoint observations, let p(A|s) be the **actual** frozen EMA transformer output. Use author checkpoint `epoch=0850-test_mean_score=0.967.ckpt`, strict state-dict loading, eval mode, float32, the checkpoint normalizer, DDPM with all 100 reverse steps, squared-cosine schedule, fixed-small variance and `clip_sample=True`. No reward ranking, added action noise, altered scheduler, or fitted normalization. Dropout is disabled by eval mode. The 10 predicted positions include a past-aligned slot: executable A is exactly upstream `action_pred[:,1:9]`, eight 2D target positions. No extra clipping is applied. Environment declared action bounds are [0,512] pixels per coordinate; the safety predicate includes them.

Observations contain nine block keypoints and pusher xy (20 values); two observations are supplied, duplicated at reset. The observation masks are all visible and do not enter the network. Privileged pusher velocity is supplied to **both safety methods only**. Frozen policy, environment parameters and native reward are unchanged. Source and model hashes are recorded in `records/asset_manifest.json`.

Let C(s) contain action chunks within the action bounds for which the pusher speed at the start and every 0.01 s simulator substep is at most v_max. Numerical membership uses an absolute 1e-5 pixel/s tolerance (also 1e-5 pixels for action bounds). No obstacle is inserted. The constraint is a pusher actuator speed restriction, not a certificate for block or whole-scene safety. The discrete simulator holds the kinematic velocity between substeps; this covers its full simulated path.

q(A|s)=p(A|s)1[A∈C(s)]/Z(s) when Z(s)>0. A failure to observe acceptance does not establish Z=0. Report per-snapshot acceptance and confidence intervals, not just pooled Z. The simulator reset has zero velocity, so staying at the initial pusher position is a feasible action sequence. For moving snapshots, test the initial velocity and solve the same feasibility problem; a safe set existing geometrically does not establish positive policy mass.

## Conditional reference and projection

For each snapshot and each of 8 replicates, generate 64 independent diffusion trajectories. Save **all** proposals, seeds, speed traces and membership. For each severity, A is the first feasible proposal in that replicate's fixed order. The first accepted proposal has exactly the conditional law given that the finite budget succeeds. Report the first-accept index, logical proposal count, actual generated count, unused proposals and budget refusals. Never select by reward. Generate whole batches for GPU efficiency; count their entire cost. Each replicate's first raw proposal also supplies B and the unfiltered descriptive baseline. Sharing that proposal couples the estimators without changing either marginal distribution. All severity levels reuse the proposal bank.

B is the unique minimizer of sum_t ||(U_t−A_t)/512||² subject to **the same entire 0.8 s feasible set**. Pusher dynamics are independent of object contact because the pusher is kinematic. At each physics step:

```
v[k+1] = v[k] + 0.01 * (100 * (U[floor(k/10)] - x[k]) - 20*v[k])
x[k+1] = x[k] + 0.01 * v[k+1]
```

Thus all 80 velocities are affine functions of the 16 decision variables. Euclidean speed constraints are second-order cones and the squared Euclidean objective is strictly convex. CVXPY 1.6.5 / Clarabel 0.10.0 solves this global convex problem, with absolute/relative gap and feasibility tolerances 1e-9, maximum 200 iterations. Already feasible proposals use the exact identity projection. Nonoptimal solver status, missing solutions and externally failed feasibility checks are separate failures; no fallback output is silently substituted. The solver targets the nominal cap; final membership uses the common stated numerical tolerance. This tiny numerical margin is the only boundary approximation.

Validate the affine trajectory against observed original Pymunk steps, including moving/contact cases. After projection or rejection, validate actual executed speed at **all 80 substeps**. Report absolute model error and unsafe outputs separately from solver failures. There is no one-step-only baseline or local nonlinear optimizer in this comparison.

## Controlled snapshots and execution

Use all 12 consecutive held-out seeds in the configuration without screening for favorable acceptance or reward. Even seeds use reset snapshots; odd seeds use a fixed 16-step (1.6 s) warmup from the unfiltered policy. Warmup seeds are `snapshot_seed+500+chunk_index`. Save the complete action prefix and numeric state; restore snapshots by exact deterministic reset-and-replay, preserving contact history. Save the two policy observations. Do not use reduced state reconstruction that omits collision state.

Control period is 0.1 s; prediction/execution horizon is eight controls = **0.8 s** for both methods. Execute the entire chunk. Continue for 72 more controls = **7.2 s** using the identical frozen unfiltered policy with 0.8 s replanning. Both branches use identical random-number seeds and batch positions at each continuation call. Completion is absorbing: record native terminal reward 1 for the remaining evaluation duration and do not simulate further. Continue to log speed violations during continuation, but safety is only promised over the initial chunk. The supplied safety restriction is active for that chunk; whole-episode safety is a subsequent experiment requiring a common constrained continuation controller.

The fixed total is 8.0 s after each snapshot. Keep native reward r=clip((intersection area / target area)/0.95,0,1), and native termination at coverage>0.95. This is target coverage, **not intersection-over-union**. Do not equate a maximum score near 1 with completing the task.

## Outcomes and uncertainty fixed before evaluation

Primary severity: **400 pixels/s**. Secondary severities: 300 and 600, selected from the development speed distributions to span restrictive and permissive cases. The primary metric is the paired difference (conditional minus projected) in mean native reward over the fixed 80-control evaluation. Minimum useful improvement: **0.05** absolute score. Primary uncertainty: 10,000 paired cluster bootstrap resamples of the 12 snapshots (averaging the 8 within-snapshot replicates first), seed 72401, 95% percentile interval. Snapshots, not proposals, are the independent unit. This is a small pilot with weak tail/power guarantees, not a definitive benchmark. No optional expansion based on significance.

Also report native maximum reward, completion proportion, first completion time (right-censored at 8 s), chunk reward, unsafe-output rate, reference refusals, solver failures, continuation speed violations and wall times. Do not discard unsuccessful episodes. An operational score assigns 0 to an absent/unsafe output and is explicitly **not** a conditional-target return estimate. If any target outputs are unavailable, report complete-pair descriptions plus worst-case [0,1] missing-outcome bounds over the full requested sample; do not claim q superiority from success-only comparisons.

Distribution diagnostics: maximum speed; proportion within 1 pixel/s of the cap; projected action distance; endpoint signed lateral displacement relative to the initial pusher-to-block vector (left/center/right with ±10 pixel thresholds); block rotation over the chunk (negative/neutral/positive with ±0.05 rad thresholds); trajectories and action first differences. The route labels are descriptive geometric bins, **not proof of multimodality**. Show their quality and contact/block-motion statistics at the same snapshots to distinguish useful alternatives from idle/jitter trajectories. Distinct bins across unrelated states do not establish conditional modes. A claim of separated coherent modes requires inspection of within-snapshot trajectory clusters and useful subsequent reward in each cluster; absence of such evidence must be stated.

Proceed toward a larger comparison only if the unmodified policy is useful, the constraint leaves usable safe support, and actual feasibility checks pass. Evidence for the target hypothesis requires the primary interval's lower endpoint exceeding +0.05, with reference coverage and coherent behaviors established. An interval spanning effects on both sides is inconclusive; an interval excluding +0.05 argues against a useful gain for this frozen design. A projection advantage or null result is valid. Other constraints or longer closed-loop horizons require a new development protocol, not post-hoc replacement of this result.

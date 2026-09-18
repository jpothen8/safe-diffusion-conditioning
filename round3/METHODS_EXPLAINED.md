# What is being compared

The input is the same observation, frozen diffusion policy and externally supplied feasible set C. A policy sample is an entire action chunk, including its ordinary denoising procedure, normalization and action clipping. The experiment executes that chunk and then gives every method the same continuation controller and unchanged task reward.

## Conditioning versus projection

**Conditioning asks: which behaviors did the policy already consider feasible?** Draw independent chunks and return the first one in C. Never rank accepted chunks by reward. Given a successful finite rejection budget, this samples

`q(A|s) = p(A|s) 1[A in C(s)] / Z(s)`.

Relative probabilities of feasible behaviors are preserved. An illustrative prior with safe behavior weights 20% and 10% becomes 2/3 and 1/3 after rejecting the unsafe remainder. This is an illustration, not measured data. Conditioning does not maximize reward and can lose to projection when the policy's feasible preferences are poor or acceptance is impractically low.

**Projection asks: what is the closest feasible modification of this particular draw?** Draw once, then minimize squared Euclidean distance over the entire feasible action chunk. Already feasible draws are unchanged. Infeasible probability mass is moved onto whichever feasible boundary points are closest. This can preserve a good maneuver, damage its timing, or create a poor boundary behavior. It generally changes relative probabilities among feasible outcomes.

The projection baseline is exact in these experiments. Pendulum uses a Cartesian torque box; global projection is coordinate clipping. The new MuJoCo experiments use a whole-chunk RMS effort ball intersected with actuator bounds. Projection is determined by its global convex optimality condition. It is not a one-step filter pretending to enforce a longer coupled constraint.

## Exact reference versus the proposed finite sampler

Rejection is the **reference target sampler**, not the proposed efficient algorithm. It lets us ask whether the conditional distribution is worth approximating, while counting all proposals and failures.

The draft's implemented **finite refinement** starts from a projected policy draw, adds learned-score drift and Gaussian noise, and projects after each update. We use gamma=1/2 with decreasing steps at three fixed noise levels. It is intended to move boundary mass back toward feasible policy behaviors. Its finite output distribution is measured against rejection; it is not assumed equal to q. A behavioral win is different from a distributional-accuracy result.

## BayesFP

[BayesFP](https://arxiv.org/abs/2606.21014) uses cost-guided denoising particles, Feynman–Kac weights and resampling to approximate a posterior proportional to `p(A|s) exp(-lambda J(A))`. We implemented its published DDPM variant because no linked official implementation was located. This is a paper-derived reimplementation, not an author-code benchmark reproduction.

In our comparison J penalizes safety violation only; it contains no task reward. At finite lambda, unsafe outcomes can retain probability. If J is zero throughout C and positive outside, its ideal large-lambda limit connects to hard conditioning. Practical finite samplers still require measurement. We report both **raw BayesFP** and **BayesFP followed by the same exact terminal projection**. The latter is strictly feasible but has a different output law.

## How to read the results

| Arm | Selection or modification | What the comparison tells us |
|---|---|---|
| First-feasible rejection | Retain the first feasible base-policy draw | Does the conditional target preserve useful behavior? |
| Full-horizon projection | Make one base draw as close as possible to feasible | What does ordinary geometric correction change? |
| Finite refinement | Add score/noise updates with repeated exact projection | Does the draft's finite mechanism recover behavior and approximate q? |
| Raw BayesFP | Guide and resample particles using safety cost | What quality and violations does the soft posterior method produce? |
| BayesFP + projection | Apply the identical hard repair to BayesFP output | How competitive is BayesFP when both outputs must be feasible? |

Return comparisons use all requested cases; unsafe outputs and refusals are reported separately. MMD, boundary concentration and behavioral-family occupancy address distribution shape. Timing and score counts address reference/practical cost. None alone establishes universal superiority.

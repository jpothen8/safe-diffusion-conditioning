# Methods and the question each comparison answers

The frozen policy includes its learned weights, 100-step sampler, normalization and physical action clipping. Call its actual output law p_sampler(A|s). Every arm sees the same observation and safety information, generates the same physical chunk duration, then uses the same continuation controller and unchanged native reward.

| Method | Operation | Interpretation |
|---|---|---|
| Conditioning Q | Independently propose chunks and return the first feasible one | Exact conditional target given budget success; never select by reward |
| Projection P | Map the first base draw to its globally nearest feasible chunk | Changes the law by moving excluded mass, often onto a boundary |
| Refinement R | Start from P, add learned-score drift and noise, project after each update | Finite annealed implementation of the draft's mechanism; not assumed equal to Q |
| Raw BayesFP B | Guide and reweight denoising particles using safety cost | Approximate soft cost-tilted target; can violate the hard predicate |
| BayesFP + projection | Apply the identical hard projection to B | Feasible practical baseline with its own output law |

Q is proportional to p_sampler times the feasibility indicator, provided safe support has positive probability. Projection preserves a feasible base sample but modifies an infeasible one. Either can have better native return: preserving the prior's feasible preferences is not reward optimization.

Rejection is the target reference, not the proposed new sampler. We save every proposal, first-accept index, logical sampling cost and refusal. The current code represents refused conditional outputs as missing/NaN, with an explicit validity mask; it never invents a conditional sample. No refusals occurred in the delivered comparisons.

Pendulum uses signed torque boxes over 12 controls/.6 s, whose exact full-horizon Euclidean projection is coordinate clipping. MuJoCo uses the physical [-1,1] action box intersected with a whole-chunk RMS command budget. From a bounded base draw, global projection is radial scaling. For arbitrary internal refinement states, the exact KKT solution is clip(c*A,-1,1), with c found by scalar bisection. The constraint couples the full chunk; it is not a one-step approximation.

The finite refinement uses gamma=.5 at DDPM indices 20,5,0, with 256,256,512 updates. Each level uses alpha_k=.15*sigma²/(1+k/64)^.6. The VP epsilon prediction is converted to a VE score before the update. This tests one finite annealed variant on convex constraints; it does not establish the draft's all-gamma or nonconvex claims.

BayesFP is a paper-derived implementation of [Algorithm 1 and Appendix G.4](https://arxiv.org/html/2606.21014v1), with no official author code found during the recorded search. It uses the native DDPM update, Feynman–Kac weights and systematic resampling. Its safety-only cost contains no task reward. K=32 and strength20 remain fixed in the new HalfCheetah replication. Raw and projected outputs are both reported. This is a controlled comparison of our implementations, not reproduction of the authors' benchmark scores.

For HalfCheetah, all methods execute eight controls/.4 s and use the same SAC expert through 4 s. The .72 RMS constraint applies only to the initial chunk's normalized actuator commands. It is an effort proxy, not mechanical energy or a certificate against all future hazards. HalfCheetah has no native unhealthy termination.

Two comparisons must stay distinct: Q versus P tests target utility; R versus Q/P/B+P tests finite-sampler behavior and accuracy. Return, boundary concentration, action-kernel MMD, safety, failures and cost answer different questions. The latest replication finds a modest R gain without a Q gain and with R farther from Q by the diagnostic, so greater conditional accuracy cannot explain the result.

The theory's ideal demonstrator density, the score-implied law, and the finite clipped sampler are not interchangeable. In particular physical clipping creates boundary mass. The experiments condition the actual sampler and measure newly imposed boundaries separately; they do not prove missing asymptotic results in the supplied draft.

# Additional tasks, BayesFP, and incorporation into the draft

All work reported here actually ran on the target Threadripper 9980X / RTX 5070 Ti workstation. This extends, rather than replaces, the [earlier Pendulum confirmation](../pendulum/RESULTS.md) and [original Push-T pilot](../../RESULTS.md). The new experiments include trained diffusion policies on Walker2d and HalfCheetah, and a paper-derived BayesFP comparison on all three tasks.

**Recommendation:** use Pendulum as the primary motivating example for conditioning; include BayesFP as a competitive baseline and the mixed locomotion results in the main evidence. The specifically named next check before a stronger locomotion claim is **independent-model replication of HalfCheetah at RMS cap 0.72**. The current finite refinement beats repaired BayesFP there, but the conditional target itself does not beat ordinary projection.

## What the methods mean

- **Conditioning (Q):** return the first feasible independent sample from the frozen policy. This is the exact conditional reference given successful rejection, with no reward ranking.
- **Projection (P):** take the first base sample and solve the global nearest-feasible problem over the entire chunk. It can move excluded probability mass onto a boundary.
- **Refinement (R):** the draft's score/noise/projection mechanism, implemented as a finite annealed gamma=.5 variant. Its output is measured against Q; equality is not assumed.
- **BayesFP (B):** a safety-cost-guided particle sampler reimplemented from the [published DDPM algorithm](https://arxiv.org/abs/2606.21014). Raw B can violate the hard constraint. **B+P** applies the same exact projection and is the relevant hard-feasible competitor. No task reward enters its cost.

See [METHODS_EXPLAINED.md](https://github.com/jpothen8/safe-diffusion-conditioning/blob/pre-consolidation/round3/METHODS_EXPLAINED.md) and the pinned [implementation](https://github.com/jpothen8/safe-diffusion-conditioning/blob/pre-consolidation/round3/src/bayesfp.py). No official BayesFP repository was located in the paper links or recorded repository search. These are our controlled experiments using a paper-derived implementation, not reproduced author benchmark scores.

## Measured comparisons

Native return differences, higher favoring the first method. B uses 32 particles. Brackets show state-cluster bootstrap intervals; their levels differ according to the locked protocols below.

| Setting | Q − P | R − (B+P) | Raw B unsafe |
|---|---:|---:|---:|
| Pendulum, negative torque | +56.37 [51.11, 61.54] | +0.38 [−0.99, 2.06] | 71/512 |
| Pendulum, positive torque | +47.77 [43.08, 52.37] | −1.20 [−2.44, 0.21] | 92/512 |
| Walker2d, RMS .72 | −17.57 [−29.68, −5.94] | +21.35 [11.06, 31.82] | 74/128 |
| Walker2d, RMS .75 | −6.24 [−12.45, −0.41] | −4.09 [−95.29, 32.03] | 25/128 |
| HalfCheetah, RMS .70 | +1.19 [−30.11, 35.40] | +14.01 [−21.90, 49.63] | 49/128 |
| HalfCheetah, RMS .72 | −0.73 [−8.99, 9.38] | +30.61 [5.44, 57.42] | 23/128 |

Pendulum Q−P intervals are descriptive 95%; R−(B+P) intervals are 98.75%, adjusting four comparisons including the 12-particle arm. MuJoCo intervals are 99.375%, adjusting eight primary contrasts across the four task/cap settings. These are bootstrap approximations, conditional on frozen models. They do not account for training-seed variation or turn the exploratory task search into a universal confirmatory study. [Generated table and CSV](../TABLE.md), [return figure](return_comparisons.pdf), [full JSON table](../table.json).

**Pendulum:** R improves over P by 57.04 and 47.65 points, recovering the earlier positive finding on new states. Projection touches the newly imposed zero-torque boundary in 64.6%/65.4% of outputs; Q and R do not. Q/R preserve coherent swing motion, while P often suppresses it. All 7,168 evaluated outputs eventually stabilize, including unsafe raw B outputs; the gain concerns continuous return and timing. B+P achieves comparable return. With 12 particles, R−(B+P) is +0.50 [−0.84,2.04] / −1.71 [−3.14,−0.20], still no useful R advantage under the five-point rule. The latter is a small advantage for BayesFP.

**Walker2d:** Q loses slightly to P at both caps. R's 21.35-point gain over B+P at .72 is below the fixed 168.94-point useful-effect threshold. At .75, one R output terminates unhealthy on control 13 (0.104 s), immediately after the 12-control constrained chunk; this outcome remains in the return calculation. All other evaluated Walker arms have zero native unhealthy terminations. Effort feasibility is not a guarantee against falling.

**HalfCheetah:** at .72, R gains 30.61 points over B+P, above the fixed 26.19-point mean-effect threshold, with positive adjusted interval. The interval also admits gains smaller than that threshold. R−P is only +14.34 [−8.50,39.15], so superiority over ordinary projection is not established. Q−P is inconclusive. This is a positive finite-sampler result, not evidence that more accurate truncation caused the gain. HalfCheetah does not terminate for unhealthy posture, so its zero native terminations cannot establish stability.

Across new comparisons there are 9,728 method-output rollouts: 7,168 Pendulum and 2,560 MuJoCo. Q, P, R and repaired B produce **7,168 hard-feasible outputs with zero chunk violations**. There are zero rejection budget failures. Raw B violations total 319/2,048 on Pendulum (both particle counts) and 171/512 on MuJoCo. Raw outputs are executed and retained in this simulation; no unsafe or unsuccessful outputs were selected away. One Walker R fall occurs among the hard-feasible outputs. These counts exclude base-policy diagnostics, development, and duplicate reproduction runs.

## Distribution diagnostics and cost

Refinement reduces boundary concentration without necessarily reproducing Q. On Pendulum, action-kernel MMD to Q is .251/.236 for R, versus .127/.171 for B+P and .622/.679 for P. On HalfCheetah at .72, it is .157 for R, .119 for B+P and .077 for P; independent Q-versus-Q discrepancy is .113. The apparent P advantage there is within a noisy finite-sample diagnostic. R is not proven closer to Q. The pooled action kernel is not a complete test of every state-conditional distribution. [Boundary/MMD figure](distribution_diagnostics.pdf).

Pendulum predominantly retains one useful safe swing direction under each sign. The two signs are mirrored cases within one task. MuJoCo nearest-expert-template occupancy is recorded, but distances are substantial and those labels do not prove coherent distinct diffusion modes. Preservation of relative weights among multiple surviving safe modes remains unverified.

R uses 1,024 score updates plus its 100-step base sample. B uses 3,200 scores for K=32, or 1,200 for K=12. Recorded batch R time is about 1.41–1.43 s for 512 Pendulum outputs; B32 takes .78–.87 s and B12 .71–.72 s. For 128 HalfCheetah outputs, R takes 5.89–6.71 s and B32 .43–.60 s. R timings exclude the already shared base draw; B timings include its full sampling. Even with that accounting, no efficiency advantage for R appears. These are batch timings on a shared GPU, not isolated per-control online latency. Exact rejection has healthy acceptance and is a competitive practical option in these particular cases; all pre-generated bank cost and logical first-accept cost are separately recorded.

Acceptance is 33.3%/34.2% for Pendulum; MuJoCo state-wise and aggregate values, logical proposals, all pre-generated proposals and particle lineages are in each comparison summary. No same-compute restriction was imposed on the reference. Original comparison runtimes including proposals, simulation and saved diagnostics were 18.93 s Pendulum, 71.06 s Walker2d and 26.13 s HalfCheetah.

## New policy acquisition and gates

We downloaded modern [Walker2d expert](https://huggingface.co/farama-minari/Walker2d-v5-SAC-expert), [Walker2d medium](https://huggingface.co/farama-minari/Walker2d-v5-SAC-medium), [HalfCheetah expert](https://huggingface.co/farama-minari/HalfCheetah-v5-SAC-expert), and [HalfCheetah medium](https://huggingface.co/farama-minari/HalfCheetah-v5-SAC-medium) SAC archives. These are RL demonstrators, not pretrained diffusion policies. Explicit revision URLs, SHA256s, original serialized metadata and missing license declarations are in [records/assets.json](../../records/sources/assets.json). No existing checkpoint or local demonstration dataset was assumed. A MountainCar archive was acquired but **not executed**; it supplies no result.

Untouched 1,000-control smoke returns were approximately 6,933–6,955 / 6,237–6,252 for Walker expert/medium and 11,336–11,535 / 3,756–4,089 for HalfCheetah expert/medium. Shared-state controller probes then established useful alternatives before training. These separate feedback controllers establish feasibility, not multimodality of the learned chunk distribution.

Each new task used 128 complete 1,000-control episodes, 64 paired groups split 48/8/8 before chunking, with training-only observation normalization and identity physical action normalization. The first fixed 16,000-step fit failed the quality gate: Walker retained about 30% of expert first-chunk/common-continuation return with 32.8% survival, and HalfCheetah retained about 74.5% of expert return. A single bounded 48,000-step refit increased early-state training and shortened Walker's chunk from .4 s to .096 s. Final development return was 3,326.42 versus 3,378.74 expert for Walker and 448.52 versus 523.76 for HalfCheetah. The final models were then frozen. Original data, initial weights, failed gates and refits are retained.

The paired comparison is deliberately an initial-chunk perturbation study with an RL continuation controller. It does **not** establish robust multi-chunk closed-loop locomotion by these diffusion models. Constraints bound normalized actuator-command RMS over the full initial chunk, not real mechanical energy or all future hazards. Native rewards were never rewritten to favor conditioning.

## Verification and reproducibility

[audit.json](../audit.json) validates all first-feasible selections, second independent conditional diagnostics, exact raw-sample projections, output constraints, native-return summaries, bootstrap intervals, 29 locked input references and 13 downloaded asset hashes. It independently replays two complete demonstration episodes per new task with zero observation/action-time or reward discrepancy, and checks exact training-only normalizers and disjoint group splits. Saved full MuJoCo snapshots replay exactly. Pendulum replay matches native Gym to at most 8.9e-16 in the fresh comparison.

The intersection projector was independently checked against 36 SOCPs: maximum solution RMS discrepancy 7.9e-6 and float32 constraint excess 8.4e-8, below the 1e-6 feasibility tolerance. One conic reference solve reported `optimal_inaccurate`; this is retained in its log. The actual baseline is analytic/global and has no iterative solver failures. Zero-guidance BayesFP matched the original DDPM bitwise in the development check; cost gradients were checked against autograd. These implementation checks do not prove posterior accuracy.

The additional [adapter audit](../adapter_checks.json) confirms bitwise zero-guidance equivalence for both new MuJoCo models after applying the identical physical postprocessing already used in evaluation. An initial check omitted that final clip and detected a 1.4e-5 DDPM scheduler overshoot; its log is retained. Applying the actual policy's documented clipping resolves the mismatch without changing any experiment. Both safety-cost gradients agree with autograd within 7.5e-9, and uniform/one-hot resampling checks pass. The [reproduction audit](../../records/unified_reproduction.json) checks all arrays in 17 HalfCheetah NPZ files and the full snapshots for exact equality.

The dependency overlay uses Gymnasium 1.2.0, MuJoCo 3.3.5, SB3 2.7.0 and the preserved parent PyTorch 2.7.1+cu128 stack; the inspected card is compute capability 12.0 and driver 595.84. See [requirements.lock.txt](https://github.com/jpothen8/safe-diffusion-conditioning/blob/pre-consolidation/round3/requirements.lock.txt), [parent host record](../../TASK_SELECTION.md#provenance-and-hardware), and [README.md](https://github.com/jpothen8/safe-diffusion-conditioning/blob/pre-consolidation/round3/README.md). Numerical outcomes of the fresh-directory HalfCheetah command reproduction match the original; it is not a new independent sample. No physical hardware was operated and no results were published.

The draft-specific deliverables are [PAPER_INTEGRATION.md](https://github.com/jpothen8/safe-diffusion-conditioning/blob/pre-consolidation/round3/PAPER_INTEGRATION.md), [paper_experiments.tex](../../docs/paper/pilot_experiments.tex) and automatically generated [paper_results.tex](../../docs/paper/pilot_results.tex). They separate the supported empirical claims from the draft's unresolved theoretical proofs.

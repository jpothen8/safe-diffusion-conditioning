# Independent-model HalfCheetah replication

**The preregistered useful-advantage criterion was NOT met.** Refinement minus BayesFP+projection is **+16.88 [+6.06, +28.29]** native-return points (95% crossed model/state bootstrap). The model-level t interval is **+16.88 [+8.83, +24.94]**. Its mean is below the unchanged **26.19**-point useful-effect threshold.

All **8 independently trained models**, **64 fresh shared simulator states**, and **4 requests/state/model** are included. Refinement beats repaired BayesFP in **7/8 model means**. This replicates training randomness on one fixed demonstration dataset; it does not replicate demonstration collection or establish universal superiority.

| Training seed | Base | Q | P | R | B+P | R − (B+P) |
|---|---:|---:|---:|---:|---:|---:|
| 120100 | 454.47 | 450.67 | 450.94 | 452.52 | 428.59 | 23.93 |
| 120101 | 444.38 | 448.08 | 448.08 | 466.85 | 444.71 | 22.14 |
| 120102 | 439.07 | 445.24 | 440.64 | 454.68 | 439.27 | 15.41 |
| 120103 | 442.75 | 446.96 | 444.59 | 461.47 | 437.63 | 23.84 |
| 120104 | 431.42 | 435.69 | 430.71 | 460.88 | 445.19 | 15.69 |
| 120105 | 438.90 | 435.28 | 436.98 | 464.82 | 438.81 | 26.01 |
| 120106 | 432.56 | 433.59 | 436.86 | 448.26 | 451.52 | -3.26 |
| 120107 | 438.52 | 443.07 | 441.94 | 458.24 | 446.93 | 11.31 |

Q is first-feasible rejection; P is exact global full-chunk projection; R is the fixed finite annealed gamma=.5 refinement; B+P is paper-derived BayesFP K=32/strength20 followed by the identical projection. Every arm executes .4 s and shares the same public expert continuation through 4 s. The RMS .72 command limit applies to the initial chunk.

## Secondary descriptive comparisons

| Contrast | Mean [crossed 95% interval] | Model t 95% interval |
|---|---:|---:|
| Conditioning − projection | +0.98 [-2.70, +4.88] | +0.98 [-1.44, +3.40] |
| Refinement − projection | +17.12 [+6.19, +28.18] | +17.12 [+9.56, +24.68] |

These secondary intervals are descriptive, without a new family of superiority claims. A refinement gain over BayesFP does not establish accurate conditional sampling or a conditioning-target advantage.

## All requested outcomes

| Arm | Mean return | Unsafe/requested | Native terminations | Nonpositive-return rollouts | Boundary fraction | MMD to Q |
|---|---:|---:|---:|---:|---:|---:|
| conditional | 442.32 | 0/2048 | 0 | 52 | 0.000 | 0.000 |
| projection | 441.34 | 0/2048 | 0 | 52 | 0.169 | 0.047 |
| refinement | 458.46 | 0/2048 | 0 | 40 | 0.041 | 0.145 |
| bayesfp | 440.82 | 320/2048 | 0 | 45 | 0.000 | 0.085 |
| bayesfp_projected | 441.58 | 0/2048 | 0 | 42 | 0.156 | 0.085 |

Rejection refusals: **0**. Hard-output violations: **0**. All 10,240 method requests are retained. HalfCheetah has no native unhealthy termination; zero terminations does not certify stable posture. Nonpositive-return counts retain visibly poor trajectories without inventing a new reward.

The mean independent Q-versus-Q MMD is 0.080. These pooled action-kernel diagnostics are not proof of state-conditional equality or multiple surviving coherent modes. The base models retain 82.5–87.0% of the fresh-bank expert return (522.63), using one generated chunk and common continuation. No model was excluded or refitted.

## Cost and provenance

| Item | Mean seconds/model |
|---|---:|
| base_proposal_bank | 1.201 |
| projection | 0.000 |
| refinement | 1.171 |
| bayesfp | 0.155 |

Each bank contains 32,768 proposals (128/request), all saved. Logical first-accept costs are saved separately. R uses 1,024 refinement scores plus its shared 100-step base draw; B uses 3,200 scores. Timing is batched on the recorded workstation, not an isolated online latency benchmark. Sampling settings were not retuned on these new models or states.

[Locked protocol](../../docs/protocols/halfcheetah_replication.md) · [Config/code/asset hashes](lock.json) · [All-model summary](summary.json) · [CSV](table.csv) · [Audit](audit.json) · [Figure](model_comparison.pdf)

The consolidation first matched the historical base sampler, refinement, BayesFP particles and native rollouts bit for bit. The fresh public-controller collection reproduced the original complete-episode archive byte for byte. These checks protect implementation fidelity; the eight new training seeds provide the independent-model evidence.

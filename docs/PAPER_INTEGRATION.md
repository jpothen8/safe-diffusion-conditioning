# Current paper integration

Use [the original experimental section](paper/pilot_experiments.tex) and [tables](paper/pilot_results.tex), then append the generated [independent-model replication subsection](../results/halfcheetah_replication/paper_replication.tex). The independent-model replication is complete; use the current results below when revising claims.

The new subsection should immediately follow the original HalfCheetah result. Report both the original +30.61 one-model estimate and the independent-model estimate **+16.88**, 95% model/state interval **[6.06,28.29]**. State explicitly that the mean did not meet the unchanged **26.19** useful-effect threshold. Seven of eight model means favor refinement; one favors repaired BayesFP. Do not describe this as replication of the stronger useful-effect claim.

The native-return improvement over ordinary projection is **+17.12 [6.19,28.18]**, a secondary descriptive result. Conditioning versus projection is **+0.98 [−2.70,4.88]**. Thus the paper still has one clear task supporting the conditioning motivation—Pendulum—and modest locomotion evidence for the finite sampler. Refinement's larger MMD to rejection means the locomotion gain cannot be attributed to more accurate hard conditioning.

Suggested replacement discussion paragraph:

> Across eight independently trained HalfCheetah policies, finite refinement improved native return over repaired BayesFP by 16.88 points, with a 95% crossed model/state bootstrap interval of [6.06,28.29]. This effect was smaller than the original pilot estimate and below the predefined useful-effect threshold. Conditioning itself remained indistinguishable from projection, and distributional diagnostics did not show that refinement was closer to the conditional reference. These results support a modest setting-specific finite-sampler benefit while limiting claims about the practical value of exact truncation in locomotion.

The scope is independent initialization/optimization seeds on a fixed demonstration dataset, not independent demonstration collection. Include all 10,240 requested outputs, the 320/2,048 raw BayesFP violations, zero hard-output violations/refusals, poor-return outcomes and batch timing. HalfCheetah has no native unhealthy termination, so avoid equating zero terminations with stability.

Keep the draft's proof TODOs separate. Simulations on convex constraints at gamma=.5 do not establish the theorem for every relaxation or nonconvex set. Distinguish the actual finite clipped policy law from the ideal demonstrator and score-implied laws. Neither the positive Pendulum result nor the new replication proves exact sampling or improved online efficiency.

The [per-model figure](../results/halfcheetah_replication/model_comparison.pdf) shows every model and the aggregate interval. It should accompany the text, rather than showing only the favorable seed. A [compiled combined preview](paper_preview/experiments.pdf) includes the original experimental section and the new subsection; its old one-model estimates are explicitly identified as the pilot.

## Separate correctness of projection from correctness of its set

Append the generated [safe-set mismatch subsection](../results/safe_set_mismatch/paper_mismatch.tex) and [sensitivity figure](../results/safe_set_mismatch/boundary_sensitivity.pdf). The projection baseline is already the exact global nearest point over the whole chunk. The new analysis changes the true predicate used in evaluation while preserving the estimated set supplied to the methods. Label it exploratory and retrospective; the eight policies and original trajectories are reused.

The mechanism is narrower than universal performance superiority: projecting infeasible mass onto an estimated boundary can make a small inward boundary error unsafe for a finite fraction of outputs. Conditioning has no added boundary atom, but can still place substantial probability in the incorrectly included strip. A finite sampler can retain both boundary and near-boundary mass. The [mechanism note](PROJECTION_AND_SET_ERROR.md) gives the assumptions and equations.

Report the full error curve, including the reversal for finite refinement. At .1% error Q/P/R/B+P have .88/17.53/6.79/16.11% violations. At 1% they have 8.20/22.75/24.46/21.04%. The R−P difference at 1% is inconclusive; exact Q has fewer violations across all eight model means. These normalized effort-budget excesses are small and must not be called observed collisions or physical harm.

Include conservative projection prominently. A fixed 1% margin gives exact oracle projection when the true cap is 1% lower. It eliminates violations up to that error with no detectable native-return loss; conditioning on the same smaller set has no detectable return advantage. This additional error bound is supplied equally to both controls. It would be misleading to compare a method using the true set with a competitor restricted to an inaccurate set without identifying this information advantage.

Revise any introduction claim that projection always harms useful behavior. The evidence instead supports two distinct motivations: behavior preservation in Pendulum with an accurate set, and sensitivity to boundary error in the HalfCheetah cap probe. Whether boundary error causes most practical harm remains a question for a simulator-based velocity, posture or clearance constraint with an independently calibrated estimator. The theoretical draft's empirical-measure result also does not guarantee that a finite chain's final iterate inherits exact conditioning's robustness; the observed refinement curve illustrates this distinction.

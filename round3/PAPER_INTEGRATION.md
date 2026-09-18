# Incorporating the experiments into the supplied draft

The supplied starter repository was inspected at commit `e2b2546d948c04f63ae7232933c768956872a94f`. It provides RL demonstrator acquisition/evaluation. The diffusion training, conditional references, full-horizon projections and finite refinement used here were implemented separately and are fully recorded. Do not imply these experiments were already provided by the starter code.

Use two experimental questions rather than combining motivation and algorithm accuracy:

1. **Target utility:** first-feasible rejection versus exact projection of the same frozen policy. This isolates whether replacing the policy distribution by its feasible conditional law can improve native task quality.
2. **Finite sampler performance:** refinement versus rejection, projection, and BayesFP. This measures quality, feasibility, approximation error and cost. A positive answer to question 1 does not establish a positive answer to every comparison in question 2.

The Pendulum mechanism figure belongs immediately after the target definition or at the start of Experiments. Show the same starting state, an excluded coherent swing direction being projected toward zero torque, and conditioning/refinement retaining useful motion. Accompany it with the aggregate held-out table; do not rely on a selected animation alone. Describe the two signed constraints as mirrored cases within one task.

Place the independent negative Push-T result and new MuJoCo outcomes in the main cross-task table or an equally visible ablation, not only among omitted runs. Report failed first diffusion fits in the reproducibility appendix and explain the bounded quality repair before the final model was frozen. A poor base policy cannot support a strong safety-filter comparison.

The new BayesFP result must temper the method claim: on Pendulum, current finite refinement has no useful return advantage over BayesFP with final projection. Raw BayesFP is not always feasible, but its repaired arm is; do not compare only against its unsafe outputs. BayesFP is faster in the recorded batch test and can be closer to rejection by the chosen action-kernel diagnostic. Label the adapter as a paper-derived reimplementation and publish its exact equations, cost, temperature, particle counts and validation limits alongside results.

Change the empirical notation to distinguish `p_sampler` (the actual frozen finite DDPM including clipping) from the ideal demonstrator density `p0` and the score-implied density. Rejection conditions `p_sampler`. The theory's smooth-density and zero-boundary-mass assumptions are not literally satisfied by an action-clipped sampler. Measure the newly imposed safety boundary separately from pre-existing physical saturation.

Describe the implemented refinement as a finite **annealed gamma=1/2 variant**, with the exact 256/256/512 step schedule and VP-to-VE conversion. It is not evidence for all gamma, all nonconvex sets, or endpoint convergence under the draft's empirical-measure theorem. Leave theorem/proof TODOs unresolved unless independently proved; simulation cannot fill those gaps.

The supported contribution is currently a controlled empirical study showing when conditioning avoids harmful projection concentration, plus a finite sampler that can recover useful behavior. Do not yet claim superior finite efficiency, exact conditional sampling, preservation of multiple surviving safe modes, or universal reward optimality. Treat the BayesFP comparison as a substantive related-work and empirical baseline, not merely a citation.

The accompanying [paper_results.tex](paper_results.tex) and [paper_experiments.tex](paper_experiments.tex) contain completed measured runs, with protocols, failure counts and limitations. No BayesFP author-reported manipulation scores are mixed into our tables, because policies, environments and safety predicates differ.

## Concrete placement in this draft

Review the compiled [experimental-section preview](paper_preview/paper_preview.pdf), containing the measured tables and two exportable figures. It is a standalone integration preview, not a compiled copy of the full ICLR draft.

1. **Introduction/contributions:** add an empirical contribution phrased as “We isolate target utility from finite-sampler performance in controlled simulation, finding useful conditioning gains on Pendulum and task-dependent outcomes on manipulation and locomotion.” Avoid a claim that all per-step filters preserve behavior less well: the BayesFP comparison does not support it.
2. **Problem formulation:** retain the ideal `p0` for theory, introduce `p_sampler` for experiments, and state that rejection conditions the actual sampler. The learned score, the demonstrator density and the finite clipped sampler are three different objects. Refinement is not automatically targeting the last of them.
3. **Practical method:** add the finite annealed schedule, VP-to-VE conversion, exact box/ball projections and a pointer to approximation diagnostics. Do not describe this as the literal single-final-level algorithm or an evaluation of every gamma.
4. **Experiments:** replace the commented-out Experiments placeholder with `\input{paper_experiments.tex}`. Put `paper_results.tex` beside it, or adjust the relative `\input` path. The tables require the draft's existing `booktabs` package. Use [return_comparisons.pdf](figures/return_comparisons.pdf) for the cross-task comparison, [distribution_diagnostics.pdf](figures/distribution_diagnostics.pdf) for boundary/MMD diagnostics, and the existing [Pendulum animation](../results/pendulum/confirmation/examples.gif) as supplementary illustration.
5. **Discussion/limitations:** state that the new HalfCheetah cap .72 result favors refinement over repaired BayesFP by **30.61** points, adjusted CI **[5.44,57.42]**, but exact conditioning does not beat projection there. The point estimate passes the fixed 26.19-point useful-effect rule; its interval does not establish that the true gain exceeds that threshold. Refinement is farther from rejection by the recorded MMD, so attributing this gain to greater conditional accuracy would be unsupported.
6. **Appendix/reproducibility:** retain failed initial fits, development tuning, all raw BayesFP violations, the Walker2d fall, proposal costs and locks. HalfCheetah has no native unhealthy termination; zero native terminations is not a general stability guarantee. Report that intervals reflect state/sampling uncertainty for one frozen model per task, not training-seed uncertainty.

## Claim ledger

| Claim | Current evidence | Appropriate wording |
|---|---|---|
| Truncation can preserve useful behavior better than projection | Replicated Pendulum results; full-horizon global projection; unchanged native reward | Supported motivating example |
| Conditioning always improves quality | Walker2d conditioning loses slightly; HalfCheetah is inconclusive; Push-T null/negative | Reject this claim |
| The proposed finite refinement can recover the motivating gain | Pendulum refinement gains about 48–57 return points | Supported for this implementation and protocol |
| Refinement beats BayesFP in general | No useful Pendulum advantage; one useful-mean HalfCheetah setting; Walker mixed | Report setting-specific results |
| Refinement accurately samples q | Measurable MMD gaps; sometimes BayesFP/ordinary projection is closer | Not established |
| Multiple surviving safe modes keep their relative weights | Pendulum predominantly retains one swing direction; MuJoCo template labels are only diagnostics | Still missing |
| Practical online efficiency improves | BayesFP is faster in these batch measurements; exact rejection acceptance is healthy | Not established |
| The draft's theorem and nonconvex extension are proved | Draft retains explicit proof TODOs; experiments only use convex sets and gamma=.5 | Must be resolved mathematically, independently of these results |

Suggested abstract sentence: “Controlled simulations show that conditioning can avoid harmful boundary concentration and improve native task quality, while broader comparisons reveal that this benefit is task dependent and that finite refinement remains distinct from exact conditional sampling.” Keep the HalfCheetah algorithm result separate from that motivation.

The next specifically named scientific check is **independent-model replication of the HalfCheetah RMS-0.72 comparison**: fix this task, cap, methods and analysis before training new seeds; evaluate the same arm set on new state seeds and report training-level uncertainty. Re-running identical arrays verifies code execution but does not supply that evidence. For a claim about multimodal truncation itself, a distinct additional requirement remains: demonstrate at least two coherent useful modes surviving the same constraint from shared states.

# Evidence and provenance

The root [RESULTS.md](../RESULTS.md) consolidates the conclusions. Per-experiment records retain all positive, negative and inconclusive findings.

| Experiment | Protocol | Numerical results |
|---|---|---|
| Released Push-T policy | [FIRST_EXPERIMENT.md](../FIRST_EXPERIMENT.md) | [Report](../results/heldout/RESULTS.md), [summary](../results/heldout/summary.json) |
| Pendulum confirmation | [Protocol](protocols/pendulum_confirmation.md) | [Report](../results/pendulum/RESULTS.md), [summary](../results/pendulum/confirmation/summary.json) |
| Fresh Pendulum / BayesFP | [Protocol](protocols/pendulum_bayesfp.md) | [Summary](../results/pendulum_comparison/summary.json) |
| Walker2d and HalfCheetah pilots | [Protocol](protocols/locomotion_pilot.md) | [Walker](../results/Walker2d_comparison/summary.json), [HalfCheetah](../results/HalfCheetah_comparison/summary.json) |
| Eight-model HalfCheetah replication | [Protocol](protocols/halfcheetah_replication.md) | [Report](../results/halfcheetah_replication/RESULTS.md), [audit](../results/halfcheetah_replication/audit.json) |
| Supplied-safe-set error | [Exploratory protocol](protocols/safe_set_mismatch.md) | [Report](../results/safe_set_mismatch/RESULTS.md), [audit](../results/safe_set_mismatch/audit.json) |

Public external weights are described in [assets.json](../records/assets.json); they are downloaded with hashes, never bundled in the repository or research release. The prepared, currently unpublished archive contains generated episodes, our trained diffusion weights and raw arrays for the reported experiments. The [reproduction guide](REPRODUCTION.md) explains restoration and regeneration.

[requirements.lock.txt](../requirements.lock.txt) is the single tested environment. [path_migration.json](../records/path_migration.json) resolves old artifact names. The `pre-consolidation` Git tag contains exact old scientific source/protocol bytes. No original proposal or failed outcome was dropped by moving files; duplicate reruns and dependency caches are not publication assets.

The [combined paper preview](paper_preview/experiments.pdf) contains the original pilot, all-model replication and exploratory boundary-error result. Its [integration guide](PAPER_INTEGRATION.md) states the allowed claims and limitations.

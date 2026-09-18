# Artifact index

| Study | Protocol | Results | Implementation status |
|---|---|---|---|
| Original released Push-T pilot | [FIRST_EXPERIMENT.md](../FIRST_EXPERIMENT.md) | [Retained report](history/2026-09-14_RESULTS.md), `results/heldout/` | Frozen root `src/` drivers |
| Pendulum target/refinement confirmation | [PENDULUM_CONFIRMATION.md](../PENDULUM_CONFIRMATION.md) | [ROUND2_RESULTS.md](../ROUND2_RESULTS.md), `results/pendulum/confirmation/` | Frozen root `src/pendulum*.py` |
| Fresh Pendulum versus BayesFP | [Round3 Pendulum protocol](../round3/PENDULUM_PROTOCOL.md) | [Round3 report](../round3/RESULTS.md), `round3/results/pendulum_comparison/` | Frozen `round3/src/` |
| Walker2d/HalfCheetah pilot | [Round3 locomotion protocol](../round3/LOCOMOTION_PROTOCOL.md) | [Round3 report](../round3/RESULTS.md), `round3/results/*_comparison/` | Frozen `round3/src/` |
| Eight-model HalfCheetah replication | [New protocol](HALFCHEETAH_REPLICATION.md) | [Full report](../results/halfcheetah_replication/RESULTS.md), [audit](../results/halfcheetah_replication/audit.json) | Maintained `safety_conditioning/` package |
| Safe-set error sensitivity | [Fixed exploratory grid](SAFE_SET_MISMATCH.md) | [Report](../results/safe_set_mismatch/RESULTS.md), [audit](../results/safe_set_mismatch/audit.json), [figure](../results/safe_set_mismatch/boundary_sensitivity.pdf) | `mismatch.py`, `mismatch_diagnostics.py`; original actions plus conservative-control rollouts |

The [current RESULTS.md](../RESULTS.md) is the evidence overview. Earlier positive, negative, failed-fit and precision-correction records remain intact. The original protocol locks still point to their original paths; nothing was silently moved or overwritten to make the refactor pass.

Public assets and environment:

- [Original asset manifest](../records/asset_manifest.json), [Pendulum assets](../records/followup_assets.json), [modern MuJoCo/BayesFP sources](../round3/records/assets.json).
- [Original host inspection](../records/host.json), [replication host inspection](../records/replication_host.json).
- [Parent dependency lock](../requirements.lock.txt), [active overlay lock](../round3/requirements.lock.txt), [package metadata](../pyproject.toml).
- Checkpoint license declarations missing from model metadata remain recorded as missing. Public access does not imply permission to redistribute weights.

New verification:

- [Numerical refactor compatibility](../records/consolidation_checks.json): base DDPM, refinement, BayesFP particles and native rollouts match frozen code bit for bit on both locomotion tasks.
- [Collection reproduction](../results/collection_reproduction/audit.json): the complete 128-episode HalfCheetah archive matches the original bytes.
- [Eight-model audit](../results/halfcheetah_replication/audit.json): source/asset/checkpoint hashes, first-feasible selection, global projection, normalizers, bootstrap and native replay.
- [Consolidation record](CONSOLIDATION.md) explains the package layout and historical compatibility.

The generated [paper replication text](../results/halfcheetah_replication/paper_replication.tex), [CSV](../results/halfcheetah_replication/table.csv) and [figure](../results/halfcheetah_replication/model_comparison.pdf) come from saved numerical results. The [paper integration guide](PAPER_INTEGRATION.md) describes their placement alongside the original experimental section.

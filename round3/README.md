# New tasks and BayesFP comparison

Start with [RESULTS.md](RESULTS.md), [METHODS_EXPLAINED.md](METHODS_EXPLAINED.md), and [PAPER_INTEGRATION.md](PAPER_INTEGRATION.md). Tables and exportable figures are generated from the completed, locked comparisons. The paper text is in [paper_experiments.tex](paper_experiments.tex); it inputs [paper_results.tex](paper_results.tex). A compiled standalone [PDF preview](paper_preview/paper_preview.pdf) includes the experimental text, tables and figures; this does not compile the supplied full draft or its unfinished proofs.

## Reproduce in this project

Commands below run from the parent `safety_conditioning_pilot` directory. The delivered models are locally trained diffusion policies; public SAC checkpoints supply demonstrations. Keep each reproduction output directory fresh.

```bash
rtk proxy bash scripts/setup_pendulum.sh
rtk proxy bash round3/scripts/setup.sh
rtk proxy round3/.venv/bin/python round3/scripts/fetch_assets.py
rtk proxy round3/.venv/bin/python round3/src/audit.py
rtk proxy round3/.venv/bin/python round3/src/build_report.py
rtk proxy round3/.venv/bin/python round3/scripts/replicate.py HalfCheetah --output-root results/my_halfcheetah_replication
rtk proxy round3/.venv/bin/python round3/scripts/replicate.py Walker2d --output-root results/my_walker_replication
rtk proxy round3/.venv/bin/python round3/scripts/replicate.py Pendulum --output-root results/my_bayes_pendulum_replication
```

The HalfCheetah wrapper was executed end to end in `round3/results/command_reproduction`. These commands retain seeds and repeat the recorded experiment; they do not create new statistical evidence. The wrappers use verified original sources/assets and write only to fresh output directories. Under the managed agent sandbox, downloads and CUDA runs require authorized host execution. The target's existing driver supports the pinned CUDA 12.8 PyTorch runtime; setup does not change the driver.

The parent setup recreates the original isolated stack. The round3 venv is an explicit overlay, adding the parent's site directory with `site.addsitedir` and installing the complete recorded dependency pins. Its absolute path is generated during setup, not assumed portable as an existing venv. `fetch_assets.py` restores pinned checkpoint URLs and verifies hashes. `acquire.py` records the original discovery workflow; use `fetch_assets.py` for reproduction, since discovery queries can change.

## Recreate the new diffusion models from public assets

```bash
rtk proxy round3/.venv/bin/python round3/scripts/train_recorded.py Walker2d --output-root results/my_walker_training
rtk proxy round3/.venv/bin/python round3/scripts/train_recorded.py HalfCheetah --output-root results/my_halfcheetah_training
```

These commands collect complete expert/medium episodes, reproduce the original group split, fit the initial model, and execute the bounded refit. Each saves all episodes, both models and training summaries under its fresh directory. Original collection/training/refit drivers all ran; the additional fresh-directory training wrapper is provided but was not itself executed in this round. No training rerun is needed to inspect or repeat the frozen comparison. Do not overwrite the delivered weights with a new training run and call it the same frozen experiment.

Useful entry points:

- [PENDULUM_PROTOCOL.md](PENDULUM_PROTOCOL.md), [LOCOMOTION_PROTOCOL.md](LOCOMOTION_PROTOCOL.md), and `configs/`: frozen hypotheses, controls, seeds, sample sizes and decision rules.
- `src/bayesfp.py`: the paper-derived DDPM/Feynman–Kac particle adapter, safety costs and diagnostics.
- `src/effort.py`: exact full-chunk box/ball projection and finite annealed refinement.
- `src/locomotion_env.py`: full integration-state snapshots and common-controller rollouts.
- `src/locomotion_probe.py`, `locomotion_train.py`, `refit_chunk.py`, `locomotion_quality.py`: original smoke, data collection, bounded fits and quality gates.
- `src/*bayes_dev.py`: all development strengths, retained independently of confirmation.
- `src/pendulum_compare.py`, `locomotion_compare.py`: frozen arm comparisons; refuse to overwrite existing locks.
- `src/audit.py`, `check_effort.py`, `build_report.py`: independent checks and reproducible figures/tables.
- `results/*_comparison/`: every proposal, full snapshots, first-feasible indices, all arm chunks/outcomes, particle clouds, protocol hashes and numerical summaries.
- `records/assets.json`: public URLs, resolved revisions, hashes and license status. Checkpoint licenses were not declared in inspected model metadata; no permission to redistribute weights is inferred from code licenses.

The native MuJoCo reward/physics are untouched. The new safety predicate is an externally supplied initial-chunk effort budget. It does not impose a fall-prevention constraint or certify the continuation. Public environment definitions: [Walker2d](https://gymnasium.farama.org/environments/mujoco/walker2d/), [HalfCheetah](https://gymnasium.farama.org/environments/mujoco/half_cheetah/). BayesFP source: [paper v1](https://arxiv.org/html/2606.21014v1), CC BY 4.0. Source and asset provenance for Pendulum, Push-T and the supplied starter repository remain in the parent project.

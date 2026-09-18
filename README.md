# Safety conditioning experiments

This project compares conditioning a frozen diffusion policy on safety with projecting its samples, and evaluates finite projected-Langevin refinement against a paper-derived BayesFP implementation. All work is in simulation.

**Start here:** [current results](RESULTS.md) · [methods](docs/METHODS.md) · [paper integration](docs/PAPER_INTEGRATION.md) · [artifact index](docs/ARTIFACTS.md).

The independent-model study trained eight HalfCheetah diffusion policies and evaluated 10,240 method requests. Refinement improved return over repaired BayesFP by **16.88 [6.06,28.29]**, but did not reach the fixed **26.19-point useful-effect threshold**. Conditioning versus projection remained inconclusive. Pendulum remains the strongest positive example for the conditional target.

The latest [safe-set error probe](results/safe_set_mismatch/RESULTS.md) distinguishes exact projection onto the supplied set from oracle projection onto the true set. A .1% tighter true effort cap produces violations on **17.53%** of projected outputs versus **.88%** of conditional outputs. Finite refinement reaches **6.79%**, but its advantage disappears at larger errors. A valid 1% projection margin eliminates violations up to that error without detectable return loss. This retrospective probe tests boundary sensitivity; it does not establish physical harm or estimator performance.

## One entry point

Run these from this project directory. `./run` uses the recorded `round3/.venv` environment. The same interface is available as `python -m safety_conditioning` or the installed `safety-conditioning` command.

```bash
rtk proxy ./run status
rtk proxy ./run audit --input results/halfcheetah_replication
rtk proxy ./run report --input results/halfcheetah_replication
rtk proxy ./run study --output results/my_halfcheetah_replication
```

The final command repeats the full eight-model study in a fresh directory. Its [protocol](docs/HALFCHEETAH_REPLICATION.md) and [config](configs/halfcheetah_replication.json) fix all settings and seeds. Reusing those seeds reproduces the experiment; it does not supply new independent evidence. All model seeds are included and existing output directories are protected. The recorded study took about 6.3 minutes on the inspected workstation.

Other supported workflows:

```bash
rtk proxy ./run collect --task HalfCheetah --output results/new_collection/episodes.npz
rtk proxy ./run train --task HalfCheetah --seed 180100 --output results/new_training
rtk proxy ./run check
rtk proxy ./run reproduce pendulum --output results/new_pendulum_reproduction
rtk proxy ./run reproduce halfcheetah --output results/new_original_halfcheetah_reproduction
rtk proxy ./run mismatch --input results/halfcheetah_replication --output results/new_mismatch_probe
rtk proxy ./run audit --input results/new_mismatch_probe
rtk proxy ./run report --input results/new_mismatch_probe
```

`collect` recreates complete public-expert episodes and the original group split. `train` uses the recorded dataset by default; use `--dataset PATH` to choose another explicitly. `check` verifies numerical equivalence to the frozen historical samplers and simulator. `reproduce` dispatches an unchanged historical experiment; choices are `pusht`, `pendulum`, `walker2d`, and `halfcheetah`. Historical Pendulum/locomotion reproduction includes the BayesFP arms.

`mismatch` uses existing proposal banks and outcomes plus 4,096 new conservative-control rollouts on CPU; it does not train new models. `audit` and `report` recognize both study types. The [fixed sensitivity protocol](docs/SAFE_SET_MISMATCH.md) defines the error grid and [mechanism note](docs/PROJECTION_AND_SET_ERROR.md) explains the comparison.

## Setup

Python 3.12 on Linux x86_64 is required for the recorded environment. The inspected host is a Threadripper 9980X with RTX 5070 Ti, driver 595.84, and PyTorch 2.7.1+cu128. No driver change is needed. Downloads and CUDA calls require host execution when launched through the restricted agent sandbox.

```bash
rtk proxy bash scripts/setup_project.sh
```

Setup uses the exact [parent](requirements.lock.txt) and [overlay](round3/requirements.lock.txt) dependency records, installs this package without upgrading dependencies, verifies pinned public checkpoint archives and runs `pip check`. The overlay is retained for compatibility with frozen experiments; it is the single environment used by the active CLI. Existing environments are not rebuilt unnecessarily. A fresh installation of the older Push-T environment needs a compiler for Pymunk 6.2.1.

The demonstration data and newly trained models are included in this workspace. They are not public pretrained diffusion checkpoints. If the HalfCheetah demonstration archive is missing, restore the SAC assets with setup, then run:

```bash
rtk proxy ./run collect --task HalfCheetah --output round3/results/HalfCheetah_training/episodes.npz
```

This command reproduced the archive byte for byte. The study trains its models from that data. Public RL URLs, revisions, hashes and undeclared checkpoint-license status are in [assets.json](round3/records/assets.json). Historical Pendulum/Push-T setup remains in the [archived README](docs/history/2026-09-14_README.md).

## Maintained code

| Module in `safety_conditioning/` | Responsibility |
|---|---|
| `policy.py` | Architecture, DDPM scheduler and complete policy postprocessing |
| `constraints.py`, `samplers.py` | Global projection, first-feasible reference, refinement and BayesFP |
| `collection.py`, `data.py`, `training.py` | Complete episodes, splits, normalization and seeded training |
| `simulator.py`, `evaluation.py` | Full snapshots and matched native-reward rollouts |
| `statistics.py`, `study.py` | Independent-model protocol and uncertainty |
| `mismatch.py`, `mismatch_diagnostics.py` | Safe-set error probe, conservative controls, independent audit and figures |
| `audit.py`, `validation.py`, `reporting.py` | Verification, compatibility and paper artifacts |
| `artifacts.py`, `cli.py` | Paths, provenance, protected outputs and commands |

New work belongs in this package. Root `src/`, historical scripts, and `round3/src/` remain frozen reproduction records. Their scientific files and results are unchanged; old protocol locks reference their paths. [CONSOLIDATION.md](docs/CONSOLIDATION.md) records the cleanup and checks.

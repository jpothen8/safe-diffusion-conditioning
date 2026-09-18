# Reproduction

Run commands from the repository root. Python 3.12, Linux x86_64 and the pinned dependency lock reproduce the tested stack. GPU experiments use CUDA 12.8 wheels compatible with the inspected RTX 5070 Ti. CPU-only checks do not need GPU access. In a restricted agent sandbox, downloads and GPU calls may require host execution.

```bash
bash scripts/setup.sh
./run check
./run fetch-assets --task all
./run fetch-results
```

There is one environment, `.venv`. `fetch-assets` downloads only public author assets and the pinned Push-T source; choices are `locomotion`, `pendulum`, `pusht`, `all`. Existing mismatches are errors, never silently overwritten. `fetch-results` restores the generated research artifacts from the public release with a recorded archive hash. It does not include third-party checkpoint ZIPs or the Push-T checkpoint.

## Reproduce the saved experiments

```bash
./run reproduce pendulum --output results/repeat_pendulum
./run reproduce halfcheetah --output results/repeat_halfcheetah
./run reproduce walker2d --output results/repeat_walker2d
./run reproduce pusht --output results/repeat_pusht
./run study --output results/repeat_eight_models
./run mismatch --input results/halfcheetah_replication --output results/repeat_mismatch
./run audit --input results/repeat_eight_models
./run audit --input results/repeat_mismatch
./run report --input results/repeat_eight_models
./run report --input results/repeat_mismatch
```

The task commands run consolidated task drivers with the recorded checkpoints/configurations. Pendulum includes both signed constraints and 12-/32-particle BayesFP. The locomotion pilots include two RMS caps each. The eight-model study trains eight new model initializations on the fixed demonstration archive, retains all models and records all requests. All requested output directories must be new. Repeating seeds is not additional independent evidence.

## Collect and train without the research release

```bash
./run collect --task HalfCheetah --output data/HalfCheetah_episodes.npz
./run collect --task Walker2d --output data/Walker2d_episodes.npz
./run study --output results/from_scratch_eight_models
./run train --task HalfCheetah --seed 92300 --output results/train_halfcheetah --checkpoint assets/HalfCheetah_diffusion_v2.pt
./run train --task Walker2d --seed 92300 --output results/train_walker2d --checkpoint assets/Walker2d_diffusion_v2.pt
./run train --task Pendulum --seed 62001 --output results/train_pendulum --checkpoint assets/pendulum_diffusion.pt
```

Use these on a fresh checkout before restoring generated assets; existing checkpoint destinations are protected. MuJoCo collection recreates 128 complete episodes per task, with paired expert/medium initial-state groups, splits before overlapping chunks and train-only normalization. Pendulum's fixed recipe collects 640 complete original/reflected-expert episodes and trains its smaller network. Push-T uses the released diffusion policy and requires no training. Training/sampling nondeterminism across hardware or package changes is outside the recorded reproduction guarantee.

## Audits and historical hashes

`./run check` compares all three state-policy samplers and BayesFP particle clouds against small CPU fixtures captured before removing duplicate implementations. It also tests explicit rejection refusals, global projection against independent SOCP solves, Pendulum dynamics against native Gym, and complete MuJoCo snapshot replay.

`./run audit` verifies source/checkpoint hashes, every stored first-feasible selection, global projection, normalizers, numerical outcomes, bootstrap intervals and native replay. Raw artifacts are required. `./run report` regenerates figures/tables from saved summaries. `python scripts/verify_repository.py` verifies the repository layout, local links, retained scientific data and publication exclusions.

Historical lock files retain original path strings. `records/path_migration.json` maps moved artifacts; source files removed or edited during consolidation are verified against `git show pre-consolidation:PATH`. Fetch the tag if using a shallow clone. This audits the historical experiment's source bytes; CPU fixtures and full task reruns separately check equivalence of the maintained implementation.

The safety-error grid reuses the original outputs at multiple true-cap labels. Only its two conservative controls add simulation: 4,096 new rollouts. It is exploratory and retrospective, not a new independent estimator evaluation.

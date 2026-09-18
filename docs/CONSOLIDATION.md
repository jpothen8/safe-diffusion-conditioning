# Consolidation completed September 17, 2026

The active code is now the installable `safety_conditioning` package, invoked through `./run`, `python -m safety_conditioning`, or the `safety-conditioning` entry point. Paths, output protection and JSON/hash records are centralized in `artifacts.py`. Training, sampling, simulation, metrics and reporting have separate modules with explicit inputs. New runs no longer depend on mutating module-global project roots or copying scripts into each output directory.

The consolidation extracted the fixed DDPM architecture, full sampling postprocessing, exact box/ball projector, annealed refinement and BayesFP adapter from the completed locomotion experiment. It retains their numerical behavior. Collection is now independent of training, so reproducing complete episodes does not require an unnecessary initial model fit. The first-feasible reference explicitly masks refusals and cannot silently return a zero-action placeholder. Simulator traces include an executed-action mask, distinguishing controls after termination from controls actually applied.

The eight-model study is one JSON config and one scientific protocol. It freezes code/data/experts/dependencies before training and checkpoints before evaluation, keeps every model, and computes uncertainty at model and shared-state levels. Reports, tables, figures and paper text are regenerated from its saved summaries. Human-readable root README/RESULTS provide the current view; previous versions are retained under `docs/history/` with repaired relative links.

No historical scientific source, model, dataset, protocol lock or measured result was deleted or modified. Root `src/` and `round3/src/` remain at their original paths as frozen reproduction code, because their hashes and paths are part of old experiment records. Historical reproduction is exposed through the same CLI. This intentionally preserves some source duplication as an audit trail; it is not a second maintained implementation.

Verification performed:

- Refactored base sampling, refinement, BayesFP output/particle clouds and simulator rollouts match the frozen implementations bit for bit for both Walker2d and HalfCheetah.
- First-feasible selection, exhausted-budget handling and exact global projection were checked directly.
- Recollecting 128 complete HalfCheetah episodes reproduced the original archive SHA256 exactly.
- All eight new models completed the full frozen recipe, have distinct checkpoint hashes, and retain exact training-only normalizers.
- An independent audit recomputed the crossed bootstrap with resampling multiplicities, checked every saved projection/selection and replayed each arm's recorded controls with zero native reward/state error.
- Package installation, CLI dispatch and dependency consistency were checked in the recorded environment. The existing CUDA stack and driver were retained.

The two-layer environment is retained so historical code remains executable with its pinned dependencies. All current workflows use `round3/.venv`; setup provides one command and does not create additional environments. PyTorch's CUDA 12.8 wheel source remains explicit in the lock-based setup. Installed dependency versions and public-asset hashes remain the source of truth, not latest package releases.

The codebase is a controlled simulation research artifact. The finite sampler is not presented as an exact implementation of every theorem in the supplied draft. In particular the new replication keeps both modest positive results and the failed useful-effect criterion visible.

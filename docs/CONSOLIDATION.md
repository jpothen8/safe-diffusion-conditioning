# Consolidated implementation

The separate experiment-round folder and root `src/` tree have been removed. One installable `safety_conditioning` package and one `.venv` now support the project. Configuration, assets, generated data, results, provenance and protocols have shared locations.

The common modules implement the DDPM architecture/scheduler/sampling loop, first-feasible reference, annealed refinement, BayesFP particles, box/ball projection, model/state statistics and artifact handling. Task modules retain only their distinct dynamics, action scaling, demonstration recipe, metrics and experiment drivers. The Pendulum denoiser uses the same configurable architecture with its recorded smaller width. Its refinement and BayesFP call the shared implementation. Both locomotion drivers use the shared policy, simulator and constraints. Push-T retains its distinct upstream transformer and full-horizon conic projector.

Unused exploratory drivers, repeated setup scripts, copied samplers, duplicate reports and the overlay environment are removed from the active tree. The `pre-consolidation` Git tag preserves them, rather than retaining a second executable codebase. Recorded measured data and model checkpoints were moved without changing their contents; old locks use the migration map and Git history for verification.

The [reproduction guide](REPRODUCTION.md) is the command reference; [RESULTS.md](../RESULTS.md) is the consolidated scientific conclusion. The public release separates generated raw arrays and trained research checkpoints from Git. Public external checkpoints remain source downloads with explicit license uncertainty.

Validation records: [CPU/native/projection checks](../records/unified_checks.json), [task reproduction](../records/unified_reproduction.json), [repository verification](../records/repository_audit.json). These checks are generated from the executed cleanup, not assumptions of numerical equivalence.

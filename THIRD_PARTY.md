# Third-party resources

This repository contains research implementation and generated simulation findings. Public access to a checkpoint is not treated as a license to redistribute it.

- **Diffusion Policy / Push-T:** [official repository](https://github.com/real-stanford/diffusion_policy), revision `5ba07ac6661db573af695b419a7947ecb704690f`, MIT source license. Setup downloads the repository with its license unchanged. Released Push-T weights are downloaded separately; no checkpoint-specific redistribution license was found.
- **SAC demonstrators:** [Farama Minari](https://huggingface.co/farama-minari), pinned Walker2d-v5 and HalfCheetah-v5 expert/medium models. Inspected model metadata did not declare checkpoint licenses. Archives are fetched from the publishers, not redistributed.
- **TD3 Pendulum demonstrator:** [SB3 model release](https://huggingface.co/sb3/td3-Pendulum-v1), pinned revision in the asset manifest. No separate checkpoint redistribution license is assumed. The compatibility adapter extracts the public actor and preserves its scaling.
- **BayesFP:** [paper, arXiv:2606.21014v1](https://arxiv.org/html/2606.21014v1). The implementation is derived from Algorithm 1 / Appendix G.4. No official implementation was located in the recorded search; this is not claimed to reproduce the authors' benchmark results. The paper PDF is not republished here.
- **Starter context:** [safe-diffusion-mujoco](https://github.com/CadenzaCoda/safe-diffusion-mujoco), inspected revision `e2b2546d948c04f63ae7232933c768956872a94f`. It supplied RL demonstrator context; this project's conditional references, diffusion fits and refinement comparisons are separate work.

Dependency versions and installed license metadata are recorded in the project. External licenses remain applicable to their respective packages. No new blanket license grant for the user's original research is inferred merely from publishing the repository.

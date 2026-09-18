"""Write reproducible provenance for acquired and generated follow-up assets."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
paths=[('assets/td3-Pendulum-v1.zip','https://huggingface.co/sb3/td3-Pendulum-v1/resolve/f200e871a887a68eec2265f1c6ff366cfbf40ea8/td3-Pendulum-v1.zip',
        'f200e871a887a68eec2265f1c6ff366cfbf40ea8','No weights license declared in inspected model card/metadata; SB3 code license does not imply a weights grant.'),
       ('assets/pendulum_diffusion.pt','Local training: src/pendulum_train.py via scripts/train_pendulum_recorded.py','seed62000-data_seed62001-training',
        'Locally generated checkpoint; no separate license assigned; derived from the recorded public expert.'),
       ('results/pendulum/demonstrations.npz','Local complete-episode simulation of public expert and reflected expert','seed62000',
        'Locally generated simulation data; no separate license assigned.'),
       ('records/pendulum_initial_dynamics.py','Local original data-collection implementation','preserved before evaluation precision correction',
        'Locally authored code; retained to reproduce the exact collected demonstrations.')]
assets=[{'path':p,'source_url_or_procedure':u,'revision':r,'license':l,'bytes':(ROOT/p).stat().st_size,'sha256':digest(ROOT/p)} for p,u,r,l in paths]
repo=ROOT/'vendor/safe-diffusion-mujoco'
hashes={str(p.relative_to(repo)):digest(p) for p in repo.rglob('*') if p.is_file() and '.git' not in p.parts}
(ROOT/'records/user_repository_hashes.json').write_text(json.dumps(hashes,indent=2))
manifest={'local_assets':assets,'user_repository':{'url':'https://github.com/CadenzaCoda/safe-diffusion-mujoco',
    'revision':'e2b2546d948c04f63ae7232933c768956872a94f','source_hashes':'records/user_repository_hashes.json',
    'license':'No LICENSE file found at inspected revision. No license inferred.',
    'use':'Read-only research context. Contains expert acquisition/evaluation code; not the refinement implementation.'},
    'environment':{'gym':'0.26.2','pendulum_id':'Pendulum-v1','dependencies':'requirements.lock.txt','host':'records/host.json'},
    'verification':{'training_reproduced_checkpoint_sha256':digest(ROOT/'results/training_reproduction/assets/pendulum_diffusion.pt'),
                    'training_reproduced_demonstrations_sha256':digest(ROOT/'results/training_reproduction/results/pendulum/demonstrations.npz')},
    'primary_sources':['https://gymnasium.farama.org/environments/classic_control/pendulum/',
        'https://github.com/openai/gym/blob/0.26.2/gym/envs/classic_control/pendulum.py',
        'https://huggingface.co/sb3/td3-Pendulum-v1',
        'https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/schedulers/scheduling_ddpm.py']}
assert manifest['verification']['training_reproduced_checkpoint_sha256']==assets[1]['sha256']
assert manifest['verification']['training_reproduced_demonstrations_sha256']==assets[2]['sha256']
(ROOT/'records/followup_assets.json').write_text(json.dumps(manifest,indent=2))
print('Recorded provenance; exact training checkpoint and dataset reproduction verified.')

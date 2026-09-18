"""Upstream policy/environment adapters; no changes to the vendored code."""

import os
import sys
from ...artifacts import PROJECT as ROOT

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(ROOT / "vendor/diffusion_policy"))
import dill
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv

KP = None


def make_env(seed):
    global KP
    if KP is None:
        KP = PushTKeypointsEnv.genenerate_keypoint_manager_params()
    env = PushTKeypointsEnv(
        legacy=True, keypoint_visible_rate=1.0, agent_keypoints=False, **KP
    )
    env.seed(int(seed))
    obs = env.reset()[:20].astype(np.float32)
    return env, [obs.copy(), obs.copy()]


def load_policy(device="cuda"):
    torch.set_num_threads(4)
    # Public author checkpoint; its serialized OmegaConf requires ordinary pickle.
    payload = torch.load(
        ROOT / "assets/pusht_transformer.ckpt",
        map_location="cpu",
        pickle_module=dill,
        weights_only=False,
    )
    cfg = payload["cfg"]
    policy = hydra.utils.instantiate(cfg.policy)
    key = "ema_model" if cfg.training.use_ema else "model"
    policy.load_state_dict(payload["state_dicts"][key], strict=True)
    policy.eval().to(device)
    for p in policy.parameters():
        p.requires_grad_(False)
    (ROOT / "records/loaded_config.yaml").write_text(OmegaConf.to_yaml(cfg))
    return policy


def sample(policy, histories, seed):
    torch.manual_seed(int(seed))
    if policy.device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    observations = np.asarray(histories, dtype=np.float32)
    with torch.inference_mode():
        result = policy.predict_action(
            {"obs": torch.as_tensor(observations, device=policy.device)}
        )
    # This is the unmodified upstream executable slice [1:9], not all 10 predictions.
    actions = result["action"].cpu().numpy()
    assert actions.shape[1:] == (8, 2)
    assert np.isfinite(actions).all()
    assert np.allclose(actions, result["action_pred"][:, 1:9].cpu().numpy())
    return actions


def step(env, history, action):
    obs, reward, done, info = env.step(np.asarray(action, dtype=np.float64))
    history[:] = [history[-1], obs[:20].astype(np.float32)]
    return float(reward), bool(done), info


def snapshot(seed, prefix=()):
    env, history = make_env(seed)
    for action in prefix:
        step(env, history, action)
    return env, history


def numeric_state(env):
    return {
        "agent_position": list(env.agent.position),
        "agent_velocity": list(env.agent.velocity),
        "block_position": list(env.block.position),
        "block_velocity": list(env.block.velocity),
        "block_angle": env.block.angle,
        "block_angular_velocity": env.block.angular_velocity,
    }

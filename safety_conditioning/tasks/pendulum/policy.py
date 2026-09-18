"""Pendulum architecture/scaling and the recorded complete-episode recipe."""

import copy, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from ...artifacts import PROJECT
from ...policy import ChunkDDPM, scheduler, sample_latent
from . import environment as pd

HORIZON = 12


class Denoiser(ChunkDDPM):
    def __init__(self):
        super().__init__(HORIZON, 3, width=256)
        self.register_buffer("action_center", torch.zeros(1))
        self.register_buffer("action_scale", torch.ones(1) * 2)


def sample(model, observations, seed):
    latent = sample_latent(model, observations, seed)
    return (
        (latent * model.action_scale + model.action_center)
        .clamp(-2, 2)
        .cpu()
        .numpy()[..., None]
    )


def load(device="cuda", checkpoint=None):
    payload = torch.load(
        checkpoint or PROJECT / "assets/pendulum_diffusion.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = Denoiser()
    model.load_state_dict(payload["state_dict"])
    return model.eval().to(device)


def train(output, checkpoint, device="cuda"):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(62000)
    # Each initial-state group has both complete, coherent expert episodes.
    groups = 320
    states = np.column_stack(
        [np.pi + rng.uniform(-0.15, 0.15, groups), rng.uniform(-0.15, 0.15, groups)]
    )
    # Add broad reset episodes for continuation coverage. Both mirror policies share every reset.
    states[240:] = np.column_stack(
        [rng.uniform(-np.pi, np.pi, 80), rng.uniform(-1, 1, 80)]
    )
    actor = pd.actor()
    data = [pd.rollout(actor, states, reflection=mode) for mode in [False, True]]
    obs = np.stack([d["observations"] for d in data], axis=1)
    actions = np.stack([d["actions"] for d in data], axis=1)
    reward = np.stack([d["rewards"] for d in data], axis=1)
    permutation = rng.permutation(groups)
    split = {
        "train": permutation[:256],
        "validation": permutation[256:288],
        "test": permutation[288:],
    }
    np.savez_compressed(
        out / "demonstrations.npz",
        initial_states=states,
        observations=obs,
        actions=actions,
        rewards=reward,
        **{k + "_groups": v for k, v in split.items()},
    )
    # No overlap or mirrored-group leakage is possible after this split.
    assert not set(split["train"]) & set(split["validation"])
    assert not set(split["train"]) & set(split["test"])
    train_obs = obs[split["train"]].reshape(-1, 3)
    train_act = actions[split["train"]].reshape(-1, 1)

    def chunks(ids):
        oo = obs[ids, :, : 200 - HORIZON + 1, :].reshape(-1, 3)
        aa = np.lib.stride_tricks.sliding_window_view(
            actions[ids, ..., 0], HORIZON, axis=-1
        ).reshape(-1, HORIZON)
        assert len(oo) == len(aa)
        # Action a[t] accompanies pre-action o[t], including the first action of each chunk.
        np.testing.assert_array_equal(aa[0], actions[ids[0], 0, :HORIZON, 0])
        np.testing.assert_array_equal(oo[0], obs[ids[0], 0, 0])
        return torch.tensor(oo, device=device), torch.tensor(aa, device=device)

    torch.manual_seed(62001)
    torch.cuda.manual_seed_all(62001)
    torch.set_num_threads(4)
    model = Denoiser().to(device)
    model.obs_mean.copy_(torch.tensor(train_obs.mean(axis=0), device=device))
    model.obs_std.copy_(
        torch.tensor(np.maximum(train_obs.std(axis=0), 1e-4), device=device)
    )
    model.action_center.copy_(
        torch.tensor((train_act.max(axis=0) + train_act.min(axis=0)) / 2, device=device)
    )
    model.action_scale.copy_(
        torch.tensor((train_act.max(axis=0) - train_act.min(axis=0)) / 2, device=device)
    )
    ema = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    train_o, train_a = chunks(split["train"])
    val_o, val_a = chunks(split["validation"])
    train_a = (train_a - model.action_center) / model.action_scale
    val_a = (val_a - model.action_center) / model.action_scale
    sched = scheduler()
    logs = []
    start = time.perf_counter()
    for step in range(12000):
        idx = torch.randint(len(train_o), (512,), device=device)
        noise = torch.randn((512, HORIZON), device=device)
        t = torch.randint(0, 100, (512,), device=device)
        x = sched.add_noise(train_a[idx], noise, t)
        loss = (model(x, t, train_o[idx]) - noise).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), model.parameters()):
                pe.lerp_(p, 0.005)
        if (step + 1) % 1000 == 0:
            with torch.no_grad():
                vi = torch.randint(len(val_o), (1024,), device=device)
                vn = torch.randn((1024, HORIZON), device=device)
                vt = torch.randint(0, 100, (1024,), device=device)
                vl = (
                    (ema(sched.add_noise(val_a[vi], vn, vt), vt, val_o[vi]) - vn)
                    .square()
                    .mean()
                )
            row = {
                "step": step + 1,
                "train_loss": float(loss),
                "validation_loss": float(vl),
                "seconds": time.perf_counter() - start,
            }
            logs.append(row)
            print(json.dumps(row), flush=True)
    ema.eval()
    torch.save(
        {"state_dict": {k: v.cpu() for k, v in ema.state_dict().items()}},
        Path(checkpoint),
    )
    result = {
        "groups": groups,
        "episodes": groups * 2,
        "episode_length": 200,
        "horizon": HORIZON,
        "train_groups": len(split["train"]),
        "validation_groups": len(split["validation"]),
        "test_groups": len(split["test"]),
        "train_chunks": len(train_o),
        "normalization": "training episodes only",
        "steps": 12000,
        "parameters": sum(p.numel() for p in model.parameters()),
        "seconds": time.perf_counter() - start,
        "checkpoint_sha256": hashlib.sha256(
            (Path(checkpoint)).read_bytes()
        ).hexdigest(),
        "logs": logs,
    }
    (out / "training.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

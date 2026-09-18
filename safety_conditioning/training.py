"""One fixed training recipe; seed is explicit and validation never selects weights."""
import copy
import time
import numpy as np
import torch
from .artifacts import sha256, write_json
from .data import EpisodeDataset
from .policy import ChunkDDPM, scheduler, seed_torch
from .simulator import CONTROL_PERIOD


def train_policy(dataset_path, task, seed, output, *, horizon=8, steps=48000, device="cuda"):
    output.mkdir(parents=True, exist_ok=False)
    dataset = EpisodeDataset(dataset_path)
    train_obs, train_actions, early = dataset.windows("train", horizon, CONTROL_PERIOD[task])
    val_obs, val_actions, _ = dataset.windows("validation", horizon, CONTROL_PERIOD[task])
    seed_torch(seed)
    torch.set_num_threads(4)
    model = ChunkDDPM(horizon * 6).to(device)
    mean, std = dataset.normalizer()
    model.obs_mean.copy_(torch.tensor(mean, device=device))
    model.obs_std.copy_(torch.tensor(std, device=device))
    ema = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    train_obs, train_actions, val_obs, val_actions = [
        torch.as_tensor(value, device=device) for value in (train_obs, train_actions, val_obs, val_actions)
    ]
    early = torch.as_tensor(np.flatnonzero(early), device=device)
    schedule, logs = scheduler(), []
    started = time.perf_counter()
    for step in range(steps):
        indices = torch.cat([
            torch.randint(len(train_obs), (256,), device=device),
            early[torch.randint(len(early), (256,), device=device)],
        ])
        noise = torch.randn((512, horizon * 6), device=device)
        timestep = torch.randint(0, 100, (512,), device=device)
        loss = (model(schedule.add_noise(train_actions[indices], noise, timestep), timestep, train_obs[indices]) - noise).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        with torch.no_grad():
            for average, current in zip(ema.parameters(), model.parameters()):
                average.lerp_(current, .005)
        if (step + 1) % 8000 == 0:
            with torch.no_grad():
                indices = torch.randint(len(val_obs), (512,), device=device)
                noise = torch.randn((512, horizon * 6), device=device)
                timestep = torch.randint(0, 100, (512,), device=device)
                val_loss = (ema(schedule.add_noise(val_actions[indices], noise, timestep), timestep, val_obs[indices]) - noise).square().mean()
            row = {"step": step + 1, "loss": float(loss), "validation_loss": float(val_loss),
                   "seconds": time.perf_counter() - started}
            logs.append(row)
            print(f"train seed={seed} step={step + 1} validation={float(val_loss):.4f}", flush=True)
    checkpoint = output / "policy.pt"
    torch.save({"dimension": horizon * 6,
                "state_dict": {key: value.cpu() for key, value in ema.state_dict().items()}}, checkpoint)
    summary = {
        "task": task, "seed": seed, "steps": steps, "batch": 512, "horizon_controls": horizon,
        "horizon_seconds": horizon * CONTROL_PERIOD[task], "dataset_sha256": sha256(dataset_path),
        "checkpoint_sha256": sha256(checkpoint), "seconds": time.perf_counter() - started,
        "selection": "Final fixed-step EMA; no selection by validation, reward or safety.", "logs": logs,
    }
    write_json(output / "summary.json", summary)
    return checkpoint


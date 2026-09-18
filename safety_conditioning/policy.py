"""State DDPM and its complete sampling law, including physical clipping."""
import numpy as np
import torch
from torch import nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


class ChunkDDPM(nn.Module):
    """Architecture and state-dict keys match the frozen round3 models."""

    def __init__(self, dimension, obs_dimension=17):
        super().__init__()
        self.dimension = dimension
        self.obs_dimension = obs_dimension
        self.net = nn.Sequential(
            nn.Linear(dimension + obs_dimension + 32, 512), nn.SiLU(),
            nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU(),
            nn.Linear(512, dimension),
        )
        self.register_buffer("obs_mean", torch.zeros(obs_dimension))
        self.register_buffer("obs_std", torch.ones(obs_dimension))

    def forward(self, actions, timestep, observations):
        timestep = torch.as_tensor(timestep, device=actions.device).expand(len(actions)).float()
        frequencies = torch.exp(torch.arange(16, device=actions.device) * (-np.log(10000) / 15))
        phase = timestep[:, None] * frequencies[None]
        features = torch.cat([
            actions, (observations - self.obs_mean) / self.obs_std,
            phase.sin(), phase.cos(),
        ], dim=-1)
        return self.net(features)


def scheduler():
    return DDPMScheduler(
        num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
        clip_sample=True, prediction_type="epsilon", variance_type="fixed_small",
    )


def seed_torch(seed):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def load_policy(path, device="cuda"):
    payload = torch.load(path, weights_only=True, map_location="cpu")
    model = ChunkDDPM(payload["dimension"])
    model.load_state_dict(payload["state_dict"])
    return model.eval().to(device)


def sample_policy(model, observations, seed):
    seed_torch(seed)
    device = next(model.parameters()).device
    observations = torch.as_tensor(observations, device=device, dtype=torch.float32)
    actions = torch.randn((len(observations), model.dimension), device=device)
    schedule = scheduler()
    schedule.set_timesteps(100)
    with torch.inference_mode():
        for timestep in schedule.timesteps:
            noise = model(actions, timestep, observations)
            actions = schedule.step(noise, timestep, actions).prev_sample
    # The final scheduler coefficient can overshoot the physical box by 1.4e-5.
    return actions.clamp(-1, 1).cpu().numpy()


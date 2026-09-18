"""Conditional reference, annealed refinement and paper-derived BayesFP.

BayesFP source: arXiv:2606.21014v1, Algorithm 1 / Appendix G.4.
All coordinates here are normalized actions. No task reward enters sampling.
"""
import numpy as np
import torch
from .policy import scheduler, seed_torch


def rejection_from_bank(bank, constraint):
    """First feasible independent proposal, with explicit finite-budget failures.

    Failed rows contain NaN, never a fabricated conditional sample.
    """
    feasible = constraint.feasible(bank, numerical_tolerance=False)
    valid = feasible.any(-1)
    indices = feasible.argmax(-1)
    actions = bank[np.arange(len(bank)), indices].copy()
    actions[~valid] = np.nan
    second = feasible.copy()
    second[np.arange(len(bank)), indices] = False
    second_valid = second.any(-1)
    second_actions = bank[np.arange(len(bank)), second.argmax(-1)].copy()
    second_actions[~second_valid] = np.nan
    return {
        "actions": actions, "feasible": feasible, "valid": valid,
        "indices": np.where(valid, indices, -1),
        "proposals_used": np.where(valid, indices + 1, bank.shape[1]),
        "second_actions": second_actions, "second_valid": second_valid,
    }


def refine(model, observations, raw, constraint, seed):
    seed_torch(seed)
    device = next(model.parameters()).device
    observations = torch.as_tensor(observations, dtype=torch.float32, device=device)
    actions = constraint.project_torch(torch.as_tensor(raw, dtype=torch.float32, device=device))
    alpha_bars = scheduler().alphas_cumprod.to(device)
    with torch.inference_mode():
        for level, steps in [(20, 256), (5, 256), (0, 512)]:
            alpha_bar = alpha_bars[level]
            sigma = ((1 - alpha_bar) / alpha_bar).sqrt()
            for step in range(steps):
                rate = .15 * sigma.square() / (1 + step / 64)**.6
                score = -model(alpha_bar.sqrt() * actions, level, observations) / sigma
                proposal = actions + rate * score + (2 * rate).sqrt() * torch.randn_like(actions)
                actions = constraint.project_torch(proposal)
    return actions.cpu().numpy()


def box_cost_gradient(actions, lower, upper):
    residual = actions - actions.clamp(lower, upper)
    return residual.square().mean(-1), 2 * residual / actions.shape[-1]


def ball_cost_gradient(actions, radius):
    rms = actions.square().mean(-1).clamp_min(1e-20).sqrt()
    residual = (rms - radius).clamp_min(0)
    gradient = 2 * residual[..., None] * actions / (rms[..., None] * actions.shape[-1])
    return residual.square(), gradient


def systematic_resample(weights):
    batch, particles = weights.shape
    cumulative = weights.cumsum(-1)
    cumulative[:, -1] = 1
    positions = (torch.rand((batch, 1), device=weights.device)
                 + torch.arange(particles, device=weights.device)) / particles
    return torch.searchsorted(cumulative.contiguous(), positions.contiguous()).clamp_max(particles - 1)


def bayesfp(model, observations, seed, *, strength=20, particles=32,
            radius=None, lower=-float("inf"), upper=0):
    """Return raw normalized DDPM output and particle diagnostics.

    Physical clipping belongs to the policy postprocessing in the caller.
    This keeps the diagnostic particle cloud identical to the frozen adapter.
    """
    seed_torch(seed)
    device = next(model.parameters()).device
    observations = torch.as_tensor(observations, device=device, dtype=torch.float32)
    batch = len(observations)
    observations = observations.repeat_interleave(particles, 0)
    actions = torch.randn((batch * particles, model.dimension), device=device)
    weights = torch.zeros((batch, particles), device=device)
    schedule = scheduler()
    schedule.set_timesteps(100)
    alpha_bars = schedule.alphas_cumprod.to(device)
    effective_sizes = []
    ancestors = torch.arange(particles, device=device)[None].expand(batch, particles).clone()
    with torch.inference_mode():
        for index, timestep in enumerate(schedule.timesteps):
            step = int(timestep)
            alpha_bar = alpha_bars[step]
            std = (1 - alpha_bar).sqrt()
            noise = model(actions, timestep, observations)
            if strength:
                _, gradient = (box_cost_gradient(actions, lower, upper) if radius is None
                               else ball_cost_gradient(actions, radius))
                delta = 1 - alpha_bar / (alpha_bars[step - 1] if step > 0 else 1.)
                score = -noise / std
                increment = (-.5 * strength * delta * (gradient * (score + actions)).sum(-1)).reshape(batch, particles)
                increment = increment - increment.mean(-1, keepdim=True)
                noise = noise + .5 * strength * std * gradient
            actions = schedule.step(noise, timestep, actions).prev_sample
            active = .05 <= index / 99 <= .95
            if strength and active:
                weights = weights + increment
                probabilities = weights.softmax(-1)
                effective_sizes.append((1 / probabilities.square().sum(-1)).cpu().numpy())
                indices = systematic_resample(probabilities)
                cloud = actions.reshape(batch, particles, model.dimension)
                actions = cloud.gather(1, indices[..., None].expand(-1, -1, model.dimension)).reshape(batch * particles, model.dimension)
                ancestors = ancestors.gather(1, indices)
                weights.zero_()
            elif not active:
                weights.zero_()
        selected = torch.multinomial(weights.softmax(-1), 1).squeeze(-1)
        cloud = actions.reshape(batch, particles, model.dimension)
        output = cloud[torch.arange(batch, device=device), selected]
    diagnostics = {
        "particles": particles, "strength": strength,
        "score_evaluations_per_output": 100 * particles,
        "ess_mean": float(np.mean(effective_sizes)) if effective_sizes else float(particles),
        "ess_min": float(np.min(effective_sizes)) if effective_sizes else float(particles),
        "unique_initial_ancestors_mean": float(np.mean([len(torch.unique(row)) for row in ancestors])),
    }
    return output.cpu().numpy(), cloud.cpu().numpy(), diagnostics

"""Global full-chunk projection onto an RMS command budget and action box."""
from dataclasses import dataclass
import numpy as np
import torch


@dataclass(frozen=True)
class EffortLimit:
    radius: float
    tolerance: float = 1e-6

    def __post_init__(self):
        if not 0 < self.radius <= 1:
            raise ValueError("The normalized RMS limit must be in (0, 1].")

    def feasible(self, actions, *, numerical_tolerance=True):
        tolerance = self.tolerance if numerical_tolerance else 0.0
        return (
            np.isfinite(actions).all(-1)
            & (np.sqrt(np.mean(actions * actions, axis=-1)) <= self.radius + tolerance)
            & (np.abs(actions).max(-1) <= 1 + tolerance)
        )

    def project(self, actions):
        actions = np.asarray(actions)
        rms = np.sqrt(np.mean(actions * actions, axis=-1, keepdims=True))
        if np.max(np.abs(actions)) <= 1 + 1e-7:
            return actions * np.minimum(1, self.radius / np.maximum(rms, 1e-20))
        clipped = np.clip(actions, -1, 1)
        inside = np.mean(clipped**2, axis=-1, keepdims=True) <= self.radius**2
        low, high = np.zeros_like(rms), np.ones_like(rms)
        for _ in range(50):
            mid = (low + high) / 2
            trial = np.clip(actions * mid, -1, 1)
            feasible = np.mean(trial * trial, axis=-1, keepdims=True) <= self.radius**2
            low, high = np.where(feasible, mid, low), np.where(feasible, high, mid)
        return np.where(inside, clipped, np.clip(actions * low, -1, 1))

    def project_torch(self, actions):
        clipped = actions.clamp(-1, 1)
        inside = clipped.square().mean(-1, keepdim=True) <= self.radius**2
        rms = actions.square().mean(-1, keepdim=True).clamp_min(1e-20).sqrt()
        radial = actions * (self.radius / rms).clamp_max(1)
        simple = radial.abs().amax(-1, keepdim=True) <= 1
        complicated = (~inside & ~simple).squeeze(-1)
        result = torch.where(inside, clipped, radial)
        if bool(complicated.any()):
            values = actions[complicated]
            low = torch.zeros((len(values), 1), device=actions.device)
            high = torch.ones_like(low)
            for _ in range(32):
                mid = (low + high) / 2
                trial = (values * mid).clamp(-1, 1)
                feasible = trial.square().mean(-1, keepdim=True) <= self.radius**2
                low = torch.where(feasible, mid, low)
                high = torch.where(feasible, high, mid)
            result[complicated] = (values * low).clamp(-1, 1)
        return result


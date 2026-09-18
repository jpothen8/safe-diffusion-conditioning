"""Distribution diagnostics and uncertainty at the actual sampling units."""

import numpy as np
from scipy.stats import t as student_t


def action_mmd(first, second, bandwidth=0.2):
    if not len(first) or not len(second):
        return None
    first = np.asarray(first, dtype=float).reshape(len(first), -1)
    second = np.asarray(second, dtype=float).reshape(len(second), -1)

    def kernel(x, y):
        squared = (
            (x * x).sum(1)[:, None] + (y * y).sum(1)[None] - 2 * x @ y.T
        ) / x.shape[1]
        return np.exp(-np.maximum(squared, 0) / (2 * bandwidth**2))

    return float(
        np.sqrt(
            max(
                0,
                kernel(first, first).mean()
                + kernel(second, second).mean()
                - 2 * kernel(first, second).mean(),
            )
        )
    )


def crossed_intervals(differences, seed, draws=20000):
    """Equal-weight model/state means; shared state resample across models.

    Input: independent training seeds x common held-out states x replicates.
    The t interval reflects between-model variation on the fixed state bank;
    the crossed bootstrap also resamples the shared state bank.
    """
    values = np.asarray(differences, dtype=float).mean(-1)
    if not np.isfinite(values).all():
        return {
            "complete": False,
            "reason": "At least one requested contrast is missing.",
        }
    models, states = values.shape
    rng = np.random.default_rng(seed)
    model_indices = rng.integers(0, models, (draws, models))
    state_indices = rng.integers(0, states, (draws, states))
    distribution = values[model_indices[:, :, None], state_indices[:, None, :]].mean(
        (1, 2)
    )
    model_means = values.mean(1)
    mean = float(model_means.mean())
    halfwidth = float(
        student_t.ppf(0.975, models - 1) * model_means.std(ddof=1) / np.sqrt(models)
    )
    return {
        "complete": True,
        "mean": mean,
        "crossed_bootstrap_ci95": np.quantile(distribution, [0.025, 0.975]).tolist(),
        "model_t_ci95": [mean - halfwidth, mean + halfwidth],
        "model_means": model_means.tolist(),
        "positive_models": int((model_means > 0).sum()),
        "models": models,
        "states": states,
        "replicates": differences.shape[-1],
    }

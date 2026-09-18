import numpy as np
import gym
from ...statistics import action_mmd


def mmd(a, b):
    return action_mmd(np.asarray(a) / 2, np.asarray(b) / 2, bandwidth=0.1)


def metrics(roll, chunk, valid):
    theta = (roll["states"][..., 0] + np.pi) % (2 * np.pi) - np.pi
    near = (np.abs(theta) < 0.1) & (np.abs(roll["states"][..., 1]) < 0.5)
    windows = np.lib.stride_tricks.sliding_window_view(near, 20, axis=1).all(-1)
    stabilized = windows.any(1) & valid
    stabilization = np.where(stabilized, windows.argmax(1) * 0.05, 10.0)
    mode = chunk.mean((1, 2))
    return {
        "return_mean": float(roll["rewards"][valid].sum(1).mean()),
        "return_std": float(roll["rewards"][valid].sum(1).std(ddof=1)),
        "stabilized": int(stabilized.sum()),
        "unsuccessful_including_refusals": int((~stabilized).sum()),
        "mean_capped_stabilization_time": float(stabilization.mean()),
        "added_boundary_fraction": float(
            np.isclose(chunk[valid], 0, atol=1e-5).any((1, 2)).mean()
        ),
        "physical_boundary_fraction": float(
            np.isclose(np.abs(chunk[valid]), 2, atol=1e-5).any((1, 2)).mean()
        ),
        "mean_abs_first_chunk_displacement": float(
            np.abs(roll["states"][valid, 12, 0] - roll["states"][valid, 0, 0]).mean()
        ),
        "mode_negative": float((mode[valid] < -1).mean()),
        "mode_positive": float((mode[valid] > 1).mean()),
        "mode_intermediate": float((np.abs(mode[valid]) <= 1).mean()),
        "mean_torque": float(chunk[valid].mean()),
        "mean_chunk_torque_std": float(chunk[valid].std(1).mean()),
    }


def compare(a, b, valid, cfg, boot):
    delta = (a - b).reshape(cfg["states"], cfg["replicates"])
    mask = valid.reshape(delta.shape)
    means = np.array([d[m].mean() if m.any() else np.nan for d, m in zip(delta, mask)])
    usable = means[np.isfinite(means)]
    if len(usable) != cfg["states"]:
        return {
            "valid_pairs": int(valid.sum()),
            "error": "Refusals invalidate the prespecified complete-state bootstrap",
            "mean_difference": float(np.nanmean(means)),
        }
    distribution = means[boot].mean(1)
    interval = np.quantile(distribution, [0.00625, 0.99375]).tolist()
    mean = float(means.mean())
    return {
        "mean_difference": mean,
        "ci95": np.quantile(distribution, [0.025, 0.975]).tolist(),
        "family_ci98_75": interval,
        "per_state_differences": means.tolist(),
        "valid_pairs": int(valid.sum()),
        "passes_effect_criterion": bool(
            mean >= cfg["minimum_useful_return_gain"] and interval[0] > 0
        ),
    }


def verify_gym(states, roll):
    error = 0.0
    for index in [0, len(states) // 3, 2 * len(states) // 3]:
        env = gym.make("Pendulum-v1")
        env.reset(seed=0)
        env.unwrapped.state = states[index].copy()
        for t in range(200):
            error = max(
                error,
                float(np.max(np.abs(env.unwrapped.state - roll["states"][index, t]))),
            )
            _, reward, _, _, _ = env.step(roll["actions"][index, t])
            error = max(error, abs(float(reward) - float(roll["rewards"][index, t])))
        env.close()
    return error

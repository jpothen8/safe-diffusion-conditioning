"""Recreate complete public-expert episodes without an unnecessary model fit."""
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from .artifacts import sha256, write_json
from .simulator import load_expert


def collect_episodes(task, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    controllers = {level: load_expert(task, level) for level in ("expert", "medium")}
    observations, actions, rewards, lengths, groups, levels, terminals = [], [], [], [], [], [], []
    for group in range(64):
        for level, actor in controllers.items():
            env = gym.make(task + "-v5")
            obs, _ = env.reset(seed=92000 + group)
            episode_obs, episode_actions, episode_rewards = [], [], []
            try:
                while True:
                    episode_obs.append(obs.copy())
                    action, _ = actor.predict(obs, deterministic=True)
                    episode_actions.append(action)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    episode_rewards.append(reward)
                    if terminated or truncated:
                        break
            finally:
                env.close()
            observations.append(np.array(episode_obs, dtype=np.float32))
            actions.append(np.array(episode_actions, dtype=np.float32))
            rewards.append(np.array(episode_rewards))
            lengths.append(len(episode_rewards))
            groups.append(group)
            levels.append(level)
            terminals.append(terminated)
    permutation = np.random.default_rng(92100).permutation(64)
    ends = np.cumsum(lengths)
    np.savez_compressed(
        output, observations=np.concatenate(observations), actions=np.concatenate(actions),
        rewards=np.concatenate(rewards), episode_starts=np.r_[0, ends[:-1]],
        episode_lengths=lengths, groupids=groups, levels=levels, terminated=terminals,
        train_groups=permutation[:48], validation_groups=permutation[48:56], test_groups=permutation[56:],
    )
    summary = {"task": task, "episodes": len(lengths), "controls": sum(lengths),
               "collection_seed_start": 92000, "split_seed": 92100, "sha256": sha256(output)}
    write_json(output.with_suffix(".json"), summary)
    print(summary, flush=True)


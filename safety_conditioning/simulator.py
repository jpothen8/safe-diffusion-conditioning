"""Native MuJoCo snapshots and first-chunk/common-controller evaluation."""
import numpy as np
import gymnasium as gym
import mujoco
from stable_baselines3 import SAC
from .artifacts import expert_path

CONTROL_PERIOD = {"Walker2d": .008, "HalfCheetah": .05}


def load_expert(task, level="expert"):
    return SAC.load(expert_path(task, level), device="cpu",
                    custom_objects={"learning_rate": 0., "lr_schedule": lambda _: 0.})


def snapshot(task, seed, actor, warm_seconds=0):
    env = gym.make(task + "-v5")
    try:
        observation, _ = env.reset(seed=seed)
        for _ in range(round(warm_seconds / CONTROL_PERIOD[task])):
            action, _ = actor.predict(observation, deterministic=True)
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                raise RuntimeError("Snapshot warmup terminated; do not replace its seed.")
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(env.unwrapped.model, spec))
        mujoco.mj_getState(env.unwrapped.model, env.unwrapped.data, state, spec)
        return {
            "task": task, "seed": seed, "warm_seconds": warm_seconds,
            "integration_state": state.tolist(), "state_spec": int(spec),
            "observation": observation.tolist(),
        }
    finally:
        env.close()


def restore(saved):
    env = gym.make(saved["task"] + "-v5")
    env.reset(seed=saved["seed"])
    mujoco.mj_setState(env.unwrapped.model, env.unwrapped.data,
                      np.array(saved["integration_state"]), saved["state_spec"])
    mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
    return env


def rollout(task, actor, snapshots, chunks=None, seconds=4.):
    """Accrue native reward; actions after termination are marked unexecuted."""
    envs = [restore(saved) for saved in snapshots]
    count = len(envs)
    controls = round(seconds / CONTROL_PERIOD[task])
    horizon = chunks.shape[1] if chunks is not None else 0
    if chunks is not None and (chunks.shape != (count, horizon, 6) or horizon > controls):
        raise ValueError("Chunks must fit the evaluation interval and six-action environment.")
    observations = np.array([env.unwrapped._get_obs() for env in envs])
    rewards = np.zeros((count, controls))
    active = np.ones(count, dtype=bool)
    terminated = np.zeros(count, dtype=bool)
    lengths = np.zeros(count, dtype=int)
    initial_x = np.array([env.unwrapped.data.qpos[0] for env in envs])
    states, actions, executed = [], [], []
    try:
        for step in range(controls):
            states.append(np.array([np.r_[env.unwrapped.data.qpos, env.unwrapped.data.qvel] for env in envs]))
            if chunks is not None and step < horizon:
                action = chunks[:, step]
            else:
                action, _ = actor.predict(observations, deterministic=True)
            actions.append(action.copy())
            executed.append(active.copy())
            for index in np.flatnonzero(active):
                observations[index], rewards[index, step], term, trunc, _ = envs[index].step(action[index])
                lengths[index] += 1
                if term or trunc:
                    active[index] = False
                    terminated[index] = term
        progress = np.array([env.unwrapped.data.qpos[0] for env in envs]) - initial_x
        return {
            "rewards": rewards, "actions": np.stack(actions, 1),
            "states": np.stack(states, 1), "executed": np.stack(executed, 1),
            "terminated": terminated, "lengths": lengths, "progress": progress,
        }
    finally:
        for env in envs:
            env.close()


"""Episode-separated windows and training-only observation normalization."""

from pathlib import Path
import numpy as np


class EpisodeDataset:
    def __init__(self, path):
        self.path = Path(path)
        with np.load(self.path) as archive:
            self.data = {key: archive[key] for key in archive.files}
        splits = [
            set(self.data[name + "_groups"]) for name in ("train", "validation", "test")
        ]
        if any(splits[i] & splits[j] for i in range(3) for j in range(i)):
            raise ValueError("Episode group splits overlap.")

    def windows(self, split, horizon, control_period):
        observations, actions, early = [], [], []
        data = self.data
        for start, length, group in zip(
            data["episode_starts"], data["episode_lengths"], data["groupids"]
        ):
            if group not in data[split + "_groups"] or length < horizon:
                continue
            episode = data["actions"][start : start + length]
            windows = np.lib.stride_tricks.sliding_window_view(
                episode, horizon, axis=0
            ).transpose(0, 2, 1)
            if not np.array_equal(windows[0], episode[:horizon]):
                raise ValueError("Action/time alignment error.")
            observations.append(data["observations"][start : start + len(windows)])
            actions.append(windows.reshape(len(windows), -1))
            early.append(np.arange(len(windows)) < round(2 / control_period))
        return (
            np.concatenate(observations),
            np.concatenate(actions),
            np.concatenate(early),
        )

    def normalizer(self):
        data = self.data
        indices = np.concatenate(
            [
                np.arange(start, start + length)
                for start, length, group in zip(
                    data["episode_starts"], data["episode_lengths"], data["groupids"]
                )
                if group in data["train_groups"]
            ]
        )
        observations = data["observations"][indices]
        return observations.mean(0), np.maximum(observations.std(0), 1e-4)

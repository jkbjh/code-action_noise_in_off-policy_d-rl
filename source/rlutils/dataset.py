from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import numpy.typing as npt
import copy

SARSDI_T = Tuple[npt.ArrayLike, npt.ArrayLike, float, npt.ArrayLike, bool, Dict]
EPISODE_T = List[SARSDI_T]


class Dataset(object):
    def __init__(self, episode_buffer: List[EPISODE_T] = None):
        self.episode_buffer: List[EPISODE_T] = episode_buffer or []

    def prepend(self, dataset: Dataset):
        self.episode_buffer = (
            copy.deepcopy(dataset.episode_buffer) + self.episode_buffer
        )

    def extend(self, dataset: Dataset):
        self.episode_buffer = self.episode_buffer + copy.deepcopy(
            dataset.episode_buffer
        )

    def limit_last(self, minimum_samples: int):
        lens = self.collect_episode_lens()
        required_items = np.maximum(
            np.argmax(np.flip(np.cumsum(np.flip(lens))) <= minimum_samples) - 1, 0
        )
        new_buffer = self.episode_buffer[required_items:]
        self.episode_buffer = new_buffer

    def collect_all_states(self) -> Optional[np.ndarray]:
        if self.episode_buffer is None:
            return None
        states = []
        for episode in self.episode_buffer:
            s, _, _, _, _, _ = episode[0]
            episode_states = [s] + [s1 for s_, a_, r_, s1, d_, i_ in episode]
            states.extend(episode_states)
        return np.vstack(states)

    def collect_all_returns(self) -> Optional[np.ndarray]:
        if self.episode_buffer is None:
            return None
        returns = []
        for episode in self.episode_buffer:
            episode_rewards = [r for s_, a_, r, s1_, d_, i_ in episode]
            episode_return = np.sum(episode_rewards)
            returns.append(episode_return)
        return np.vstack(returns)

    def collect_all_rewards(self) -> Optional[np.ndarray]:
        if self.episode_buffer is None:
            return None
        rewards = []
        for episode in self.episode_buffer:
            episode_rewards = [r for s_, a_, r, s1_, d_, i_ in episode]
            rewards.extend(episode_rewards)
        return np.vstack(rewards)

    def collect_episode_lens(self) -> Optional[np.ndarray]:
        return np.array([len(episode) for episode in self.episode_buffer])

    def collect_all_actions(self) -> Optional[np.ndarray]:
        if self.episode_buffer is None:
            return None
        actions = []
        for episode in self.episode_buffer:
            episode_actions = [a for s_, a, r_, s1_, d_, i_ in episode]
            actions.extend(episode_actions)
        return np.vstack(actions)

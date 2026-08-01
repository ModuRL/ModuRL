"""Generate a deterministic flat-terrain BipedalWalker-v3 trajectory."""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np


class DeterministicRng:
    def uniform(self, low=0.0, high=1.0, size=None):
        value = (float(low) + float(high)) / 2.0
        return np.full(size, value) if size is not None else value

    def integers(self, low, high=None, size=None):
        value = int(low)
        return np.full(size, value, dtype=np.int64) if size is not None else value

    def random(self, size=None):
        return np.zeros(size) if size is not None else 0.0


if __name__ == "__main__":
    transition_actions = [
        np.zeros(4, dtype=np.float32),
        np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32),
        np.array([-0.1, -0.1, -0.1, -0.1], dtype=np.float32),
        np.array([0.2, -0.2, 0.2, -0.2], dtype=np.float32),
    ]
    initial_observation = None
    observations = []
    rewards = []
    terminated = []
    for action in transition_actions:
        environment = gym.make("BipedalWalker-v3").unwrapped
        environment.np_random = DeterministicRng()
        reset_observation, _ = environment.reset()
        if initial_observation is None:
            initial_observation = reset_observation
        observation, reward, done, _, _ = environment.step(action)
        observations.append(observation.tolist())
        rewards.append(float(reward))
        terminated.append(bool(done))
        environment.close()

    sequential_actions = [
        np.zeros(4, dtype=np.float32),
        np.full(4, 0.01, dtype=np.float32),
        np.full(4, -0.01, dtype=np.float32),
        np.array([0.01, -0.01, 0.01, -0.01], dtype=np.float32),
        np.array([-0.01, 0.01, -0.01, 0.01], dtype=np.float32),
        np.zeros(4, dtype=np.float32),
        np.array([0.02, 0.0, 0.02, 0.0], dtype=np.float32),
        np.array([-0.02, 0.0, -0.02, 0.0], dtype=np.float32),
    ]
    sequential_observations = []
    sequential_rewards = []
    sequential_terminated = []
    environment = gym.make("BipedalWalker-v3").unwrapped
    environment.np_random = DeterministicRng()
    environment.reset()
    for action in sequential_actions:
        observation, reward, done, _, _ = environment.step(action)
        sequential_observations.append(observation.tolist())
        sequential_rewards.append(float(reward))
        sequential_terminated.append(bool(done))
    environment.close()
    target = Path(__file__).parent / "bipedal_walker"
    target.mkdir(exist_ok=True)
    (target / "trajectory.json").write_text(
        json.dumps(
            {
                "initial_observation": initial_observation.tolist(),
                "actions": [action.tolist() for action in transition_actions],
                "observations": observations,
                "rewards": rewards,
                "terminated": terminated,
                "sequential_actions": [
                    action.tolist() for action in sequential_actions
                ],
                "sequential_observations": sequential_observations,
                "sequential_rewards": sequential_rewards,
                "sequential_terminated": sequential_terminated,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

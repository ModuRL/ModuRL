"""Generate exact-state parity fixtures for Pendulum-v1 and Acrobot-v1."""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np


ROOT = Path(__file__).parent


def pendulum() -> None:
    env = gym.make("Pendulum-v1").unwrapped
    transitions = []
    for step in range(24):
        state = np.array(
            [np.sin(step * 0.47) * np.pi, np.cos(step * 0.31) * 7.5],
            dtype=np.float64,
        )
        action = np.array([2.5 * np.sin(step * 0.73)], dtype=np.float32)
        env.state = state.copy()
        observation, reward, _, _, _ = env.step(action)
        transitions.append(
            {
                "state": state.tolist(),
                "action": float(action[0]),
                "observation": observation.tolist(),
                "reward": float(reward),
            }
        )
    target = ROOT / "pendulum"
    target.mkdir(exist_ok=True)
    (target / "trajectory.json").write_text(
        json.dumps(transitions, indent=2) + "\n", encoding="utf-8"
    )


def acrobot() -> None:
    env = gym.make("Acrobot-v1").unwrapped
    transitions = []
    for step in range(30):
        state = np.array(
            [
                np.sin(step * 0.29) * 2.8,
                np.cos(step * 0.37) * 2.6,
                np.sin(step * 0.41) * 11.0,
                np.cos(step * 0.23) * 24.0,
            ],
            dtype=np.float64,
        )
        action = step % 3
        env.state = state.copy()
        observation, reward, terminated, _, _ = env.step(action)
        transitions.append(
            {
                "state": state.tolist(),
                "action": action,
                "observation": observation.tolist(),
                "reward": float(reward),
                "terminated": bool(terminated),
            }
        )
    target = ROOT / "acrobot"
    target.mkdir(exist_ok=True)
    (target / "trajectory.json").write_text(
        json.dumps(transitions, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    pendulum()
    acrobot()

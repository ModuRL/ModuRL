"""Generate deterministic Gymnasium v5 trajectory fixtures.

Run from the repository root. The Rust tests consume the JSON files and do not
need Python or Gymnasium at test time.
"""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np


ROOT = Path(__file__).parent
ENVIRONMENTS = {
    "half_cheetah": ("HalfCheetah-v5", 20),
    "hopper": ("Hopper-v5", 20),
    # Stop before the first simultaneous two-foot impact. That degenerate
    # contact can legitimately choose different solvers/orderings across the
    # official Python and mujoco-rs binary builds of the same engine version.
    "walker2d": ("Walker2d-v5", 12),
}


def generate(folder: str, environment_id: str, number_of_steps: int) -> None:
    env = gym.make(environment_id).unwrapped
    env.reset(seed=7)

    # Use the model's exact reference configuration and zero velocity. This
    # bypasses reset RNG differences and isolates model/physics/reward parity.
    qpos = env.init_qpos.copy()
    qvel = np.zeros(env.model.nv, dtype=np.float64)
    env.set_state(qpos, qvel)

    actions = []
    states = []
    for step in range(number_of_steps):
        states.append(
            {
                "qpos": env.data.qpos.copy().tolist(),
                "qvel": env.data.qvel.copy().tolist(),
            }
        )
        indices = np.arange(env.model.nu, dtype=np.float64)
        action = (0.35 * np.sin(0.37 * step + 0.61 * indices)).astype(np.float32)
        env.step(action)
        actions.append(action.tolist())

    # Test each transition from a clean solver state. Contact solvers can
    # amplify tiny compiler/platform differences over a long rollout; this
    # still covers states along that rollout while comparing the environment's
    # actual one-step transition contract at tight tolerance.
    outputs = []
    for state, action in zip(states, actions, strict=True):
        mujoco.mj_resetData(env.model, env.data)
        env.set_state(np.asarray(state["qpos"]), np.asarray(state["qvel"]))
        observation, reward, terminated, truncated, _ = env.step(
            np.asarray(action, dtype=np.float32)
        )
        outputs.append(
            {
                "observation": observation.tolist(),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )

    target = ROOT / folder
    target.mkdir(exist_ok=True)
    (target / "trajectory.json").write_text(
        json.dumps(
            {
                "gymnasium_version": gym.__version__,
                "mujoco_version": mujoco.__version__,
                "environment_id": environment_id,
                "qpos": qpos.tolist(),
                "qvel": qvel.tolist(),
                "actions": actions,
                "states": states,
                "outputs": outputs,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    env.close()


if __name__ == "__main__":
    for folder_name, (env_id, steps) in ENVIRONMENTS.items():
        generate(folder_name, env_id, steps)

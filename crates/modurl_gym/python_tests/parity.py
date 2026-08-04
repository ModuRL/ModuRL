"""Generate fixtures and run focused Gymnasium parity tests.

Pass one environment name to test it, or omit the name to test every registered
environment. Rust tests consume the generated JSON and do not invoke Python.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


ROOT = Path(__file__).resolve().parent
CRATE_ROOT = ROOT.parent
REPOSITORY_ROOT = CRATE_ROOT.parents[1]
EXPECTED_GYMNASIUM_VERSION = "1.2.1"
EXPECTED_PYBOX2D_VERSION = "2.3.5"


def reference_modules(*, box2d: bool = False):
    try:
        import gymnasium as gym
        import numpy as np
        if box2d:
            import Box2D
        else:
            Box2D = None
    except ImportError as error:
        extra = "[box2d]" if box2d else ""
        raise RuntimeError(
            "install the reference dependencies first: "
            f'python -m pip install "gymnasium{extra}=={EXPECTED_GYMNASIUM_VERSION}"'
        ) from error

    if gym.__version__ != EXPECTED_GYMNASIUM_VERSION:
        raise RuntimeError(
            f"parity fixtures require gymnasium=={EXPECTED_GYMNASIUM_VERSION}; "
            f"found gymnasium=={gym.__version__}"
        )
    if box2d and getattr(Box2D, "__version__", None) != EXPECTED_PYBOX2D_VERSION:
        raise RuntimeError(
            f"Box2D parity fixtures require box2d-py=={EXPECTED_PYBOX2D_VERSION}; "
            f"found box2d-py=={getattr(Box2D, '__version__', 'unknown')}"
        )
    return gym, np, Box2D


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"generated {path.relative_to(CRATE_ROOT)}")


def generate_discrete_rollout(
    name: str,
    gymnasium_id: str,
    *,
    reset: Callable | None = None,
    extract_info: Callable | None = None,
) -> None:
    gym, _, _ = reference_modules(box2d=name == "lunar_lander")
    target = ROOT / name
    actions = json.loads((target / "inputs.json").read_text(encoding="utf-8"))
    environment = gym.make(gymnasium_id)

    def reset_environment():
        if reset is not None:
            return reset(environment)
        return environment.reset(seed=123, options={"low": 0.0, "high": 0.0})

    outputs = []
    try:
        reset_environment()
        for action in actions:
            observation, reward, terminated, truncated, info = environment.step(action)
            output = {
                "observation": observation.tolist(),
                "reward": float(reward),
                "done": bool(terminated),
                "truncated": bool(truncated),
            }
            if extract_info is not None:
                output["info"] = extract_info(environment, info, observation)
            outputs.append(output)
            if terminated or truncated:
                reset_environment()
    finally:
        environment.close()

    write_json(target / "output.json", outputs)


def generate_cartpole() -> None:
    generate_discrete_rollout("cartpole", "CartPole-v1")


def generate_mountain_car() -> None:
    generate_discrete_rollout("mountain_car", "MountainCar-v0")


def generate_pendulum() -> None:
    gym, np, _ = reference_modules()
    environment = gym.make("Pendulum-v1").unwrapped
    transitions = []
    try:
        for index in range(24):
            state = np.array(
                [np.sin(index * 0.47) * np.pi, np.cos(index * 0.31) * 7.5],
                dtype=np.float64,
            )
            action = np.array([2.5 * np.sin(index * 0.73)], dtype=np.float32)
            environment.state = state.copy()
            observation, reward, _, _, _ = environment.step(action)
            transitions.append(
                {
                    "state": state.tolist(),
                    "action": float(action[0]),
                    "observation": observation.tolist(),
                    "reward": float(reward),
                }
            )
    finally:
        environment.close()
    write_json(ROOT / "pendulum" / "trajectory.json", transitions)


def generate_acrobot() -> None:
    gym, np, _ = reference_modules()
    environment = gym.make("Acrobot-v1").unwrapped
    transitions = []
    try:
        for index in range(30):
            state = np.array(
                [
                    np.sin(index * 0.29) * 2.8,
                    np.cos(index * 0.37) * 2.6,
                    np.sin(index * 0.41) * 11.0,
                    np.cos(index * 0.23) * 24.0,
                ],
                dtype=np.float64,
            )
            action = index % 3
            environment.state = state.copy()
            observation, reward, terminated, _, _ = environment.step(action)
            transitions.append(
                {
                    "state": state.tolist(),
                    "action": action,
                    "observation": observation.tolist(),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                }
            )
    finally:
        environment.close()
    write_json(ROOT / "acrobot" / "trajectory.json", transitions)


VIEWPORT_W = 600.0
VIEWPORT_H = 400.0
SCALE = 30.0
LUNAR_LANDER_INITIAL_STATE = (
    VIEWPORT_W / SCALE / 2.0,
    VIEWPORT_H / SCALE * 0.8,
    0.0,
    -1.0,
    0.0,
    0.0,
    0.0,
    0.0,
)


class ZeroDispersionGenerator:
    """Preserve RNG APIs while fixing LunarLander engine dispersion."""

    def __init__(self, generator, numpy):
        self.generator = generator
        self.numpy = numpy

    def uniform(self, low=0.0, high=1.0, size=None):
        midpoint = (low + high) / 2.0
        if size is None:
            return float(midpoint)
        return self.numpy.full(size, midpoint, dtype=self.numpy.float64)

    def __getattr__(self, name):
        return getattr(self.generator, name)


def reset_lunar_lander(environment):
    _, info = environment.reset(seed=42, options={})
    unwrapped = environment.unwrapped
    _, np, _ = reference_modules(box2d=True)
    unwrapped.np_random = ZeroDispersionGenerator(unwrapped.np_random, np)
    unwrapped.helipad_y = (VIEWPORT_H / SCALE) / 4.0

    initial = LUNAR_LANDER_INITIAL_STATE
    lander = unwrapped.lander
    lander.position = (initial[0], initial[1])
    lander.angle = initial[4]
    lander.linearVelocity = (initial[2], initial[3])
    lander.angularVelocity = initial[5]
    lander.linearDamping = 0.0
    lander.angularDamping = 0.0
    lander.awake = True

    leg_away = 20.0 / SCALE
    for index, leg in enumerate(unwrapped.legs):
        direction = -1.0 if index == 0 else 1.0
        leg.position = (initial[0] - direction * leg_away, initial[1])
        leg.angle = initial[4] + direction * 0.05
        leg.linearVelocity = (initial[2], initial[3])
        leg.angularVelocity = initial[5]

    if hasattr(unwrapped, "wind_idx"):
        unwrapped.wind_idx = 0
    if hasattr(unwrapped, "torque_idx"):
        unwrapped.torque_idx = 0
    unwrapped.game_over = False
    unwrapped.prev_shaping = None

    leg_down = 18.0 / SCALE
    observation = np.array(
        [
            (initial[0] - VIEWPORT_W / SCALE / 2.0) / (VIEWPORT_W / SCALE / 2.0),
            (initial[1] - (unwrapped.helipad_y + leg_down))
            / (VIEWPORT_H / SCALE / 2.0),
            initial[2] * (VIEWPORT_W / SCALE / 2.0) / 50.0,
            initial[3] * (VIEWPORT_H / SCALE / 2.0) / 50.0,
            initial[4],
            20.0 * initial[5] / 50.0,
            initial[6],
            initial[7],
        ],
        dtype=np.float32,
    )
    return observation, info


def lunar_lander_info(environment, info, observation):
    unwrapped = environment.unwrapped
    lander = unwrapped.lander
    result = dict(info)
    result.update(
        {
            "raw_lander_pos_x": float(lander.position[0]),
            "raw_lander_pos_y": float(lander.position[1]),
            "raw_lander_angle": float(lander.angle),
            "raw_lander_vel_x": float(lander.linearVelocity[0]),
            "raw_lander_vel_y": float(lander.linearVelocity[1]),
            "raw_lander_angular_vel": float(lander.angularVelocity),
            "lander_awake": bool(lander.awake),
        }
    )
    for index, leg in enumerate(unwrapped.legs):
        result.update(
            {
                f"raw_leg{index}_pos_x": float(leg.position[0]),
                f"raw_leg{index}_pos_y": float(leg.position[1]),
                f"raw_leg{index}_angle": float(leg.angle),
                f"raw_leg{index}_vel_x": float(leg.linearVelocity[0]),
                f"raw_leg{index}_vel_y": float(leg.linearVelocity[1]),
                f"raw_leg{index}_angular_vel": float(leg.angularVelocity),
            }
        )
    result.update(
        {
            "leg0_contact": float(observation[6]),
            "leg1_contact": float(observation[7]),
            "helipad_y": float(unwrapped.helipad_y),
            "helipad_x1": float(unwrapped.helipad_x1),
            "helipad_x2": float(unwrapped.helipad_x2),
        }
    )
    if hasattr(unwrapped, "wind_idx"):
        result["wind_idx"] = int(unwrapped.wind_idx)
    if hasattr(unwrapped, "torque_idx"):
        result["torque_idx"] = int(unwrapped.torque_idx)
    result.update(
        {
            "game_over": bool(unwrapped.game_over),
            "prev_shaping": (
                float(unwrapped.prev_shaping)
                if unwrapped.prev_shaping is not None
                else None
            ),
            "VIEWPORT_W": VIEWPORT_W,
            "VIEWPORT_H": VIEWPORT_H,
            "SCALE": SCALE,
        }
    )
    return result


def generate_lunar_lander() -> None:
    generate_discrete_rollout(
        "lunar_lander",
        "LunarLander-v3",
        reset=reset_lunar_lander,
        extract_info=lunar_lander_info,
    )


class DeterministicRng:
    def __init__(self, numpy):
        self.numpy = numpy

    def uniform(self, low=0.0, high=1.0, size=None):
        value = (float(low) + float(high)) / 2.0
        return self.numpy.full(size, value) if size is not None else value

    def integers(self, low, high=None, size=None):
        value = int(low)
        if size is None:
            return value
        return self.numpy.full(size, value, dtype=self.numpy.int64)

    def random(self, size=None):
        return self.numpy.zeros(size) if size is not None else 0.0


def generate_bipedal_walker() -> None:
    gym, np, box2d = reference_modules(box2d=True)
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
        try:
            environment.np_random = DeterministicRng(np)
            reset_observation, _ = environment.reset()
            if initial_observation is None:
                initial_observation = reset_observation
            observation, reward, done, _, _ = environment.step(action)
            observations.append(observation.tolist())
            rewards.append(float(reward))
            terminated.append(bool(done))
        finally:
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
    try:
        environment.np_random = DeterministicRng(np)
        environment.reset()
        for action in sequential_actions:
            observation, reward, done, _, _ = environment.step(action)
            sequential_observations.append(observation.tolist())
            sequential_rewards.append(float(reward))
            sequential_terminated.append(bool(done))
    finally:
        environment.close()

    write_json(
        ROOT / "bipedal_walker" / "trajectory.json",
        {
            "generator": {
                "python": platform.python_version(),
                "gymnasium": gym.__version__,
                "pybox2d": box2d.__version__,
            },
            "initial_observation": initial_observation.tolist(),
            "actions": [action.tolist() for action in transition_actions],
            "observations": observations,
            "rewards": rewards,
            "terminated": terminated,
            "sequential_actions": [action.tolist() for action in sequential_actions],
            "sequential_observations": sequential_observations,
            "sequential_rewards": sequential_rewards,
            "sequential_terminated": sequential_terminated,
        },
    )


@dataclass(frozen=True)
class ParityCase:
    gymnasium_id: str
    generate: Callable[[], None]
    rust_filter: str


ENVIRONMENTS = {
    "acrobot": ParityCase(
        "Acrobot-v1", generate_acrobot, "classic_control::acrobot::tests::parity"
    ),
    "bipedal_walker": ParityCase(
        "BipedalWalker-v3",
        generate_bipedal_walker,
        "box_2d::bipedal_walker::tests::parity",
    ),
    "cartpole": ParityCase(
        "CartPole-v1", generate_cartpole, "classic_control::cartpole::tests::parity"
    ),
    "lunar_lander": ParityCase(
        "LunarLander-v3",
        generate_lunar_lander,
        "box_2d::lunar_lander::tests::parity",
    ),
    "mountain_car": ParityCase(
        "MountainCar-v0",
        generate_mountain_car,
        "classic_control::mountain_car::tests::parity",
    ),
    "pendulum": ParityCase(
        "Pendulum-v1", generate_pendulum, "classic_control::pendulum::tests::parity"
    ),
}


def run_rust_tests(test_filter: str) -> None:
    command = [
        "cargo",
        "test",
        "--locked",
        "-p",
        "modurl_gym",
        "--lib",
        test_filter,
    ]
    print(f"running {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def parse_args(arguments: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Gymnasium fixtures and run focused Rust parity tests."
    )
    parser.add_argument(
        "environment",
        nargs="?",
        choices=["all", *ENVIRONMENTS],
        default="all",
        help="environment to test (default: all)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--rust-only",
        action="store_true",
        help="test existing fixtures without importing Gymnasium or Box2D",
    )
    mode.add_argument(
        "--generate-only",
        action="store_true",
        help="regenerate fixtures without running Rust tests",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list environment names and exit",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    if args.list:
        for name, case in ENVIRONMENTS.items():
            print(f"{name:<16} {case.gymnasium_id}")
        return 0

    all_environments = args.environment == "all"
    names = list(ENVIRONMENTS) if all_environments else [args.environment]
    if not args.rust_only:
        for name in names:
            ENVIRONMENTS[name].generate()
    if not args.generate_only:
        test_filter = (
            "::tests::parity"
            if all_environments
            else ENVIRONMENTS[args.environment].rust_filter
        )
        run_rust_tests(test_filter)
    return 0


def cli(arguments: Sequence[str] | None = None) -> int:
    try:
        return main(arguments)
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())

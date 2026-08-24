"""Generate fixtures and run focused Gymnasium parity tests.

Pass one environment name to test it, or omit the name to test every registered
environment. Rust tests consume the generated JSON and do not invoke Python.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent
CRATE_ROOT = ROOT.parent
REPOSITORY_ROOT = CRATE_ROOT.parents[1]
EXPECTED_GYMNASIUM_VERSION = "1.2.1"
EXPECTED_MUJOCO_VERSION = "3.9.0"
PARITY_STEPS = 64


@dataclass(frozen=True)
class ParityCase:
    gymnasium_id: str


ENVIRONMENTS = {
    "ant": ParityCase("Ant-v5"),
    "half_cheetah": ParityCase("HalfCheetah-v5"),
    "hopper": ParityCase("Hopper-v5"),
    "humanoid": ParityCase("Humanoid-v5"),
    "walker2d": ParityCase("Walker2d-v5"),
}


def require_reference_versions(gymnasium_version: str, mujoco_version: str) -> None:
    found = (gymnasium_version, mujoco_version)
    expected = (EXPECTED_GYMNASIUM_VERSION, EXPECTED_MUJOCO_VERSION)
    if found != expected:
        raise RuntimeError(
            "parity fixtures require "
            f"gymnasium=={EXPECTED_GYMNASIUM_VERSION} and "
            f"mujoco=={EXPECTED_MUJOCO_VERSION}; found "
            f"gymnasium=={gymnasium_version} and mujoco=={mujoco_version}"
        )


def generate(name: str, case: ParityCase) -> Path:
    try:
        import gymnasium as gym
        import mujoco
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "install the reference dependencies first: "
            f"python -m pip install gymnasium=={EXPECTED_GYMNASIUM_VERSION} "
            f"mujoco=={EXPECTED_MUJOCO_VERSION}"
        ) from error

    require_reference_versions(gym.__version__, mujoco.__version__)
    env = gym.make(case.gymnasium_id).unwrapped
    try:
        env.reset(seed=7)

        # Use the model's exact reference configuration and zero velocity. This
        # bypasses reset RNG differences and isolates model/physics/reward parity.
        qpos = env.init_qpos.copy()
        qvel = np.zeros(env.model.nv, dtype=np.float64)
        env.set_state(qpos, qvel)

        actions = []
        states = []
        for step in range(PARITY_STEPS):
            states.append(
                {
                    "qpos": env.data.qpos.copy().tolist(),
                    "qvel": env.data.qvel.copy().tolist(),
                }
            )
            indices = np.arange(env.model.nu, dtype=np.float64)
            action = (0.35 * np.sin(0.37 * step + 0.61 * indices)).astype(
                np.float32
            )
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

        target = ROOT / name / "trajectory.json"
        target.write_text(
            json.dumps(
                {
                    "gymnasium_version": gym.__version__,
                    "mujoco_version": mujoco.__version__,
                    "environment_id": case.gymnasium_id,
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
            newline="\n",
        )
        print(f"generated {target.relative_to(CRATE_ROOT)}")
        return target
    finally:
        env.close()


def run_rust_tests(name: str | None) -> None:
    command = [
        "cargo",
        "test",
        "--locked",
        "-p",
        "modurl_mujoco",
        "--test",
        "parity",
    ]
    if name is not None:
        command.extend([name, "--", "--exact"])
    print(f"running {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def parse_args(arguments: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Gymnasium fixtures and run focused Rust parity tests."
    )
    parser.add_argument(
        "environment",
        nargs="?",
        choices=["all", *sorted(ENVIRONMENTS)],
        default="all",
        help="environment to test (default: all)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--rust-only",
        action="store_true",
        help="test the existing fixture without importing Gymnasium or MuJoCo",
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
            print(f"{name:<13} {case.gymnasium_id:<18} {PARITY_STEPS} transitions")
        return 0

    all_environments = args.environment == "all"
    names = list(ENVIRONMENTS) if all_environments else [args.environment]
    if not args.rust_only:
        for name in names:
            generate(name, ENVIRONMENTS[name])
    if not args.generate_only:
        run_rust_tests(None if all_environments else args.environment)
    return 0


def cli(arguments: Sequence[str] | None = None) -> int:
    try:
        return main(arguments)
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())

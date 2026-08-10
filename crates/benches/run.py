#!/usr/bin/env python3
"""Run a matched Rust/Python RL benchmark and save raw samples as JSON."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BENCH_DIR = Path(__file__).resolve().parent
WORKSPACE = BENCH_DIR.parents[1]
PYTHON_BENCH = BENCH_DIR / "python" / "benchmark.py"
FRAMEWORKS = ("modurl", "sb3", "cleanrl", "tianshou")
RESULT_PREFIX = "BENCH_RESULT="


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=("ppo", "dqn", "sac"), default="ppo")
    parser.add_argument("--device", choices=("cpu", "cuda", "metal"), default="cpu")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--framework",
        action="append",
        choices=FRAMEWORKS,
        dest="frameworks",
        help="Run only this framework; repeat the flag to select several.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python executable containing the pinned benchmark packages.",
    )
    args = parser.parse_args()
    default_steps, default_warmup = {
        "ppo": (20_480, 2_048),
        "dqn": (4_096, 1_024),
        "sac": (1_024, 256),
    }[args.algorithm]
    args.steps = args.steps or default_steps
    args.warmup_steps = args.warmup_steps or default_warmup
    return args


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def git_revision() -> str:
    safe_workspace = WORKSPACE.as_posix()
    environment = os.environ.copy()
    environment["XDG_CONFIG_HOME"] = str(WORKSPACE / "target" / "bench-git-config")
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={safe_workspace}", "rev-parse", "HEAD"],
        cwd=WORKSPACE,
        env=environment,
        text=True,
    ).strip()


def command_for(framework: str, args: argparse.Namespace) -> list[str]:
    common = [
        "--algorithm",
        args.algorithm,
        "--device",
        args.device,
        "--steps",
        str(args.steps),
        "--warmup-steps",
        str(args.warmup_steps),
        "--threads",
        str(args.threads),
    ]
    if framework == "modurl":
        command = ["cargo", "run", "--release", "-p", "modurl-benches"]
        if args.device != "cpu":
            command.extend(["--features", args.device])
        command.extend(["--", *common])
        return command
    return [
        str(args.python),
        str(PYTHON_BENCH),
        "--framework",
        framework,
        *common,
    ]


def run_one(framework: str, args: argparse.Namespace) -> dict[str, Any]:
    command = command_for(framework, args)
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "42"
    environment["RAYON_NUM_THREADS"] = str(args.threads)
    completed = subprocess.run(
        command,
        cwd=WORKSPACE,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(completed.stdout, end="")
    if completed.returncode:
        raise subprocess.CalledProcessError(
            completed.returncode, command, output=completed.stdout
        )
    result_lines = [
        line[len(RESULT_PREFIX) :]
        for line in completed.stdout.splitlines()
        if line.startswith(RESULT_PREFIX)
    ]
    if len(result_lines) != 1:
        raise RuntimeError(f"{framework} emitted {len(result_lines)} benchmark results")
    return json.loads(result_lines[0])


def summarize(samples: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_framework: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        by_framework.setdefault(sample["framework"], []).append(sample)
    summary: dict[str, dict[str, float]] = {}
    for framework, rows in by_framework.items():
        times = [float(row["elapsed_seconds"]) for row in rows]
        median_seconds = statistics.median(times)
        summary[framework] = {
            "median_seconds": median_seconds,
            "min_seconds": min(times),
            "max_seconds": max(times),
            "median_steps_per_second": rows[0]["measured_steps"] / median_seconds,
        }
    if "modurl" in summary:
        modurl_time = summary["modurl"]["median_seconds"]
        for values in summary.values():
            values["modurl_speedup"] = values["median_seconds"] / modurl_time
    return summary


def print_markdown(algorithm: str, summary: dict[str, dict[str, float]]) -> None:
    print(f"\n{algorithm.upper()} results")
    print("\n| Framework | Median time | Transitions/s | ModuRL speedup |")
    print("| --- | ---: | ---: | ---: |")
    for framework in FRAMEWORKS:
        if framework not in summary:
            continue
        row = summary[framework]
        speedup = row.get("modurl_speedup")
        speedup_text = f"{speedup:.2f}×" if speedup is not None else "n/a"
        print(
            f"| {framework} | {row['median_seconds']:.3f} s | "
            f"{row['median_steps_per_second']:,.0f} | {speedup_text} |"
        )


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.steps < 1 or args.warmup_steps < 1:
        raise ValueError("--steps and --warmup-steps must be at least 1")
    required_warmup = {"ppo": 2_048, "dqn": 1_024, "sac": 256}[args.algorithm]
    if args.warmup_steps != required_warmup:
        raise ValueError(
            f"{args.algorithm.upper()} requires --warmup-steps {required_warmup}"
        )
    if args.algorithm == "ppo" and args.steps % 2_048:
        raise ValueError("PPO --steps must be a multiple of 2048")
    if args.algorithm == "dqn" and args.steps % 4:
        raise ValueError("DQN --steps must be a multiple of 4")
    frameworks = list(dict.fromkeys(args.frameworks or FRAMEWORKS))
    samples: list[dict[str, Any]] = []
    order_rng = random.Random(42)
    for repeat in range(args.repeats):
        order = frameworks.copy()
        order_rng.shuffle(order)
        for framework in order:
            print(f"\n[{repeat + 1}/{args.repeats}] {framework} ({args.device})", flush=True)
            sample = run_one(framework, args)
            sample["repeat"] = repeat + 1
            samples.append(sample)

    configs = {json.dumps(sample["config"], sort_keys=True) for sample in samples}
    if len(configs) != 1:
        raise RuntimeError("runners reported different benchmark configurations")
    summary = summarize(samples)
    framework_metadata: dict[str, dict[str, str]] = {}
    raw_samples: dict[str, list[dict[str, float | int]]] = {}
    for sample in samples:
        framework = sample["framework"]
        framework_metadata[framework] = {
            "version": sample["framework_version"],
            "backend": sample["backend"],
        }
        raw_samples.setdefault(framework, []).append(
            {
                "repeat": sample["repeat"],
                "elapsed_seconds": sample["elapsed_seconds"],
                "steps_per_second": sample["steps_per_second"],
            }
        )
    for rows in raw_samples.values():
        rows.sort(key=lambda row: row["repeat"])
    generated_at = datetime.now(timezone.utc)
    report = {
        "schema_version": 2,
        "generated_at": generated_at.isoformat(),
        "git_revision": git_revision(),
        "dirty_worktree": bool(
            subprocess.check_output(
                [
                    "git",
                    "-c",
                    f"safe.directory={WORKSPACE.as_posix()}",
                    "status",
                    "--porcelain",
                ],
                cwd=WORKSPACE,
                env={
                    **os.environ,
                    "XDG_CONFIG_HOME": str(WORKSPACE / "target" / "bench-git-config"),
                },
                text=True,
            ).strip()
        ),
        "system": {
            "platform": platform.platform(),
            "cpu": cpu_name(),
            "logical_cpus": os.cpu_count(),
            "python": platform.python_version(),
        },
        "algorithm": args.algorithm,
        "device": args.device,
        "threads": args.threads,
        "repeats": args.repeats,
        "measured_steps_per_repeat": args.steps,
        "warmup_steps_per_repeat": args.warmup_steps,
        "config": samples[0]["config"],
        "frameworks": framework_metadata,
        "samples": raw_samples,
        "summary": summary,
    }
    output = args.output
    if output is None:
        output = BENCH_DIR / "results" / (
            f"{args.algorithm}-{args.device}-{platform.system().lower()}-{generated_at:%Y%m%d}.json"
        )
    if not output.is_absolute():
        output = WORKSPACE / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print_markdown(args.algorithm, summary)
    print(f"\nRaw results: {output}")


if __name__ == "__main__":
    main()

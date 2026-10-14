#!/usr/bin/env python3
"""Run a matched Rust/Python RL benchmark and save raw samples as JSON."""

from __future__ import annotations

import argparse
import hashlib
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
FRAMEWORKS = ("modurl", "sb3", "cleanrl")
RESULT_PREFIX = "BENCH_RESULT="


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=("ppo", "dqn", "all"), default="ppo")
    parser.add_argument("--rom", type=Path, required=True, help="Path to a locally supplied Breakout ROM.")
    parser.add_argument("--device", choices=("cpu", "cuda", "metal"), default="cuda")
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
        "ppo": (40_960, 4_096),
        "dqn": (4_096, 80_004),
        "all": (None, None),
    }[args.algorithm]
    args.steps = default_steps if args.steps is None else args.steps
    args.warmup_steps = default_warmup if args.warmup_steps is None else args.warmup_steps
    if not args.rom.is_file():
        parser.error(f"ROM file does not exist: {args.rom}")
    args.rom = args.rom.resolve()
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
        "--rom",
        str(args.rom),
    ]
    if framework == "modurl":
        command = ["cargo", "run", "--locked", "--release", "-p", "modurl-benches"]
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
    log_directory = WORKSPACE / "target" / "benchmark-logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    log_path = log_directory / f"{args.algorithm}-{framework}-{args.device}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        print(completed.stdout, end="")
        raise subprocess.CalledProcessError(
            completed.returncode, command, output=completed.stdout
        )
    result_lines = [
        line[len(RESULT_PREFIX) :]
        for line in completed.stdout.splitlines()
        if line.startswith(RESULT_PREFIX)
    ]
    if len(result_lines) != 1:
        raise RuntimeError(f"{framework} emitted {len(result_lines)} benchmark results; see {log_path}")
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


def run_algorithm(args: argparse.Namespace) -> None:
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.steps < 1 or args.warmup_steps < 1:
        raise ValueError("--steps and --warmup-steps must be at least 1")
    required_warmup = {"ppo": 4_096, "dqn": 80_004}[args.algorithm]
    if args.warmup_steps != required_warmup:
        raise ValueError(
            f"{args.algorithm.upper()} requires --warmup-steps {required_warmup}"
        )
    if args.algorithm == "ppo" and args.steps % 4_096:
        raise ValueError("PPO --steps must be a multiple of 4096")
    if args.algorithm == "dqn" and args.steps % 4:
        raise ValueError("DQN --steps must be a multiple of 4")
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")
    frameworks = list(dict.fromkeys(args.frameworks or FRAMEWORKS))
    samples: list[dict[str, Any]] = []
    order_rng = random.Random(42)
    for repeat in range(args.repeats):
        order = frameworks.copy()
        order_rng.shuffle(order)
        for framework in order:
            print(f"\n[{repeat + 1}/{args.repeats}] {framework} ({args.device})", flush=True)
            sample = run_one(framework, args)
            expected = json.loads((BENCH_DIR / "workloads.json").read_text())[args.algorithm]
            if sample["config"] != expected:
                raise RuntimeError(f"{framework} reported a different workload configuration")
            if sample["measured_steps"] != args.steps or sample["warmup_steps"] != args.warmup_steps:
                raise RuntimeError(f"{framework} reported different collection counts")
            expected_updates = args.steps // 4096 * 16 if args.algorithm == "ppo" else args.steps // 4
            if sample["optimizer_steps"] != expected_updates:
                raise RuntimeError(f"{framework} reported different optimizer counts")
            sample["repeat"] = repeat + 1
            samples.append(sample)

    configs = {json.dumps(sample["config"], sort_keys=True) for sample in samples}
    if len(configs) != 1:
        raise RuntimeError("runners reported different benchmark configurations")
    summary = summarize(samples)
    framework_metadata: dict[str, dict[str, str | int]] = {}
    raw_samples: dict[str, list[dict[str, float | int]]] = {}
    for sample in samples:
        framework = sample["framework"]
        framework_metadata[framework] = {
            "version": sample["framework_version"],
            "backend": sample["backend"],
            "worker_kind": sample["worker_kind"],
            "environment_workers": sample["environment_workers"],
            "ale_version": sample.get("ale_version", "0.6 (bundled)"),
            "fusion": sample["fusion"],
        }
        raw_samples.setdefault(framework, []).append(
            {
                "repeat": sample["repeat"],
                "elapsed_seconds": sample["elapsed_seconds"],
                "steps_per_second": sample["steps_per_second"],
                "optimizer_steps": sample["optimizer_steps"],
            }
        )
    for rows in raw_samples.values():
        rows.sort(key=lambda row: row["repeat"])
    generated_at = datetime.now(timezone.utc)
    report = {
        "schema_version": 3,
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
        "environment": "Breakout",
        "rom_sha256": hashlib.sha256(args.rom.read_bytes()).hexdigest(),
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
            f"{args.algorithm}-breakout-{args.device}-{platform.system().lower()}-{generated_at:%Y%m%d}.json"
        )
    if not output.is_absolute():
        output = WORKSPACE / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print_markdown(args.algorithm, summary)
    for framework, metadata in framework_metadata.items():
        print(f"{framework} fusion: {metadata['fusion']}")
    print(f"\nRaw results: {output}")


def main() -> None:
    args = parse_args()
    algorithms = ("ppo", "dqn") if args.algorithm == "all" else (args.algorithm,)
    if len(algorithms) > 1 and args.output is not None:
        raise ValueError("--output requires one algorithm; omit it for separate dated reports")
    for algorithm in algorithms:
        selected = argparse.Namespace(**{**vars(args), "algorithm": algorithm})
        default_steps, default_warmup = {"ppo": (40960, 4096), "dqn": (4096, 80004)}[algorithm]
        selected.steps = default_steps if selected.steps is None else selected.steps
        selected.warmup_steps = default_warmup if selected.warmup_steps is None else selected.warmup_steps
        run_algorithm(selected)


if __name__ == "__main__":
    main()

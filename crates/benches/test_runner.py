import argparse
import sys
import unittest
from unittest import mock
from pathlib import Path

import run


class BackendCommandTests(unittest.TestCase):
    def test_default_ppo_warmup_is_ten_rollouts_and_explicit_override_still_works(self):
        arguments = ["run.py", "--rom", str(Path(__file__))]
        with mock.patch.object(sys, "argv", arguments):
            self.assertEqual(run.parse_args().warmup_steps, 40960)
        with mock.patch.object(sys, "argv", arguments + ["--warmup-steps", "16384"]):
            self.assertEqual(run.parse_args().warmup_steps, 16384)

    def arguments(self, backend):
        return argparse.Namespace(
            algorithm="ppo", device="cuda", backend=backend, steps=40960,
            warmup_steps=4096, threads=1, rom=Path("breakout.bin"), python=Path("python"),
            trace_dir=None, trace_sync=False,
            repeats=1, frameworks=["modurl"],
        )

    def test_libtorch_selects_its_feature_and_preserves_the_cuda_workload(self):
        command = run.command_for("modurl", self.arguments("libtorch"))
        self.assertEqual(command[command.index("--features") + 1], "libtorch")
        self.assertEqual(command[command.index("--backend") + 1], "libtorch")
        self.assertEqual(command[command.index("--device") + 1], "cuda")
        self.assertEqual(command[command.index("--steps") + 1], "40960")

    def test_native_cuda_keeps_its_feature(self):
        command = run.command_for("modurl", self.arguments("native"))
        self.assertEqual(command[command.index("--features") + 1], "cuda")

    def test_trace_enables_instrumentation_and_records_synchronization_mode(self):
        args = self.arguments("libtorch")
        args.trace_dir = Path("target/traces")
        args.trace_sync = True
        command = run.command_for("modurl", args)
        self.assertEqual(command[command.index("--features") + 1], "libtorch,tracing")
        self.assertEqual(command[command.index("--trace-sync") + 1], "true")
        self.assertEqual(Path(command[command.index("--trace") + 1]).parent, args.trace_dir)

    def test_python_commands_do_not_change_with_the_modurl_backend(self):
        for framework in ("sb3", "cleanrl"):
            self.assertEqual(
                run.command_for(framework, self.arguments("libtorch")),
                run.command_for(framework, self.arguments("native")),
            )

    def test_longer_ppo_warmup_reaches_the_runner(self):
        args = self.arguments("native")
        args.steps = 4096
        args.warmup_steps = 16384
        with mock.patch.object(run, "run_one", side_effect=RuntimeError("validated warmup")) as runner:
            with self.assertRaisesRegex(RuntimeError, "validated warmup"):
                run.run_algorithm(args)
            runner.assert_called_once_with("modurl", args)

    def test_ppo_warmup_rejects_incomplete_rollouts_and_zero_steps(self):
        for warmup in (0, 16000):
            with self.subTest(warmup=warmup):
                args = self.arguments("native")
                args.warmup_steps = warmup
                with self.assertRaises(ValueError):
                    run.run_algorithm(args)


if __name__ == "__main__":
    unittest.main()

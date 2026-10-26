import argparse
import sys
import unittest
from unittest import mock
from pathlib import Path

import run


class BackendCommandTests(unittest.TestCase):
    def test_warmup_is_fixed_per_algorithm(self):
        arguments = ["run.py", "--rom", str(Path(__file__))]
        with mock.patch.object(sys, "argv", arguments):
            self.assertEqual(run.parse_args().warmup_steps, 40960)
        with mock.patch.object(sys, "argv", arguments + ["--algorithm", "dqn"]):
            self.assertEqual(run.parse_args().warmup_steps, 80004)

    def test_warmup_override_is_rejected(self):
        arguments = ["run.py", "--rom", str(Path(__file__)), "--warmup" + "-steps", "16384"]
        with mock.patch.object(sys, "argv", arguments):
            with self.assertRaises(SystemExit):
                run.parse_args()

    def arguments(self, backend):
        return argparse.Namespace(
            algorithm="ppo", device="cuda", backend=backend, steps=40960,
            warmup_steps=40960, threads=1, rom=Path("breakout.bin"), python=Path("python"),
            trace_dir=None, trace_sync=False,
            repeats=1, frameworks=["modurl"], output=None,
        )

    def test_all_runs_use_each_algorithms_fixed_warmup(self):
        args = self.arguments("native")
        args.algorithm = "all"
        args.steps = None
        args.warmup_steps = None
        with mock.patch.object(run, "parse_args", return_value=args):
            with mock.patch.object(run, "run_algorithm") as runner:
                run.main()
        self.assertEqual(
            [(call.args[0].algorithm, call.args[0].steps, call.args[0].warmup_steps)
             for call in runner.call_args_list],
            [("ppo", 40960, 40960), ("dqn", 4096, 80004)],
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
            self.assertNotIn("--warmup" + "-steps", run.command_for(framework, self.arguments("native")))
        self.assertNotIn("--warmup" + "-steps", run.command_for("modurl", self.arguments("native")))

    def test_nonpositive_steps_are_rejected(self):
        for steps in (0, -1):
            with self.subTest(steps=steps):
                args = self.arguments("native")
                args.steps = steps
                with self.assertRaisesRegex(ValueError, "--steps must be at least 1"):
                    run.run_algorithm(args)


if __name__ == "__main__":
    unittest.main()

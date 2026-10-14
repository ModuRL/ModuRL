import argparse
import unittest
from pathlib import Path

import run


class BackendCommandTests(unittest.TestCase):
    def arguments(self, backend):
        return argparse.Namespace(
            algorithm="ppo", device="cuda", backend=backend, steps=40960,
            warmup_steps=4096, threads=1, rom=Path("breakout.bin"), python=Path("python"),
            trace_dir=None, trace_sync=False,
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


if __name__ == "__main__":
    unittest.main()

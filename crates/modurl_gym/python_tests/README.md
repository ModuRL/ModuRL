# Gymnasium parity fixtures

From the `modurl_gym` crate directory, regenerate one environment's fixtures
and run only its Rust parity tests:

```powershell
python python_tests/parity.py cartpole
```

Omit the environment to regenerate and test every registered environment:

```powershell
python python_tests/parity.py
```

Use `--rust-only` to test committed fixtures without Python dependencies, or
`--generate-only` to refresh fixtures without running Cargo. List supported
environment names with `python python_tests/parity.py --list`.

Every environment uses the same public baseline of 64 reference transitions.
The Python runner generates both the actions and expected outputs, and the Rust
tests reject fixtures with any other length. BipedalWalker also has a separate
8-step uninterrupted rollout check for short-horizon Box2D solver drift.

| Environment | Baseline | Reference setup | Additional comparison |
| --- | ---: | --- | --- |
| Acrobot | 64 transitions | Independent deterministic state/action probes | None |
| BipedalWalker | 64 transitions | Deterministic flat reset with a varied in-space action per probe | 8-step uninterrupted rollout |
| CartPole | 64 transitions | Seeded, balanced actions with the previous reference observation restored before each comparison | None |
| LunarLander | 64 transitions | Seeded, balanced actions with the previous reference physics state restored before each comparison | The same 64 transitions are also replayed uninterrupted |
| MountainCar | 64 transitions | Seeded, balanced actions with the previous reference observation restored before each comparison | None |
| Pendulum | 64 transitions | Independent deterministic state/action probes with in-space actions | None |

The equal baseline is the generated transition count, not a claim that every
physics engine can use the same state setup or numerical tolerance.

Fixture generation requires Gymnasium 1.2.1. LunarLander and BipedalWalker also
require box2d-py 2.3.5; install the matching extras with:

```powershell
python -m pip install "gymnasium[box2d]==1.2.1"
```

Ordinary Rust tests read the committed JSON fixtures and never invoke Python.

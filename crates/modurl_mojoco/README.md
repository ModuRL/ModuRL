# modurl_mojoco

Native Rust MuJoCo environments for [ModuRL](https://github.com/ModuRL/ModuRL), using Candle tensors and [`mujoco-rs`](https://github.com/davidhozic/mujoco-rs) physics.

Implemented Gymnasium v5 environments:

- `AntV5` — observation `(105,)`, action `(8,)`
- `HalfCheetahV5` — observation `(17,)`, action `(6,)`
- `HopperV5` — observation `(11,)`, action `(3,)`
- `HumanoidV5` — observation `(348,)`, action `(17,)`
- `Walker2dV5` — observation `(17,)`, action `(6,)`

## Installation

MuJoCo 3.9 is automatically downloaded by `mujoco-rs` on Windows and Linux. A clone of this repository works without extra configuration because [`.cargo/config.toml`](../../.cargo/config.toml) supplies a project-local cache. On Windows, this crate's build script also copies `mujoco.dll` next to Cargo-built executables and tests.

When adding `modurl_mojoco` as a dependency, Cargo does not read configuration files from dependencies. Add this to the consuming project's `.cargo/config.toml`:

```toml
[env]
MUJOCO_DOWNLOAD_DIR = { value = "target/mujoco", relative = true, force = false }
```

That limitation comes from `mujoco-rs`, which requires an absolute `MUJOCO_DOWNLOAD_DIR` while its dependency build script is running. macOS requires a manual MuJoCo installation; see the [`mujoco-rs` installation guide](https://mujoco-rs.readthedocs.io/en/latest/installation.html).

## Usage

```rust
use candle_core::Device;
use modurl_mojoco::prelude::*;

let mut environment = HalfCheetahV5::builder().device(&Device::Cpu).build()?;
let observation = environment.reset()?.state;
let action = environment.action_space().sample(&Device::Cpu)?;
let transition = environment.step(action)?;
# Ok::<(), MujocoError>(())
```

Every builder defaults to the corresponding Gymnasium v5 configuration. The
behavioral parameters can be overridden without constructing a separate config
object:

```rust
let environment = HopperV5::builder()
    .frame_skip(4)
    .forward_reward_weight(1.0)
    .ctrl_cost_weight(1e-3)
    .healthy_reward(1.0)
    .terminate_when_unhealthy(false)
    .healthy_z_range((0.6, 2.2))
    .reset_noise_scale(0.005)
    .exclude_current_positions_from_observation(false)
    .build()?;
# Ok::<(), MujocoError>(())
```

The 1,000-step Gymnasium time limit is intentionally not built into the five
Gymnasium-compatible environments; apply it in an environment wrapper.

## Rendering

Enable the optional interactive MuJoCo viewer in your dependency:

```toml
modurl_mojoco = { version = "0.1", features = ["rendering"] }
```

Then opt an environment into rendering through its builder:

```rust
let mut environment = HalfCheetahV5::builder().render(true).build()?;
environment.reset()?;
# Ok::<(), MujocoError>(())
```

The same `.render(true)` option is available on `AntV5`, `HopperV5`,
`HumanoidV5`, and `Walker2dV5`.
The viewer updates after resets, exact state changes, and simulation steps. Closing
the window stops rendering while leaving the environment usable. Interactive
viewers should be created on the application's main thread. Without the
`rendering` feature, the viewer dependencies and builder option are omitted.

## License

`modurl_mojoco` is MIT-licensed and does not depend on GPL-licensed ALE code.
MuJoCo is Apache-2.0, `mujoco-rs` is used under its MIT option, and the
Gymnasium-derived XML models are MIT. See
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) for the dependency audit
and binary-redistribution notices. When available, the build places MuJoCo's
license and third-party notice files beside Cargo output automatically.

## Parity tests

The Python parity runner regenerates fixtures and executes the matching Rust
tests. Pass one environment name to focus the entire workflow on it. For
example, from this crate's directory:

```powershell
python python_tests/parity.py ant
```

That command regenerates `ant/trajectory.json` with Gymnasium, then runs the
exact `ant` Rust parity test. To run the Rust test against the existing fixture
without requiring Python dependencies:

```powershell
python python_tests/parity.py ant --rust-only
```

List the supported names or run every environment by omitting the name:

```powershell
python python_tests/parity.py --list
python python_tests/parity.py
```

Install the pinned reference dependencies before regenerating fixtures:

```powershell
python -m pip install gymnasium==1.2.1 mujoco==3.9.0
```

Use `--generate-only` when you only want to refresh fixture JSON. Ordinary
`cargo test` uses the committed fixtures and therefore does not require Python
or Gymnasium.

Every environment uses the same public baseline of 64 reference transitions.
The runner prints that count with `--list`, and the Rust tests reject fixtures
with any other length.

Each fixture follows the same procedure: Gymnasium produces a deterministic
64-step reference trajectory, then every recorded state/action pair is compared
as a one-step transition from a clean solver state.

| Environment | Baseline | Observation tolerance | Reward tolerance | Additional check |
| --- | ---: | ---: | ---: | --- |
| Ant | 64 transitions | `1e-5` | `1e-5` | Contact-force observations must be exercised |
| HalfCheetah | 64 transitions | `1e-5` | `1e-5` | None |
| Hopper | 64 transitions | `1e-5` | `1e-5` | None |
| Humanoid | 64 transitions | `1e-5` | `1e-5` | Contact-force observations must be exercised |
| Walker2d | 64 transitions | `4e-3` positions; `6.6e-1` impact velocities | `2e-2` | Includes simultaneous foot impacts |

Observation bounds also include one `f32` conversion epsilon scaled by the
reference value. Walker2d keeps all impact transitions in the baseline; its
velocity-only allowance accounts for solver-order differences between the
official Python and `mujoco-rs` binary builds instead of shortening the fixture.

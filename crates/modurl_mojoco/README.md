# modurl_mojoco

Native Rust MuJoCo environments for [ModuRL](https://github.com/ModuRL/ModuRL), using Candle tensors and [`mujoco-rs`](https://github.com/davidhozic/mujoco-rs) physics.

Implemented Gymnasium v5 environments:

- `HalfCheetahV5` — observation `(17,)`, action `(6,)`
- `HopperV5` — observation `(11,)`, action `(3,)`
- `Walker2dV5` — observation `(17,)`, action `(6,)`

## Installation

MuJoCo 3.9 is automatically downloaded by `mujoco-rs` on Windows and Linux. A clone of this repository works without extra configuration because [`.cargo/config.toml`](.cargo/config.toml) supplies a project-local cache. On Windows, this crate's build script also copies `mujoco.dll` next to Cargo-built executables and tests.

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
# Ok::<(), Box<dyn std::error::Error>>(())
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
# Ok::<(), Box<dyn std::error::Error>>(())
```

The 1,000-step Gymnasium time limit is intentionally not built into the environments; apply it in an environment wrapper.

## Rendering

Enable the optional interactive MuJoCo viewer in your dependency:

```toml
modurl_mojoco = { version = "0.1", features = ["rendering"] }
```

Then opt an environment into rendering through its builder:

```rust
let mut environment = HalfCheetahV5::builder().render(true).build()?;
environment.reset()?;
# Ok::<(), Box<dyn std::error::Error>>(());
```

The same `.render(true)` option is available on `HopperV5` and `Walker2dV5`.
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

`python_tests/generate_parity.py` creates deterministic Gymnasium v5 fixtures using MuJoCo 3.9. Ordinary Rust tests use the committed JSON and therefore do not require Python:

```powershell
cargo test
```

Regenerate fixtures only with matching dependencies:

```powershell
python -m pip install gymnasium==1.2.1 mujoco==3.9.0
python python_tests/generate_parity.py
cargo test --test parity
```

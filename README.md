# ModuRL

[![Workspace CI](https://github.com/ModuRL/ModuRL/actions/workflows/rust.yml/badge.svg)](https://github.com/ModuRL/ModuRL/actions/workflows/rust.yml)

Composable deep reinforcement learning in Rust, built on
[Candle](https://github.com/huggingface/candle).

ModuRL provides training agents, environments, models, distributions, replay
buffers, parameter schedules, and logging without hiding how they fit together.
It is aimed at Rust developers who want control over tensor shapes, devices,
networks, and training configuration.

> [!WARNING]
> ModuRL is early in development. The APIs may change between revisions.

[Guide](https://modurl.github.io/ModuRL/dev/) ·
[Examples](crates/examples/examples) ·
[Issues](https://github.com/ModuRL/ModuRL/issues)

## At a Glance

Create an environment, reset it, sample a valid action, and take one step using
the same `Gym` and `Space` contracts that training agents use:

```rust
use candle_core::Device;
use modurl::prelude::*;
use modurl_gym::classic_control::cartpole::CartPoleV1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let mut env = CartPoleV1::builder().device(&device).build();

    let observation = env.reset()?.state;
    let action = env.action_space().sample(&device)?;
    let transition = env.step(action)?;

    println!("observation: {:?}", observation.to_vec1::<f32>()?);
    println!("reward: {}", transition.reward);
    Ok(())
}
```

Training agents apply these same traits to vectorized environments.

## Architectural Principles

ModuRL keeps algorithm behavior explicit and puts reusable responsibilities
behind small public contracts:

- **Agents coordinate training.** An `Agent` owns the algorithm loop and brings
  together models, optimizers, schedules, rollout or replay storage, action
  selection, and update logic. It learns against `MultiGym`, not a
  concrete environment type.
- **Single and batched environments are separate boundaries.** `Gym` describes
  one environment. `MultiGym` describes batched collection. The supplied
  synchronous and multithreaded wrappers auto-reset completed environments
  while preserving their terminal observations.
- **Models produce tensors; policies give them meaning.** Candle modules remain
  ordinary tensor-to-tensor functions. `ProbabilisticPolicyModel<D>` combines a
  module with a `Distribution`, and `Space` converts the resulting policy
  representation into an environment action. Applications can supply their own
  modules and distributions.
- **Network topology stays visible.** PPO and A2C can use separate actor and
  critic networks or a shared trunk with distinct heads. Optimizers and
  variables remain attached to the matching topology instead of being hidden
  behind a global runtime.
- **Training state is explicit.** On-policy agents use rollout storage;
  off-policy agents use replay storage. Parameter schedules advance over a
  declared training horizon and retain progress across repeated `learn` calls.
- **Device placement is part of configuration.** Tensors, models, and
  environments use Candle devices directly. Replay-based agents can keep
  storage and optimization on one device or split them with
  `ReplayDeviceStrategy`.
- **Observability is pluggable.** Algorithm-specific logger traits expose
  collection and optimization data without coupling the core training loop to
  a particular UI or file format.

This design favors inspectable composition over a one-line black-box trainer.
It also lets ModuRL integrate deeply with Candle instead of abstracting over
every tensor backend.

## Features

| Area | Included |
| --- | --- |
| Algorithms | PPO, A2C, SAC, DDPG, TD3, DQN, and Double DQN |
| Policies and models | Categorical and Gaussian policies, transformed distributions, MLPs, dueling networks, and user-supplied Candle modules |
| Experience | Rollout buffers, replay buffers, configurable replay devices, and parameter schedules |
| Environments | Single, vectorized, and multithreaded interfaces; Acrobot, BipedalWalker, CartPole, LunarLander, Mountain Car, Pendulum, MuJoCo, and Atari integrations |
| Wrappers | Time limits, observation and reward normalization, reward clipping, frame stacking, max-and-skip, and raw reward recording |
| Backends | CPU, CUDA, cuDNN, Metal, and MKL through Candle |
| Logging | Algorithm-specific hooks, terminal graphs, and TensorBoard event output |

Atari support is provided by the separately licensed `modurl_ale` crate. See
[Licensing](#licensing) before distributing a binary that uses it.

## Algorithms

| Algorithm | Family | Action Space | Guide |
| --- | --- | --- | --- |
| PPO | On-policy actor-critic | Discrete or continuous | [PPO](https://modurl.github.io/ModuRL/dev/ppo.html) |
| A2C | On-policy actor-critic | Discrete or continuous | [A2C](https://modurl.github.io/ModuRL/dev/a2c.html) |
| SAC | Off-policy actor-critic | Discrete or continuous | [SAC](https://modurl.github.io/ModuRL/dev/sac.html) |
| DDPG | Off-policy deterministic actor-critic | Continuous | [DDPG](https://modurl.github.io/ModuRL/dev/ddpg.html) |
| TD3 | Off-policy deterministic actor-critic | Continuous | [TD3](https://modurl.github.io/ModuRL/dev/td3.html) |
| DQN | Off-policy value-based | Discrete | [DQN](https://modurl.github.io/ModuRL/dev/dqn.html) |
| Double DQN | Off-policy value-based | Discrete | [Double DQN](https://modurl.github.io/ModuRL/dev/ddqn.html) |

## Quick Start

Install a current stable Rust toolchain, clone the repository, and train PPO on
eight CartPole environments:

```sh
git clone https://github.com/ModuRL/ModuRL.git
cd ModuRL
cargo run --release -p examples --example ppo_cartpole
```

The [Getting Started guide](https://modurl.github.io/ModuRL/dev/getting-started.html)
builds a smaller PPO program one component at a time. The
[examples directory](crates/examples/examples) contains complete programs for
the other supported algorithms and environments.

## Workspace Crates

| Crate | Purpose | License |
| --- | --- | --- |
| `modurl` | Agents, models, distributions, buffers, spaces, schedules, and environment traits | MIT |
| `modurl_gym` | Acrobot, BipedalWalker, CartPole, LunarLander, Mountain Car, Pendulum, rendering, and Gym utilities | MIT |
| `modurl_mojoco` | Ant, HalfCheetah, Hopper, Humanoid, Walker2d, SumoAnts, and SumoHumans MuJoCo environments | MIT; third-party notices apply |
| `modurl_ale` | Atari Learning Environment integration and wrappers | GPL-2.0-only |
| `modurl_logger` | Terminal graphs and TensorBoard event logging | MIT |
| `examples` | Runnable training programs; not published as a crate | MIT |

## Cargo Features

The core `modurl` crate has no default features.

| Feature | Purpose |
| --- | --- |
| `cuda` | Enable Candle CUDA support |
| `cudnn` | Enable Candle cuDNN support |
| `metal` | Enable Candle Metal support |
| `mkl` | Enable Candle's Intel MKL backend |
| `multithreading` | Enable multithreaded vectorized environments |

Workspace crates expose additional features for rendering and environment
selection. Check the relevant crate manifest before combining them.

## Documentation

- The [ModuRL Guide](https://modurl.github.io/ModuRL/dev/) teaches complete
  workflows and the concepts behind them.
- The [environment guides](https://modurl.github.io/ModuRL/dev/environments.html)
  cover vectorized environments and implementing a custom `Gym`.
- The [training-run guides](https://modurl.github.io/ModuRL/dev/understand-ppo-training.html)
  explain the metrics emitted by PPO, SAC, deterministic actor-critic, and
  value-based agents.
- Rustdoc contains the precise contracts for public traits, structs, and
  builders.

## Contributing

Issues and pull requests are welcome. Because the public API is still evolving,
open an [issue](https://github.com/ModuRL/ModuRL/issues) before starting a large
change so its direction can be discussed first.

The main local checks mirror the workspace CI:

```sh
cargo fmt --all --check
cargo test --locked -p modurl -p modurl_gym
cargo test --locked --all-targets -p examples
```

## Licensing

Licensing is defined per crate, not once for the entire workspace:

- `modurl`, `modurl_gym`, `modurl_logger`, and the unpublished `examples`
  package use the repository's [MIT License](LICENSE).
- `modurl_mojoco` is available under its own
  [MIT License](crates/modurl_mojoco/LICENSE). Its native MuJoCo dependency is
  Apache-2.0, and its Gymnasium-derived environment models are MIT-licensed.
  See the [MuJoCo third-party notices](crates/modurl_mojoco/THIRD_PARTY_LICENSES.md)
  for redistribution requirements.
- `modurl_ale` is
  [GPL-2.0-only](crates/modurl_ale/LICENSE) because it contains and links the
  GPL-covered ALE/Stella native code. Distributed binaries that use this crate
  must comply with GPL-2.0. See the
  [ALE third-party notices](crates/modurl_ale/THIRD_PARTY_LICENSES.md).

Where a crate provides its own license file, use that license rather than
assuming the repository-level MIT license applies. Dependencies and bundled
third-party components remain subject to their own terms. This summary is not
legal advice.

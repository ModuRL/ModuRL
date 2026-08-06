# ModuRL

[![Workspace CI](https://github.com/ModuRL/ModuRL/actions/workflows/rust.yml/badge.svg)](https://github.com/ModuRL/ModuRL/actions/workflows/rust.yml)

`ModuRL` is a deep reinforcement learning framework for Rust, built on [Candle](https://github.com/huggingface/candle).

It provides training algorithms, environments, models, policy distributions, storage, schedules, and logging. Tensor shapes, devices, networks, and optimizer configuration remain explicit.

> [!WARNING]
> ModuRL is early in development. Public APIs may change between revisions.

## Quick start

Install Git and a Rust toolchain that supports edition 2024. Then clone the repository and train PPO on eight CartPole environments:

```console
git clone https://github.com/ModuRL/ModuRL.git
cd ModuRL
cargo run --release -p examples --example ppo_cartpole
```

The same `Gym` and `Space` traits used by training agents also support direct environment interaction:

```rust
use candle_core::Device;
use modurl::prelude::*;
use modurl_gym::classic_control::cartpole::CartPoleV1;
use modurl_gym::EnvironmentError;

fn main() -> Result<(), EnvironmentError> {
    let device = Device::Cpu;
    let mut env = CartPoleV1::builder().device(&device).build()?;

    let observation = env.reset()?.state;
    let action = env.action_space().sample(&device)?;
    let transition = env.step(action)?;

    println!("observation shape: {:?}", observation.dims());
    println!("reward: {}", transition.reward);
    Ok(())
}
```

The example uses the CPU by default. [Additional examples](crates/examples/examples) cover other algorithms and environments.

## What ModuRL includes

| Area | Included |
| --- | --- |
| Training algorithms | PPO, A2C, SAC, DDPG, TD3, DQN, and Double DQN |
| Models and policies | MLPs, shared or separate actor-critic networks, categorical and Gaussian policies, transformed distributions, and user-supplied Candle modules |
| Experience | Rollout buffers, replay buffers, configurable replay devices, and parameter schedules |
| Environments | Single and vectorized interfaces, classic control, Box2D, MuJoCo, and Atari integrations |
| Wrappers | Time limits, observation and reward normalization, reward clipping, frame stacking, max-and-skip, and raw reward recording |
| Logging | Algorithm-specific hooks, terminal graphs, and TensorBoard event output |

## Architecture and choices

ModuRL keeps training decisions at the call site. Typed builders require each algorithm's components and provide defaults for tuning options. Agent builders also validate algorithm-specific settings in `build()`.

After creating the Candle modules and optimizers, a separate-network PPO agent is assembled like this:

```rust
let policy =
    ProbabilisticPolicyModel::<CategoricalDistribution>::new(Box::new(actor));

let networks = PPONetworkInfo::Separate(
    SeparatePPONetwork::builder()
        .actor_network(Box::new(policy))
        .critic_network(Box::new(critic))
        .actor_optimizer(actor_optimizer)
        .critic_optimizer(critic_optimizer)
        .build(),
);

let mut agent = PPOAgent::builder()
    .action_space(action_space)
    .network_info(networks)
    .batch_size(2_048)
    .mini_batch_size(64)
    .clip_range(Box::new(ConstantSchedule::new(0.2)))
    .training_horizon(100_000)
    .device(device)
    .logging_info(&mut logger)
    .build()?;
```

The public boundaries make each part replaceable:

| Decision | Available choices |
| --- | --- |
| Environment execution | Implement `Gym` for one environment or `MultiGym` for a batch. Collect `Gym` implementations with `VectorizedGymWrapper::from(envs)`, or enable `multithreading` and use `MultithreadedVectorizedGymWrapper`. |
| Network topology | Use `PPONetworkInfo::Separate` for independent actor and critic modules, or `PPONetworkInfo::Shared` for one trunk with policy and value heads. |
| Policy representation | Wrap a Candle module in `ProbabilisticPolicyModel<D>`. Choose a categorical, Gaussian, transformed, or application-defined `Distribution`. |
| Training state and devices | On-policy agents use rollout storage. Replay agents select `ReplayDeviceStrategy::OneDevice` or split storage and optimization with `Hybrid`. |
| Schedules and logging | Use constant, linear, exponential, or application-defined `ParameterSchedule` values. Pass a logger implementation to the agent builder, or omit logging. |

Agents train against these contracts instead of concrete environment, network, or logger types. See [Core Concepts](https://modurl.github.io/ModuRL/dev/core-concepts.html) for how the parts interact.

## Workspace crates

| Crate | Purpose | Pure Rust? | License |
| --- | --- | --- | --- |
| [`modurl`](src/lib.rs) | Agents, models, distributions, buffers, spaces, schedules, and environment traits | Yes, with default features | [MIT](LICENSE) |
| [`modurl_gym`](crates/modurl_gym) | Classic-control and Box2D environments, rendering, and Gym utilities | Yes, with default features | MIT |
| [`modurl_mojoco`](crates/modurl_mojoco) | Gymnasium-compatible MuJoCo v5 environments | No; uses the native MuJoCo library | [MIT](crates/modurl_mojoco/LICENSE); [Gymnasium asset license](crates/modurl_mojoco/assets/LICENSE) |
| [`modurl_ale`](crates/modurl_ale) | Arcade Learning Environment integration and Atari wrappers | No; builds bundled C++ | [GPL-2.0-only](crates/modurl_ale/LICENSE); [third-party notices](crates/modurl_ale/THIRD_PARTY_LICENSES.md) |
| [`modurl_logger`](crates/modurl_logger) | Terminal graphs and TensorBoard event logging | Yes | MIT |
| [`examples`](crates/examples/Cargo.toml) | Runnable training programs; not published as a crate | Mixed; depends on the example | MIT |

The `cuda`, `cudnn`, `metal`, and `mkl` features can add native platform or
accelerator dependencies to crates that are pure Rust with default features.

## Cargo features

The core `modurl` crate has no default Cargo features.

| Feature | Purpose |
| --- | --- |
| `multithreading` | Enable multithreaded vectorized and stacked environments |
| `cuda` | Enable Candle CUDA support |
| `cudnn` | Enable Candle cuDNN support |
| `metal` | Enable Candle Metal support |
| `mkl` | Enable Candle's Intel MKL backend |

Workspace crates define additional Cargo features for rendering and environment support. Check the relevant `Cargo.toml` before combining them.

## Documentation

The [ModuRL Guide](https://modurl.github.io/ModuRL/dev/) covers the training algorithms, environment interfaces, models, policies, distributions, and device configuration.

Build the core API documentation locally:

```console
cargo doc -p modurl --no-deps --open
```

## Development

Run the main checks used by workspace CI:

```console
cargo fmt --all --check
cargo test --locked -p modurl -p modurl_gym
cargo test --locked --all-targets -p examples
```

MuJoCo and Atari checks have additional native requirements. See the [`modurl_mojoco` README](crates/modurl_mojoco/README.md) and [`modurl_ale` README](crates/modurl_ale/README.md) before working on those crates.

Issues and pull requests are welcome. Open an [issue](https://github.com/ModuRL/ModuRL/issues) before starting a large API change.

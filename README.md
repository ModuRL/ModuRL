# <img src="docs/modurl-logo-transparent.png" width="70px" height="70px" align="bottom" alt="ModuRL mascot icon"> ModuRL

[![Workspace CI](https://github.com/ModuRL/ModuRL/actions/workflows/rust.yml/badge.svg)](https://github.com/ModuRL/ModuRL/actions/workflows/rust.yml)
[![crates.io](https://img.shields.io/crates/v/modurl.svg)](https://crates.io/crates/modurl)
[![docs.rs](https://docs.rs/modurl/badge.svg)](https://docs.rs/modurl)
[![book](https://img.shields.io/github/actions/workflow/status/ModuRL/ModuRL/book-pages.yml?branch=master&label=book)](https://modurl.github.io/ModuRL/)

`ModuRL` is a deep reinforcement learning framework for Rust.

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
use burn::tensor::Device;
use modurl::prelude::*;
use modurl_gym::classic_control::cartpole::CartPoleV1;
use modurl_gym::EnvironmentError;

fn main() -> Result<(), EnvironmentError> {
    let device = Device::flex();
    let mut env = CartPoleV1::builder().seed(7).build()?;

    let observation = env.reset()?.observation;
    let action = env.action_space().sample_batch(1, &device)?.squeeze_dim::<1>(0);
    let transition = env.step(action)?;

    println!("observation shape: {:?}", observation.dims());
    println!("reward: {}", transition.reward);
    Ok(())
}
```

The example uses the CPU by default. [Additional examples](crates/examples/examples) cover other algorithms and environments.

## Atari training benchmarks

The current benchmark suite measures end-to-end Breakout PPO and DQN throughput in ModuRL, Stable-Baselines3, and adapted CleanRL runners. It uses matched Atari preprocessing and training workloads with U8 observations.

The latest PPO comparison processed 40,960 Breakout transitions:

| Runner | Time | Throughput |
| --- | ---: | ---: |
| ModuRL | 6.86 s | 6,000 transitions/s |
| Stable-Baselines3 | 10.96 s | 3,737 transitions/s |
| CleanRL | 9.8 s | 4,178 transitions/s |

The suite measures throughput, not final reward or sample efficiency. Emulator versions differ across frameworks, so trajectories and rewards are not expected to match. See the [benchmark workloads, adaptations, fairness controls, and run commands](crates/benches/README.md).

## What ModuRL includes

- Training algorithms: PPO, A2C, SAC, DDPG, TD3, DQN, and Double DQN.
- Models and policies: MLPs, shared or separate actor-critic networks, categorical and Gaussian policies, transformed distributions, and user-supplied models.
- Experience: rollout buffers, replay buffers, configurable replay devices, and parameter schedules.
- Environments: single and vectorized interfaces, classic control, Box2D, MuJoCo, and Atari integrations.
- Wrappers: time limits, observation and reward normalization, reward clipping, frame stacking, max-and-skip, and raw reward recording.
- Logging: algorithm-specific hooks, terminal graphs, and TensorBoard event output.

## Architecture and choices

ModuRL keeps training decisions at the call site. Typed builders require each algorithm's components and provide defaults for tuning options. Agent builders also validate algorithm-specific settings in `build()`.

After creating the models and optimizers, a separate-network PPO agent is assembled like this:

```rust
let policy =
    ProbabilisticPolicyModel::new(actor, CategoricalDistribution);

let networks = SeparatePPONetwork::builder()
        .actor_network(policy)
        .critic_network(critic)
        .actor_optimizer(actor_optimizer)
        .critic_optimizer(critic_optimizer)
        .actor_learning_rate(3e-4)
        .critic_learning_rate(3e-4)
        .build()?;

let mut agent = PPOAgent::builder()
    .observation_space(observation_space)
    .action_space(action_space)
    .network_info(networks)
    .batch_size(2_048)
    .mini_batch_size(64)
    .clip_range(ConstantSchedule::new(0.2))
    .training_horizon(100_000)
    .logging_info(&mut logger)
    .build()?;
```

The public boundaries make each part replaceable:

| Decision | Available choices |
| --- | --- |
| Environment execution | Implement `Gym` for one environment or `MultiGym` for a batch. Collect `Gym` implementations with `VectorizedGymWrapper::from(envs)`, or enable `multithreading` and use `MultithreadedVectorizedGymWrapper`. |
| Network topology | Use `SeparatePPONetwork` for independent actor and critic modules, or `SharedPPONetwork` for one trunk with policy and value heads. |
| Policy representation | Wrap a model in `ProbabilisticPolicyModel`. Choose a categorical, Gaussian, transformed, or application-defined `Distribution`. |
| Training state and devices | On-policy agents use rollout storage. Replay agents optimize on the model device and store replay there or on an optional `.replay_device(..)`; observations retain their native dtype. |
| Schedules and logging | Use constant, linear, exponential, or application-defined `ParameterSchedule` values. Pass a logger implementation to the agent builder, or omit logging. |

Agents train against these contracts instead of concrete environment, network, or logger types. See [Core Concepts](https://modurl.github.io/ModuRL/dev/core-concepts.html) for how the parts interact.

## Workspace crates

| Crate | Purpose | Pure Rust? | License |
| --- | --- | --- | --- |
| [`modurl`](src/lib.rs) | Agents, models, distributions, buffers, spaces, schedules, and environment traits | Yes, with default features | [MIT](LICENSE) |
| [`modurl_gym`](crates/modurl_gym) | Classic-control and Box2D environments, rendering, and Gym utilities | Yes, with default features | MIT |
| [`modurl_mujoco`](crates/modurl_mujoco) | Gymnasium-compatible MuJoCo v5 environments | No; uses the native MuJoCo library | [MIT](crates/modurl_mujoco/LICENSE); [Gymnasium asset license](crates/modurl_mujoco/assets/LICENSE) |
| [`modurl_ale`](crates/modurl_ale) | Arcade Learning Environment integration and Atari wrappers | No; builds bundled C++ | [GPL-2.0-only](crates/modurl_ale/LICENSE); [third-party notices](crates/modurl_ale/THIRD_PARTY_LICENSES.md) |
| [`modurl_logger`](crates/modurl_logger) | Terminal graphs and TensorBoard event logging | Yes | MIT |
| [`examples`](crates/examples/Cargo.toml) | Runnable training programs; not published as a crate | Mixed; depends on the example | MIT |
| [`modurl-benches`](crates/benches/Cargo.toml) | Atari PPO and DQN cross-framework throughput benchmarks; not published as a crate | Mixed; includes Python runners | MIT |

The core `modurl` crate has no default Cargo features. Enable its
`multithreading` feature for multithreaded vectorized and stacked environments.
Enable `tracing` to emit performance spans for agent learning, rollout
collection, and optimization. All spans use the `modurl::performance` target;
install a `tracing` subscriber in the application to record or display them.
Workspace crates define additional features for rendering and environment
support.

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

MuJoCo and Atari checks have additional native requirements. See the [`modurl_mujoco` README](crates/modurl_mujoco/README.md) and [`modurl_ale` README](crates/modurl_ale/README.md) before working on those crates.

Issues and pull requests are welcome. Open an [issue](https://github.com/ModuRL/ModuRL/issues) before starting a large API change.

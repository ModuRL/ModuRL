# ModuRL

ModuRL is a Rust-native reinforcement learning library built on
[Candle](https://github.com/huggingface/candle). It focuses on fast,
composable training components for users who want explicit control over
agents, environments, models, distributions, schedules, logging, and devices.

The project is currently early and API stability is not guaranteed.

## Goals

ModuRL is built around a few long-term goals:

- **Be the fastest Rust-native RL library for supported algorithms.** We want
  PPO, DQN, and future supported algorithms to compete on training throughput,
  memory efficiency, and backend utilization.
- **Make RL systems modular without making them slow.** Users should be able to
  swap policies, critics, distributions, environments, schedules, loggers, and
  buffers while staying close to hand-written training-loop performance.
- **Provide the clearest Rust API for building RL agents.** ModuRL should feel
  natural to Rust users: explicit types, predictable ownership, clear
  contracts, and composable traits.
- **Make reproducible RL experiments easier.** Runs should be configurable,
  inspectable, seedable where backends support it, and easy to compare across
  examples, algorithms, and benchmarks.
- **Support serious research-style iteration.** It should be straightforward to
  implement a new agent, distribution, environment wrapper, schedule, logging
  backend, or algorithm variant without rewriting the whole library.

## Non-Goals

ModuRL intentionally does not try to be everything:

- **Not a one-line black-box training framework.** Convenience APIs are welcome,
  but the core library should remain explicit, inspectable, and composable.
- **Not a Python RL framework clone.** ModuRL learns from libraries like
  CleanRL, Stable-Baselines3, and RLlib, but the API should fit Rust, Candle,
  and Rust ownership patterns.
- **Not backend-agnostic at all costs.** Candle is a core design choice. ModuRL
  should integrate deeply with Candle rather than dilute the API around every
  possible tensor backend.

## What Works Today

| Area | Status |
| --- | --- |
| PPO | Implemented |
| DQN | Implemented |
| DDQN | Implemented |
| Vectorized environments | Implemented |
| Multithreaded vectorized environments | Available behind the `multithreading` feature |
| Candle CPU backend | Supported |
| Candle CUDA / Metal backends | Available through feature flags |

## Running an Example

The examples use `modurl_gym` environments.

```sh
cargo run --example ppo_bench
```

For a PPO example with logging and terminal plots:

```sh
cargo run --example ppo_cartpole_with_graphs
cargo run --release --example ppo_mujoco_with_graphs -- --env HalfCheetah-v5
cargo run --release --example ppo_mujoco_with_graphs -- --env Hopper-v5
cargo run --release --example ppo_mujoco_with_graphs -- --env Walker2d-v5
```

For DQN training with terminal plots for loss, exploration, Q-values, and
episode performance:

```sh
cargo run --example dqn_cartpole_with_graphs
```

For value-based CartPole programs, read the [DQN guide](docs/src/dqn.md) or
[Double DQN guide](docs/src/ddqn.md). Both pages include a complete program and
explain the required replay, exploration, and target-network configuration.

For GPU-backed builds, enable the matching Candle backend feature:

```sh
cargo run --features cuda --example ppo_bench
cargo run --features metal --example ppo_bench
```

## Examples

| Example | Shows |
| --- | --- |
| `examples/ppo_bench.rs` | PPO on CartPole with separate actor and critic networks |
| `examples/ppo_cartpole_with_graphs.rs` | PPO training metrics through `PPOLogger` |
| `examples/ppo_mujoco_with_graphs.rs` | CleanRL-style continuous PPO on selectable MuJoCo environments with progress and terminal plots |
| `examples/dqn_cartpole_with_graphs.rs` | DQN training metrics and episode graphs through `DQNLogger` |
| `examples/rendered_lunar_lander_ppo.rs` | PPO on LunarLander with rendering and learning-rate schedules |

## Core Concepts

ModuRL is organized around a few public building blocks:

- `Agent`: an agent that can `act` from observations and `learn` from a
  vectorized environment.
- `Gym`: a single environment interface.
- `VectorizedGym`: a batched environment interface used by training loops.
- `Space`: a contract for sampling, validating, and converting policy outputs
  into environment actions.
- `Distribution`: a policy distribution used by probabilistic policies.
- `ParameterSchedule`: a scalar schedule for values such as PPO clipping and
  learning rates.

These APIs are intentionally explicit. Users are expected to understand tensor
shapes, Candle devices, and the model modules they pass into agents.

## Feature Flags

| Feature | Purpose |
| --- | --- |
| `cuda` | Enable Candle CUDA support |
| `cudnn` | Enable Candle cuDNN support |
| `metal` | Enable Candle Metal support |
| `multithreading` | Enable multithreaded vectorized environment support |

## Documentation

The documentation surface is organized as follows:

- README: short project overview, status, and first successful path.
- mdBook: user guide, concepts, examples, and tensor/device contracts.
- rustdoc: precise API contracts for public traits, structs, and builders.

## License

ModuRL is licensed under the MIT License. See [`LICENSE`](LICENSE).

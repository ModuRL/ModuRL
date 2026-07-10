# ModuRL

ModuRL is a Rust-native reinforcement learning library built on
[Candle](https://github.com/huggingface/candle). It focuses on fast,
composable training components for users who want explicit control over
agents, environments, models, distributions, schedules, logging, and devices.

The project is currently early and API stability is not guaranteed. PPO is the
best-supported path today; DQN and DDQN are implemented but need stronger docs,
examples, and benchmarks.

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
| PPO | Implemented with separate and shared-network paths |
| DQN | Implemented, needs public guide/example coverage |
| DDQN | Implemented, needs public guide/example coverage |
| Vectorized environments | Implemented |
| Multithreaded vectorized environments | Available behind the `multithreading` feature |
| Candle CPU backend | Supported |
| Candle CUDA / Metal backends | Available through feature flags |

## Running an Example

The examples currently use `modurl_gym` environments and the local development
dependency layout from this repository.

```sh
cargo run --example ppo_bench
```

For a PPO example with logging and terminal plots:

```sh
cargo run --example ppo_cartpole_with_graphs
```

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

## Documentation Status

The current README and examples are not the final documentation surface. The
planned direction is:

- README: short project overview, status, and first successful path.
- mdBook: user guide, concepts, examples, and tensor/device contracts.
- rustdoc: precise API contracts for public traits, structs, and builders.

Known documentation gaps are tracked in [`Doc_Holes.md`](Doc_Holes.md).

## License

ModuRL is licensed under the MIT License. See [`LICENSE`](LICENSE).

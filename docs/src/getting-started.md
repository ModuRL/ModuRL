# Getting Started

This page gets you from a fresh project to one working PPO training run.

## Prerequisites

Install a Rust toolchain that supports edition 2024.

From the `modurl` repository, check that the basic crate builds:

```sh
cargo check
```

## Run an Existing Example

The shortest supported path is the PPO CartPole benchmark example:

```sh
cargo run --example ppo_bench
```

For terminal plots of PPO metrics, run:

```sh
cargo run --example ppo_cartpole_with_graphs
```

For GPU-backed builds, enable the matching feature:

```sh
cargo run --features cuda --example ppo_bench
cargo run --features metal --example ppo_bench
```

## Create a Small Program

Now build the same shape of program from a blank binary crate:

```sh
cargo new modurl-hello
cd modurl-hello
```

Start by adding the libraries used by this example:

```toml
[package]
name = "modurl-hello"
version = "0.1.0"
edition = "2024"

[dependencies]
modurl = "0.1"
modurl_gym = "0.1"
candle-core = "0.9.1"
candle-nn = "0.9.1"
candle-optimisers = "0.9.0"
```

The snippets below are consecutive pieces of `src/main.rs`. Add each one in
order; the file compiles after the final `Train` snippet.

### Imports

First, bring the Candle types, optimizer, ModuRL traits, and CartPole
environment into scope:

```rust
use candle_core::Device;
use candle_nn::{Optimizer, VarBuilder, VarMap};
use candle_optimisers::adam::{Adam, ParamsAdam};
use modurl::prelude::*;
use modurl_gym::classic_control::cartpole::CartPoleV1;
```

`modurl::prelude::*` brings the common ModuRL traits and training types into
scope. `PPOActor` is the training algorithm. `Actor` and `VectorizedGym` are
traits that make `.learn()`, `.observation_space()`, and `.action_space()`
available. `CategoricalDistribution` is the right policy distribution for
CartPole because CartPole has a discrete action space.

### Choose a Device

Start `main` by choosing where tensors and models will live:

```rust
fn main() {
    let device = Device::Cpu;
```

The CPU device is the simplest first run. CUDA and Metal are covered later on
this page.

### Create Environments

PPO learns from batches of environment steps, so create several CartPole
environments and wrap them as one vectorized environment:

```rust
    let envs = (0..4)
        .map(|_| CartPoleV1::builder().device(&device).build())
        .collect::<Vec<_>>();
    let mut env = VectorizedGymWrapper::from(envs);
```

Each inner `CartPoleV1` is a single environment. `VectorizedGymWrapper` stacks
them so the actor can collect multiple transitions per step.

### Read the Spaces

Ask the environment for its observation and action spaces before building the
networks:

```rust
    let observation_space = env.observation_space();
    let action_space = env.action_space();
```

The observation space determines the actor and critic input size. The action
space determines how many action logits the actor must produce.

### Build the Actor Network

The actor maps observations to action logits:

```rust
    let actor_var_map = VarMap::new();
    let actor_vb = VarBuilder::from_varmap(&actor_var_map, candle_core::DType::F32, &device);
    let actor_network = MLP::builder()
        .input_size(observation_space.shape()[0])
        .output_size(action_space.shape()[0])
        .vb(actor_vb)
        .activation(Box::new(tanh))
        .hidden_layer_sizes(vec![64, 64])
        .name("actor".to_string())
        .build()
        .expect("failed to build actor network");
```

The actor output size is the number of action logits. For CartPole, this
produces two logits: one for pushing left and one for pushing right. Those
logits are interpreted by `CategoricalDistribution`.

### Build the Critic Network

The critic maps observations to one value estimate:

```rust
    let critic_var_map = VarMap::new();
    let critic_vb = VarBuilder::from_varmap(&critic_var_map, candle_core::DType::F32, &device);
    let critic_network = MLP::builder()
        .input_size(observation_space.shape()[0])
        .output_size(1)
        .vb(critic_vb)
        .activation(Box::new(tanh))
        .hidden_layer_sizes(vec![64, 64])
        .name("critic".to_string())
        .build()
        .expect("failed to build critic network");
```

PPO uses the critic to estimate how good each observed state is. That is why the
critic has the same input size as the actor, but only one output.

### Create Optimizers

Give the actor and critic separate Adam optimizers:

```rust
    let mut optimizer_config = ParamsAdam::default();
    optimizer_config.lr = 3e-4;

    let actor_optimizer = Adam::new(actor_var_map.all_vars(), optimizer_config.clone())
        .expect("failed to build actor optimizer");
    let critic_optimizer = Adam::new(critic_var_map.all_vars(), optimizer_config)
        .expect("failed to build critic optimizer");
```

Each optimizer receives the variables from the matching network's `VarMap`.

### Assemble PPO

Wrap the actor network in `ProbabilisticActorModel` so PPO can sample actions
and evaluate their log probabilities:

```rust
    let network_info = PPONetworkInfo::Separate(
        SeparatePPONetwork::builder()
            .actor_network(Box::new(ProbabilisticActorModel::<CategoricalDistribution>::new(
                Box::new(actor_network),
            )))
            .critic_network(Box::new(critic_network))
            .actor_optimizer(actor_optimizer)
            .critic_optimizer(critic_optimizer)
            .build(),
    );
```

`SeparatePPONetwork` means the actor and critic are independent networks with
independent optimizers.

### Train

Finally, build the actor and run learning:

```rust
    let mut actor = PPOActor::builder()
        .action_space(action_space)
        .network_info(network_info)
        .batch_size(1024)
        .mini_batch_size(64)
        .clip_range(Box::new(ConstantSchedule::new(0.2)))
        .device(device)
        .build();

    actor.learn(&mut env, 10_000).expect("PPO learning failed");
}
```

`batch_size` controls how many transitions PPO collects before an update.
`mini_batch_size` controls how those transitions are split during optimization.
The clip range is the PPO policy-update bound.

Run the program with:

```sh
cargo run
```

### Complete File

After applying the pieces above, `src/main.rs` should look like this:

```rust
use candle_core::Device;
use candle_nn::{Optimizer, VarBuilder, VarMap};
use candle_optimisers::adam::{Adam, ParamsAdam};
use modurl::prelude::*;
use modurl_gym::classic_control::cartpole::CartPoleV1;

fn main() {
    let device = Device::Cpu;

    let envs = (0..4)
        .map(|_| CartPoleV1::builder().device(&device).build())
        .collect::<Vec<_>>();
    let mut env = VectorizedGymWrapper::from(envs);

    let observation_space = env.observation_space();
    let action_space = env.action_space();

    let actor_var_map = VarMap::new();
    let actor_vb = VarBuilder::from_varmap(&actor_var_map, candle_core::DType::F32, &device);
    let actor_network = MLP::builder()
        .input_size(observation_space.shape()[0])
        .output_size(action_space.shape()[0])
        .vb(actor_vb)
        .activation(Box::new(tanh))
        .hidden_layer_sizes(vec![64, 64])
        .name("actor".to_string())
        .build()
        .expect("failed to build actor network");

    let critic_var_map = VarMap::new();
    let critic_vb = VarBuilder::from_varmap(&critic_var_map, candle_core::DType::F32, &device);
    let critic_network = MLP::builder()
        .input_size(observation_space.shape()[0])
        .output_size(1)
        .vb(critic_vb)
        .activation(Box::new(tanh))
        .hidden_layer_sizes(vec![64, 64])
        .name("critic".to_string())
        .build()
        .expect("failed to build critic network");

    let mut optimizer_config = ParamsAdam::default();
    optimizer_config.lr = 3e-4;

    let actor_optimizer = Adam::new(actor_var_map.all_vars(), optimizer_config.clone())
        .expect("failed to build actor optimizer");
    let critic_optimizer = Adam::new(critic_var_map.all_vars(), optimizer_config)
        .expect("failed to build critic optimizer");

    let network_info = PPONetworkInfo::Separate(
        SeparatePPONetwork::builder()
            .actor_network(Box::new(ProbabilisticActorModel::<CategoricalDistribution>::new(
                Box::new(actor_network),
            )))
            .critic_network(Box::new(critic_network))
            .actor_optimizer(actor_optimizer)
            .critic_optimizer(critic_optimizer)
            .build(),
    );

    let mut actor = PPOActor::builder()
        .action_space(action_space)
        .network_info(network_info)
        .batch_size(1024)
        .mini_batch_size(64)
        .clip_range(Box::new(ConstantSchedule::new(0.2)))
        .device(device)
        .build();

    actor.learn(&mut env, 10_000).expect("PPO learning failed");
}
```

## Recap

The program has five moving parts:

- `CartPoleV1` creates single environments.
- `VectorizedGymWrapper` batches multiple environments for training.
- `MLP` builds the actor and critic networks.
- `ProbabilisticActorModel<CategoricalDistribution>` turns actor-network logits
  into sampled discrete actions.
- `PPOActor` collects rollouts and optimizes both networks.

PPO needs both pieces: the actor chooses actions, and the critic estimates the
value of the states PPO is learning from.

## Backend Features

ModuRL exposes Candle backend features through Cargo:

| Feature | Purpose |
| --- | --- |
| `cuda` | Enable Candle CUDA support |
| `cudnn` | Enable Candle cuDNN support |
| `metal` | Enable Candle Metal support |
| `multithreading` | Enable multithreaded vectorized environments |

This example uses `Device::Cpu`. GPU setup depends on your Candle backend, so
enable the feature here and construct the corresponding Candle device in
`main`.

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
scope. `Agent` and `VectorizedGym` are traits that make `.learn()`,
`.observation_space()`, and `.action_space()` available.

### A Few Terms

Before we build the program, it is worth separating two similar names.

In ModuRL, an `Agent` is the object that can act in an environment and learn
from it. For this example, the agent is a `PPOAgent`. It contains the PPO
rollout logic, loss calculations, optimizers, policy distribution, and neural
networks.

The actor network is the source of the policy's action scores. It produces
logits, which are raw scores for each action.
`ProbabilisticPolicyModel<CategoricalDistribution>` interprets those scores as
a categorical policy: it can sample an action from them and later compute the
log probability of that action. PPO needs both operations during training.

You will also see `MLP` below. `MLP` builds a dense feed-forward neural network:
linear layers with an activation between them. We use one `MLP` for the actor
network and one `MLP` for the critic network.

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
them so PPO can collect multiple transitions per step.

### Read the Spaces

Ask the environment for its observation and action spaces before building the
networks:

```rust
    let observation_space = env.observation_space();
    let action_space = env.action_space();
```

The observation space determines the input size for both networks. The action
space determines how many action logits the actor network must produce.

### Build the Actor Network

The actor network maps observations to action logits:

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

The `MLP` here has two hidden dense layers of width 64. Its output size is the
number of action logits. For CartPole, that means two logits: one for pushing
left and one for pushing right. Later, `CategoricalDistribution` will interpret
those logits as a discrete policy.

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

This is another `MLP` with the same hidden-layer shape. PPO uses the critic to
estimate how good each observed state is, so the critic network has the same
input size as the actor network but only one output.

### Create Optimizers

Give the actor network and critic network separate Adam optimizers:

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

Wrap the actor network in `ProbabilisticPolicyModel` so `PPOAgent` can sample
actions and evaluate their log probabilities:

```rust
    let policy =
        ProbabilisticPolicyModel::<CategoricalDistribution>::new(Box::new(actor_network));

    let network_info = PPONetworkInfo::Separate(
        SeparatePPONetwork::builder()
            .actor_network(Box::new(policy))
            .critic_network(Box::new(critic_network))
            .actor_optimizer(actor_optimizer)
            .critic_optimizer(critic_optimizer)
            .build(),
    );
```

`policy` is the probabilistic policy PPO trains. `SeparatePPONetwork` means the
actor network and critic network are independent neural networks with
independent optimizers.

### Train

Finally, build the `PPOAgent` and run learning:

```rust
    let mut agent = PPOAgent::builder()
        .action_space(action_space)
        .network_info(network_info)
        .batch_size(1024)
        .mini_batch_size(64)
        .clip_range(Box::new(ConstantSchedule::new(0.2)))
        .device(device)
        .build();

    agent.learn(&mut env, 10_000).expect("PPO learning failed");
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

    let policy =
        ProbabilisticPolicyModel::<CategoricalDistribution>::new(Box::new(actor_network));

    let network_info = PPONetworkInfo::Separate(
        SeparatePPONetwork::builder()
            .actor_network(Box::new(policy))
            .critic_network(Box::new(critic_network))
            .actor_optimizer(actor_optimizer)
            .critic_optimizer(critic_optimizer)
            .build(),
    );

    let mut agent = PPOAgent::builder()
        .action_space(action_space)
        .network_info(network_info)
        .batch_size(1024)
        .mini_batch_size(64)
        .clip_range(Box::new(ConstantSchedule::new(0.2)))
        .device(device)
        .build();

    agent.learn(&mut env, 10_000).expect("PPO learning failed");
}
```

## Recap

The program has five moving parts:

- `CartPoleV1` creates single environments.
- `VectorizedGymWrapper` batches multiple environments for training.
- `MLP` builds dense feed-forward neural networks for the policy and critic.
- `ProbabilisticPolicyModel<CategoricalDistribution>` turns actor-network logits
  into sampled discrete actions.
- `PPOAgent` is the agent: the full training object that collects rollouts,
  computes PPO losses, and optimizes both networks.

PPO needs both model roles: the actor network produces policy logits used to
choose actions, and the critic network estimates the value of the states PPO is
learning from. The `PPOAgent` ties those networks together with the rest of the
PPO algorithm.

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

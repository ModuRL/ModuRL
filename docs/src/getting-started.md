# Getting Started

This page builds a small PPO training program in a new Cargo binary crate.

## What You Will Build

The program trains PPO on several CartPole environments at once. It uses an
`MLP` to produce policy logits, a probabilistic policy model to sample actions,
and another `MLP` to estimate state values.

## Prerequisites

Install a Rust toolchain that supports edition 2024.

## Create a Small Program

Create a new binary crate:

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
order; the file compiles after the final `Train` section.

### Imports

First, bring the Candle types, optimizer, ModuRL traits, and CartPole
environment into scope:

```rust,ignore
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

```rust,ignore
fn main() {
    let device = Device::Cpu;
```

The CPU device is the simplest first run. See [Run on CUDA or Metal](./run-on-cuda-or-metal.md) when the CPU version works.

### Create Environments

PPO learns from batches of environment steps, so create several CartPole
environments and wrap them as one vectorized environment:

```rust,ignore
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

```rust,ignore
    let observation_space = env.observation_space();
    let action_space = env.action_space();
```

The observation space determines the input size for both networks. The action
space determines how many action logits the actor network must produce.

### Build the Actor Network

The actor network maps observations to action logits:

```rust,ignore
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

```rust,ignore
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

```rust,ignore
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

```rust,ignore
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

`policy` is the probabilistic policy PPO trains. In this configuration, the
policy model and value model are independent neural networks with independent
optimizers.

### Train

Finally, build the `PPOAgent` and run learning:

```rust,ignore
    let mut agent = PPOAgent::builder()
        .action_space(action_space)
        .network_info(network_info)
        .batch_size(1024)
        .mini_batch_size(64)
        .clip_range(Box::new(ConstantSchedule::new(0.2)))
        .training_horizon(10_000)
        .device(device)
        .build();

    agent.learn(&mut env, 10_000).expect("PPO learning failed");
    println!("Training complete.");
}
```

`batch_size` controls how many transitions PPO collects before an update.
`mini_batch_size` controls how those transitions are split during optimization.
The clip range is the PPO policy-update bound.
`training_horizon` defines the total number of environment transitions over
which parameter schedules progress. This way, the schedules can span multiple `learn` calls.

Run the program with:

```sh
cargo run
```

When the training loop finishes, the program prints `Training complete.`. The
default run collects 10,000 environment steps, so it does more than compile the
program but is smaller than a longer experiment.

### Complete File

After applying the pieces above, `src/main.rs` should look like this:

```rust,ignore
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
        .training_horizon(10_000)
        .device(device)
        .build();

    agent.learn(&mut env, 10_000).expect("PPO learning failed");
    println!("Training complete.");
}
```

## Where to Go Next

You now have a PPO training program with vectorized CartPole environments, a
stochastic categorical policy, and a value model. Read [Understand a PPO
Training Run](./understand-ppo-training.md) to learn what the example's metrics
show, or [Use Vectorized Environments](./vectorized-environments.md) to work
directly with batched environments.

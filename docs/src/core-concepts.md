# Core Concepts

This chapter explains the names you will see in ModuRL examples. The goal is
not to explain all of reinforcement learning at once. The goal is to make the
types and builder arguments in the code feel less mysterious.

## Agent

An `Agent` is the top-level object that interacts with an environment.

In ModuRL, an agent can do two things:

- `act` from observations
- `learn` from a vectorized environment

ModuRL provides agent types named after the algorithms they implement.
`PPOAgent` implements PPO, `DQNAgent` implements DQN, and `DDQNAgent` implements
Double DQN.

An agent coordinates the pieces needed for training: model modules, optimizers,
schedules, buffers, distributions, and update logic.

## Environments and Spaces

An environment is the world the agent interacts with.

A single environment implements the `Gym` interface. It can reset, accept an
action, and return the next observation and reward.

Training usually needs more than one transition at a time, so ModuRL also uses
vectorized environments. A `VectorizedGym` behaves like a batch of environments.
When the agent sends one batch of actions, the vectorized environment steps each
inner environment and returns a batch of results.

`VectorizedGymWrapper` is the simple wrapper used in the getting-started
example. It turns several single environments into one vectorized environment.
With the `multithreading` feature enabled,
`MultithreadedVectorizedGymWrapper` can step the inner environments on worker
threads. Applications can also implement the public `VectorizedGym` trait when
they need a different way to manage or step a batch of environments.

Vectorized environments auto-reset each inner environment when it returns
`done` or `truncated`. The `states` field in `VectorizedStepInfo` contains the
next state to continue training from. If an inner environment ended, that next
state is already the reset state for the next episode.

A `Space` describes what observations or actions look like.

For observations, the space tells you the input shape your networks need. For
actions, the space tells you what kind of actions the environment accepts.

CartPole has a discrete action space. That is why the PPO example uses
`CategoricalDistribution`: the policy chooses from a fixed set of actions.

Spaces also help convert policy outputs into environment actions. In PPO, the
policy produces action information as tensors. The action space defines how
those tensors map to valid actions for the environment.

## Models and Policies

Models are neural network modules.

`MLP` builds a dense feed-forward network. Dense means each linear layer connects
all of its inputs to all of its outputs. In the getting-started example, both
networks are `MLP`s with two hidden layers.

`MLP` does not know about PPO by itself. It maps input tensors to output
tensors. The agent and policy model decide what those outputs mean.

A policy is the rule used to choose actions. A stochastic policy samples from
a distribution. PPO wraps its actor module in
`ProbabilisticPolicyModel<D>`, where `D` defines how to interpret the actor's
output. `D` can be a distribution supplied by ModuRL or a user-defined type
that implements the public `Distribution` trait. This gives PPO the operations
it needs:

- sample an action representation
- compute the log probability of an action
- compute the policy entropy

For DQN and DDQN, an `MLP` is a *Q-network*. Its output has one value for each
discrete action. The agent selects an action from those values with
epsilon-greedy exploration instead of a probability distribution.

Read [Models, Policies, and
Distributions](./models-policies-and-distributions.md) for how tensors move
between actors, distributions, and action spaces. Read [Value-Based
Training](./q-learning.md) for Q-networks.

## Tensors and Devices

ModuRL uses Candle tensors.

A tensor has a shape, data type, and device. The shape must match what the model
or environment expects. The device says where the tensor lives, such as CPU,
CUDA, or Metal.

Most examples start with:

```rust,ignore
let device = Device::Cpu;
```

The important rule is that tensors and models used together should live on the
same device. If a model is on the CPU, the observations passed to it should also
be on the CPU.

## Training Configuration

An optimizer updates model parameters.

In PPO with separate networks, the two model modules can have separate
optimizers. The optimizer receives variables from the matching `VarMap`.

A schedule is a value that changes during training. For example,
`ConstantSchedule` keeps the PPO clip range fixed. `LinearSchedule` can move a
value from one number to another over training progress.

## In the CartPole Example

The CartPole PPO example configures a probabilistic policy that wraps a model
which produces policy scores. It also configures a separate model that estimates
values. `PPOAgent` uses the policy and value model while it collects experience
and updates their parameters.

These are roles in the PPO configuration, not separate library architecture
types. The public types in the example are `MLP`,
`ProbabilisticPolicyModel<CategoricalDistribution>`, and `PPOAgent`.

Read [Understand a PPO Training Run](./understand-ppo-training.md) to inspect
the results of the example, or [Use Vectorized Environments](./vectorized-environments.md) to work with batched environment steps directly.

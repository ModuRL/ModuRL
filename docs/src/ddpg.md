# DDPG

Deep Deterministic Policy Gradient (DDPG) trains one deterministic actor and
one Q-value critic from replayed transitions. This page shows how the complete
MuJoCo example connects those networks to `DDPGAgent`.

You should already understand the actor, critic, target-network, and replay
contracts in [Deterministic Actor-Critic
Training](./deterministic-actor-critic.md).

## Run the DDPG Example

The example supports Ant, HalfCheetah, Hopper, and Walker2d:

```sh
cargo run --release -p examples --example ddpg_mujoco --features ant
cargo run --release -p examples --example ddpg_mujoco --features half-cheetah
cargo run --release -p examples --example ddpg_mujoco --features hopper
cargo run --release -p examples --example ddpg_mujoco --features walker2d
```

Enable exactly one environment feature. The program selects CUDA when it is
available and otherwise uses the CPU. It trains for one million collected
transitions, then displays terminal graphs for optimization and episode
metrics.

The complete source is `crates/examples/examples/ddpg_mujoco.rs`. It contains
four parts:

1. Separate online and target actor networks.
2. One online/target critic pair.
3. A bounded continuous action space and replay-device strategy.
4. A `DDPGAgent` that owns collection and optimization.

`actor` and `critic` are private helper functions defined in that example, not
functions provided by ModuRL. `actor` builds an actor MLP. `critic` builds the
online and target critic networks and packages them as a
`DeterministicCritic`.

## Build the Online and Target Actors

Both actor networks must have the same architecture and parameter names. Only
the online actor has an optimizer:

```rust,ignore
let online_actor_variables = VarMap::new();
let mut target_actor_variables = VarMap::new();

let online_actor = actor(
    &online_actor_variables,
    observation_size,
    action_size,
    &device,
)?;
let target_actor = actor(
    &target_actor_variables,
    observation_size,
    action_size,
    &device,
)?;
let actor_optimizer = AdamW::new(
    online_actor_variables.all_vars(),
    actor_optimizer_parameters,
)?;
```

`DDPGAgent` copies the online actor parameters to the target actor during
construction. Later, it moves each target parameter toward the corresponding
online parameter by `tau` after every replay update.

## Build One Critic

DDPG requires exactly one `DeterministicCritic`. Its online network learns from
replay. Its target network supplies the next-state Q estimate:

```rust,ignore
let critic = DeterministicCritic::builder()
    .online_network(Box::new(ScalarStateActionCritic::new(Box::new(
        online_critic,
    ))))
    .target_network(Box::new(ScalarStateActionCritic::new(Box::new(
        target_critic,
    ))))
    .online_vars(&online_critic_variables)
    .target_vars(&mut target_critic_variables)
    .optimizer(critic_optimizer)
    .build()?;
```

The actor learns to maximize the online critic's mean Q estimate. Equivalently,
the optimizer minimizes the negative mean Q value.

## Assemble the Agent

The example passes all owned modules and borrowed parameter maps to the
builder:

```rust,ignore
let mut agent = DDPGAgent::builder()
    .online_actor(Box::new(online_actor))
    .target_actor(Box::new(target_actor))
    .online_actor_vars(&online_actor_variables)
    .target_actor_vars(&mut target_actor_variables)
    .actor_optimizer(actor_optimizer)
    .critic(critic)
    .action_space(action_space)
    .observation_space(observation_space)
    .device_strategy(ReplayDeviceStrategy::OneDevice(device))
    .gamma(0.99)
    .tau(0.005)
    .exploration_noise(0.1)
    .replay_capacity(1_000_000)
    .batch_size(256)
    .training_start(10_000)
    .training_horizon(TOTAL_TIMESTEPS)
    .logger(&mut grapher)
    .build()?;

agent.learn(&mut env, TOTAL_TIMESTEPS)?;
```

The `action_space` must be a `BoxSpace`. Its shape must match the actor output,
and its bounds must match the environment's accepted actions. The included
MuJoCo environments use one-dimensional action vectors bounded by
`-1.0..=1.0`; their vector length depends on the selected environment.

Important defaults and constraints are:

| Setting | Default | Constraint or effect |
| --- | --- | --- |
| `gamma` | `0.99` | Finite and between zero and one |
| `tau` | `0.005` | Target-network update coefficient from zero to one |
| `exploration_noise` | `0.1` | Non-negative Gaussian standard deviation |
| `replay_capacity` | `1_000_000` | At least `batch_size` |
| `batch_size` | `256` | Nonzero |
| `update_frequency` | `1` | Nonzero transition interval |
| `training_start` | `1_000` | Random-action transitions before optimization |
| `training_horizon` | Required | Nonzero global transition horizon |

The example overrides `training_start` to collect 10,000 random transitions.
Those transitions remain in replay. After warm-up, the agent adds exploration
noise to online-actor actions and updates the actor, critic, and target
networks at every selected replay update.

## Evaluate Without Exploration Noise

`Agent::act` includes Gaussian exploration noise. Use `act_deterministic` for
evaluation:

```rust,ignore
let actions = agent.act_deterministic(&observations)?;
```

The observations must be shaped `[batch, ...observation_shape]`. The returned
actions are shaped `[batch, ...action_shape]` and are clamped to the configured
`BoxSpace`.

Read [Understand a Deterministic Actor-Critic Training
Run](./understand-deterministic-actor-critic-training.md) to add a logger and
interpret replay-update metrics. Read [TD3](./td3.md) to add twin critics,
target-policy smoothing, and delayed actor updates.

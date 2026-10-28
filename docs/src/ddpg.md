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

Normally enable one environment feature; add `rendering` to it to open a
viewer. Additive builds with several environment features select `ant`,
`half-cheetah`, `hopper`, then `walker2d`. The program uses the CPU by default. Enable `cuda` or `metal` to select an accelerator. It trains for one million collected
transitions, then displays terminal graphs for optimization and episode
metrics.

The complete source is `crates/examples/examples/ddpg_mujoco.rs`. It contains
four parts:

1. An owned actor model.
2. One `ActionValueCritic` with an owned online network and optimizer.
3. A bounded continuous action space and replay-device strategy.
4. A `DDPGAgent` that owns collection and optimization.

## Build the Actor and Critic

The actor implements `Module` and `Forward<O, A>`.
Observation rank `O` and action rank `A` include the batch axis.
The agent creates a detached target actor automatically.
`ActionValueCritic` creates its own detached target network.

```rust,ignore
use modurl::agents::action_value_critic::ScalarStateActionCritic;

let online_actor = MLP::builder()
    .input_size(observation_size)
    .output_size(action_size)
    .hidden_layer_sizes(vec![64, 64])
    .activation(Relu)
    .output_activation(Tanh)
    .options((&device, DType::F32))
    .build()?;
let critic = ActionValueCritic::builder()
    .online_network(ScalarStateActionCritic::new(online_critic))
    .optimizer(AdamWConfig::new().with_weight_decay(0.0).init())
    .learning_rate(1e-3)
    .build()?;
```

The critic receives observations `[batch_size, ...observation_shape]` and actions `[batch_size, ...action_shape]`.
It returns Q values `[batch_size, 1]`.
`ScalarStateActionCritic` flattens event axes and joins Float observations and actions.
Its inputs must share dtype and device. Custom critics can accept other native observation kinds.

After each replay update, targets move toward online parameters by `tau`.

The actor learns to maximize the online critic's mean Q estimate. Equivalently,
the optimizer minimizes the negative mean Q value.

## Assemble the Agent

The builder accepts owned models and optimizers:

```rust,ignore
let mut agent = DDPGAgent::builder()
    .online_actor(online_actor)
    .actor_optimizer(AdamWConfig::new().with_weight_decay(0.0).init())
    .actor_learning_rate(1e-4)
    .critic(critic)
    .action_space(action_space)
    .observation_space(observation_space)
    .replay_storage_config(ReplayStorageConfig::new(
        ReplayDeviceStrategy::OneDevice(device),
    ))
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

The `action_space` implements `ActionMap<A, A>` with Float actions.
`BoxSpace` clamps actor outputs to the environment bounds.
Actor outputs must match the policy event shape. Environment actions must match the action event shape. The included
MuJoCo environments use one-dimensional action vectors bounded by
`-1.0..=1.0`; their vector length depends on the selected environment.

Observations keep their native kind and dtype in inputs and replay.
Models must perform any required conversion explicitly.
The `dtype` setting controls actions, rewards, and Q values, including F64 where supported.
Actions and Q values must share the configured dtype; mismatches return an error.
Replay rewards are converted to that dtype.
Model parameters must use the optimization device.
Replay stores detached transitions and uses terminal observations for truncation targets.

Important defaults and constraints are:

| Setting | Default | Constraint or effect |
| --- | --- | --- |
| `actor_learning_rate` | Required | Finite and positive |
| `gamma` | `0.99` | Finite and between zero and one |
| `tau` | `0.005` | Target-network update coefficient from zero to one |
| `exploration_noise` | `0.1` | Non-negative Gaussian standard deviation |
| `replay_capacity` | `1_000_000` | At least `batch_size`; at first `learn`, larger than and divisible by `env.num_envs()` |
| `batch_size` | `256` | Nonzero |
| `update_frequency` | `1` | Nonzero transition interval |
| `training_start` | `1_000` | Random-action transitions before optimization |
| `training_horizon` | First `learn` call's `num_timesteps` | Nonzero global transition horizon |

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

# TD3

Twin Delayed Deep Deterministic Policy Gradient (TD3) keeps DDPG's
deterministic actor and replay loop, then adds three safeguards:

1. Two critics reduce optimistic target estimates.
2. Noise on target actions smooths the critic target.
3. Delayed actor and target-network updates give the critics more update steps.

This page starts from the network and replay contracts in [Deterministic
Actor-Critic Training](./deterministic-actor-critic.md) and focuses on the
TD3-specific choices.

## Run the TD3 Example

The example supports Ant, HalfCheetah, Hopper, and Walker2d:

```sh
cargo run --release -p examples --example td3_mujoco --features ant
cargo run --release -p examples --example td3_mujoco --features half-cheetah
cargo run --release -p examples --example td3_mujoco --features hopper
cargo run --release -p examples --example td3_mujoco --features walker2d
```

Normally enable one environment feature; add `rendering` to it to open a
viewer. Additive builds with several environment features select `ant`,
`half-cheetah`, `hopper`, then `walker2d`. The program selects CUDA when it is
available and otherwise uses the CPU. It trains for one million collected
transitions, then displays terminal graphs for optimization and episode
metrics.

The complete source is `crates/examples/examples/td3_mujoco.rs`. Its actor
construction is the same as the DDPG example. The difference is the critic
ensemble and the TD3 builder settings.

`actor` and `critic` are private helper functions defined in that example, not
functions provided by ModuRL. `actor` builds an actor MLP. `critic` builds one
pair of online and target critic networks and packages them as a
`DeterministicCritic`.

## Build Two Independent Critics

Canonical TD3 uses two critics. Build each critic with separate online and
target networks, parameter maps, and optimizers:

```rust,ignore
let critic_1 = critic(
    &online_critic_variables_1,
    &mut target_critic_variables_1,
    observation_size,
    action_size,
    &optimizer_parameters,
    &device,
)?;
let critic_2 = critic(
    &online_critic_variables_2,
    &mut target_critic_variables_2,
    observation_size,
    action_size,
    &optimizer_parameters,
    &device,
)?;
```

Do not share a `VarMap` or optimizer between the critics. Their independent
errors are what make the smaller of the two target estimates useful.

## Configure the Three TD3 Safeguards

Pass both critics and the TD3-specific update settings:

```rust,ignore
let mut agent = TD3Agent::builder()
    .online_actor(online_actor)
    .target_actor(target_actor)
    .online_actor_vars(&online_actor_variables)
    .target_actor_vars(&mut target_actor_variables)
    .actor_optimizer(actor_optimizer)
    .critics(vec![critic_1, critic_2])
    .action_space(action_space)
    .observation_space(observation_space)
    .replay_storage_config(ReplayStorageConfig::new(
        ReplayDeviceStrategy::OneDevice(device),
    ))
    .gamma(0.99)
    .tau(0.005)
    .exploration_noise(0.1)
    .target_policy_noise(0.2)
    .target_noise_clip(0.5)
    .actor_update_interval(2)
    .replay_capacity(1_000_000)
    .batch_size(256)
    .training_start(10_000)
    .training_horizon(TOTAL_TIMESTEPS)
    .logger(&mut grapher)
    .build()?;

agent.learn(&mut env, TOTAL_TIMESTEPS)?;
```

`target_policy_noise` is the standard deviation of Gaussian noise added to the
target actor's output. `target_noise_clip` limits that noise component by
component. The `BoxSpace` then clamps the noisy target action to the environment
bounds.

By default, `target_aggregation_mode` is
`SACCriticAggregationMode::Min`. The smaller target Q estimate becomes the
bootstrap value shared by both critic losses.

`actor_update_interval` defaults to `2`. The agent updates every critic on each
replay optimization, but updates the actor and all target networks only on
every second optimization. On skipped actor updates, logger fields such as
`actor_loss` are `None`.

Important defaults and constraints are:

| Setting | Default | Constraint or effect |
| --- | --- | --- |
| `gamma` | `0.99` | Finite and between zero and one |
| `tau` | `0.005` | Target-network update coefficient from zero to one |
| `exploration_noise` | `0.1` | Non-negative collection-noise deviation |
| `target_policy_noise` | `0.2` | Non-negative target-noise deviation |
| `target_noise_clip` | `0.5` | Non-negative target-noise bound |
| `actor_update_interval` | `2` | Nonzero replay-update interval |
| `replay_capacity` | `1_000_000` | At least `batch_size`; at first `learn`, larger than and divisible by `env.num_envs()` |
| `batch_size` | `256` | Nonzero |
| `update_frequency` | `1` | Nonzero transition interval |
| `training_start` | `1_000` | Random-action transitions before optimization |
| `training_horizon` | Required | Nonzero global transition horizon |

Collection exploration noise and target-policy noise solve different problems.
`exploration_noise` changes actions sent to the environment after warm-up.
`target_policy_noise` changes only next actions used to calculate replay
targets.

## Keep Canonical Actor and Target Aggregation

When `actor_aggregation_mode` is omitted, the actor maximizes the first online
critic's Q estimate. This is canonical TD3 behavior.

ModuRL supports nonempty ensembles of other sizes and explicit aggregation for
experiments:

```rust,ignore
.target_aggregation_mode(SACCriticAggregationMode::Mean)
.actor_aggregation_mode(SACCriticAggregationMode::Median)
```

`Min`, `Mean`, `Median`, and `Max` use the same elementwise aggregation
implemented for SAC critic ensembles. Setting `actor_aggregation_mode` makes
the actor optimize the selected aggregate instead of the first critic.

These choices define algorithm variants. Omit them when reproducing canonical
TD3 with two critics.

## Evaluate the Deterministic Actor

As with DDPG, `Agent::act` adds collection exploration noise. Evaluate the
online actor with:

```rust,ignore
let actions = agent.act_deterministic(&observations)?;
```

Read [Understand a Deterministic Actor-Critic Training
Run](./understand-deterministic-actor-critic-training.md) to distinguish critic
updates from delayed actor updates in logs.

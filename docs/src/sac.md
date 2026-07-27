# Soft Actor-Critic

Soft Actor-Critic (SAC) trains one stochastic actor and two Q-value critics
from replayed transitions. The algorithm traditionally known as SAC almost
always uses exactly two critics. ModuRL accepts any nonempty critic ensemble,
but use two unless you are intentionally studying another aggregation scheme.

SAC is primarily a continuous-control algorithm. This chapter follows the
continuous MuJoCo example and explains the actor, critic, entropy, replay, and
device choices you need to adapt it. You should already be familiar with
`VectorizedGym`, Candle `VarMap`s, and probabilistic policies from [Core
Concepts](./core-concepts.md).

## Run Continuous SAC

The example supports HalfCheetah, Hopper, and Walker2d:

```sh
cargo run --release -p examples --example sac_mujoco --features half-cheetah
cargo run --release -p examples --example sac_mujoco --features hopper
cargo run --release -p examples --example sac_mujoco --features walker2d
```

Enable exactly one environment feature. The program selects CUDA when it is
available and otherwise uses the CPU. It trains for one million collected
transitions, then displays terminal graphs for episode performance and SAC
diagnostics.

The complete source is `crates/examples/examples/sac_mujoco.rs`. It contains
four parts:

1. A Gaussian parameter module and squashed distribution for bounded actions.
2. Two independently optimized online/target critic pairs.
3. Automatic entropy tuning.
4. An `SACAgent` that owns replay and runs collection and optimization.

The agent stores every collected transition in replay, including transitions
from the initial random-action phase. Once `collection_timestep` reaches
`training_start`, each new transition triggers one replay optimization step.
This gives the implementation an update-to-data ratio of one. A vectorized step
with `N` environments stores `N` transitions and, after the threshold, triggers
`N` updates.

## Build the Continuous Policy

`GaussianModule` returns distribution parameters rather than an environment
action. `ProbabilisticPolicyModel` combines the module with a distribution to
form the actor's policy. For an action with `A` components,
`GaussianDistribution` expects `A` means followed by `A` log standard
deviations.

The module uses state-dependent means and one trainable log standard deviation
per action component:

```rust,ignore
struct GaussianModule {
    mean: MLP,
    log_std: Tensor,
}

impl Module for GaussianModule {
    fn forward(&self, observations: &Tensor) -> candle_core::Result<Tensor> {
        let mean = self.mean.forward(observations)?;
        let log_std = self
            .log_std
            .broadcast_as(mean.shape())?
            .clamp(-20.0, 2.0)?;
        Tensor::cat(&[mean, log_std], 1)
    }
}
```

Build the distribution with the environment's action shape:

```rust,ignore
let action_shape = action_space.shape();
let action_size = action_shape.iter().product();

let actor_vars = VarMap::new();
let actor_vb = VarBuilder::from_varmap(
    &actor_vars,
    DType::F32,
    &device,
);
let mean = sac_mlp(
    actor_vb.pp("mean"),
    observation_size,
    action_size,
    0.01,
    "mlp",
)?;
let log_std = actor_vb.get_with_hints(
    (1, action_size),
    "log_std",
    Init::Const(0.0),
)?;

let distribution = TransformedDistribution::new(
    GaussianDistribution::new(action_shape)?,
    TanhTransform,
);
let policy = ProbabilisticPolicyModel::with_distribution(
    Box::new(GaussianModule { mean, log_std }),
    distribution,
);
```

`TanhTransform` keeps policy candidates between `-1` and `1`, matching the
included MuJoCo environments' action bounds. The transform also includes its
Jacobian correction in action log probabilities.

## Build Two Critics

Each `SACCritic` contains an online network, a target network, the corresponding
parameter maps, and an optimizer for the online parameters. The constructor
copies the online parameters into the target parameter map. Both maps must
contain the same parameter names.

A continuous SAC critic receives a state and an action and returns one Q-value.
`ScalarStateActionCritic` concatenates each state/action pair before calling the
wrapped module:

```rust,ignore
let critic = SACCritic::builder()
    .online_network(Box::new(ScalarStateActionCritic::new(Box::new(
        online_network,
    ))))
    .target_network(Box::new(ScalarStateActionCritic::new(Box::new(
        target_network,
    ))))
    .online_vars(&online_vars)
    .target_vars(&mut target_vars)
    .optimizer(critic_optimizer)
    .build()?;
```

Create this pair twice with separate online parameters, target parameters, and
optimizers. The agent aggregates both critics when it builds targets and
updates the actor.

The default aggregation mode is `SACCriticAggregationMode::Min`. This is the
traditional SAC choice: using the lower estimate helps reduce optimistic
Q-value errors.

ModuRL also exposes `Mean`, `Median`, and `Max` for experiments with different
critic ensembles:

| Choice | Use |
| --- | --- |
| `Min` | Traditional SAC with two critics |
| `Mean` | Arithmetic mean of the critic ensemble |
| `Median` | Middle estimate for an ensemble with several critics |
| `Max` | Optimistic experimental estimate |

## Configure Entropy

The entropy coefficient, usually written as alpha, balances expected return
against policy entropy. Larger values put more weight on keeping the policy's
action distribution stochastic. Use automatic entropy tuning for the standard
SAC workflow:

```rust,ignore
let log_alpha = Var::from_vec(vec![0.0f32], (), &device)?;
let alpha_optimizer = AdamW::new(
    vec![log_alpha.clone()],
    optimizer_parameters.clone(),
)?;
let entropy = SACEntropyConfiguration::automatic(
    log_alpha,
    alpha_optimizer,
    None,
);
```

Passing `None` asks the policy for its default target entropy. The built-in
Gaussian policy uses the negative action-component count. An affine
distribution transform adjusts that default for its scale.

Pass a `ParameterSchedule` when the desired entropy should change during the
run:

```rust,ignore
let target_entropy = LinearSchedule::new(-2.0, -1.0);
let entropy = SACEntropyConfiguration::automatic(
    log_alpha,
    alpha_optimizer,
    Some(Box::new(target_entropy)),
);
```

`LinearSchedule` moves the target entropy from `-2.0` to `-1.0`. Choose values
appropriate for the environment's action dimension rather than copying these
illustrative values.

Use a fixed coefficient when you need a known constant or want to reproduce a
configuration without an entropy optimizer:

```rust,ignore
let entropy = SACEntropyConfiguration::<AdamW>::fixed(0.2);
```

The optimizer type annotation is required because the fixed variant does not
contain an optimizer value. A fixed coefficient must be finite and
non-negative.

## Assemble the Agent

The continuous example assembles the pieces as follows:

```rust,ignore
let mut agent = SACAgent::builder()
    .policy(Box::new(policy))
    .actor_optimizer(actor_optimizer)
    .critics(vec![critic_1, critic_2])
    .entropy_configuration(entropy)
    .action_space(action_space)
    .observation_space(observation_space)
    .aggregation_mode(SACCriticAggregationMode::Min)
    .training_horizon(total_timesteps)
    .device_strategy(ReplayDeviceStrategy::OneDevice(device))
    .build()?;

agent.learn(&mut env, total_timesteps)?;
```

`training_horizon` defines how many collected transitions parameter schedules
take to reach their final values. Schedule progress continues across `learn`
calls and stops at the end of the horizon.

Important defaults and constraints are:

| Setting | Default | Constraint or effect |
| --- | --- | --- |
| `gamma` | `0.99` | Finite and between zero and one |
| `tau` | `0.005` | Target-network update coefficient from zero to one |
| `replay_capacity` | `1_000_000` | At least `batch_size` |
| `batch_size` | `256` | Nonzero |
| `training_start` | `1_000` | Random-action collection before optimization |
| `samples` | `1` | Continuous expectation candidates per state |
| `training_horizon` | Required | Nonzero schedule horizon in transitions |

`samples` controls the Monte Carlo estimate used by continuous distributions.
Larger values cost more critic evaluations but use more candidates to estimate
the actor and target expectations.

Q-value clipping is an optional experimental stabilization:

```rust,ignore
.q_value_clip(0.5)
```

The value is a positive bound on a critic's update around its target-network
estimate. Its useful scale depends on environment rewards, so it remains
opt-in.

## Implement a Custom Critic Adapter

Use `ScalarStateActionCritic` unless your critic has a different input or
output layout. A custom `SACCriticNetwork` must follow these contracts:

| Operation | Inputs | Output |
| --- | --- | --- |
| `replay_values` | states `[B, ...state]`, actions `[B, ...action]` | `[B]` |
| `policy_values` | states `[B, ...state]`, candidates `[B, K, ...action]` | `[B, K]` |
| `actor_values` | states `[B, ...state]`, candidates `[B, K, ...action]` | `[B, K]` |

`actor_values` must preserve gradients through differentiable candidate
actions while excluding gradients to critic parameters.

## Discrete SAC Is an API Variant

`SACAgent` can also work with categorical policies and
`DiscreteVectorHeadCritic`. ModuRL retains that support for experiments with
discrete-SAC variants, including `SACStabilizationConfiguration::stable_discrete`.
It is not the canonical example because traditional SAC is a continuous-control
algorithm.

Read [Understand an SAC Training Run](./understand-sac-training.md) to add a
logger and interpret its metrics. Read [Run on CUDA or
Metal](./run-on-cuda-or-metal.md) when replay storage should remain on the CPU
while optimization runs on an accelerator.

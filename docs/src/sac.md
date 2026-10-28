# Soft Actor-Critic

The SAC agent uses Burn and is available through `modurl::agents::sac` and the prelude. Its critics use the generic action-value API in `modurl::agents::action_value_critic`.

`SACModelGroup` groups the policy, critic, and execution tensor types.
The builder infers this group from the supplied models and spaces.
Explicit agent annotations use `SACAgent<'a, G, OS, AS, GE, I, A>`, where `G` is the model group.
The observation and candidate ranks are inferred; `A` is the environment action batch rank.

Soft Actor-Critic (SAC) trains one stochastic actor and two Q-value critics
from replayed transitions. The algorithm traditionally known as SAC almost
always uses exactly two critics. ModuRL accepts any nonempty critic ensemble,
but use two unless you are intentionally studying another aggregation scheme.

SAC is primarily a continuous-control algorithm. This chapter follows the
continuous MuJoCo example and explains the actor, critic, entropy, replay, and
device choices you need to adapt it. You should already be familiar with
`MultiGym`, native Burn models, and probabilistic policies from [Core
Concepts](./core-concepts.md).

## Run Continuous SAC

The example supports Ant, HalfCheetah, Hopper, and Walker2d:

```sh
cargo run --release -p examples --example sac_mujoco --features ant
cargo run --release -p examples --example sac_mujoco --features half-cheetah
cargo run --release -p examples --example sac_mujoco --features hopper
cargo run --release -p examples --example sac_mujoco --features walker2d
```

Normally enable one environment feature; add `rendering` to it to open a
viewer. Additive builds with several environment features select `ant`,
`half-cheetah`, `hopper`, then `walker2d`. The program uses CPU by default. Enable `cuda` or `metal` to select an accelerator. It trains for one million collected
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

The module uses observation-dependent means and one trainable log standard deviation
per action component:

```rust,ignore
#[derive(Module, Debug)]
struct GaussianModule {
    mean: MLP,
    log_std: Param<Tensor<2>>,
}

impl Forward<2, 2> for GaussianModule {
    type InputKind = Float;
    type OutputKind = Float;
    type Error = ModelError;

    fn forward(&self, observations: Tensor<2>) -> Result<Tensor<2>, Self::Error> {
        let mean = self.mean.forward(observations)?;
        let log_std = self.log_std.val().expand(mean.dims()).clamp(-20.0, 2.0);
        Ok(Tensor::cat(vec![mean, log_std], 1))
    }
}
```

Build the distribution with the environment's action shape:

```rust,ignore
let action_size = action_space.shape()[0];
let mean = sac_mlp(observation_size, action_size, 0.01, &device)?;
let log_std = Param::from_tensor(Tensor::zeros([1, action_size], (&device, DType::F32)));
let distribution = TransformedDistribution::new(
    GaussianDistribution::<2>::new([action_size])?,
    TanhTransform,
);
let policy = ProbabilisticPolicyModel::new(
    GaussianModule { mean, log_std },
    distribution,
);
```

`TanhTransform` keeps policy candidates between `-1` and `1`, matching the
included MuJoCo environments' action bounds. The transform also includes its
Jacobian correction in action log probabilities.

## Build Two Critics

Each `ActionValueCritic` owns an online Burn model, a detached target snapshot, and an optimizer.
The constructor clones the online model to initialize the target. Each critic has its own explicit learning rate.

A continuous critic receives observations and actions and returns one Q-value per batch item.
`ScalarStateActionCritic` concatenates flattened observation/action pairs before calling the model.
The model implements Burn's `Module` and ModuRL's `Forward<2, 2>`.
It accepts Float `[batch_size, observation_size + action_size]` and returns `[batch_size, 1]`.
Observations and actions must share dtype and device; the adapter does not cast them.

```rust,ignore
let critic = ActionValueCritic::builder()
    .online_network(ScalarStateActionCritic::new(online_network))
    .optimizer(critic_optimizer)
    .learning_rate(3e-4)
    .build()?;
```

Create each critic with its own online model and optimizer. Separate target models and parameter maps are unnecessary.
`backward_step(&loss)` optimizes the online model from a Float loss `[1]` and leaves its target unchanged.
`hard_update()` copies the online model into a detached target snapshot.
`polyak_update(tau)` blends each floating-point target parameter with its matching online parameter.
Both target-evaluation methods detach their results. Actor evaluation retains action gradients while excluding critic-parameter gradients.

`ActionValueCriticNetwork<O, A, C>` describes native observation and action kinds.
Ranks O and A include batch and event axes; rank C adds a candidate axis to rank A.
The methods retain borrowed tensor inputs. For a scalar critic, use `online_replay_values::<2, 2, 3>(&observations, &actions)` for rank-2 batches.

`DiscreteVectorHeadCritic` accepts native observation kinds and a Float model output `[batch_size, action_count]`.
It requires Int replay actions `[batch_size, 1]` or Int candidates `[batch_size, candidate_count, 1]`.
The adapter checks index bounds before gathering values, reading one Boolean result to the host.
The model must perform any required observation conversion explicitly.

The default aggregation mode is `ActionValueCriticAggregationMode::Min`. This is the
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
let log_alpha = Param::from_tensor(Tensor::zeros([1], (&device, DType::F32)));
let alpha_optimizer = AdamWConfig::new().with_weight_decay(0.0).init();
let entropy = SACEntropyConfiguration::automatic(log_alpha, alpha_optimizer, 3e-4)?;
```

`automatic` asks the policy for its default target entropy. The built-in
Gaussian policy uses the negative action-component count. An affine
distribution transform adjusts that default for its scale.

Pass a `ParameterSchedule` when the desired entropy should change during the
run:

```rust,ignore
let target_entropy = LinearSchedule::new(-2.0, -1.0);
let entropy = SACEntropyConfiguration::automatic_with_target_schedule(
    log_alpha,
    alpha_optimizer,
    3e-4,
    target_entropy,
)?;
```

`LinearSchedule` moves the target entropy from `-2.0` to `-1.0`. Choose values
appropriate for the environment's action dimension rather than copying these
illustrative values.

Use a fixed coefficient when you need a known constant or want to reproduce a
configuration without an entropy optimizer:

```rust,ignore
let entropy = SACEntropyConfiguration::fixed(0.2)?;
```

Entropy configuration constructors return a typed error for invalid parameters.
A fixed coefficient must be finite and nonnegative. Automatic tuning requires an explicit learning rate and Float log alpha `[1]`.

## Assemble the Agent

The continuous example assembles the pieces as follows:

```rust,ignore
let mut agent = SACAgent::builder()
    .policy(policy)
    .actor_optimizer(actor_optimizer)
    .actor_learning_rate(3e-4)
    .critics(vec![critic_1, critic_2])
    .entropy_configuration(entropy)
    .action_space(action_space)
    .observation_space(observation_space)
    .aggregation_mode(ActionValueCriticAggregationMode::Min)
    .training_horizon(total_timesteps)
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
| `replay_capacity` | `1_000_000` | At least `batch_size`; at first `learn`, larger than and divisible by `env.num_envs()` |
| `batch_size` | `256` | Nonzero |
| `training_start` | `1_000` | Random-action collection before optimization |
| `samples` | `1` | Continuous expectation candidates per observation |
| `training_horizon` | First `learn` call's `num_timesteps` | Nonzero schedule horizon in transitions |
| `actor_learning_rate` | Required | Finite and positive |
| `dtype` | `F32` | Floating-point training statistics; does not cast observations |

Models and automatic log alpha must use the optimization device. Critic outputs must use the configured compute dtype.
Replay storage is allocated for the first collected transition and retained across `learn` calls.
Replay preserves native observation kind and first-batch dtype; later observation dtype changes return an error.
User models or explicit preprocessing perform input conversions. Rewards and policy statistics are converted to compute dtype for optimization.
The agent owns concrete native policy and critic models; the builder returns `SACConfigurationError` for invalid construction.
Runtime failures retain policy, critic, environment, space, tensor-read, and replay causes in `SACError`.

A categorical CPU example is also available:

```console
cargo run --release --locked -p examples --example sac_cartpole
```

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
output layout. A custom `ActionValueCriticNetwork<O, A, C>` must implement Burn's `Module` and follow these contracts:

| Operation | Inputs | Output |
| --- | --- | --- |
| `replay_values` | observations `[B, ...observation_shape]`, actions `[B, ...action]` | `[B, 1]` |
| `policy_values` | observations `[B, ...observation_shape]`, candidates `[B, K, ...action]` | `[B, K, 1]` |
| `actor_values` | observations `[B, ...observation_shape]`, candidates `[B, K, ...action]` | `[B, K, 1]` |

`actor_values` must preserve gradients through differentiable candidate
actions while excluding gradients to critic parameters.

## Discrete SAC Is an API Variant

`SACAgent` can also work with categorical policies and
`DiscreteVectorHeadCritic`. Experimental discrete-SAC configurations can select
mean critic aggregation with `.aggregation_mode(ActionValueCriticAggregationMode::Mean)`
and add an entropy-drift penalty with `.entropy_change_penalty(0.5)`. It is not
the canonical example because traditional SAC is a continuous-control algorithm.

Read [Understand an SAC Training Run](./understand-sac-training.md) to add a
logger and interpret its metrics. Read [Run on CUDA or
Metal](./run-on-cuda-or-metal.md) when replay storage should remain on the CPU
while optimization runs on an accelerator.

# Deterministic Actor-Critic Training

Use DDPG or TD3 when the environment has a continuous action space and you
want a deterministic policy trained from replay. Both algorithms use an actor
that returns one action directly and a critic that estimates the value of a
state-action pair.

This chapter explains the pieces shared by `DDPGAgent` and `TD3Agent`. Read
[DDPG](./ddpg.md) for the smaller, single-critic configuration. Read
[TD3](./td3.md) when you want twin critics, target-policy smoothing, and delayed
actor updates.

## Choose DDPG or TD3

DDPG and TD3 share the same collection and replay loop. TD3 changes how the
agent calculates targets and when it updates the actor:

| Behavior | `DDPGAgent` | Canonical `TD3Agent` |
| --- | --- | --- |
| Critics | Exactly one | Two |
| Target Q estimate | The sole target critic | The smaller target-critic estimate |
| Target action | Target actor action | Target actor action plus clipped noise |
| Actor update | Every replay update | Every second replay update |
| Target-network update | Every replay update | With each delayed actor update |
| Actor objective | The sole online critic | The first online critic |

DDPG is the direct deterministic actor-critic baseline. TD3 adds safeguards
against overestimated Q values and an actor that exploits narrow errors in the
critic. Start with TD3 when those safeguards are appropriate. Choose DDPG when
you need the single-critic algorithm or a simpler baseline for comparison.

ModuRL also lets `TD3Agent` use any nonempty critic ensemble and explicit
aggregation modes. Those settings are experimental variants, not canonical
TD3.

## Follow the Shared Training Loop

Both agents collect and train in the same order:

1. Before `training_start`, sample actions uniformly from the `BoxSpace`.
2. After `training_start`, run the online actor and add Gaussian exploration
   noise.
3. Clamp the resulting actions to the action-space bounds and store each
   transition in replay.
4. On collection timesteps selected by `update_frequency`, sample a replay
   batch and update every online critic.
5. Update the online actor at the algorithm's actor-update interval.
6. Polyak-update the target actors and critics whenever the actor is updated.

Collection timesteps count transitions, not calls to the vectorized
environment. A step with `N` environments adds `N` transitions. Each transition
whose global index is a multiple of `update_frequency` causes one replay update
after the warm-up threshold.

`training_horizon` records the intended number of collected transitions and
tracks progress across calls to `learn`. DDPG and TD3 do not currently expose
scheduled hyperparameters, but the shared progress counter still determines
the global collection timestep used by training and logging.

## Match the Actor Contract

The online and target actors are ordinary Candle `Module`s. For observations
shaped `[batch, ...observation_shape]`, each actor must return values shaped
`[batch, ...action_shape]`.

The supplied `BoxSpace` clamps those values to its bounds. The MuJoCo examples
use `tanh` as the actor's output activation and bounds of `-1.0..=1.0`:

```rust,ignore
let actor = MLP::builder()
    .input_size(observation_size)
    .output_size(action_size)
    .vb(actor_vb)
    .hidden_layer_sizes(vec![64, 64])
    .activation(Box::new(Tensor::relu))
    .output_activation(Box::new(Tensor::tanh))
    .name("actor".to_owned())
    .build()?;
```

Build the online and target actors with separate `VarMap`s and identical
parameter names. Agent construction copies the online parameters into the
target map. Construction returns
`DeterministicActorCriticError::ActorParameterMapMismatch` if the names differ.

The exploration-noise standard deviation is measured in the actor's output
units. ModuRL adds that noise before `BoxSpace` clamps the action. Use
`act_deterministic` when evaluating a trained policy without exploration noise.

## Match the Critic Contract

`DeterministicCritic` is the deterministic-agent name for `SACCritic`. Each
critic owns:

- an online state-action network
- a target state-action network
- separate online and target `VarMap`s
- an optimizer for the online parameters

A scalar critic receives a state and action and returns one Q value. Wrap a
module whose input width is `observation_size + action_size` with
`ScalarStateActionCritic`:

```rust,ignore
let critic = DeterministicCritic::builder()
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

As with the actors, the online and target critic parameter names must match.
The constructor initializes the target parameters from the online parameters.
Each TD3 critic needs its own networks, parameter maps, and optimizer.

## Keep Replay and Models on the Intended Devices

`ReplayDeviceStrategy::OneDevice(device)` stores replay and runs optimization
on the same device. `ReplayDeviceStrategy::Hybrid` can keep detached replay
transitions on one device, such as the CPU, and move sampled batches to the
optimization device.

The strategy does not move models or environments. Build the environment,
actors, critics, target networks, and optimizers on the optimization device.
Read [Run on CUDA or Metal](./run-on-cuda-or-metal.md) before splitting replay
storage from optimization.

The [DDPG chapter](./ddpg.md) uses these pieces to build the single-critic
baseline. The [TD3 chapter](./td3.md) shows how to use two critics,
target-policy noise, and delayed actor updates.

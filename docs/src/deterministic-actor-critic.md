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

DDPG and TD3 use owned models. Each actor implements `Module` and `Forward<O, A>`.
Observations use `[batch_size, ...observation_shape]`; actions use `[batch_size, ...action_shape]`.
The agent creates a detached target actor during construction.
The [DDPG chapter](./ddpg.md) shows the complete builder configuration.

The supplied `BoxSpace` clamps policy outputs to its bounds.
The exploration-noise standard deviation is measured in policy output units.
The agent adds noise before mapping policy outputs to environment actions.
Use `act_deterministic` for evaluation without exploration noise.

## Match the Critic Contract

DDPG requires one `ActionValueCritic` with an owned online network and optimizer.
TD3 accepts a nonempty ensemble of these critics.
The critic creates a detached target network automatically.
A scalar critic maps observations and actions to Q values `[batch_size, 1]`.
Wrap an MLP with input width `observation_size + action_size` in `ScalarStateActionCritic`.
The [DDPG chapter](./ddpg.md) shows this construction.

## Keep Replay and Models on the Intended Devices

Optimization runs on the device of the actor and critic parameters, which must
all share one device. Replay is stored there too unless `.replay_device(..)`
keeps detached transitions on another device, such as the CPU, and moves
sampled batches to the optimization device.

The agent does not move models or environments. Build actors and critics on the optimization device.
Target networks inherit their online network device.
Keep CPU environments on CPU and use `DeviceMultiGymWrapper` for transfers.
Read [Run on CUDA or Metal](./run-on-cuda-or-metal.md) before splitting replay
storage from optimization.

The [DDPG chapter](./ddpg.md) uses these pieces to build the single-critic
baseline. The [TD3 chapter](./td3.md) shows how to use two critics,
target-policy noise, and delayed actor updates.

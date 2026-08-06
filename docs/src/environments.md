# Environments

An environment defines the interaction loop an agent learns from: reset to an
initial observation, apply an action, and return the next observation and
reward. ModuRL represents one environment with `Gym` and a batch of environments
with `MultiGym`.

The getting-started example uses `CartPoleV1` for individual environments and
`VectorizedGymWrapper` to train from several of them at once.

`modurl_gym` includes these Gymnasium-compatible environments:

| Module | Environment | Action space | Observation shape |
| --- | --- | --- | --- |
| `classic_control::acrobot` | `AcrobotV1` | Discrete, 3 actions | `[6]` |
| `classic_control::cartpole` | `CartPoleV1` | Discrete, 2 actions | `[4]` |
| `classic_control::mountain_car` | `MountainCarV0` | Discrete, 3 actions | `[2]` |
| `classic_control::pendulum` | `PendulumV1` | Continuous `[1]` | `[3]` |
| `box_2d::bipedal_walker` | `BipedalWalkerV3` | Continuous `[4]` | `[24]` |
| `box_2d::lunar_lander` | `LunarLanderV3` | Discrete, 4 actions | `[8]` |

`BipedalWalkerV3` implements the standard environment with uneven grass terrain;
the hardcore obstacle variant is not included.

These structs expose Gymnasium's unwrapped dynamics. Registry time limits are
applied explicitly with `TimeLimitGym`: use 500 steps for `AcrobotV1`, 200 for
`PendulumV1`, and the registry horizon appropriate to the other environment.
Keeping the limit in a wrapper makes truncation visible and composable.

`modurl_mojoco` provides the Gymnasium v5 `AntV5`, `HalfCheetahV5`, `HopperV5`,
`HumanoidV5`, and `Walker2dV5` environments. See that crate's README for model,
installation, metadata, and parity details.

Read [Use Vectorized Environments](./vectorized-environments.md) before writing
manual training or evaluation loops. Read [Build a Custom Gym Environment](./custom-gym-environment.md) when you need a new environment type.

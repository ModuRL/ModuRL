# Environments

An environment defines the interaction loop an agent learns from: reset to an
initial observation, apply an action, and return the next observation and
reward. ModuRL represents one environment with `Gym` and a batch of environments
with `VectorizedGym`.

The getting-started example uses `CartPoleV1` for individual environments and
`VectorizedGymWrapper` to train from several of them at once.

Read [Use Vectorized Environments](./vectorized-environments.md) before writing
manual training or evaluation loops. Read [Build a Custom Gym Environment](./custom-gym-environment.md) when you need a new environment type.

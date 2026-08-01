# PPO

`PPOAgent` is a ModuRL agent that implements Proximal Policy Optimization
(PPO). It collects transitions from a `MultiGym`, then updates a stochastic
policy and a value model from that experience.

The getting-started program uses the separate-network PPO configuration. It
gives the policy model and the value model their own `MLP`, `VarMap`, and Adam
optimizer. `PPONetworkInfo::Shared` is also available when a configuration needs
one shared model followed by separate policy and value heads.

Start with [Getting Started](./getting-started.md) for the complete CartPole
program. Then read [Understand a PPO Training Run](./understand-ppo-training.md)
before changing its configuration.

The continuous-control example can also train a shared policy through
competitive SumoAnts self-play:

```console
cargo run --release -p examples --example ppo_mujoco --features sumo-ants
```

Both Ant player rows use the same actor and critic while interacting in one
MuJoCo simulation. The other supported features are `ant`, `half-cheetah`,
`hopper`, `humanoid`, and `walker2d`.

PPO can use any compatible `Distribution` implementation. ModuRL currently
supplies categorical and Gaussian distributions, and applications can define
their own. [Models, Policies, and
Distributions](./models-policies-and-distributions.md) explains the extension
point, the built-in tensor layouts, and how sampled representations become
environment actions.

For value-based agents in discrete action spaces, read [Value-Based
Training](./q-learning.md).

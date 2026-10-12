# PPO

`PPOAgent` is a ModuRL agent that implements Proximal Policy Optimization
(PPO). It collects transitions from a `MultiGym`, then updates a stochastic
policy and a value model from that experience.

The separate-network PPO configuration owns the policy and value models, such
as Burn `MLP` models. Each model has its own `ModuleOptimizer` and explicit
learning rate. `PPONetworkInfo::Shared` is also
available when a configuration needs one shared model followed by separate
policy and value heads.

`SeparatePPONetwork::builder()` requires `actor_learning_rate` and
`critic_learning_rate`. `SharedPPONetwork::builder()` requires `learning_rate`.
Both builders return `Result`. PPO's builder also requires `observation_space`
and returns `Result`. The constructor names remain `new`.

Use `PPONetworkInfo::<_, _>::Separate(network)` for separate models. Its unused
shared model defaults to `FakeModule`. A shared configuration infers the shared
model type from `PPONetworkInfo::Shared(network)`.

The agent infers `PPOModelGroup` from its models and native tensor types.
Observations keep `[batch_size, ...observation_shape]`; PPO casts float, integer,
and boolean observations to its configured floating-point dtype before model
execution. Values, log probabilities, and advantages use `[batch_size, 1]`.
Losses use `[1]`. Raw reward logs preserve the environment's reward dtype.
`reset_current_observations()` clears observations retained between training calls.

Start with [Getting Started](./getting-started.md) for the complete CartPole
program. Then read [Understand a PPO Training Run](./understand-ppo-training.md)
before changing its configuration.

The continuous-control example supports `ant`, `half-cheetah`, `hopper`, and
`walker2d`. Normally enable one environment feature; `rendering` may be added
to that feature to open a viewer. Cargo features are additive, so builds that
enable several environment features use the priority `ant`, `half-cheetah`,
`hopper`, then `walker2d`. PPO writes TensorBoard events below
`runs/ppo_mujoco/` and also displays terminal graphs.

PPO can use any compatible `Distribution` implementation. ModuRL currently
supplies categorical and Gaussian distributions, and applications can define
their own. [Models, Policies, and
Distributions](./models-policies-and-distributions.md) explains the extension
point, the built-in tensor layouts, and how sampled representations become
environment actions.

For value-based agents in discrete action spaces, read [Value-Based
Training](./q-learning.md).

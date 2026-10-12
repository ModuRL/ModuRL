# PPO

`PPOAgent` is a ModuRL agent that implements Proximal Policy Optimization
(PPO). It collects transitions from a `MultiGym`, then updates a stochastic
policy and a value model from that experience.

The separate-network PPO configuration owns the policy and value models, such
as Burn `MLP` models. Each model has its own `ModuleOptimizer` and explicit
learning rate. `SharedPPONetwork` is also
available when a configuration needs one shared model followed by separate
policy and value heads.

`SeparatePPONetwork::builder()` requires `actor_learning_rate` and
`critic_learning_rate`. `SharedPPONetwork::builder()` requires `learning_rate`.
Both builders return `Result`. PPO's builder also requires `observation_space`
and returns `Result`. The constructor names remain `new`.

Pass a `SeparatePPONetwork` or `SharedPPONetwork` directly to `.network_info(network)`.

The agent infers `PPOModelGroup` from its models and native tensor types.
Observations keep `[batch_size, ...observation_shape]`, their native kind, and their dtype.
Models perform any required conversions. Separate models accept observations directly;
shared trunks produce Float features, and shared heads accept Float inputs. The compiler rejects incompatible kind connections.
PPO has no `dtype` setting: critic outputs must match the environment reward dtype.
Training rejects mismatches without casting rewards or critic outputs. Returns and advantages share the reward dtype.
Values, log probabilities, and advantages use `[batch_size, 1]`.
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

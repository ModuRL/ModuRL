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

PPO estimates advantages with `GAE` (lambda `0.95`) by default. Pass any
`AdvantageEstimator` to `.advantage_estimator(..)`, for example
`.advantage_estimator(GAE::new(0.9)?)`. An estimator returns an
`AdvantageEstimate` with the advantages and the critic's value targets; `GAE`
uses `values + advantages` as those targets.

The trait method is `estimate(&self, rollout: AdvantageInput<'_>, gamma: f32) -> AdvantageEstimate`.
Outputs are flat `[rows, 1]` tensors in collection order. `AdvantageInput::Dense` preserves
`[time, env_count, 1]` inputs for identity-slot rollouts without building a slot layout.
`AdvantageInput::Ragged` carries a validated `RaggedRollout`: flat rewards and values,
termination flags, a `RolloutLayout`, and values for live tails in flat row order.
Build the layout with `RolloutLayout::from_steps`, passing each step's slot slice;
its offsets delimit compact chunks and its successors link the same slot across gaps.
Construct ragged inputs with `RaggedRollout::new`, which returns recoverable validation errors.
Both variants receive rewards already corrected by `gamma * V(terminal)` for
non-done truncations. Done and truncated rows stop the recurrence, including when
both flags are true. PPO evaluates terminal observations and live-tail next observations
in one compact critic batch during rollout preparation. Dead tails and slots with
no stored rows require no bootstrap evaluation. Custom estimators can use
`rollout.rewards_and_values()` to obtain flat tensors for either variant.
The built-in `NStep::new(n)?` estimator is an alternative for n-step returns.
Use `GAE::new(0.95)?.with_value_lambda(1.0)?` to use a separate lambda for value targets.
The built-in `UPGO::new(GAE::new(0.95)?)` estimator is also available for upgoing policy updates.

The agent infers its model group from `PPONetwork::ModelGroup`.
Custom `PPONetwork<O, H, L>` implementations must set `ModelGroup` to
`PPOModelGroup<Self, Tensor<O, Self::InputKind>, Tensor<H, Self::LatentKind>, Tensor<L>>`.
The network must implement `Module`. Its module operations must traverse model parameters and training flags.
PPO calls `Module::valid()` and removes autodiff from inference observations without changing their kind or dtype.
It reuses one validation network for collection and fixed value estimates, then releases it before optimizer updates.
The built-in networks supply these implementations directly. Their learning-rate schedules must implement `Send + Sync`.
Schedules are shared through `Arc` when a validation network is created.
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

Atari training saves Burn checkpoints with the `.bpk` extension.
`ppo_atari_play` loads the shared encoder, actor, and critic from that checkpoint.
Candle safetensors checkpoints require conversion before use.
`agent.network().models()` returns a snapshot of the current owned models for saving.
For shared networks, the snapshot contains the encoder, actor head, and critic head.
For separate networks, it contains the actor and critic.

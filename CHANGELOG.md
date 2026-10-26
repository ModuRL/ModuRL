# Changelog

This file tracks public API and caller-visible changes by release.

## 0.2.0 — Unreleased

Changes below are relative to the `v0.1.0` tag and include the current working tree.
The Burn migration is incomplete. Entries describe APIs already changed in source;
they do not imply that all workspace packages build together. See [Remaining migration work](#remaining-migration-work).

### Rust and tensor backend

- The minimum Rust version increases from 1.88 to 1.95. The Rust edition remains 2024.
- Migrated core APIs use Burn 0.22.0 instead of Candle tensors, devices, dtypes, and model parameters.
  Burn tensors include rank and kind in their type: `Tensor<R>`, `Tensor<R, Int>`, and `Tensor<R, Bool>`.
  Candle tensors cannot be passed directly to these APIs.
- Many tensor inputs now take owned tensors instead of references. Clone a tensor when the caller must retain it.
  Fallible component operations still return `Result`; ordinary Burn tensor operations often return tensors directly.
  Do not assume every backend shape or device failure is represented by a ModuRL error.
- Seed migrated tensor sampling with Burn's `Device::seed` instead of Candle's device seeding API.
  Seeded sample sequences can differ between backends, and some backends share RNG state across devices.
- Added `modurl::tensor_rank::{NextRank, PrevRank}` to infer tensor types with one more or one fewer axis.
  These traits preserve kind without converting values. `NextRank` covers ranks 1–1024; `PrevRank` covers ranks 2–1025.

### Tensor shapes and space traits

- Removed the combined `Space` trait. Implement `ObservationSpace<O>` for observation checks,
  `ActionSpace<A>` for action checks and sampling, and `ActionMap<L, A>` for policy-to-environment conversion.
- Both space roles have an associated tensor `Kind`. `ObservationSpace` has no associated error type;
  `ActionSpace` retains `Error`. `contains()` returns `bool`, and `shape()` returns the item shape without a batch axis.
- Removed unbatched `Space::sample`. Use `ActionSpace::sample_batch(batch_size, device)`.
  Observation-space sampling is no longer part of the observation trait.
- Moved `tensor_from_neurons` to `ActionMap`. Added `policy_shape()` to distinguish policy output shape from environment action shape.
  Continuous mapping still clamps values; discrete mapping selects the largest category score.
- Scalars now have item shape `[1]`, batch shape `[batch_size, 1]`, and frame-stack shape `[stack_size, 1]`.
  The old scalar layouts `[]` and `[batch_size]` are no longer accepted at migrated environment boundaries.
  Rewards, values, log probabilities, entropy, and termination masks also use `[batch_size, 1]`.
  Rollout scalars use `[time, num_envs, 1]`; candidate scalars use `[batch_size, candidate_count, 1]`.
  Reductions over action components retain a trailing scalar axis. Overall scalar losses remain `[1]`.
- `Discrete::shape()` now returns `[1]`, rather than a category-logit shape.
  `Discrete::policy_shape()` returns `[action_count]`, including `[1]` for a one-category space.
  Discrete batches use `Int` tensors `[batch_size, 1]`; `sample_batch` produces I32 indices.
  Sampling rejects category counts outside `1..=2^24` with `SpaceError::InvalidCategoryCount`.
- `BoxSpace` becomes `BoxSpace<R = 2>`. Bounds must be Float tensors `[1, ...item_shape]`
  with matching shapes, dtypes, and devices. Membership checks accept batches, not unbatched values.
  `new_with_universal_bounds` and `new_unbounded` take fixed-size shape arrays including the size-one batch axis.
  For vector bounds, replace `[features]` with `[1, features]`.
- Added `SpaceError` for invalid category counts and policy output shapes.
  Sampling and action mapping retain the bounds or policy dtype as documented, including supported F64 paths.

### Environments and agents

- `Gym`, `MultiGym`, and `Agent` gain observation and action rank parameters:
  `Gym<I = (), O = 2, A = 2>`, `MultiGym<I = (), O = 2, A = 2>`, and `Agent<I = (), O = 2, A = 2>`.
  These const parameters count batch axes even for `Gym`, whose step and reset values are unbatched.
  `PrevRank` determines each unbatched tensor type.
- `Gym` and `MultiGym` replace `SpaceError` and boxed combined spaces with associated `ObservationSpace` and `ActionSpace` types.
  Their space accessors return those types directly. Trait objects must specify both associated space types.
  `Agent` also adds associated space types, while retaining `SpaceError` for its action-space error.
- Renamed `ResetInfo::state` and `StepInfo::state` to `observation`.
  Both structures gain an observation tensor type parameter: `ResetInfo<I, T>` and `StepInfo<I, T>`.
- Renamed `MultiGymStepInfo::states` to `observations`, `terminal_states` to `terminal_observations`,
  and `transition_next_states()` to `transition_next_observations()`.
  `MultiGymStepInfo<I, T>` uses an unbatched terminal tensor type `T` and stores batched observations as `T::Next`.
  Rewards are Float `Tensor<1>`; termination and truncation remain separate host flags.
  The renamed transition helper returns a tensor directly instead of `candle_core::Result<Tensor>`.
- Vectorized environments retain auto-reset behavior and save terminal observations for transition targets.
  Call the transition helper when targets must use the final observation instead of the reset observation.
- `VectorizedGymWrapper<G, I, BO, BA>` gains batch-rank parameters.
  `new(Vec<G>)` now returns `Result` and rejects an empty environment list.
  `From<Vec<G>>` is replaced by `TryFrom<Vec<G>>`; `From<G>` remains available for one environment.
  `envs()` and `envs_mut()` now return slices instead of references to `Vec`.
- `StackedMultiGym<G, I, O, A>` gains rank parameters and retains fallible construction and `TryFrom<Vec<G>>`.
  Inner groups must have matching item shapes and nonzero slot counts.
- Threaded wrappers infer space types from the inner environment instead of taking public space-type and space-error parameters.
  Their type forms are `MultithreadedStackedMultiGym<G, I, O, A>` and
  `MultithreadedVectorizedGymWrapper<G, I, BO, BA>`.
  Constructors take the associated space values; threaded vectorized construction now returns `Result` and rejects an empty list.
  These wrappers remain behind `multithreading`.
- Updated `VectorizedGymError` and `StackedMultiGymError` for the Burn interfaces.
  Removed `Batch` and `InvalidActionBatch` from both enums, and `ChangedBatchSize` and `InvalidOutputShape` from stacked errors.
  Added `VectorizedGymError::Empty` and feature-gated `WorkerDisconnected` variants for threaded failures.
  Candle tensor-error conversions are removed;
  callers must satisfy the documented tensor layouts and backend requirements.

### Environment wrappers and device placement

- Replaced `TensorMapMultiGymWrapper` with `InputMapMultiGymWrapper` and `OutputMapMultiGymWrapper`.
  Compose the two wrappers when both directions need mapping.
- Input mapping receives the complete action batch. Output mapping has separate reset and step callbacks;
  the step callback receives the complete `MultiGymStepInfo`, including terminal observations, rewards, flags, and metadata.
  Callbacks return `Result` with a caller-defined mapping error. `TensorMapMultiGymError<E, M = Infallible>`
  separates wrapped environment failures from mapping failures.
  Mapping preserves native tensor kinds and layouts; callbacks control dtypes and devices.
  Space descriptions pass through unchanged, so mapped values must remain consistent with them.
- Added `DeviceMultiGymWrapper::new(gym, environment_device, agent_device)`.
  It sends actions to the environment device and returns observations, rewards, and terminal observations on the agent device.
  Transfers preserve shapes, kinds, and dtypes. Input, output, and device wrappers expose `inner`, `inner_mut`, and `into_inner`.
- Observation, reward, normalization, statistics, and time-limit wrappers now implement the rank-aware Burn environment traits
  and forward associated space types instead of boxed `Space` values.
  Tensor-read failures use Burn errors where applicable.
- `FrameStackGym` becomes `FrameStackGym<G, BF = 3, S = BoxSpace<BF>>`.
  Its constructor accepts a pre-stacked observation-space type `S`; the space's first item axis must equal `stack_size`.
  Stacking adds an item axis and preserves native observation kind, including integer and boolean frames.
- Classic-control and Box2D builders in `modurl_gym` rename `.device(...)` to `.rng_device(...)`.
  This setting controls random draws only. Observations, space bounds, and physics remain on CPU.
- `AntV5`, `HalfCheetahV5`, `HopperV5`, `HumanoidV5`, and `Walker2dV5` remove `.device(...)` from their builders.
  MuJoCo observations and space bounds remain on CPU; environment seeds still control simulation randomness.
  Use a device wrapper around a compatible batched environment for accelerator transfers.

### Models and initialization

- Added `Forward<I, O, K = Float>` for fallible execution from `Tensor<I, K>` to Float `Tensor<O>`.
  Parameter-owning models also implement Burn's `Module`, which supplies parameter traversal and optimizer integration.
  Implementations exist for `MLP`, `DuelingMLP`, and supported native Burn layers and activations.
- `MLP` and `DuelingMLP` own native Burn layers and parameters instead of Candle layers backed by `VarMap`.
  Builder `.vb(...)` and `.name(...)` are removed. Supply `.options(device)` or `.options((device, dtype))`.
  The configurable constructors are named `builder`; callers still use `Type::builder().option(...).build()`.
- Builder activations use Burn's `Activation` values instead of boxed Candle modules or function callbacks.
  MLP output activation remains optional. Shape-preserving MLP activations reject projected SwiGlu;
  SwiGlu remains usable through its separate `Forward` implementation.
- Removed `MLPInitializer`, `DefaultMLPInitializer`, and `OrthogonalMLPInitializer`.
  Configure `.hidden_initializer(...)`, `.output_initializer(...)`, and optional `.bias_initializer(...)`
  with Burn's `Initializer` instead of `.initializer(...)`.
- Default weights change from Candle linear initialization to Kaiming normal with gain `sqrt(2)`.
  Biases default to input-width-based uniform values; orthogonal weights default to zero biases.
  Explicit bias initialization applies to all linear layers.
- Removed `modurl::init`, including `linear_ortho`, `conv2d_ortho`, and `orthogonal_init`.
  Use native Burn initializers and layer configuration.
- Added `ModelError` for construction and input failures. MLP inputs must match parameter dtype and device,
  as well as feature count. Construction validates feature counts, supported dtypes, initializers, and activation compatibility.

### Policies and distributions

- `ProbabilisticPolicy<O = 2, A = 2>` adds ranks and an associated `ObservationKind`.
  Sampling, mode, and evaluation take owned observation tensors. Policy actions remain Float tensors;
  an action map converts them to native environment actions when needed.
- `ExpectationPolicy<O = 2, A = 2, C = 3>` adds candidate rank and `CandidateKind`.
  Candidate layouts are `[batch_size, candidate_count, ...event_shape]`;
  weights and log probabilities use `[batch_size, candidate_count, 1]`.
- Added `PolicyTypes`. `ProbabilisticPolicyModel<T>` replaces the distribution-only type parameter and boxed model
  with owned model and distribution types plus native observation, parameter, and action tensor types.
  The tuple form is `(M, D, (Tensor<O, K>, Tensor<P>, Tensor<A>))`.
  `new` uses a default distribution; `with_distribution` accepts one explicitly. Added the `module()` accessor.
  The policy implements Burn's `Module` and visits only the owned model's parameters.
- `ProbabilisticPolicyModelError<ME, DE>` now preserves model and distribution errors separately.
  The policy delegates tensor-contract checks to its components; conversions can be composed explicitly outside or inside the model.
- `Distribution<P = 2, A = 2>` takes rank parameters and owned parameter/action tensors.
  `DifferentiableExpectation<P = 2, A = 2, C = 3>` adds `CandidateKind`.
  `ExpectationTerms<D = 3, K = Float>` stores typed candidate tensors.
- `DistEval::new` and `ExpectationTerms::new` now return `Result<_, DistributionTensorError>`.
  Constructors check relevant dimensions, dtype, and device; expectation terms reject zero candidates.
  Expectation weights must already be normalized.
- Categorical samples and modes remain Float score tensors `[batch_size, categories]`.
  Exact expectations now use Int category indices `[batch_size, categories, 1]`, rather than floating-point indices.
  Added categorical errors for unsupported logit dtypes and category counts outside the index range.
- `GaussianDistribution<A = 2>::new` takes a fixed-size event-shape array instead of `Vec<usize>`.
  Action rank must equal event rank plus one. Scalars require `[1]`; empty event shapes are no longer supported.
  Parameters remain `[batch_size, 2 * event_size]`, with means followed by log standard deviations.
  Candidate rank is inferred through `NextRank`.
- `DistributionTransform` methods now take owned, const-rank Float tensors and preserve rank.
  `AffineTransform<E = 1>` fixes event rank in its type; `from_bounds` takes owned bounds.
  Event dimensions must match exactly; fixed parameters broadcast over documented batch and candidate axes.
  Transformed distributions retain the base distribution's ranks and support Float candidates.
- Distribution and transform error payloads replace Candle tensor errors with `DistributionTensorError`
  or Burn tensor-read errors. Update exhaustive matches and custom transform implementations.

### Objectives, sampling, buffers, and replay

- `bellman_targets` accepts owned Float rewards/next values and a Bool termination mask, all `[batch_size, 1]`.
  It returns `Tensor<2>` directly. Termination blocks bootstrapping; truncation alone does not.
- `clipped_value_loss` accepts owned rank-2 `[batch_size, 1]` tensors and returns a loss tensor `[1]` directly.
  Both objective helpers leave gradient detachment to their callers.
- `sample_u32_inclusive` and `shuffle_with_device_rng` use Burn devices and return `SamplingError`.
  Inclusive integer sampling now rejects endpoints above `2^24 - 1` and reversed ranges.
- `RolloutBuffer::new` takes a Burn device. Shuffling failures become `RolloutBufferError::SamplingError`
  instead of `TensorError`. The public `Experience` trait remains unchanged;
  added `ExperienceBatchError` for incompatible tensor or boolean fields.
- Replay tensor storage uses native rank/kind types and removes autodiff associations from stored experience.
  Public `ExperienceReplay` and `ReplayStorage` method names remain unchanged.
  `ExperienceReplayError::TensorError` and `ReplayStorageError::TensorError`, `MissingBatchDimension`,
  and `ItemShapeMismatch` are removed. `UninitializedEnvironmentCount` becomes `EnvironmentCountNotSet`.
  Added `ReplayStorageError::IndexTooLarge` for indices outside the signed 64-bit range.
- `ReplayDeviceStrategy` and `ReplayStorageConfig` now use Burn devices.
  Removed `ReplayStorageConfig::with_observation_dtype`; configuration selects devices only.
  Migrated Q-learning replay preserves incoming observation kind and the first batch's dtype, rejecting later dtype changes.
  Environment output mapping, model input conversion, or another explicit caller composition can select the representation.
  The unmigrated SAC and deterministic actor-critic storage paths retain their former default F32 observation storage.
- Public Q-learning logs now use Burn tensors: loss `[1]`, selected Q values `[batch_size, 1]`,
  replay rewards `[batch_size, 1]`, and collection rewards `[num_envs, 1]`.
  `QAgentError<GE, ME>` replaces the generic space-error parameter with a model-error parameter;
  space errors use concrete `SpaceError`, and tensor errors use `TensorReadError`.
  Added `QLearningReplayError` and configuration errors for compute dtype, learning rate, and observation rank.

### DQN and DDQN

- `DQNAgent<'a, M, S, GE, I = (), R = 2>` and `DDQNAgent` own a concrete Burn model `M`
  and observation space `S`. `R` includes the batch axis; native observation kinds and dtypes are preserved.
- Constructors use `Type::builder()`. Supply `.online_q_network(model)`, a Burn optimizer through `.optimizer(...)`,
  and an explicit `.learning_rate(...)`. Removed `.target_q_network(...)`, `.online_vars(...)`, and `.target_vars(...)`.
  The engine creates a detached target copy and replaces it at each hard-update interval.
- `get_observation_space()` returns `&S`. Both agents implement the rank-aware `Agent<I, R, 2>` contract
  with concrete `Discrete` actions and `QAgentError<GE, M::Error>`.
  `act` accepts native observations `[batch_size, ...observation_shape]` and returns U32 Int actions `[batch_size, 1]`.
- DQN still maximizes target-network values. DDQN still selects actions with the online network and evaluates them with the target network.
  Rewards, termination masks, and targets remain `[batch_size, 1]`; only termination suppresses bootstrapping.
- The CartPole executable and Q-learning documentation examples use Burn models, optimizers, and devices.
  The Q grapher accumulates Burn scalars and converts averaged metrics to the existing logger representation.
  CUDA and Metal features enable the corresponding Burn backends. CartPole requires the remaining environment migration.

### Custom MuJoCo environments

- Added `CustomMujoco<T>`, `MujocoTask`, `MujocoState`, and `TaskStep` in `modurl_mujoco` and its prelude.
  `CustomMujoco::builder()` requires an XML path, task, and fixed observation dimension.
  Optional frame skip defaults to 1; viewer construction remains behind `rendering`.
- `MujocoTask` supplies `observation` and `transition`, with optional `reset_state`, `reset`, and `before_step` hooks.
  Tasks can override `action_dim`, `map_action`, and `physical_control_targets`.
  Default policy actions use `[-1, 1]` and map limited actuators to their XML control ranges.
- `MujocoState` exposes positions, velocities, controls, actuator forces, sensors, contacts per physics substep,
  contact body pairs, body positions and parents, site positions and velocities, site body IDs,
  world contact forces, subtree angular momenta, joint position limits, and the physics timestep.
  `TaskStep` supplies reward, termination, and truncation separately.
- Custom environments expose `seed`, `actuator_count`, `viewer_running`, and `model`.
  Environments loaded from the same XML path share a compiled model while keeping separate simulation state.
- Added `CustomMujoco::edit_model` for fallible numerical physics edits between steps.
  Edits use a private model candidate, persist across resets, and leave the live model unchanged on closure errors or panics.
  Structural model changes require rebuilding the environment. Re-exported `MjModel` and `MjtObj` for model inspection and edits.

### Prelude changes

- Added exports for `Forward`, `ModelError`, `PolicyTypes`, `ObservationSpace`, `ActionSpace`, `ActionMap`, `SpaceError`,
  `DistributionTensorError`, `QLearningReplayError`, `NextRank`, `PrevRank`, and the new input/output/device wrappers.
- Removed exports for `Space`, `TensorMapMultiGymWrapper`, and the three removed MLP initializer types.
  Buffer and sampling errors remain available through their defining modules.

### Remaining migration work

- A2C, PPO, SAC, DDPG, TD3, concrete environment integrations, loggers, and other runnable examples still contain Candle APIs.
  The updated core traits do not provide automatic Candle compatibility.
- Atari output mapping to U8 and the forward wrapper for CNN input conversion are deferred.
  Removing the replay dtype setter does not complete that example's migration.
- Update this section and the relevant entries as the remaining public APIs migrate; 0.2 is not yet released.

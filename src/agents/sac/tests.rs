use super::*;
use crate::{
    distributions::{
        CategoricalDistribution, GaussianDistribution, TanhTransform, TransformedDistribution,
    },
    gym::{Gym, ResetInfo, StepInfo, VectorizedGymWrapper},
    models::{Forward, probabilistic_model::ProbabilisticPolicyModel},
    spaces::{BoxSpace, Discrete},
    tensor_rank::TensorBatchTypes,
};
use burn::{
    optim::SgdConfig,
    tensor::{Float, Int},
};
use std::convert::Infallible;

#[derive(Module, Debug)]
struct LinearModel {
    weight: Param<Tensor<2>>,
}

impl Forward<2, 2> for LinearModel {
    type InputKind = Float;
    type OutputKind = Float;
    type Error = Infallible;

    /// Maps Float observations `[batch_size, features]` through weights `[features, outputs]` to `[batch_size, outputs]`, retaining gradients.
    fn forward(&self, input: Tensor<2>) -> Result<Tensor<2>, Self::Error> {
        Ok(input.matmul(self.weight.val()))
    }
}

#[derive(Module, Debug)]
struct IntegerModel {
    weight: Param<Tensor<2>>,
}

impl Forward<2, 2> for IntegerModel {
    type InputKind = Int;
    type OutputKind = Float;
    type Error = Infallible;

    /// Explicitly converts native I64 observations `[batch_size, 1]` to the parameter dtype and returns Float `[batch_size, 2]`.
    fn forward(&self, input: Tensor<2, Int>) -> Result<Tensor<2>, Self::Error> {
        assert_eq!(input.dtype(), DType::I64);
        Ok(input
            .float()
            .cast(self.weight.val().dtype())
            .matmul(self.weight.val()))
    }
}

#[derive(Module, Debug)]
struct BooleanModel {
    weight: Param<Tensor<2>>,
}

impl Forward<2, 2> for BooleanModel {
    type InputKind = Bool;
    type OutputKind = Float;
    type Error = Infallible;

    /// Explicitly converts Bool observations `[batch_size, 1]` to the parameter dtype and returns Float `[batch_size, 2]`.
    fn forward(&self, input: Tensor<2, Bool>) -> Result<Tensor<2>, Self::Error> {
        Ok(input
            .float()
            .cast(self.weight.val().dtype())
            .matmul(self.weight.val()))
    }
}

struct BooleanSpace;

impl TensorBatchTypes for BooleanSpace {
    type Kind = Bool;
    type Item = Tensor<1, Bool>;
    type Batch = Tensor<2, Bool>;
}

impl ObservationSpace<2> for BooleanSpace {
    /// Checks the Bool scalar batch layout `[batch_size, 1]` without converting values.
    fn contains(&self, values: &Tensor<2, Bool>) -> bool {
        values.dims()[1] == 1
    }

    fn shape(&self) -> Vec<usize> {
        vec![1]
    }
}

/// Creates Float parameters `[features, outputs]` with a fixed initial value on device.
fn linear_model(features: usize, outputs: usize, value: f64, device: &Device) -> LinearModel {
    LinearModel {
        weight: Param::from_tensor(Tensor::full([features, outputs], value, device)),
    }
}

/// Creates a discrete critic whose weights `[1, 2]` are `[1, 3]`, retaining F64 when requested.
fn discrete_critic(
    device: &Device,
    dtype: DType,
) -> SACCritic<DiscreteVectorHeadCritic<LinearModel>> {
    SACCritic::builder()
        .online_network(DiscreteVectorHeadCritic::new(LinearModel {
            weight: Param::from_tensor(Tensor::from_data([[1.0f64, 3.0]], (device, dtype))),
        }))
        .optimizer(SgdConfig::new().init())
        .learning_rate(0.1)
        .build()
        .unwrap()
}

/// Reads rank-D Float tensor values as a flat F64 vector without changing the source shape or device.
fn numbers<const D: usize>(tensor: Tensor<D>) -> Vec<f64> {
    tensor
        .into_data()
        .convert::<f64>()
        .try_to_vec::<f64>()
        .unwrap()
}

/// Creates a detached optimization batch with observations/actions `[2, 1]`, Float statistics `[2, 1]`, and one terminated item.
fn categorical_batch(device: &Device, dtype: DType) -> SACOptimizationBatch<2, 2, Float, Int> {
    SACOptimizationBatch {
        observations: Tensor::ones([2, 1], (device, dtype)),
        next_observations: Tensor::full([2, 1], 2.0, (device, dtype)),
        actions: Tensor::zeros([2, 1], device),
        rewards: Tensor::ones([2, 1], (device, dtype)),
        terminated: Tensor::from_data([[false], [true]], device),
        collection_policy_entropies: Tensor::zeros([2, 1], (device, dtype)),
        entropy_change_weights: Tensor::zeros([2, 1], (device, dtype)),
    }
}

struct EndingEnv {
    device: Device,
    terminated: bool,
    terminal_width: usize,
}

impl Gym for EndingEnv {
    type Error = Infallible;
    type ObservationSpace = BoxSpace<2>;
    type ActionSpace = Discrete;

    /// Accepts an Int scalar action `[1]`, returning a Float terminal vector `[terminal_width]` filled with 10 and reward 2.
    fn step(&mut self, _action: Tensor<1, Int>) -> Result<StepInfo, Self::Error> {
        Ok(StepInfo {
            observation: Tensor::full([self.terminal_width], 10.0, &self.device),
            reward: 2.0,
            done: self.terminated,
            truncated: !self.terminated,
            info: (),
        })
    }

    /// Returns reset Float observation `[1]` equal to 1 on the fixture CPU device.
    fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
        Ok(ResetInfo {
            observation: Tensor::from_data([1.0f32], &self.device),
            info: (),
        })
    }

    fn observation_space(&self) -> BoxSpace<2> {
        BoxSpace::new(
            Tensor::zeros([1, 1], &self.device),
            Tensor::full([1, 1], 100.0, &self.device),
        )
    }

    fn action_space(&self) -> Discrete {
        Discrete::new(2)
    }
}

#[derive(Default)]
struct RecordingLogger {
    updates: Vec<usize>,
    temperature_updates: usize,
    entropy_penalties: usize,
    collection_times: Vec<usize>,
    episodes: Vec<SACEpisodeLogEntry>,
}

impl SACLogger for RecordingLogger {
    fn log_update(&mut self, entry: &SACLogEntry) {
        self.updates.push(entry.collection_timestep);
        self.temperature_updates += usize::from(entry.alpha_loss.is_some());
        self.entropy_penalties += usize::from(entry.entropy_change_loss.is_some());
        assert!(!entry.actor_loss.is_require_grad());
        assert!(!entry.bellman_targets.is_require_grad());
        assert_eq!(entry.policy_weights.dims()[2], 1);
    }

    fn log_collection(&mut self, entry: &SACCollectionLogEntry) {
        self.collection_times.push(entry.collection_timestep);
        self.episodes
            .extend(entry.completed_episodes.iter().cloned());
    }
}

#[test]
fn alpha_and_entropy_drift_losses_preserve_reference_gradients() {
    let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
    let device = Device::flex().autodiff();
    let log_alpha = Tensor::from_data([0.0f64], (&device, DType::F64)).require_grad();
    let random = Tensor::from_data([[-2.0f64]], (&device, DType::F64)).require_grad();
    let gradients = sac_alpha_loss(&log_alpha, &random, 1.0).backward();
    assert_eq!(numbers(log_alpha.grad(&gradients).unwrap()), [1.0]);
    assert!(random.grad(&gradients).is_none());
    let deterministic = Tensor::from_data([[-0.25f64]], (&device, DType::F64));
    assert_eq!(
        numbers(
            log_alpha
                .grad(&sac_alpha_loss(&log_alpha, &deterministic, 1.0).backward())
                .unwrap()
        ),
        [-0.75]
    );
    let current = Tensor::from_data([[0.2f64]], (&device, DType::F64)).require_grad();
    let collection = Tensor::from_data([[0.8f64]], (&device, DType::F64)).require_grad();
    let mask = Tensor::ones([1, 1], (&device, DType::F64)).require_grad();
    let loss = sac_entropy_change_loss(&current, &collection, &mask, 0.5);
    assert!((numbers(loss.clone())[0] - 0.18).abs() < 1e-12);
    let gradients = loss.backward();
    assert!((numbers(current.grad(&gradients).unwrap())[0] + 0.6).abs() < 1e-12);
    assert!(collection.grad(&gradients).is_none());
    assert!(mask.grad(&gradients).is_none());
    assert_eq!(
        numbers(sac_entropy_change_loss(
            &current,
            &collection,
            &Tensor::zeros([1, 1], (&device, DType::F64)),
            0.5
        )),
        [0.0]
    );
}

#[test]
fn entropy_configuration_rejects_invalid_parameters() {
    let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
    for alpha in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            SACEntropyConfiguration::fixed(alpha),
            Err(SACConfigurationError::InvalidAlpha)
        ));
    }
    let device = Device::flex().autodiff();
    for rate in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            SACEntropyConfiguration::automatic(
                Param::from_tensor(Tensor::zeros([1], &device)),
                SgdConfig::new().init(),
                rate
            ),
            Err(SACConfigurationError::InvalidLearningRate)
        ));
    }
    assert!(matches!(
        SACEntropyConfiguration::automatic(
            Param::from_tensor(Tensor::zeros([2], &device)),
            SgdConfig::new().init(),
            0.1
        ),
        Err(SACConfigurationError::InvalidLogAlphaShape)
    ));
    assert!(matches!(
        SACEntropyConfiguration::automatic(
            Param::from_tensor(Tensor::full([1], f64::NAN, &device)),
            SgdConfig::new().init(),
            0.1
        ),
        Err(SACConfigurationError::InvalidLogAlpha)
    ));
    assert!(matches!(
        SACEntropyConfiguration::automatic_with_target_schedule(
            Param::from_tensor(Tensor::zeros([1], &device)),
            SgdConfig::new().init(),
            0.1,
            |_| f64::NAN
        ),
        Err(SACConfigurationError::InvalidTargetEntropySchedule)
    ));
}

#[test]
fn categorical_bellman_targets_and_half_mse_preserve_f64() {
    let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
    let device = Device::flex().autodiff();
    let actor = LinearModel {
        weight: Param::from_tensor(Tensor::zeros([1, 2], (&device, DType::F64))),
    };
    let mut agent = SACAgent::<_, _, _, _, Infallible>::builder()
        .policy(ProbabilisticPolicyModel::new(
            actor,
            CategoricalDistribution,
        ))
        .actor_optimizer(SgdConfig::new().init())
        .actor_learning_rate(0.1)
        .critics(vec![discrete_critic(&device, DType::F64)])
        .entropy_configuration(SACEntropyConfiguration::fixed(0.5).unwrap())
        .action_space(Discrete::new(2))
        .observation_space(BoxSpace::new(
            Tensor::zeros([1, 1], &device),
            Tensor::full([1, 1], 100.0, &device),
        ))
        .replay_storage_config(ReplayStorageConfig::new(
            super::super::ReplayDeviceStrategy::OneDevice(device.clone()),
        ))
        .dtype(DType::F64)
        .gamma(0.9)
        .training_horizon(4)
        .build()
        .unwrap();
    let mut batch = categorical_batch(&device, DType::F64);
    let targets = agent.compute_bellman_targets(&batch).unwrap();
    assert_eq!(targets.dtype(), DType::F64);
    assert!(!targets.is_require_grad());
    let values = numbers(targets);
    assert!((values[0] - (1.0 + 0.9 * (4.0 + 0.5 * 2.0f64.ln()))).abs() < 1e-12);
    assert_eq!(values[1], 1.0);
    batch.observations = Tensor::ones([2, 1], (&device, DType::F64));
    let targets = Tensor::full([2, 1], 3.0, (&device, DType::F64));
    let losses = agent.optimize_critics(&batch, &targets).unwrap();
    assert_eq!(numbers(losses[0].clone()), [2.0]);
    assert_eq!(
        numbers(
            agent.critics[0]
                .online_replay_values::<2, 2, 3>(&batch.observations, &batch.actions)
                .unwrap()
        ),
        [1.2; 2]
    );
    assert_eq!(
        numbers(
            agent.critics[0]
                .target_replay_values::<2, 2, 3>(&batch.observations, &batch.actions)
                .unwrap()
        ),
        [1.0; 2]
    );
    agent.q_value_clip = Some(0.05);
    let losses = agent.optimize_critics(&batch, &targets).unwrap();
    assert!((numbers(losses[0].clone())[0] - 1.90125).abs() < 1e-12);
    assert_eq!(
        numbers(
            agent.critics[0]
                .online_replay_values::<2, 2, 3>(&batch.observations, &batch.actions)
                .unwrap()
        ),
        [1.2; 2]
    );
}

#[test]
fn agent_builder_rejects_invalid_sizes_ranges_and_dtype() {
    let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
    let device = Device::flex().autodiff();
    for case in 0..12 {
        let critics = if case == 10 {
            vec![]
        } else {
            vec![discrete_critic(&device, DType::F32)]
        };
        let result = SACAgent::<_, _, _, _, Infallible>::builder()
            .policy(ProbabilisticPolicyModel::new(
                linear_model(1, 2, 0.0, &device),
                CategoricalDistribution,
            ))
            .actor_optimizer(SgdConfig::new().init())
            .actor_learning_rate(if case == 5 { 0.0 } else { 0.1 })
            .critics(critics)
            .entropy_configuration(SACEntropyConfiguration::fixed(0.2).unwrap())
            .action_space(Discrete::new(if case == 11 { 0 } else { 2 }))
            .observation_space(BoxSpace::new(
                Tensor::zeros([1, 1], &device),
                Tensor::ones([1, 1], &device),
            ))
            .replay_storage_config(ReplayStorageConfig::new(
                super::super::ReplayDeviceStrategy::OneDevice(device.clone()),
            ))
            .replay_capacity(if case == 0 {
                0
            } else if case == 2 {
                1
            } else {
                16
            })
            .batch_size(if case == 1 { 0 } else { 2 })
            .gamma(if case == 3 { f64::NAN } else { 0.99 })
            .tau(if case == 4 { 1.1 } else { 0.005 })
            .dtype(if case == 6 { DType::I64 } else { DType::F32 })
            .entropy_change_penalty(if case == 7 { -0.1 } else { 0.0 })
            .q_value_clip(if case == 8 { 0.0 } else { 0.5 })
            .training_horizon(if case == 9 { 0 } else { 4 })
            .build();
        let error = result.err().unwrap();
        assert!(matches!(
            (case, error),
            (0, SACConfigurationError::ZeroReplayCapacity)
                | (1, SACConfigurationError::ZeroBatchSize)
                | (2, SACConfigurationError::ReplayCapacityBelowBatchSize)
                | (3, SACConfigurationError::InvalidGamma)
                | (4, SACConfigurationError::InvalidTau)
                | (5, SACConfigurationError::InvalidLearningRate)
                | (6, SACConfigurationError::InvalidDType(DType::I64))
                | (7, SACConfigurationError::InvalidEntropyChangePenalty)
                | (8, SACConfigurationError::InvalidQValueClip)
                | (9, SACConfigurationError::ZeroTrainingHorizon)
                | (10, SACConfigurationError::NoCritics)
                | (11, SACConfigurationError::EmptySpace)
        ));
    }
}

#[test]
fn mismatched_critic_dtype_is_rejected_before_training_arithmetic() {
    let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
    let device = Device::flex().autodiff();
    let mut agent = SACAgent::<_, _, _, _, Infallible>::builder()
        .policy(ProbabilisticPolicyModel::new(
            linear_model(1, 2, 0.0, &device),
            CategoricalDistribution,
        ))
        .actor_optimizer(SgdConfig::new().init())
        .actor_learning_rate(0.1)
        .critics(vec![discrete_critic(&device, DType::F64)])
        .entropy_configuration(SACEntropyConfiguration::fixed(0.2).unwrap())
        .action_space(Discrete::new(2))
        .observation_space(BoxSpace::new(
            Tensor::zeros([1, 1], &device),
            Tensor::ones([1, 1], &device),
        ))
        .replay_storage_config(ReplayStorageConfig::new(
            super::super::ReplayDeviceStrategy::OneDevice(device.clone()),
        ))
        .training_horizon(4)
        .build()
        .unwrap();
    let mut batch = categorical_batch(&device, DType::F32);
    // Supply native F64 observations to the F64 critic; the algorithm's statistics remain F32.
    batch.observations = batch.observations.cast(DType::F64);
    let targets = Tensor::ones([2, 1], &device);
    assert!(matches!(
        agent.optimize_critics(&batch, &targets),
        Err(SACError::DType {
            field: "critic values",
            expected: DType::F32,
            actual: DType::F64
        })
    ));
}

#[test]
fn actor_update_changes_policy_without_updating_critics_or_targets() {
    let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
    let device = Device::flex().autodiff();
    let actor = linear_model(1, 2, 0.0, &device);
    let mut agent = SACAgent::<_, _, _, _, Infallible>::builder()
        .policy(ProbabilisticPolicyModel::new(
            actor,
            CategoricalDistribution,
        ))
        .actor_optimizer(SgdConfig::new().init())
        .actor_learning_rate(0.1)
        .critics(vec![discrete_critic(&device, DType::F32)])
        .entropy_configuration(SACEntropyConfiguration::fixed(0.5).unwrap())
        .action_space(Discrete::new(2))
        .observation_space(BoxSpace::new(
            Tensor::zeros([1, 1], &device),
            Tensor::full([1, 1], 100.0, &device),
        ))
        .replay_storage_config(ReplayStorageConfig::new(
            super::super::ReplayDeviceStrategy::OneDevice(device.clone()),
        ))
        .training_horizon(4)
        .build()
        .unwrap();
    let batch = categorical_batch(&device, DType::F32);
    let before = numbers(agent.policy.module().weight.val());
    let q_before = numbers(
        agent.critics[0]
            .online_replay_values::<2, 2, 3>(&batch.observations, &batch.actions)
            .unwrap(),
    );
    let update = agent.optimize_actor(&batch).unwrap();
    assert!(!update.loss.is_require_grad());
    assert_ne!(numbers(agent.policy.module().weight.val()), before);
    assert_eq!(
        numbers(
            agent.critics[0]
                .online_replay_values::<2, 2, 3>(&batch.observations, &batch.actions)
                .unwrap()
        ),
        q_before
    );
    assert_eq!(
        numbers(
            agent.critics[0]
                .target_replay_values::<2, 2, 3>(&batch.observations, &batch.actions)
                .unwrap()
        ),
        q_before
    );
}

#[test]
fn vector_collection_preserves_terminal_targets_warmup_schedules_and_logging() {
    let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
    let environment_device = Device::flex();
    let device = environment_device.clone().autodiff();
    let mut environment = VectorizedGymWrapper::new(vec![
        EndingEnv {
            device: environment_device.clone(),
            terminated: false,
            terminal_width: 1,
        },
        EndingEnv {
            device: environment_device,
            terminated: true,
            terminal_width: 1,
        },
    ])
    .unwrap();
    let entropy = SACEntropyConfiguration::automatic_with_target_schedule(
        Param::from_tensor(Tensor::zeros([1], &device)),
        SgdConfig::new().init(),
        0.01,
        |progress| 1.0 - progress,
    )
    .unwrap();
    let mut logger = RecordingLogger::default();
    let mut agent = SACAgent::builder()
        .policy(ProbabilisticPolicyModel::new(
            linear_model(1, 2, 0.0, &device),
            CategoricalDistribution,
        ))
        .actor_optimizer(SgdConfig::new().init())
        .actor_learning_rate(0.01)
        .critics(vec![
            discrete_critic(&device, DType::F32),
            discrete_critic(&device, DType::F32),
        ])
        .entropy_configuration(entropy)
        .action_space(environment.action_space())
        .observation_space(environment.observation_space())
        .replay_storage_config(ReplayStorageConfig::new(
            super::super::ReplayDeviceStrategy::Hybrid {
                optimization_device: device,
                storage_device: Device::flex(),
            },
        ))
        .replay_capacity(16)
        .batch_size(2)
        .training_start(2)
        .training_horizon(4)
        .aggregation_mode(SACCriticAggregationMode::Mean)
        .entropy_change_penalty(0.5)
        .q_value_clip(0.5)
        .logger(&mut logger)
        .build()
        .unwrap();
    assert!(agent.replay.is_none());
    agent.learn(&mut environment, 0).unwrap();
    assert!(agent.replay.is_none());
    agent.learn(&mut environment, 2).unwrap();
    let batch = agent.replay.as_ref().unwrap().sample().unwrap();
    assert_eq!(numbers(batch.next_observations), [10.0; 2]);
    assert_eq!(numbers(batch.collection_policy_entropies), [0.0; 2]);
    assert_eq!(numbers(batch.entropy_change_weights), [0.0; 2]);
    agent.learn(&mut environment, 2).unwrap();
    assert_eq!(agent.replay.as_ref().unwrap().len(), 4);
    let mut different_environment = VectorizedGymWrapper::from(EndingEnv {
        device: Device::flex(),
        terminated: false,
        terminal_width: 1,
    });
    assert!(matches!(
        agent.learn(&mut different_environment, 1),
        Err(SACError::ReplayStorageError(SACReplayError::Storage(
            ReplayStorageError::EnvironmentCountMismatch {
                expected: 2,
                actual: 1,
            }
        )))
    ));
    assert_eq!(agent.optimization_steps, 3);
    assert_eq!(agent.schedule_progress.elapsed_steps(), 4);
    assert_eq!(logger.updates, [2, 3, 4]);
    assert_eq!(logger.temperature_updates, 3);
    assert_eq!(logger.entropy_penalties, 3);
    assert_eq!(logger.collection_times, [2, 4]);
    assert_eq!(logger.episodes.len(), 4);
    assert!(
        logger
            .episodes
            .iter()
            .all(|episode| episode.episode_return == 2.0 && episode.episode_length == 1)
    );
    assert!(logger.episodes[0].truncated);
    assert!(logger.episodes[1].terminated);
}

#[test]
fn continuous_policy_uses_reparameterized_actions_and_detached_targets() {
    let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
    let device = Device::flex().autodiff();
    let policy = ProbabilisticPolicyModel::new(
        linear_model(1, 2, 0.0, &device),
        TransformedDistribution::<GaussianDistribution, TanhTransform>::default(),
    );
    let critic = SACCritic::builder()
        .online_network(ScalarStateActionCritic::new(linear_model(
            2, 1, 1.0, &device,
        )))
        .optimizer(SgdConfig::new().init())
        .learning_rate(0.01)
        .build()
        .unwrap();
    let bounds = BoxSpace::new(
        Tensor::full([1, 1], -1.0, &device),
        Tensor::ones([1, 1], &device),
    );
    let mut agent = SACAgent::<_, _, _, _, Infallible>::builder()
        .policy(policy)
        .actor_optimizer(SgdConfig::new().init())
        .actor_learning_rate(0.01)
        .critics(vec![critic])
        .entropy_configuration(SACEntropyConfiguration::fixed(0.2).unwrap())
        .action_space(bounds)
        .observation_space(BoxSpace::new(
            Tensor::zeros([1, 1], &device),
            Tensor::full([1, 1], 100.0, &device),
        ))
        .replay_storage_config(ReplayStorageConfig::new(
            super::super::ReplayDeviceStrategy::OneDevice(device.clone()),
        ))
        .samples(NonZeroUsize::new(4).unwrap())
        .training_horizon(4)
        .build()
        .unwrap();
    let batch = SACOptimizationBatch {
        observations: Tensor::ones([2, 1], &device),
        next_observations: Tensor::ones([2, 1], &device),
        actions: Tensor::zeros([2, 1], &device),
        rewards: Tensor::ones([2, 1], &device),
        terminated: Tensor::from_data([[false], [true]], &device),
        collection_policy_entropies: Tensor::zeros([2, 1], &device),
        entropy_change_weights: Tensor::zeros([2, 1], &device),
    };
    let before = numbers(agent.policy.module().weight.val());
    let critic_before = numbers(
        agent.critics[0]
            .online_replay_values::<2, 2, 3>(&batch.observations, &batch.actions)
            .unwrap(),
    );
    let targets = agent.compute_bellman_targets(&batch).unwrap();
    assert!(!targets.is_require_grad());
    assert_eq!(numbers(targets)[1], 1.0);
    agent.optimize_actor(&batch).unwrap();
    assert_ne!(numbers(agent.policy.module().weight.val()), before);
    assert_eq!(
        numbers(
            agent.critics[0]
                .online_replay_values::<2, 2, 3>(&batch.observations, &batch.actions)
                .unwrap()
        ),
        critic_before
    );
    let actions = agent.act_deterministic(&batch.observations).unwrap();
    assert_eq!(actions.dims(), [2, 1]);
    assert!(!actions.is_require_grad());
}

#[test]
fn malformed_terminal_observations_return_errors_before_replay_insertion() {
    let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
    let device = Device::flex().autodiff();
    for terminal_width in [0, 2] {
        let mut environment = VectorizedGymWrapper::from(EndingEnv {
            device: Device::flex(),
            terminated: false,
            terminal_width,
        });
        let mut agent = SACAgent::builder()
            .policy(ProbabilisticPolicyModel::new(
                linear_model(1, 2, 0.0, &device),
                CategoricalDistribution,
            ))
            .actor_optimizer(SgdConfig::new().init())
            .actor_learning_rate(0.1)
            .critics(vec![discrete_critic(&device, DType::F32)])
            .entropy_configuration(SACEntropyConfiguration::fixed(0.2).unwrap())
            .action_space(environment.action_space())
            .observation_space(environment.observation_space())
            .replay_storage_config(ReplayStorageConfig::new(
                super::super::ReplayDeviceStrategy::OneDevice(device.clone()),
            ))
            .replay_capacity(8)
            .batch_size(1)
            .training_horizon(4)
            .build()
            .unwrap();
        assert!(matches!(
            agent.learn(&mut environment, 1),
            Err(SACError::Shape {
                field: "terminal observations",
                ..
            })
        ));
        assert!(agent.replay.is_none());
        assert_eq!(agent.optimization_steps, 0);
    }
}

#[test]
fn integer_and_boolean_observations_reach_user_models_without_casting() {
    let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
    let device = Device::flex().autodiff();
    let policy = ProbabilisticPolicyModel::new(
        IntegerModel {
            weight: Param::from_tensor(Tensor::zeros([1, 2], (&device, DType::F64))),
        },
        CategoricalDistribution,
    );
    let critic = SACCritic::builder()
        .online_network(DiscreteVectorHeadCritic::new(IntegerModel {
            weight: Param::from_tensor(Tensor::ones([1, 2], (&device, DType::F64))),
        }))
        .optimizer(SgdConfig::new().init())
        .learning_rate(0.01)
        .build()
        .unwrap();
    let mut agent = SACAgent::<_, _, _, _, Infallible>::builder()
        .policy(policy)
        .actor_optimizer(SgdConfig::new().init())
        .actor_learning_rate(0.01)
        .critics(vec![critic])
        .entropy_configuration(SACEntropyConfiguration::fixed(0.2).unwrap())
        .action_space(Discrete::new(2))
        .observation_space(Discrete::new(10))
        .replay_storage_config(ReplayStorageConfig::new(
            super::super::ReplayDeviceStrategy::OneDevice(device.clone()),
        ))
        .dtype(DType::F64)
        .training_horizon(4)
        .build()
        .unwrap();
    let observations = Tensor::<2, Int>::from_data([[1i64], [2]], (&device, DType::I64));
    assert_eq!(agent.act(&observations).unwrap().dims(), [2, 1]);
    let batch = SACOptimizationBatch {
        observations: observations.clone(),
        next_observations: observations,
        actions: Tensor::zeros([2, 1], &device),
        rewards: Tensor::ones([2, 1], (&device, DType::F64)),
        terminated: Tensor::from_data([[false], [false]], &device),
        collection_policy_entropies: Tensor::zeros([2, 1], (&device, DType::F64)),
        entropy_change_weights: Tensor::zeros([2, 1], (&device, DType::F64)),
    };
    agent.optimize_actor(&batch).unwrap();
    assert_eq!(
        agent.compute_bellman_targets(&batch).unwrap().dtype(),
        DType::F64
    );

    let policy = ProbabilisticPolicyModel::new(
        BooleanModel {
            weight: Param::from_tensor(Tensor::zeros([1, 2], &device)),
        },
        CategoricalDistribution,
    );
    let critic = SACCritic::builder()
        .online_network(DiscreteVectorHeadCritic::new(BooleanModel {
            weight: Param::from_tensor(Tensor::ones([1, 2], &device)),
        }))
        .optimizer(SgdConfig::new().init())
        .learning_rate(0.01)
        .build()
        .unwrap();
    let mut agent = SACAgent::<_, _, _, _, Infallible>::builder()
        .policy(policy)
        .actor_optimizer(SgdConfig::new().init())
        .actor_learning_rate(0.01)
        .critics(vec![critic])
        .entropy_configuration(SACEntropyConfiguration::fixed(0.2).unwrap())
        .action_space(Discrete::new(2))
        .observation_space(BooleanSpace)
        .replay_storage_config(ReplayStorageConfig::new(
            super::super::ReplayDeviceStrategy::OneDevice(device.clone()),
        ))
        .training_horizon(4)
        .build()
        .unwrap();
    assert_eq!(
        agent
            .act(&Tensor::<2, Bool>::from_data([[true], [false]], &device))
            .unwrap()
            .dims(),
        [2, 1]
    );
}

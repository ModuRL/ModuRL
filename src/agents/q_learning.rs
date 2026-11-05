use bon::bon;
use burn::{
    module::Module,
    optim::{GradientsParams, ModuleOptimizer},
    tensor::{
        Bool, Cast, DType, Device, Float, FloatDType, IndexingUpdateOp, Int, IntDType, Tensor,
        TensorData, TensorReadError,
        kind::{Autodiff, Basic},
    },
};
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
use std::marker::PhantomData;

use crate::{
    agents::ReplayStorageConfig,
    buffers::experience_replay::{
        AlignedObservationReplay, ExperienceReplay, ExperienceReplayError, ReplayStorage,
        ReplayStorageError, TensorReplayColumn, replay_index_tensor, validate_replay_alignment,
    },
    gym::{MultiGym, MultiGymStepInfo},
    models::Forward,
    parameter_schedule::{LinearSchedule, ParameterSchedule, ScheduleProgress},
    spaces::{ActionSpace, Discrete, ObservationSpace, SpaceError},
    tensor_operations::transfer_bool_tensor,
    tensor_rank::{NextRank, TensorBatchTypes},
    wrappers::split_legal_action_mask,
};

pub mod ddqn;
pub mod dqn;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum QLearningConfigurationError {
    #[error("Q-network parameters must share one device")]
    ModelDevice,
    #[error("replay capacity must be nonzero")]
    ZeroReplayCapacity,
    #[error("batch size must be nonzero")]
    ZeroBatchSize,
    #[error("target update interval must be nonzero")]
    ZeroTargetUpdateInterval,
    #[error("update frequency must be nonzero")]
    ZeroUpdateFrequency,
    #[error("training horizon must be nonzero")]
    ZeroTrainingHorizon,
    #[error("replay capacity must be at least the batch size")]
    ReplayCapacityBelowBatchSize,
    #[error("gamma must be finite and in 0..=1")]
    InvalidGamma,
    #[error("epsilon schedule values must be finite and in 0..=1")]
    InvalidEpsilon,
    #[error("Q-learning compute dtype must be floating-point, got {0:?}")]
    InvalidDType(DType),
    #[error("learning rate must be finite and positive")]
    InvalidLearningRate,
    #[error("observation space has {actual} item axes, expected {expected}")]
    ObservationRank { expected: usize, actual: usize },
    #[error("legal-action masking requires rank-2 observations, got rank {actual}")]
    LegalActionMaskRequiresRank2 { actual: usize },
}

/// Q-learning replay storage failures.
#[derive(Debug, thiserror::Error)]
pub enum QLearningReplayError {
    #[error("replay storage failed: {0}")]
    Storage(#[from] ReplayStorageError),
}

#[derive(Debug, thiserror::Error)]
pub enum QAgentError<GE, ME>
where
    GE: std::fmt::Debug,
    ME: std::fmt::Debug,
{
    #[error("Q-learning tensor operation failed: {0}")]
    TensorError(#[from] TensorReadError),
    #[error("Q-learning model failed: {0}")]
    ModelError(#[source] ME),
    #[error("replay storage failed: {0}")]
    ReplayStorageError(#[source] QLearningReplayError),
    #[error("invalid Q-learning configuration: {0}")]
    ConfigurationError(#[source] QLearningConfigurationError),
    #[error("gym failed: {0}")]
    GymError(#[source] GE),
    #[error("space operation failed: {0}")]
    SpaceError(#[from] SpaceError),
    #[error(
        "legal-action mask observation has {actual} columns, fewer than {action_count} mask columns"
    )]
    LegalActionMaskShape { actual: usize, action_count: usize },
}

impl<GE, ME> From<std::convert::Infallible> for QAgentError<GE, ME>
where
    GE: std::fmt::Debug,
    ME: std::fmt::Debug,
{
    fn from(error: std::convert::Infallible) -> Self {
        match error {}
    }
}

impl<GE, ME> From<ReplayStorageError> for QAgentError<GE, ME>
where
    GE: std::fmt::Debug,
    ME: std::fmt::Debug,
{
    fn from(error: ReplayStorageError) -> Self {
        Self::ReplayStorageError(error.into())
    }
}

impl<GE, ME> From<ExperienceReplayError<QLearningReplayError>> for QAgentError<GE, ME>
where
    GE: std::fmt::Debug,
    ME: std::fmt::Debug,
{
    fn from(error: ExperienceReplayError<QLearningReplayError>) -> Self {
        Self::ReplayStorageError(match error {
            ExperienceReplayError::ExperienceError(error) => error,
            ExperienceReplayError::InsertionExceedsCapacity { capacity, inserted } => {
                ReplayStorageError::InsertionExceedsCapacity { capacity, inserted }.into()
            }
        })
    }
}

impl<GE, ME> From<QLearningConfigurationError> for QAgentError<GE, ME>
where
    GE: std::fmt::Debug,
    ME: std::fmt::Debug,
{
    fn from(err: QLearningConfigurationError) -> Self {
        Self::ConfigurationError(err)
    }
}

pub struct QLogEntry {
    /// Float loss `[1]` in the compute dtype on the optimization device.
    pub loss: Tensor<1>,
    pub epsilon: f64,
    pub learning_rate: f32,
    /// Selected Float Q values `[batch_size, 1]` in the compute dtype.
    pub q_values: Tensor<2>,
    /// Sampled Float rewards `[batch_size, 1]` in the compute dtype.
    pub replay_rewards: Tensor<2>,
    pub update_index: usize,
    pub collection_timestep: usize,
}

pub struct QCollectionLogEntry<I = ()> {
    /// Fresh Float rewards `[num_envs, 1]` with the environment's dtype and device.
    pub collection_rewards: Tensor<2>,
    pub infos: Vec<I>,
    pub epsilon: f64,
    pub collection_timestep: usize,
    pub completed_episodes: Vec<QEpisodeLogEntry>,
}

pub struct QEpisodeLogEntry {
    pub environment_index: usize,
    pub episode_return: f32,
    pub episode_length: usize,
    pub terminated: bool,
    pub truncated: bool,
    pub collection_timestep: usize,
}

struct QEpisodeTracker {
    returns: Vec<f32>,
    lengths: Vec<usize>,
}

impl QEpisodeTracker {
    fn new(environment_count: usize) -> Self {
        Self {
            returns: vec![0.0; environment_count],
            lengths: vec![0; environment_count],
        }
    }

    fn record(
        &mut self,
        environment_index: usize,
        reward: f32,
        terminated: bool,
        truncated: bool,
        collection_timestep: usize,
    ) -> Option<QEpisodeLogEntry> {
        self.returns[environment_index] += reward;
        self.lengths[environment_index] += 1;
        if !terminated && !truncated {
            return None;
        }

        let entry = QEpisodeLogEntry {
            environment_index,
            episode_return: self.returns[environment_index],
            episode_length: self.lengths[environment_index],
            terminated,
            truncated,
            collection_timestep,
        };
        self.returns[environment_index] = 0.0;
        self.lengths[environment_index] = 0;
        Some(entry)
    }
}

pub(crate) trait QLearningLogger<I = ()> {
    fn log_update(&mut self, entry: &QLogEntry);

    fn log_collection(&mut self, entry: &QCollectionLogEntry<I>);
}

pub(crate) trait QLearningTarget {
    fn requires_online_next_q_values() -> bool;

    /// Computes Float targets `[batch_size, 1]` from Float rewards and Bool termination flags `[batch_size, 1]`.
    /// Q tensors use `[batch_size, action_count]` on the reward device and in the reward dtype.
    /// Truncation alone must not set the termination flags. The caller detaches returned targets.
    fn target_q_values(
        rewards: &Tensor<2>,
        next_dones: &Tensor<2, Bool>,
        online_next_q_values: Option<&Tensor<2>>,
        target_next_q_values: &Tensor<2>,
        gamma: f32,
    ) -> Tensor<2>;
}

struct QLearningBatch<const R: usize, K: Autodiff> {
    observations: Tensor<R, K>,
    next_observations: Tensor<R, K>,
    actions: Tensor<2, Int>,
    rewards: Tensor<2>,
    next_dones: Tensor<2, Bool>,
}

struct QLearningInsert<const R: usize, K: Autodiff> {
    observations: Tensor<R, K>,
    next_observations: Tensor<R, K>,
    actions: Tensor<2, Int>,
    rewards: Tensor<2>,
    next_dones: Tensor<2, Bool>,
    truncateds: Vec<bool>,
}

struct QLearningReplayStorage<const R: usize, K: Autodiff> {
    observations: Option<AlignedObservationReplay<R, K>>,
    actions: TensorReplayColumn<2, Int>,
    rewards: TensorReplayColumn<2>,
    next_dones: TensorReplayColumn<2, Bool>,
    observation_storage_shape: [usize; R],
    device: Device,
}

impl<const R: usize, K: Autodiff> QLearningReplayStorage<R, K> {
    /// Configures observation storage [capacity, ...observation_shape], Int actions [capacity, 1], and scalar statistics [capacity, 1].
    /// The first observation batch selects the observation dtype. All stored experience has no autodiff graph.
    fn new(observation_storage_shape: [usize; R], device: Device) -> Self {
        const {
            assert!(R >= 2, "observations require batch and item axes");
        }
        Self {
            observations: None,
            actions: TensorReplayColumn::new(
                [observation_storage_shape[0], 1],
                (&device, DType::U32),
            ),
            rewards: TensorReplayColumn::new(
                [observation_storage_shape[0], 1],
                (&device, DType::F32),
            ),
            next_dones: TensorReplayColumn::new([observation_storage_shape[0], 1], &device),
            observation_storage_shape,
            device,
        }
    }

    fn validate_environment_count(
        &self,
        environment_count: usize,
    ) -> Result<(), ReplayStorageError> {
        if let Some(observations) = &self.observations {
            observations.validate_environment_count(environment_count)
        } else {
            validate_replay_alignment(self.observation_storage_shape[0], environment_count)
                .map(|_| ())
        }
    }
}

impl<const R: usize, K: Autodiff> ReplayStorage for QLearningReplayStorage<R, K> {
    type Insert = QLearningInsert<R, K>;
    type Batch = QLearningBatch<R, K>;
    type Error = QLearningReplayError;

    fn capacity(&self) -> usize {
        self.observation_storage_shape[0]
    }

    /// Inserts native rank-R observations [num_envs, ...observation_shape], Int actions [num_envs, 1], and statistics [num_envs, 1].
    /// Observations retain their kind and first-batch dtype. Transfers to storage device and detaches every column.
    fn insert(&mut self, start: usize, transitions: Self::Insert) -> Result<usize, Self::Error> {
        let count = transitions.observations.dims()[0];
        for (field, actual) in [
            ("actions", transitions.actions.dims()[0]),
            ("rewards", transitions.rewards.dims()[0]),
            ("next dones", transitions.next_dones.dims()[0]),
        ] {
            if actual != count {
                return Err(ReplayStorageError::BatchLengthMismatch {
                    field,
                    expected: count,
                    actual,
                }
                .into());
            }
        }
        let dtype = transitions.observations.dtype();
        let mut new_observations = if self.observations.is_none() {
            Some(AlignedObservationReplay::new(
                self.observation_storage_shape,
                count,
                (&self.device, dtype),
            )?)
        } else {
            None
        };
        let observations = match (&mut self.observations, &mut new_observations) {
            (Some(observations), _) => observations,
            (None, Some(observations)) => observations,
            (None, None) => return Err(ReplayStorageError::EnvironmentCountNotSet.into()),
        };
        observations.insert(
            start,
            transitions.observations,
            transitions.next_observations,
            &transitions.truncateds,
        )?;
        if let Some(observations) = new_observations {
            self.observations = Some(observations);
        }
        self.actions.write(
            start,
            transitions.actions.to_device(&self.device).cast(DType::U32),
        )?;
        self.rewards.write(
            start,
            transitions.rewards.to_device(&self.device).cast(DType::F32),
        )?;
        self.next_dones.write(
            start,
            transfer_bool_tensor(transitions.next_dones, &self.device),
        )?;
        Ok(count)
    }

    /// Samples detached native rank-R observations [sample_count, ...observation_shape], actions [sample_count, 1], and statistics [sample_count, 1].
    /// Observation kind, dtype, and item axes remain unchanged on the storage device.
    fn gather(&self, indices: &[usize]) -> Result<Self::Batch, Self::Error> {
        let observations = self
            .observations
            .as_ref()
            .ok_or(ReplayStorageError::EnvironmentCountNotSet)?;
        let (observations, next_observations) = observations.gather(indices)?;
        let rows = replay_index_tensor(indices, &self.device)?;
        Ok(QLearningBatch {
            observations,
            next_observations,
            actions: self.actions.gather(rows.clone()),
            rewards: self.rewards.gather(rows.clone()),
            next_dones: self.next_dones.gather(rows),
        })
    }

    fn sampleable_len(&self, len: usize) -> usize {
        self.observations
            .as_ref()
            .map_or(0, |storage| storage.sampleable_len(len))
    }

    fn sample_index(&self, index: usize, len: usize) -> usize {
        self.observations
            .as_ref()
            .map_or(index, |storage| storage.sample_index(index, len))
    }
}

struct QCollectedTransitions<'a, const R: usize, K: Autodiff> {
    observations: &'a Tensor<R, K>,
    next_observations: &'a Tensor<R, K>,
    actions: &'a Tensor<2, Int>,
    rewards: &'a Tensor<2>,
    dones: &'a [bool],
    truncateds: &'a [bool],
    first_timestep: usize,
}

pub(crate) struct QLearningAgent<M, S, GE, T, const R: usize = 2>
where
    S: ObservationSpace<R>,
    S::Kind: Autodiff,
{
    online_q_network: M,
    target_q_network: M,
    target_update_interval: usize,
    optimizer: ModuleOptimizer,
    learning_rate: f64,
    current_epsilon: f64,
    epsilon_schedule: Box<dyn ParameterSchedule>,
    schedule_progress: ScheduleProgress,
    action_space: Discrete,
    observation_space: S,
    experience_replay:
        ExperienceReplay<QLearningInsert<R, S::Kind>, QLearningReplayStorage<R, S::Kind>>,
    gamma: f32,
    update_frequency: usize,
    training_start: usize,
    replay_storage_config: ReplayStorageConfig,
    dtype: DType,
    optimization_steps: usize,
    action_rng: StdRng,
    timestep_episodes: Option<QEpisodeTracker>,
    legal_action_mask: bool,
    _phantom: PhantomData<(GE, T)>,
}

#[bon]
impl<M, S, GE, T, const R: usize, ME: std::fmt::Debug> QLearningAgent<M, S, GE, T, R>
where
    S: ObservationSpace<R>,
    S::Kind: Autodiff,
    M: Module + Forward<R, 2, InputKind = S::Kind, OutputKind = Float, Error = ME>,
    GE: std::fmt::Debug,
    T: QLearningTarget,
{
    /// Creates a trainable Q model for rank-R observations [batch_size, ...observation_shape] and Q outputs [batch_size, action_count].
    /// The target starts from the online model and remains detached between hard updates.
    /// Observations retain native kind and dtype. Model Q outputs must use the configured compute dtype.
    /// Model parameters must use the optimization device. Replay matches the first observation dtype on its storage device.
    /// Replay storage configuration selects the storage and optimization devices.
    #[builder]
    pub(crate) fn new(
        action_space: Discrete,
        observation_space: S,
        online_q_network: M,
        #[builder(into)] optimizer: ModuleOptimizer,
        learning_rate: f64,
        #[builder(default = 1000)] target_update_interval: usize,
        #[builder(default = Box::new(LinearSchedule::new(1.0, 0.1)), with = |schedule: impl ParameterSchedule + 'static| Box::new(schedule))]
        epsilon_schedule: Box<dyn ParameterSchedule>,
        #[builder(default = 10000)] replay_capacity: usize,
        #[builder(default = 32)] batch_size: usize,
        #[builder(default = 0.99)] gamma: f32,
        #[builder(default = 4)] update_frequency: usize,
        #[builder(default = 1000)] training_start: usize,
        training_horizon: Option<usize>,
        replay_device: Option<Device>,
        #[builder(default = DType::F32)] dtype: DType,
        #[builder(default)] legal_action_mask: bool,
    ) -> Result<Self, QAgentError<GE, ME>> {
        const {
            assert!(R >= 2, "observations require batch and item axes");
        }
        if !dtype.is_float() {
            return Err(QLearningConfigurationError::InvalidDType(dtype).into());
        }
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(QLearningConfigurationError::InvalidLearningRate.into());
        }
        let observation_shape = observation_space.shape();
        if legal_action_mask && R != 2 {
            return Err(
                QLearningConfigurationError::LegalActionMaskRequiresRank2 { actual: R }.into(),
            );
        }
        let initial_epsilon = epsilon_schedule.value(0.0);
        QLearningConfigurationValidator::validate_configuration()
            .replay_capacity(replay_capacity)
            .batch_size(batch_size)
            .gamma(gamma)
            .initial_epsilon(initial_epsilon)
            .final_epsilon(epsilon_schedule.value(1.0))
            .update_frequency(update_frequency)
            .target_update_interval(target_update_interval)
            .maybe_training_horizon(training_horizon)
            .call()?;

        let mut observation_storage_shape = vec![replay_capacity];
        observation_storage_shape.extend(observation_shape);
        let actual = observation_storage_shape.len() - 1;
        let observation_storage_shape: [usize; R] =
            observation_storage_shape.try_into().map_err(|_| {
                QLearningConfigurationError::ObservationRank {
                    expected: R - 1,
                    actual,
                }
            })?;
        let replay_storage_config =
            ReplayStorageConfig::from_models(online_q_network.devices(), replay_device)
                .ok_or(QLearningConfigurationError::ModelDevice)?;
        let replay_storage = QLearningReplayStorage::<R, S::Kind>::new(
            observation_storage_shape,
            replay_storage_config.storage_device(),
        );
        let optimization_device = replay_storage_config.optimization_device();
        // Seed the owned exploration RNG once from the configured device RNG; action selection needs no later RNG host reads.
        let action_seed = Tensor::<1>::random(
            [1],
            burn::tensor::Distribution::Uniform(0.0, u32::MAX as f64),
            (&optimization_device, DType::F64),
        )
        .cast(IntDType::U32)
        .try_into_scalar::<u32>()?;
        let online_q_network = online_q_network.train();
        let target_q_network = online_q_network.clone().valid();
        Ok(Self {
            online_q_network,
            target_q_network,
            target_update_interval,
            optimizer,
            learning_rate,
            current_epsilon: initial_epsilon,
            epsilon_schedule,
            schedule_progress: ScheduleProgress::new(training_horizon),
            action_space,
            observation_space,
            experience_replay: ExperienceReplay::with_storage(replay_storage, batch_size),
            gamma,
            update_frequency,
            training_start,
            replay_storage_config,
            dtype,
            optimization_steps: 0,
            action_rng: StdRng::seed_from_u64(u64::from(action_seed)),
            timestep_episodes: None,
            legal_action_mask,
            _phantom: PhantomData,
        })
    }
}

impl<M, S, GE, T, const R: usize, ME: std::fmt::Debug> QLearningAgent<M, S, GE, T, R>
where
    S: ObservationSpace<R>,
    S::Kind: Autodiff,
    M: Module + Forward<R, 2, InputKind = S::Kind, OutputKind = Float, Error = ME>,
    GE: std::fmt::Debug,
    T: QLearningTarget,
{
    pub(crate) fn get_action_space(&self) -> &Discrete {
        &self.action_space
    }

    pub(crate) fn get_observation_space(&self) -> &S {
        &self.observation_space
    }

    /// Moves rank-R observations [batch_size, ...observation_shape] to the optimization device.
    /// Preserves every axis, native kind, dtype, and gradient path. The model must accept the observation dtype.
    fn model_observations(&self, observations: Tensor<R, S::Kind>) -> Tensor<R, S::Kind> {
        observations.to_device(&self.replay_storage_config.optimization_device())
    }

    /// Selects U32 Int actions [batch_size, 1] from rank-R observations [batch_size, ...observation_shape].
    /// Selection uses the optimization device and the observations' native kind and dtype.
    pub(crate) fn act(
        &mut self,
        observations: &Tensor<R, S::Kind>,
    ) -> Result<Tensor<2, Int>, QAgentError<GE, ME>>
    where
        FloatDType: Cast<2, S::Kind, OutputKind = Float>,
    {
        let observations = self.model_observations(observations.clone());
        let legal_action_mask = if self.legal_action_mask {
            let batch_size = observations.dims()[0];
            let feature_count = observations.dims()[1..].iter().product();
            let flat_observations = observations.clone().reshape([batch_size, feature_count]);
            let (_, mask) =
                split_legal_action_mask(flat_observations, self.action_space.get_possible_values())
                    .map_err(|error| QAgentError::LegalActionMaskShape {
                        actual: error.columns,
                        action_count: error.action_count,
                    })?;
            Some(mask)
        } else {
            None
        };
        epsilon_greedy_actions(
            &observations,
            self.current_epsilon,
            &self.action_space,
            &mut self.action_rng,
            legal_action_mask.as_ref(),
            |observations| self.online_q_network.forward(observations.clone()),
        )
        .map_err(|error| match error {
            EpsilonGreedyError::ModelError(error) => QAgentError::ModelError(error),
            EpsilonGreedyError::SpaceError(error) => QAgentError::SpaceError(error),
        })
    }

    /// Samples detached rank-R observations, Int actions [batch_size, 1], and statistics [batch_size, 1].
    /// Computes a Float loss [1] in compute dtype on the optimization device and updates only the online model.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            name = "q_learning.optimize",
            target = "modurl::performance",
            skip_all,
            fields(collection_timestep)
        )
    )]
    fn optimize<I>(
        &mut self,
        collection_timestep: usize,
        logger: &mut dyn QLearningLogger<I>,
    ) -> Result<(), QAgentError<GE, ME>> {
        if self.experience_replay.len() < self.experience_replay.get_batch_size() {
            return Ok(());
        }
        let optimization_device = self.replay_storage_config.optimization_device();
        let batch = self.experience_replay.sample()?;
        let observations = self.model_observations(batch.observations);
        let next_observations = self.model_observations(batch.next_observations);
        let actions = batch.actions.to_device(&optimization_device);
        let rewards = batch
            .rewards
            .to_device(&optimization_device)
            .cast(self.dtype);
        let next_dones = transfer_bool_tensor(batch.next_dones, &optimization_device);
        let target_next_q_values = self
            .target_q_network
            .forward(next_observations.clone())
            .map_err(QAgentError::ModelError)?;
        let online_next_q_values = T::requires_online_next_q_values()
            .then(|| self.online_q_network.forward(next_observations))
            .transpose()
            .map_err(QAgentError::ModelError)?;
        let target_q_values = T::target_q_values(
            &rewards,
            &next_dones,
            online_next_q_values.as_ref(),
            &target_next_q_values,
            self.gamma,
        )
        .detach();
        let q_values = self
            .online_q_network
            .forward(observations)
            .map_err(QAgentError::ModelError)?;
        let selected_q_values = selected_action_q_values(&q_values, &actions);
        let loss = (selected_q_values.clone() - target_q_values)
            .square()
            .mean();
        logger.log_update(&QLogEntry {
            loss: loss.clone(),
            epsilon: self.current_epsilon,
            learning_rate: self.learning_rate as f32,
            q_values: selected_q_values,
            replay_rewards: rewards,
            update_index: self.optimization_steps,
            collection_timestep,
        });
        self.optimization_steps += 1;
        let gradients = GradientsParams::from_grads(loss.backward(), &self.online_q_network);
        self.online_q_network =
            self.optimizer
                .step(self.learning_rate, self.online_q_network.clone(), gradients);
        Ok(())
    }

    fn update_target_network(&mut self) {
        self.target_q_network = self.online_q_network.clone().valid();
    }

    /// Stores native rank-R observations and next observations [num_envs, ...observation_shape] with actions [num_envs, 1].
    /// Rewards [num_envs, 1] feed episode metrics. Only termination blocks bootstrapping; truncation retains terminal observations.
    fn store_vectorized_transitions(
        &mut self,
        transitions: QCollectedTransitions<'_, R, S::Kind>,
        episodes: &mut QEpisodeTracker,
    ) -> Result<Vec<QEpisodeLogEntry>, QAgentError<GE, ME>> {
        let environment_count = transitions.dones.len();
        let reward_values = transitions
            .rewards
            .clone()
            .try_into_data_as::<f32>()?
            .try_to_vec::<f32>()
            .map_err(TensorReadError::from)?;
        let next_dones = Tensor::<2, Bool>::from_data(
            TensorData::new(transitions.dones.to_vec(), [environment_count, 1]),
            &self.replay_storage_config.storage_device(),
        );
        self.experience_replay.add(QLearningInsert {
            observations: transitions.observations.clone(),
            next_observations: transitions.next_observations.clone(),
            actions: transitions.actions.clone(),
            rewards: transitions.rewards.clone(),
            next_dones,
            truncateds: transitions.truncateds.to_vec(),
        })?;
        let mut completed_episodes = Vec::new();
        for (environment_index, reward) in reward_values.into_iter().enumerate() {
            let collection_timestep = transitions
                .first_timestep
                .saturating_add(environment_index + 1);
            if let Some(entry) = episodes.record(
                environment_index,
                reward,
                transitions.dones[environment_index],
                transitions.truncateds[environment_index],
                collection_timestep,
            ) {
                completed_episodes.push(entry);
            }
        }
        Ok(completed_episodes)
    }

    /// Validates one transition's leading environment batch shape.
    fn validate_timestep<I, const U: usize>(
        &mut self,
        observations: &Tensor<R, S::Kind>,
        actions: &Tensor<2, Int>,
        step: &MultiGymStepInfo<I, Tensor<U, S::Kind>>,
    ) -> Result<usize, QAgentError<GE, ME>>
    where
        Tensor<U, S::Kind>: NextRank<Next = Tensor<R, S::Kind>>,
    {
        let count = step.dones.len();
        for (field, actual) in [
            ("observations", observations.dims()[0]),
            ("training actions", actions.dims()[0]),
            ("truncation flags", step.truncateds.len()),
            ("terminal observations", step.terminal_observations.len()),
            ("environment infos", step.infos.len()),
            ("environment observations", step.observations.dims()[0]),
            ("environment rewards", step.rewards.dims()[0]),
        ] {
            if actual != count {
                return Err(ReplayStorageError::BatchLengthMismatch {
                    field,
                    expected: count,
                    actual,
                }
                .into());
            }
        }
        Ok(count)
    }

    /// Stores one already-validated transition with batch shape `[num_envs, ...]`.
    fn collect_timestep<I, const U: usize>(
        &mut self,
        observations: &Tensor<R, S::Kind>,
        actions: Tensor<2, Int>,
        step: MultiGymStepInfo<I, Tensor<U, S::Kind>>,
        episodes: &mut QEpisodeTracker,
        logger: &mut dyn QLearningLogger<I>,
    ) -> Result<(), QAgentError<GE, ME>>
    where
        Tensor<U, S::Kind>: NextRank<Next = Tensor<R, S::Kind>>,
    {
        let count = step.dones.len();
        let first_timestep = self.schedule_progress.elapsed_steps();
        let transition_next_observations = step.transition_next_observations();
        let collection_rewards = step.rewards.clone();
        let completed_episodes = self.store_vectorized_transitions(
            QCollectedTransitions {
                observations,
                next_observations: &transition_next_observations,
                actions: &actions,
                rewards: &step.rewards,
                dones: &step.dones,
                truncateds: &step.truncateds,
                first_timestep,
            },
            episodes,
        )?;
        let entry = QCollectionLogEntry {
            collection_rewards,
            infos: step.infos,
            epsilon: self.current_epsilon,
            collection_timestep: first_timestep.saturating_add(count),
            completed_episodes,
        };
        let update_result = self.run_scheduled_updates(first_timestep, count, logger);
        if update_result.is_ok() {
            logger.log_collection(&entry);
        }
        self.schedule_progress.advance_steps(count);
        self.current_epsilon = validate_epsilon(
            self.schedule_progress
                .parameter(self.epsilon_schedule.as_ref()),
        )?;
        update_result
    }

    /// Skips remaining updates after a failure but still runs scheduled target syncs.
    fn run_scheduled_updates<I>(
        &mut self,
        first_timestep: usize,
        environment_count: usize,
        logger: &mut dyn QLearningLogger<I>,
    ) -> Result<(), QAgentError<GE, ME>> {
        let mut result = Ok(());
        for offset in 1..=environment_count {
            let timestep = first_timestep.saturating_add(offset);
            if result.is_ok()
                && timestep.is_multiple_of(self.update_frequency)
                && timestep >= self.training_start
            {
                result = self.optimize(timestep, logger);
            }
            if timestep.is_multiple_of(self.target_update_interval) {
                self.update_target_network();
            }
        }
        result
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            name = "q_learning.learn",
            target = "modurl::performance",
            skip_all,
            fields(num_timesteps)
        )
    )]
    /// Collects native rank-R observations `[num_envs, ...observation_shape]` and Int actions `[num_envs, 1]`.
    /// Terminal observations omit the batch axis and preserve observation kind, dtype, and device.
    /// Each call resets environments; schedule and replay progress persist.
    pub(crate) fn learn<I, const U: usize>(
        &mut self,
        env: &mut dyn MultiGym<I, R, 2, Error = GE, ObservationSpace = S, ActionSpace = Discrete>,
        num_timesteps: usize,
        logger: &mut dyn QLearningLogger<I>,
    ) -> Result<(), QAgentError<GE, ME>>
    where
        S: ObservationSpace<R, Item = Tensor<U, <S as TensorBatchTypes>::Kind>>,
        Tensor<U, S::Kind>: NextRank<Next = Tensor<R, S::Kind>>,
        FloatDType: Cast<2, S::Kind, OutputKind = Float>,
    {
        let mut elapsed_timesteps = 0;
        let environment_count = env.num_envs();
        self.timestep_episodes = None;
        self.experience_replay
            .storage_mut()
            .validate_environment_count(environment_count)?;
        let mut observations = env.reset().map_err(QAgentError::GymError)?;
        self.schedule_progress.default_horizon(num_timesteps);
        let mut episodes = QEpisodeTracker::new(environment_count);
        while elapsed_timesteps < num_timesteps {
            self.current_epsilon = validate_epsilon(
                self.schedule_progress
                    .parameter(self.epsilon_schedule.as_ref()),
            )?;
            let actions = self.act(&observations)?;
            let step_info = env.step(actions.clone()).map_err(QAgentError::GymError)?;
            let observations_before = observations.clone();
            let collected_count =
                self.validate_timestep(&observations_before, &actions, &step_info)?;
            observations = step_info.observations.clone();
            elapsed_timesteps += collected_count;
            self.collect_timestep(
                &observations_before,
                actions,
                step_info,
                &mut episodes,
                logger,
            )?;
        }
        Ok(())
    }

    /// Processes a caller-collected transition with matching environment batch shapes.
    pub(crate) fn learn_from_timesteps<I, const U: usize>(
        &mut self,
        observations: &Tensor<R, S::Kind>,
        actions: Tensor<2, Int>,
        step: MultiGymStepInfo<I, Tensor<U, S::Kind>>,
        logger: &mut dyn QLearningLogger<I>,
    ) -> Result<(), QAgentError<GE, ME>>
    where
        Tensor<U, S::Kind>: NextRank<Next = Tensor<R, S::Kind>>,
    {
        let count = self.validate_timestep(observations, &actions, &step)?;
        self.experience_replay
            .storage_mut()
            .validate_environment_count(count)?;
        let mut episodes = match self.timestep_episodes.take() {
            Some(episodes) if episodes.returns.len() == count => episodes,
            _ => QEpisodeTracker::new(count),
        };
        let result = self.collect_timestep(observations, actions, step, &mut episodes, logger);
        self.timestep_episodes = Some(episodes);
        result
    }
}
struct QLearningConfigurationValidator;

#[bon]
impl QLearningConfigurationValidator {
    #[builder]
    pub(crate) fn validate_configuration(
        replay_capacity: usize,
        batch_size: usize,
        gamma: f32,
        initial_epsilon: f64,
        final_epsilon: f64,
        update_frequency: usize,
        target_update_interval: usize,
        training_horizon: Option<usize>,
    ) -> Result<(), QLearningConfigurationError> {
        if replay_capacity == 0 {
            return Err(QLearningConfigurationError::ZeroReplayCapacity);
        }
        if batch_size == 0 {
            return Err(QLearningConfigurationError::ZeroBatchSize);
        }
        if target_update_interval == 0 {
            return Err(QLearningConfigurationError::ZeroTargetUpdateInterval);
        }
        if update_frequency == 0 {
            return Err(QLearningConfigurationError::ZeroUpdateFrequency);
        }
        if training_horizon == Some(0) {
            return Err(QLearningConfigurationError::ZeroTrainingHorizon);
        }
        if replay_capacity < batch_size {
            return Err(QLearningConfigurationError::ReplayCapacityBelowBatchSize);
        }
        if !gamma.is_finite() || !(0.0..=1.0).contains(&gamma) {
            return Err(QLearningConfigurationError::InvalidGamma);
        }
        validate_epsilon(initial_epsilon)?;
        validate_epsilon(final_epsilon)?;
        Ok(())
    }
}

pub(crate) fn validate_epsilon(epsilon: f64) -> Result<f64, QLearningConfigurationError> {
    if !epsilon.is_finite() || !(0.0..=1.0).contains(&epsilon) {
        return Err(QLearningConfigurationError::InvalidEpsilon);
    }
    Ok(epsilon)
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum EpsilonGreedyError<E> {
    #[error("Q-network failed: {0}")]
    ModelError(#[source] E),
    #[error("action-space sampling failed: {0}")]
    SpaceError(#[from] SpaceError),
}

/// Selects U32 Int actions `[batch_size, 1]` from rank-R observations `[batch_size, ...observation_shape]`.
/// An optional legal-action mask has shape `[batch_size, action_count]`.
/// The `rng` argument only decides which rows explore; random actions use the backend device RNG, seeded with `Device::seed`.
/// The callback receives only greedy rows and returns Float Q values `[selected_batch, action_count]` on the input device.
/// The caller validates epsilon and category count.
pub(crate) fn epsilon_greedy_actions<const R: usize, K: Basic, E>(
    observations: &Tensor<R, K>,
    epsilon: f64,
    action_space: &Discrete,
    rng: &mut impl Rng,
    legal_action_mask: Option<&Tensor<2, K>>,
    forward: impl FnOnce(&Tensor<R, K>) -> Result<Tensor<2>, E>,
) -> Result<Tensor<2, Int>, EpsilonGreedyError<E>>
where
    FloatDType: Cast<2, K, OutputKind = Float>,
{
    const {
        assert!(R >= 2, "observations require batch and item axes");
    }
    let batch_size = observations.dims()[0];
    let device = observations.device();
    if epsilon == 0.0 {
        return Ok(forward(observations)
            .map_err(EpsilonGreedyError::ModelError)?
            .argmax(1)
            .cast(DType::U32));
    }
    let mut random_rows = Vec::new();
    let mut greedy_rows = Vec::with_capacity(batch_size);
    for index in 0..batch_size {
        if rng.random_bool(epsilon) {
            random_rows.push(index as i64);
        } else {
            greedy_rows.push(index as i64);
        }
    }
    if greedy_rows.len() == batch_size {
        return Ok(forward(observations)
            .map_err(EpsilonGreedyError::ModelError)?
            .argmax(1)
            .cast(DType::U32));
    }
    let mut actions = Tensor::<2, Int>::zeros([batch_size, 1], (&device, DType::U32));
    if !random_rows.is_empty() {
        let random_count = random_rows.len();
        let rows = Tensor::<1, Int>::from_data(
            TensorData::new(random_rows, [random_count]),
            (&device, DType::I64),
        );
        let random_actions = match legal_action_mask {
            Some(mask) => action_space.sample_batch_masked(&<FloatDType as Cast<2, K>>::cast(
                mask.clone().select(0, rows.clone()),
                FloatDType::F32,
            )),
            None => action_space
                .sample_batch(random_count, &device)
                .map(|actions| actions.cast(DType::U32)),
        }
        .map_err(EpsilonGreedyError::SpaceError)?;
        actions = actions.select_assign(0, rows, random_actions, IndexingUpdateOp::Assign);
    }
    if greedy_rows.is_empty() {
        return Ok(actions);
    }
    let greedy_count = greedy_rows.len();
    let rows = Tensor::<1, Int>::from_data(
        TensorData::new(greedy_rows, [greedy_count]),
        (&device, DType::I64),
    );
    let selected = observations.clone().select(0, rows.clone());
    let greedy_actions = forward(&selected)
        .map_err(EpsilonGreedyError::ModelError)?
        .argmax(1)
        .cast(DType::U32);
    Ok(actions.select_assign(0, rows, greedy_actions, IndexingUpdateOp::Assign))
}

/// Gathers Float Q values [batch_size, action_count] using Int actions [batch_size, 1].
/// Returns [batch_size, 1], preserving Q dtype, device, and gradients. Each action must be a valid column index.
pub(crate) fn selected_action_q_values(
    q_values: &Tensor<2>,
    actions: &Tensor<2, Int>,
) -> Tensor<2> {
    q_values.clone().gather(1, actions.clone())
}

#[cfg(test)]
mod tests;

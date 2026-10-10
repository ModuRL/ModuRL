use bon::bon;
use burn::{
    module::Module,
    optim::{GradientsParams, ModuleOptimizer},
    tensor::{
        Bool, DType, Device, Float, IndexingUpdateOp, Int, IntDType, Tensor, TensorData,
        TensorReadError,
        kind::{Autodiff, Basic},
    },
};
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
use std::marker::PhantomData;

use crate::{
    agents::ReplayStorageConfig,
    buffers::experience_replay::{
        AlignedObservationReplay, ExperienceReplay, ExperienceReplayError, ReplayStorage,
        ReplayStorageError, TensorReplayColumn, replay_index_tensor,
    },
    gym::{MultiGym, MultiGymStepInfo},
    models::Forward,
    parameter_schedule::{LinearSchedule, ParameterSchedule, ScheduleProgress},
    spaces::{Discrete, ObservationSpace, SpaceError},
    tensor_rank::{NextRank, PrevRank},
};

pub mod ddqn;
pub mod dqn;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum QLearningConfigurationError {
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
}

/// Q-learning replay alignment failures or a change to the established observation dtype.
#[derive(Debug, thiserror::Error)]
pub enum QLearningReplayError {
    #[error("replay storage failed: {0}")]
    Storage(#[from] ReplayStorageError),
    #[error("replay observation dtype is {actual:?}, expected {expected:?}")]
    ObservationDType { expected: DType, actual: DType },
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
    shape: [usize; R],
    device: Device,
    observation_dtype: Option<DType>,
    environment_count: Option<usize>,
}

impl<const R: usize, K: Autodiff> QLearningReplayStorage<R, K> {
    /// Configures observation storage [capacity, ...observation_shape], Int actions [capacity, 1], and scalar statistics [capacity, 1].
    /// The first observation batch selects the observation dtype. All stored experience has no autodiff graph.
    fn new(shape: [usize; R], device: Device) -> Self {
        const {
            assert!(R >= 2, "observations require batch and item axes");
        }
        Self {
            observations: None,
            actions: TensorReplayColumn::new([shape[0], 1], (&device, DType::U32)),
            rewards: TensorReplayColumn::new([shape[0], 1], (&device, DType::F32)),
            next_dones: TensorReplayColumn::new([shape[0], 1], &device),
            shape,
            device,
            observation_dtype: None,
            environment_count: None,
        }
    }

    fn initialize_environment_count(&mut self, count: usize) -> Result<(), ReplayStorageError> {
        if let Some(expected) = self.environment_count {
            return if expected == count {
                Ok(())
            } else {
                Err(ReplayStorageError::EnvironmentCountMismatch {
                    expected,
                    actual: count,
                })
            };
        }
        let capacity = self.shape[0];
        if count == 0 || capacity <= count || !capacity.is_multiple_of(count) {
            return Err(ReplayStorageError::InvalidReplayAlignment {
                capacity,
                environment_count: count,
            });
        }
        capacity
            .checked_add(count)
            .ok_or(ReplayStorageError::CapacityOverflow {
                capacity,
                additional: count,
            })?;
        self.environment_count = Some(count);
        Ok(())
    }
}

impl<const R: usize, K: Autodiff> ReplayStorage for QLearningReplayStorage<R, K> {
    type Insert = QLearningInsert<R, K>;
    type Batch = QLearningBatch<R, K>;
    type Error = QLearningReplayError;

    fn capacity(&self) -> usize {
        self.shape[0]
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
        self.initialize_environment_count(count)?;
        let dtype = transitions.observations.dtype();
        let expected = self.observation_dtype.unwrap_or(dtype);
        for actual in [dtype, transitions.next_observations.dtype()] {
            if actual != expected {
                return Err(QLearningReplayError::ObservationDType { expected, actual });
            }
        }
        if self.observations.is_none() {
            self.observations = Some(AlignedObservationReplay::new(
                self.shape,
                (&self.device, dtype),
            ));
            self.observation_dtype = Some(dtype);
        }
        let observations = self
            .observations
            .as_mut()
            .ok_or(ReplayStorageError::EnvironmentCountNotSet)?;
        observations.insert(
            start,
            transitions.observations.to_device(&self.device),
            transitions.next_observations.to_device(&self.device),
            &transitions.truncateds,
        )?;
        self.actions.write(
            start,
            transitions.actions.to_device(&self.device).cast(DType::U32),
        )?;
        self.rewards.write(
            start,
            transitions.rewards.to_device(&self.device).cast(DType::F32),
        )?;
        self.next_dones
            .write(start, transitions.next_dones.to_device(&self.device))?;
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
    _phantom: PhantomData<(GE, T)>,
}

#[bon]
impl<M, S, GE, T, const R: usize, K: Autodiff, ME: std::fmt::Debug> QLearningAgent<M, S, GE, T, R>
where
    S: ObservationSpace<R, Kind = K>,
    M: Module + Forward<R, 2, K, Error = ME>,
    GE: std::fmt::Debug,
    T: QLearningTarget,
{
    /// Creates a trainable Q model for rank-R observations [batch_size, ...observation_shape] and Q outputs [batch_size, action_count].
    /// The target starts from the online model and remains detached between hard updates.
    /// Observations retain native kind and dtype. Model Q outputs must use the configured compute dtype.
    /// Model parameters must use the optimization device. Replay matches the first observation dtype on its storage device.
    /// Replay storage configuration selects the storage and optimization devices.
    #[builder]
    pub(crate) fn builder(
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
        training_horizon: usize,
        replay_storage_config: ReplayStorageConfig,
        #[builder(default = DType::F32)] dtype: DType,
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
        if action_space.get_possible_values() == 0 || action_space.get_possible_values() > (1 << 24)
        {
            return Err(
                SpaceError::InvalidCategoryCount(action_space.get_possible_values()).into(),
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
            .training_horizon(training_horizon)
            .call()?;

        let mut shape = vec![replay_capacity];
        shape.extend(observation_space.shape());
        let actual = shape.len() - 1;
        let shape: [usize; R] =
            shape
                .try_into()
                .map_err(|_| QLearningConfigurationError::ObservationRank {
                    expected: R - 1,
                    actual,
                })?;
        let replay_storage =
            QLearningReplayStorage::<R, K>::new(shape, replay_storage_config.storage_device());
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
            _phantom: PhantomData,
        })
    }
}

impl<M, S, GE, T, const R: usize, K: Autodiff, ME: std::fmt::Debug> QLearningAgent<M, S, GE, T, R>
where
    S: ObservationSpace<R, Kind = K>,
    M: Module + Forward<R, 2, K, Error = ME>,
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
    fn model_observations(&self, observations: Tensor<R, K>) -> Tensor<R, K> {
        observations.to_device(&self.replay_storage_config.optimization_device())
    }

    /// Selects U32 Int actions [batch_size, 1] from rank-R observations [batch_size, ...observation_shape].
    /// Selection uses the optimization device and the observations' native kind and dtype.
    pub(crate) fn act(
        &mut self,
        observations: &Tensor<R, K>,
    ) -> Result<Tensor<2, Int>, QAgentError<GE, ME>> {
        let observations = self.model_observations(observations.clone());
        epsilon_greedy_actions(
            &observations,
            self.current_epsilon,
            &self.action_space,
            &mut self.action_rng,
            |observations| self.online_q_network.forward(observations.clone()),
        )
        .map_err(QAgentError::ModelError)
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
        let next_dones = batch.next_dones.to_device(&optimization_device);
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
        transitions: QCollectedTransitions<'_, R, K>,
        episodes: &mut QEpisodeTracker,
    ) -> Result<Vec<QEpisodeLogEntry>, QAgentError<GE, ME>> {
        let environment_count = transitions.dones.len();
        let reward_values = transitions
            .rewards
            .clone()
            .try_into_data_as::<f32>()?
            .try_to_vec::<f32>()
            .map_err(TensorReadError::from)?;
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
        Ok(completed_episodes)
    }

    fn run_scheduled_updates<I>(
        &mut self,
        first_timestep: usize,
        environment_count: usize,
        logger: &mut dyn QLearningLogger<I>,
    ) -> Result<(), QAgentError<GE, ME>> {
        for offset in 1..=environment_count {
            let timestep = first_timestep.saturating_add(offset);
            if timestep.is_multiple_of(self.update_frequency) && timestep >= self.training_start {
                self.optimize(timestep, logger)?;
            }
            if timestep.is_multiple_of(self.target_update_interval) {
                self.update_target_network();
            }
        }
        Ok(())
    }

    /// Collects native rank-R observations [num_envs, ...observation_shape] and Int actions [num_envs, 1].
    /// Saves terminal observations before auto-reset replacements. Each call resets environments; schedule and replay progress persist.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            name = "q_learning.learn",
            target = "modurl::performance",
            skip_all,
            fields(num_timesteps)
        )
    )]
    pub(crate) fn learn<I, const U: usize>(
        &mut self,
        env: &mut dyn MultiGym<I, R, 2, Error = GE, ObservationSpace = S, ActionSpace = Discrete>,
        num_timesteps: usize,
        logger: &mut dyn QLearningLogger<I>,
    ) -> Result<(), QAgentError<GE, ME>>
    where
        Tensor<R, K>: PrevRank<Prev = Tensor<U, K>>,
        Tensor<U, K>: NextRank<Next = Tensor<R, K>>,
    {
        let mut elapsed_timesteps = 0;
        let environment_count = env.num_envs();
        self.experience_replay
            .storage_mut()
            .initialize_environment_count(environment_count)?;
        let mut observations = env.reset().map_err(QAgentError::GymError)?;
        let mut episodes = QEpisodeTracker::new(environment_count);
        while elapsed_timesteps < num_timesteps {
            self.current_epsilon = validate_epsilon(
                self.schedule_progress
                    .parameter(self.epsilon_schedule.as_ref()),
            )?;
            let actions = self.act(&observations)?;
            let step_info = env.step(actions.clone()).map_err(QAgentError::GymError)?;
            let transition_next_observations = step_info.transition_next_observations();
            let MultiGymStepInfo {
                observations: reset_next_observations,
                rewards,
                infos,
                dones,
                truncateds,
                ..
            } = step_info;
            let collection_rewards = rewards.clone();
            let first_timestep = self.schedule_progress.elapsed_steps();
            let completed_episodes = self.store_vectorized_transitions(
                QCollectedTransitions {
                    observations: &observations,
                    next_observations: &transition_next_observations,
                    actions: &actions,
                    rewards: &rewards,
                    dones: &dones,
                    truncateds: &truncateds,
                    first_timestep,
                },
                &mut episodes,
            )?;
            observations = reset_next_observations;
            let collection_timestep = first_timestep.saturating_add(environment_count);
            let entry = QCollectionLogEntry {
                collection_rewards,
                infos,
                epsilon: self.current_epsilon,
                collection_timestep,
                completed_episodes,
            };
            elapsed_timesteps += environment_count;
            self.run_scheduled_updates(first_timestep, environment_count, logger)?;
            logger.log_collection(&entry);
            self.schedule_progress.advance_steps(environment_count);
        }
        Ok(())
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
        training_horizon: usize,
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
        if training_horizon == 0 {
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

/// Selects U32 Int actions [batch_size, 1] from rank-R observations [batch_size, ...observation_shape].
/// The callback receives only greedy rows and returns Float Q values [selected_batch, action_count] on the input device.
/// The caller validates epsilon and category count. Exploration uses the supplied RNG; native observation kind is preserved.
pub(crate) fn epsilon_greedy_actions<const R: usize, K: Basic, E>(
    observations: &Tensor<R, K>,
    epsilon: f64,
    action_space: &Discrete,
    rng: &mut impl Rng,
    forward: impl FnOnce(&Tensor<R, K>) -> Result<Tensor<2>, E>,
) -> Result<Tensor<2, Int>, E> {
    const {
        assert!(R >= 2, "observations require batch and item axes");
    }
    let batch_size = observations.dims()[0];
    let device = observations.device();
    if epsilon == 0.0 {
        return Ok(forward(observations)?.argmax(1).cast(DType::U32));
    }
    let action_count = action_space.get_possible_values() as u32;
    let mut actions = Vec::with_capacity(batch_size);
    let mut greedy_indices = Vec::with_capacity(batch_size);
    for index in 0..batch_size {
        if rng.random_bool(epsilon) {
            actions.push(rng.random_range(0..action_count));
        } else {
            actions.push(0);
            greedy_indices.push(index as i64);
        }
    }
    if greedy_indices.is_empty() {
        return Ok(Tensor::from_data(
            TensorData::new(actions, [batch_size, 1]),
            (&device, DType::U32),
        ));
    }
    if greedy_indices.len() == batch_size {
        return Ok(forward(observations)?.argmax(1).cast(DType::U32));
    }
    let greedy_count = greedy_indices.len();
    let rows = Tensor::<1, Int>::from_data(
        TensorData::new(greedy_indices, [greedy_count]),
        (&device, DType::I64),
    );
    let selected = observations.clone().select(0, rows.clone());
    let greedy_actions = forward(&selected)?.argmax(1).cast(DType::U32);
    Ok(Tensor::<2, Int>::from_data(
        TensorData::new(actions, [batch_size, 1]),
        (&device, DType::U32),
    )
    .select_assign(0, rows, greedy_actions, IndexingUpdateOp::Assign))
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
mod tests {
    use super::*;
    use crate::{
        agents::ReplayDeviceStrategy,
        gym::{Gym, ResetInfo, StepInfo, VectorizedGymError, VectorizedGymWrapper},
        objectives::bellman_targets,
        spaces::BoxSpace,
    };
    use burn::{module::Param, optim::SgdConfig};
    use std::convert::Infallible;

    struct TestTarget;

    impl QLearningTarget for TestTarget {
        fn requires_online_next_q_values() -> bool {
            false
        }

        /// Builds Float targets [batch_size, 1] from rewards and Bool termination flags [batch_size, 1] and Q values [batch_size, action_count].
        fn target_q_values(
            rewards: &Tensor<2>,
            next_dones: &Tensor<2, Bool>,
            _online: Option<&Tensor<2>>,
            target: &Tensor<2>,
            gamma: f32,
        ) -> Tensor<2> {
            bellman_targets(
                rewards.clone(),
                next_dones.clone(),
                target.clone().max_dim(1),
                f64::from(gamma),
            )
        }
    }

    #[derive(Module, Debug)]
    struct TestNetwork {
        values: Param<Tensor<2>>,
    }

    impl<const R: usize, K: Basic> Forward<R, 2, K> for TestNetwork {
        type Error = Infallible;

        /// Returns Float Q values [batch_size, 2] for native rank-R observations [batch_size, ...observation_shape].
        fn forward(&self, input: Tensor<R, K>) -> Result<Tensor<2>, Self::Error> {
            Ok(self.values.val().expand([input.dims()[0], 2]))
        }
    }

    fn network(device: &Device, dtype: DType) -> TestNetwork {
        TestNetwork {
            values: Param::from_tensor(Tensor::from_data([[0.0f64, 1.0]], (device, dtype))),
        }
    }

    struct EpisodeEnv {
        device: Device,
        steps: usize,
        truncate: bool,
    }

    impl Gym for EpisodeEnv {
        type Error = Infallible;
        type ObservationSpace = BoxSpace<2>;
        type ActionSpace = Discrete;

        /// Accepts an Int scalar action [1] and returns a Float observation [1] on the environment device.
        fn step(&mut self, _action: Tensor<1, Int>) -> Result<StepInfo, Self::Error> {
            self.steps += 1;
            Ok(StepInfo {
                observation: Tensor::from_data([self.steps as f32], &self.device),
                reward: 2.0,
                done: self.steps == 2 && !self.truncate,
                truncated: self.steps == 2 && self.truncate,
                info: (),
            })
        }

        /// Resets to Float observations [1] on the environment device.
        fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
            self.steps = 0;
            Ok(ResetInfo {
                observation: Tensor::zeros([1], &self.device),
                info: (),
            })
        }

        fn observation_space(&self) -> Self::ObservationSpace {
            BoxSpace::new_unbounded([1, 1], &self.device)
        }

        fn action_space(&self) -> Self::ActionSpace {
            Discrete::new(2)
        }
    }

    #[derive(Default)]
    struct Logger {
        updates: Vec<usize>,
        rewards: Vec<Vec<f32>>,
        episodes: Vec<QEpisodeLogEntry>,
    }

    impl QLearningLogger for Logger {
        fn log_update(&mut self, entry: &QLogEntry) {
            assert_eq!(entry.loss.dims(), [1]);
            assert_eq!(entry.q_values.dims()[1], 1);
            assert_eq!(entry.q_values.dims(), entry.replay_rewards.dims());
            assert_eq!(entry.loss.dtype(), entry.q_values.dtype());
            assert_eq!(entry.update_index, self.updates.len());
            self.updates.push(entry.collection_timestep);
        }

        fn log_collection(&mut self, entry: &QCollectionLogEntry) {
            self.rewards.push(
                entry
                    .collection_rewards
                    .clone()
                    .into_data()
                    .try_to_vec::<f32>()
                    .unwrap(),
            );
            self.episodes.extend(
                entry
                    .completed_episodes
                    .iter()
                    .map(|entry| QEpisodeLogEntry {
                        environment_index: entry.environment_index,
                        episode_return: entry.episode_return,
                        episode_length: entry.episode_length,
                        terminated: entry.terminated,
                        truncated: entry.truncated,
                        collection_timestep: entry.collection_timestep,
                    }),
            );
        }
    }

    fn float_agent(
        device: &Device,
    ) -> QLearningAgent<TestNetwork, BoxSpace<2>, VectorizedGymError<Infallible>, TestTarget> {
        QLearningAgent::builder()
            .action_space(Discrete::new(2))
            .observation_space(BoxSpace::new_unbounded([1, 1], device))
            .online_q_network(network(&device.clone().autodiff(), DType::F32))
            .optimizer(SgdConfig::new().init())
            .learning_rate(0.01)
            .epsilon_schedule(LinearSchedule::new(1.0, 0.0))
            .replay_capacity(8)
            .batch_size(1)
            .training_start(3)
            .update_frequency(2)
            .target_update_interval(4)
            .training_horizon(10)
            .replay_storage_config(ReplayStorageConfig::new(ReplayDeviceStrategy::OneDevice(
                device.clone().autodiff(),
            )))
            .build()
            .unwrap()
    }

    #[test]
    fn epsilon_greedy_preserves_layout_kind_and_seeded_host_randomness() {
        let device = Device::flex();
        let observations = Tensor::<2>::from_data(
            [[0.0f32, 2.0, 1.0], [3.0, 1.0, 2.0], [0.0, 1.0, 4.0]],
            &device,
        );
        let greedy = epsilon_greedy_actions(
            &observations,
            0.0,
            &Discrete::new(3),
            &mut StdRng::seed_from_u64(1),
            |input| Ok::<_, Infallible>(input.clone()),
        )
        .unwrap();
        assert_eq!(greedy.dims(), [3, 1]);
        assert_eq!(greedy.dtype(), DType::U32);
        assert_eq!(greedy.into_data().try_to_vec::<u32>().unwrap(), [1, 0, 2]);
        let input =
            Tensor::<3, Bool>::from_data(TensorData::new(vec![true; 128], [128, 1, 1]), &device);
        let sample = || {
            epsilon_greedy_actions(
                &input,
                1.0,
                &Discrete::new(3),
                &mut StdRng::seed_from_u64(2),
                |_| -> Result<Tensor<2>, Infallible> {
                    panic!("full exploration must not evaluate the model")
                },
            )
            .unwrap()
        };
        assert_eq!(sample().dims(), [128, 1]);
        let first = sample().into_data().try_to_vec::<u32>().unwrap();
        assert_eq!(first, sample().into_data().try_to_vec::<u32>().unwrap());
        assert!(first.into_iter().all(|action| action < 3));
    }

    #[test]
    fn mixed_exploration_forwards_only_greedy_native_observations() {
        let device = Device::flex();
        let observations = Tensor::<2, Int>::from_data(
            TensorData::new((0i64..128).collect(), [128, 1]),
            (&device, DType::I64),
        );
        let mut forwarded = 0;
        let actions = epsilon_greedy_actions(
            &observations,
            0.5,
            &Discrete::new(2),
            &mut StdRng::seed_from_u64(3),
            |input| {
                assert_eq!(input.dtype(), DType::I64);
                forwarded = input.dims()[0];
                let input = input.clone().float();
                Ok::<_, Infallible>(Tensor::cat(vec![input.clone(), input.neg()], 1))
            },
        )
        .unwrap();
        assert!(forwarded > 0 && forwarded < 128);
        assert_eq!(actions.dims(), [128, 1]);
        assert!(
            actions
                .into_data()
                .try_to_vec::<u32>()
                .unwrap()
                .into_iter()
                .all(|action| action < 2)
        );
    }

    #[test]
    fn selected_action_q_values_retains_shape_precision_and_gradient_flow() {
        let device = Device::flex().autodiff();
        let q_values =
            Tensor::<2>::from_data([[1.0f64, 5.0, 2.0], [7.0, 3.0, 4.0]], (&device, DType::F64))
                .require_grad();
        let actions = Tensor::<2, Int>::from_data([[1u32], [2]], (&device, DType::U32));
        let selected = selected_action_q_values(&q_values, &actions);
        assert_eq!(selected.dims(), [2, 1]);
        assert_eq!(selected.dtype(), DType::F64);
        assert_eq!(
            selected.clone().into_data().try_to_vec::<f64>().unwrap(),
            [5.0, 4.0]
        );
        let gradients = selected.sum().backward();
        assert_eq!(
            q_values
                .grad(&gradients)
                .unwrap()
                .into_data()
                .try_to_vec::<f64>()
                .unwrap(),
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
    }

    /// Creates Float replay rows [batch_size, 1] with scalar rewards and Bool flags [batch_size, 1].
    fn replay_insert(values: &[f64], device: &Device) -> QLearningInsert<2, Float> {
        let count = values.len();
        QLearningInsert {
            observations: Tensor::from_data(
                TensorData::new(values.to_vec(), [count, 1]),
                (device, DType::F64),
            ),
            next_observations: Tensor::from_data(
                TensorData::new(
                    values.iter().map(|value| value + 10.0).collect::<Vec<_>>(),
                    [count, 1],
                ),
                (device, DType::F64),
            ),
            actions: Tensor::zeros([count, 1], (device, DType::U32)),
            rewards: Tensor::ones([count, 1], device),
            next_dones: Tensor::from_data(TensorData::new(vec![false; count], [count, 1]), device),
            truncateds: vec![false; count],
        }
    }

    #[test]
    fn replay_matches_observation_dtype_and_does_not_alias_gathered_observations() {
        let device = Device::flex();
        let mut storage = QLearningReplayStorage::<2, Float>::new([4, 1], device.clone());
        storage.initialize_environment_count(2).unwrap();
        storage
            .insert(0, replay_insert(&[1.0, 2.0], &device))
            .unwrap();
        let before = storage.gather(&[0]).unwrap();
        storage
            .insert(2, replay_insert(&[3.0, 4.0], &device))
            .unwrap();
        storage
            .insert(0, replay_insert(&[5.0, 6.0], &device))
            .unwrap();
        assert_eq!(
            before.observations.into_data().try_to_vec::<f64>().unwrap(),
            [1.0]
        );
        let batch = storage.gather(&[0, 1, 2, 3]).unwrap();
        assert_eq!(
            batch.observations.into_data().try_to_vec::<f64>().unwrap(),
            [5.0, 6.0, 15.0, 16.0]
        );
        assert_eq!(batch.actions.dims(), [4, 1]);
        assert_eq!(batch.next_dones.dims(), [4, 1]);
        assert_eq!(batch.rewards.dims(), [4, 1]);
    }

    #[test]
    fn replay_preserves_native_integer_and_boolean_observations() {
        let device = Device::flex();
        let mut integer = QLearningReplayStorage::<2, Int>::new([4, 1], device.clone());
        let values = [u64::MAX - 1, u64::MAX - 2];
        integer
            .insert(
                0,
                QLearningInsert {
                    observations: Tensor::from_data(
                        TensorData::new(values.to_vec(), [2, 1]),
                        (&device, DType::U64),
                    ),
                    next_observations: Tensor::from_data(
                        TensorData::new(values.to_vec(), [2, 1]),
                        (&device, DType::U64),
                    ),
                    actions: Tensor::zeros([2, 1], (&device, DType::U32)),
                    rewards: Tensor::zeros([2, 1], &device),
                    next_dones: Tensor::from_data([[false], [true]], &device),
                    truncateds: vec![false; 2],
                },
            )
            .unwrap();
        let batch = integer.gather(&[0, 1]).unwrap();
        assert_eq!(batch.observations.dtype(), DType::U64);
        assert_eq!(
            batch.observations.into_data().try_to_vec::<u64>().unwrap(),
            values
        );
        let mut boolean = QLearningReplayStorage::<2, Bool>::new([4, 1], device.clone());
        boolean
            .insert(
                0,
                QLearningInsert {
                    observations: Tensor::from_data([[false], [true]], &device),
                    next_observations: Tensor::from_data([[true], [false]], &device),
                    actions: Tensor::zeros([2, 1], (&device, DType::U32)),
                    rewards: Tensor::zeros([2, 1], &device),
                    next_dones: Tensor::from_data([[false], [false]], &device),
                    truncateds: vec![true, false],
                },
            )
            .unwrap();
        let batch = boolean.gather(&[0, 1]).unwrap();
        assert_eq!(
            batch
                .observations
                .try_into_data_as::<bool>()
                .unwrap()
                .try_to_vec::<bool>()
                .unwrap(),
            [false, true]
        );
        assert_eq!(
            batch
                .next_observations
                .try_into_data_as::<bool>()
                .unwrap()
                .try_to_vec::<bool>()
                .unwrap(),
            [true, false]
        );
    }

    #[derive(Clone)]
    struct ObservationFixture<const R: usize, K> {
        _kind: PhantomData<K>,
    }

    impl<const R: usize, K: Basic> ObservationSpace<R> for ObservationFixture<R, K> {
        type Kind = K;

        /// Checks native rank-R observations [batch_size, 1, ...] against size-one item axes.
        fn contains(&self, observations: &Tensor<R, K>) -> bool {
            observations.dims()[1..].iter().all(|size| *size == 1)
        }

        fn shape(&self) -> Vec<usize> {
            vec![1; R - 1]
        }
    }

    #[test]
    fn native_high_rank_boolean_model_trains_with_matching_replay() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        let mut agent = QLearningAgent::<_, _, Infallible, TestTarget, 3>::builder()
            .action_space(Discrete::new(2))
            .observation_space(ObservationFixture::<3, Bool> { _kind: PhantomData })
            .online_q_network(network(&device, DType::F64))
            .optimizer(SgdConfig::new().init())
            .learning_rate(0.01)
            .epsilon_schedule(LinearSchedule::new(0.0, 0.0))
            .replay_capacity(8)
            .batch_size(2)
            .training_horizon(10)
            .dtype(DType::F64)
            .replay_storage_config(ReplayStorageConfig::new(ReplayDeviceStrategy::OneDevice(
                device.clone(),
            )))
            .build()
            .unwrap();
        let observations = Tensor::<3, Bool>::from_data([[[true]], [[false]]], &device);
        let actions = agent.act(&observations).unwrap();
        assert_eq!(actions.dims(), [2, 1]);
        assert_eq!(actions.into_data().try_to_vec::<u32>().unwrap(), [1, 1]);
        agent
            .experience_replay
            .add(QLearningInsert {
                observations: observations.clone(),
                next_observations: observations,
                actions: Tensor::ones([2, 1], (&device, DType::U32)),
                rewards: Tensor::full([2, 1], 2.0, (&device, DType::F64)),
                next_dones: Tensor::from_data([[false], [false]], &device),
                truncateds: vec![false; 2],
            })
            .unwrap();
        let before = agent.online_q_network.values.val().into_data();
        let batch = agent
            .experience_replay
            .storage_mut()
            .gather(&[0, 1])
            .unwrap();
        assert_eq!(batch.observations.dims(), [2, 1, 1]);
        assert_eq!(
            batch
                .observations
                .try_into_data_as::<bool>()
                .unwrap()
                .try_to_vec::<bool>()
                .unwrap(),
            [true, false]
        );
        agent.optimize(2, &mut Logger::default()).unwrap();
        assert_ne!(agent.online_q_network.values.val().into_data(), before);
        assert_eq!(agent.target_q_network.values.val().into_data(), before);
        assert!(!agent.target_q_network.values.val().is_require_grad());
    }

    #[test]
    fn matching_replay_rejects_dtype_changes_without_overwriting_existing_rows() {
        let device = Device::flex();
        let mut storage = QLearningReplayStorage::<2, Float>::new([4, 1], device.clone());
        storage
            .insert(0, replay_insert(&[1.0, 2.0], &device))
            .unwrap();
        let mut changed = replay_insert(&[3.0, 4.0], &device);
        changed.observations = changed.observations.cast(DType::F32);
        changed.next_observations = changed.next_observations.cast(DType::F32);
        assert!(matches!(
            storage.insert(0, changed),
            Err(QLearningReplayError::ObservationDType {
                expected: DType::F64,
                actual: DType::F32
            })
        ));
        let retained = storage.gather(&[0, 1]).unwrap();
        assert_eq!(retained.observations.dtype(), DType::F64);
        assert_eq!(
            retained
                .observations
                .into_data()
                .try_to_vec::<f64>()
                .unwrap(),
            [1.0, 2.0]
        );
    }

    #[test]
    fn learn_preserves_schedule_replay_alignment_target_snapshots_and_episode_metrics() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex();
        let mut env = VectorizedGymWrapper::new(vec![
            EpisodeEnv {
                device: device.clone(),
                steps: 0,
                truncate: false,
            },
            EpisodeEnv {
                device: device.clone(),
                steps: 0,
                truncate: true,
            },
        ])
        .unwrap();
        let mut agent = float_agent(&device);
        assert!(!agent.target_q_network.values.val().is_require_grad());
        let initial_target = agent.target_q_network.values.val().into_data();
        let mut logger = Logger::default();
        agent.learn(&mut env, 4, &mut logger).unwrap();
        assert_eq!(agent.schedule_progress.elapsed_steps(), 4);
        assert_eq!(agent.current_epsilon, 0.8);
        assert_eq!(logger.updates, [4]);
        assert_eq!(logger.rewards, [vec![2.0, 2.0], vec![2.0, 2.0]]);
        assert_eq!(logger.episodes.len(), 2);
        assert_eq!(
            (
                logger.episodes[0].episode_return,
                logger.episodes[0].episode_length,
                logger.episodes[0].collection_timestep
            ),
            (4.0, 2, 3)
        );
        assert!(logger.episodes[0].terminated);
        assert!(logger.episodes[1].truncated);
        assert!(!logger.episodes[1].terminated);
        assert_ne!(
            agent.online_q_network.values.val().into_data(),
            initial_target
        );
        assert_eq!(
            agent.online_q_network.values.val().into_data(),
            agent.target_q_network.values.val().into_data()
        );
        let batch = agent
            .experience_replay
            .storage_mut()
            .gather(&[2, 3])
            .unwrap();
        assert_eq!(
            batch
                .next_observations
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            [2.0, 2.0]
        );
        assert_eq!(
            batch
                .next_dones
                .try_into_data_as::<bool>()
                .unwrap()
                .try_to_vec::<bool>()
                .unwrap(),
            [true, false]
        );
        agent.learn(&mut env, 2, &mut logger).unwrap();
        assert_eq!(agent.schedule_progress.elapsed_steps(), 6);
        assert_eq!(logger.updates, [4, 6]);
        assert_ne!(
            agent.online_q_network.values.val().into_data(),
            agent.target_q_network.values.val().into_data()
        );
        assert_eq!(agent.get_action_space().get_possible_values(), 2);
        assert_eq!(agent.get_observation_space().shape(), [1]);
        let len = agent.experience_replay.len();
        let mut incompatible = VectorizedGymWrapper::from(EpisodeEnv {
            device,
            steps: 0,
            truncate: false,
        });
        assert!(matches!(
            agent.learn(&mut incompatible, 1, &mut logger),
            Err(QAgentError::ReplayStorageError(
                QLearningReplayError::Storage(ReplayStorageError::EnvironmentCountMismatch {
                    expected: 2,
                    actual: 1
                })
            ))
        ));
        assert_eq!(agent.experience_replay.len(), len);
    }

    #[test]
    fn builder_rejects_invalid_learning_rates_and_compute_dtypes() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        let build = |rate, dtype| {
            QLearningAgent::<_, _, Infallible, TestTarget>::builder()
                .action_space(Discrete::new(2))
                .observation_space(BoxSpace::new_unbounded([1, 1], &device))
                .online_q_network(network(&device, DType::F32))
                .optimizer(SgdConfig::new().init())
                .learning_rate(rate)
                .training_horizon(10)
                .dtype(dtype)
                .replay_storage_config(ReplayStorageConfig::new(ReplayDeviceStrategy::OneDevice(
                    device.clone(),
                )))
                .build()
        };
        assert!(matches!(
            build(f64::NAN, DType::F32),
            Err(QAgentError::ConfigurationError(
                QLearningConfigurationError::InvalidLearningRate
            ))
        ));
        assert!(matches!(
            build(0.01, DType::U32),
            Err(QAgentError::ConfigurationError(
                QLearningConfigurationError::InvalidDType(DType::U32)
            ))
        ));
    }
    #[test]
    fn accepts_valid_configuration() {
        assert_eq!(
            QLearningConfigurationValidator::validate_configuration()
                .replay_capacity(1_000)
                .batch_size(32)
                .gamma(0.99)
                .initial_epsilon(1.0)
                .final_epsilon(0.1)
                .update_frequency(4)
                .target_update_interval(1_000)
                .training_horizon(10_000)
                .call(),
            Ok(())
        );
    }

    #[test]
    fn rejects_invalid_configuration_values() {
        assert_eq!(
            QLearningConfigurationValidator::validate_configuration()
                .replay_capacity(0)
                .batch_size(32)
                .gamma(0.99)
                .initial_epsilon(1.0)
                .final_epsilon(0.1)
                .update_frequency(4)
                .target_update_interval(1_000)
                .training_horizon(10_000)
                .call(),
            Err(QLearningConfigurationError::ZeroReplayCapacity)
        );
        assert_eq!(
            QLearningConfigurationValidator::validate_configuration()
                .replay_capacity(1_000)
                .batch_size(0)
                .gamma(0.99)
                .initial_epsilon(1.0)
                .final_epsilon(0.1)
                .update_frequency(4)
                .target_update_interval(1_000)
                .training_horizon(10_000)
                .call(),
            Err(QLearningConfigurationError::ZeroBatchSize)
        );
        assert_eq!(
            QLearningConfigurationValidator::validate_configuration()
                .replay_capacity(1_000)
                .batch_size(32)
                .gamma(0.99)
                .initial_epsilon(1.0)
                .final_epsilon(0.1)
                .update_frequency(4)
                .target_update_interval(0)
                .training_horizon(10_000)
                .call(),
            Err(QLearningConfigurationError::ZeroTargetUpdateInterval)
        );
        assert_eq!(
            QLearningConfigurationValidator::validate_configuration()
                .replay_capacity(1_000)
                .batch_size(32)
                .gamma(0.99)
                .initial_epsilon(1.0)
                .final_epsilon(0.1)
                .update_frequency(0)
                .target_update_interval(1_000)
                .training_horizon(10_000)
                .call(),
            Err(QLearningConfigurationError::ZeroUpdateFrequency)
        );
        assert_eq!(
            QLearningConfigurationValidator::validate_configuration()
                .replay_capacity(31)
                .batch_size(32)
                .gamma(0.99)
                .initial_epsilon(1.0)
                .final_epsilon(0.1)
                .update_frequency(4)
                .target_update_interval(1_000)
                .training_horizon(10_000)
                .call(),
            Err(QLearningConfigurationError::ReplayCapacityBelowBatchSize)
        );
        assert_eq!(
            QLearningConfigurationValidator::validate_configuration()
                .replay_capacity(1_000)
                .batch_size(32)
                .gamma(1.01)
                .initial_epsilon(1.0)
                .final_epsilon(0.1)
                .update_frequency(4)
                .target_update_interval(1_000)
                .training_horizon(10_000)
                .call(),
            Err(QLearningConfigurationError::InvalidGamma)
        );
        assert_eq!(
            QLearningConfigurationValidator::validate_configuration()
                .replay_capacity(1_000)
                .batch_size(32)
                .gamma(0.99)
                .initial_epsilon(1.0)
                .final_epsilon(0.1)
                .update_frequency(4)
                .target_update_interval(1_000)
                .training_horizon(0)
                .call(),
            Err(QLearningConfigurationError::ZeroTrainingHorizon)
        );
        assert_eq!(
            validate_epsilon(1.01),
            Err(QLearningConfigurationError::InvalidEpsilon)
        );
    }
}

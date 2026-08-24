use bon::bon;
use candle_core::{DType, Error, Tensor};
use candle_nn::{Optimizer, VarMap};
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
use std::{marker::PhantomData, ops::Deref};

use crate::{
    agents::ReplayStorageConfig,
    buffers::experience_replay::{
        AlignedObservationReplay, ExperienceReplay, ExperienceReplayError, ReplayStorage,
        ReplayStorageError, TensorReplayColumn, replay_index_tensor,
    },
    gym::{MultiGym, MultiGymStepInfo},
    parameter_schedule::{LinearSchedule, ParameterSchedule, ScheduleProgress},
    spaces::{Discrete, Space},
};

pub mod ddqn;
pub mod dqn;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QLearningConfigurationError {
    ZeroReplayCapacity,
    ZeroBatchSize,
    ZeroTargetUpdateInterval,
    ZeroUpdateFrequency,
    ZeroTrainingHorizon,
    ReplayCapacityBelowBatchSize,
    InvalidGamma,
    InvalidEpsilon,
}

#[derive(Debug)]
pub enum QAgentError<GE, SE>
where
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    TensorError(candle_core::Error),
    ReplayStorageError(ReplayStorageError),
    ConfigurationError(QLearningConfigurationError),
    GymError(GE),
    SpaceError(SE),
}

impl<GE, SE> From<candle_core::Error> for QAgentError<GE, SE>
where
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    fn from(err: candle_core::Error) -> Self {
        Self::TensorError(err)
    }
}

impl<GE, SE> From<ReplayStorageError> for QAgentError<GE, SE>
where
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    fn from(error: ReplayStorageError) -> Self {
        Self::ReplayStorageError(error)
    }
}

impl<GE, SE> From<ExperienceReplayError<ReplayStorageError>> for QAgentError<GE, SE>
where
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    fn from(error: ExperienceReplayError<ReplayStorageError>) -> Self {
        Self::ReplayStorageError(error.into())
    }
}

impl<GE, SE> From<QLearningConfigurationError> for QAgentError<GE, SE>
where
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    fn from(err: QLearningConfigurationError) -> Self {
        Self::ConfigurationError(err)
    }
}

pub struct QLogEntry {
    pub loss: Tensor,
    pub epsilon: f64,
    pub learning_rate: f32,
    pub q_values: Tensor,
    pub replay_rewards: Tensor,
    pub update_index: usize,
    pub collection_timestep: usize,
}

pub struct QCollectionLogEntry<I = ()> {
    pub collection_rewards: Tensor,
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

    /// Computes targets from `rewards` and `next_dones` shaped `[batch]` and Q
    /// tensors shaped `[batch, action_count]`, returning `[batch]`.
    fn target_q_values(
        rewards: &Tensor,
        next_dones: &Tensor,
        online_next_q_values: Option<&Tensor>,
        target_next_q_values: &Tensor,
        gamma: f32,
    ) -> Result<Tensor, Error>;
}

struct QLearningBatch {
    states: Tensor,
    next_states: Tensor,
    actions: Tensor,
    rewards: Tensor,
    next_dones: Tensor,
}

struct QLearningInsert {
    states: Tensor,
    next_states: Tensor,
    actions: Tensor,
    rewards: Tensor,
    next_dones: Tensor,
    truncateds: Vec<bool>,
}

struct QLearningReplayStorage {
    observations: AlignedObservationReplay,
    actions: TensorReplayColumn,
    rewards: TensorReplayColumn,
    next_dones: TensorReplayColumn,
    capacity: usize,
    device: candle_core::Device,
}

impl QLearningReplayStorage {
    fn new(
        capacity: usize,
        observation_shape: &[usize],
        observation_dtype: DType,
        device: candle_core::Device,
    ) -> Result<Self, ReplayStorageError> {
        Ok(Self {
            observations: AlignedObservationReplay::new(
                capacity,
                observation_shape,
                observation_dtype,
                &device,
            ),
            actions: TensorReplayColumn::new(capacity, &[], DType::U32, &device)?,
            rewards: TensorReplayColumn::new(capacity, &[], DType::F32, &device)?,
            next_dones: TensorReplayColumn::new(capacity, &[], DType::F32, &device)?,
            capacity,
            device,
        })
    }

    fn initialize_environment_count(
        &mut self,
        environment_count: usize,
    ) -> Result<(), ReplayStorageError> {
        self.observations
            .initialize_environment_count(environment_count)
    }
}

impl ReplayStorage for QLearningReplayStorage {
    type Insert = QLearningInsert;
    type Batch = QLearningBatch;
    type Error = ReplayStorageError;

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn insert(&mut self, start: usize, transitions: Self::Insert) -> Result<usize, Self::Error> {
        let actions = self.actions.prepare(&transitions.actions)?;
        let rewards = self.rewards.prepare(&transitions.rewards)?;
        let next_dones = self.next_dones.prepare(&transitions.next_dones)?;
        let count = transitions.states.dim(0)?;
        for (name, tensor) in [
            ("actions", &transitions.actions),
            ("rewards", &transitions.rewards),
            ("next dones", &transitions.next_dones),
        ] {
            let actual = tensor.dim(0)?;
            if actual != count {
                return Err(ReplayStorageError::BatchLengthMismatch {
                    field: name,
                    expected: count,
                    actual,
                });
            }
        }

        self.observations.insert(
            start,
            &transitions.states,
            &transitions.next_states,
            &transitions.truncateds,
        )?;
        self.actions.write(start, &actions)?;
        self.rewards.write(start, &rewards)?;
        self.next_dones.write(start, &next_dones)?;
        Ok(count)
    }

    fn gather(&self, indices: &[usize]) -> Result<Self::Batch, Self::Error> {
        let (states, next_states) = self.observations.gather(indices)?;
        let indices = replay_index_tensor(indices, &self.device)?;

        Ok(QLearningBatch {
            states,
            next_states,
            actions: self.actions.gather(&indices)?,
            rewards: self.rewards.gather(&indices)?,
            next_dones: self.next_dones.gather(&indices)?,
        })
    }

    fn sampleable_len(&self, len: usize) -> usize {
        self.observations.sampleable_len(len)
    }

    fn sample_index(&self, index: usize, len: usize) -> usize {
        self.observations.sample_index(index, len)
    }
}

struct QCollectedTransitions<'a> {
    states: &'a Tensor,
    next_states: &'a Tensor,
    actions: &'a Tensor,
    rewards: &'a Tensor,
    dones: &'a [bool],
    truncateds: &'a [bool],
    first_timestep: usize,
}

pub(crate) struct QLearningAgent<'a, O, GE, SE, T>
where
    O: Optimizer,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
    T: QLearningTarget,
{
    online_q_network: Box<dyn candle_core::Module>,
    target_q_network: Box<dyn candle_core::Module>,
    target_vars: &'a mut VarMap,
    online_vars: &'a VarMap,
    target_update_interval: usize,
    optimizer: O,
    current_epsilon: f64,
    epsilon_schedule: Box<dyn ParameterSchedule>,
    schedule_progress: ScheduleProgress,
    action_space: Discrete,
    observation_space: Box<dyn Space<Error = SE>>,
    experience_replay: ExperienceReplay<QLearningInsert, QLearningReplayStorage>,
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
impl<'a, O, GE, SE, T> QLearningAgent<'a, O, GE, SE, T>
where
    O: Optimizer,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
    T: QLearningTarget,
{
    #[builder]
    pub(crate) fn new(
        action_space: Discrete,
        observation_space: Box<dyn Space<Error = SE>>,
        target_q_network: Box<dyn candle_core::Module>,
        online_q_network: Box<dyn candle_core::Module>,
        target_vars: &'a mut VarMap,
        online_vars: &'a VarMap,
        optimizer: O,
        #[builder(default = 1000)] target_update_interval: usize,
        #[builder(default = Box::new(LinearSchedule::new(1.0, 0.1)))] epsilon_schedule: Box<
            dyn ParameterSchedule,
        >,
        #[builder(default = 10000)] replay_capacity: usize,
        #[builder(default = 32)] batch_size: usize,
        #[builder(default = 0.99)] gamma: f32,
        #[builder(default = 4)] update_frequency: usize,
        #[builder(default = 1000)] training_start: usize,
        training_horizon: usize,
        replay_storage_config: ReplayStorageConfig,
        #[builder(default = DType::F32)] dtype: DType,
    ) -> Result<Self, QAgentError<GE, SE>> {
        assert!(
            dtype.is_float(),
            "Q-learning compute dtype must be floating-point"
        );
        let initial_epsilon = epsilon_schedule.value(0.0);
        let final_epsilon = epsilon_schedule.value(1.0);
        QLearningConfigurationValidator::validate_configuration()
            .replay_capacity(replay_capacity)
            .batch_size(batch_size)
            .gamma(gamma)
            .initial_epsilon(initial_epsilon)
            .final_epsilon(final_epsilon)
            .update_frequency(update_frequency)
            .target_update_interval(target_update_interval)
            .training_horizon(training_horizon)
            .call()?;

        let optimization_device = replay_storage_config.optimization_device();
        // Tie host-side action sampling to the configured accelerator seed. This
        // is the only device-to-host synchronization needed by epsilon-greedy.
        let action_seed = Tensor::rand(0.0f64, u32::MAX as f64, (), &optimization_device)?
            .to_dtype(DType::U32)?
            .to_scalar::<u32>()?;
        let action_rng = StdRng::seed_from_u64(u64::from(action_seed));

        let storage_device = replay_storage_config.storage_device();
        let replay_storage = QLearningReplayStorage::new(
            replay_capacity,
            &observation_space.shape(),
            replay_storage_config.observation_dtype(),
            storage_device,
        )?;

        let mut agent = Self {
            online_q_network,
            target_q_network,
            target_vars,
            online_vars,
            target_update_interval,
            optimizer,
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
            action_rng,
            _phantom: PhantomData,
        };
        agent.update_target_network();
        Ok(agent)
    }
}

impl<'a, O, GE, SE, T> QLearningAgent<'a, O, GE, SE, T>
where
    O: Optimizer,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
    T: QLearningTarget,
{
    pub(crate) fn get_action_space(&self) -> &Discrete {
        &self.action_space
    }

    pub(crate) fn get_observation_space(&self) -> &dyn Space<Error = SE> {
        &*self.observation_space
    }

    /// Selects scalar discrete actions shaped `[batch]` for observations shaped
    /// `[batch, ...observation_shape]`.
    pub(crate) fn act(&mut self, observation: &Tensor) -> Result<Tensor, QAgentError<GE, SE>> {
        let observation = observation
            .to_device(&self.replay_storage_config.optimization_device())?
            .to_dtype(self.dtype)?;
        Ok(epsilon_greedy_actions(
            &observation,
            self.current_epsilon,
            &self.action_space,
            &mut self.action_rng,
            |observations| self.online_q_network.forward(observations),
        )?)
    }

    fn optimize<I>(
        &mut self,
        collection_timestep: usize,
        logger: &mut dyn QLearningLogger<I>,
    ) -> Result<(), QAgentError<GE, SE>> {
        if self.experience_replay.len() < self.experience_replay.get_batch_size() {
            return Ok(());
        }
        let optimization_device = self.replay_storage_config.optimization_device();
        let training_batch = self.experience_replay.sample()?;
        let mut training_batch = training_batch;
        training_batch.states = training_batch
            .states
            .to_device(&optimization_device)?
            .to_dtype(self.dtype)?;
        training_batch.next_states = training_batch
            .next_states
            .to_device(&optimization_device)?
            .to_dtype(self.dtype)?;
        training_batch.actions = training_batch.actions.to_device(&optimization_device)?;
        training_batch.rewards = training_batch
            .rewards
            .to_device(&optimization_device)?
            .to_dtype(self.dtype)?;
        training_batch.next_dones = training_batch
            .next_dones
            .to_device(&optimization_device)?
            .to_dtype(self.dtype)?;
        let QLearningBatch {
            states,
            next_states,
            actions,
            rewards,
            next_dones,
        } = training_batch;

        let target_next_q_values = self.target_q_network.forward(&next_states)?;
        let online_next_q_values = T::requires_online_next_q_values()
            .then(|| self.online_q_network.forward(&next_states))
            .transpose()?;
        let target_q_values = T::target_q_values(
            &rewards,
            &next_dones,
            online_next_q_values.as_ref(),
            &target_next_q_values,
            self.gamma,
        )?
        .reshape(&[rewards.shape().dims()[0], 1])?
        .detach();

        let state_action_q_values = selected_action_q_values(
            &self.online_q_network.forward(&states)?.squeeze(1)?,
            &actions,
        )?;
        let loss = candle_nn::loss::mse(&state_action_q_values, &target_q_values)?;
        let entry = QLogEntry {
            loss: loss.clone(),
            epsilon: self.current_epsilon,
            learning_rate: self.optimizer.learning_rate() as f32,
            q_values: state_action_q_values,
            replay_rewards: rewards,
            update_index: self.optimization_steps,
            collection_timestep,
        };
        logger.log_update(&entry);
        self.optimization_steps += 1;
        self.optimizer.backward_step(&loss)?;
        Ok(())
    }

    fn update_target_network(&mut self) {
        let online_data = self.online_vars.data().lock().unwrap();
        for (name, online_var) in online_data.deref() {
            self.target_vars.set_one(name, online_var.as_tensor()).expect(
                "failed to match var names in target and online varmaps, make sure they are the same",
            );
        }
    }

    fn store_vectorized_transitions(
        &mut self,
        transitions: QCollectedTransitions<'_>,
        episodes: &mut QEpisodeTracker,
    ) -> Result<Vec<QEpisodeLogEntry>, QAgentError<GE, SE>> {
        let QCollectedTransitions {
            states,
            next_states,
            actions,
            rewards,
            dones,
            truncateds,
            first_timestep,
        } = transitions;
        let environment_count = dones.len();
        let reward_values = rewards
            .to_dtype(DType::F32)?
            .to_device(&candle_core::Device::Cpu)?
            .to_vec1::<f32>()?;
        let mut completed_episodes = Vec::new();

        for environment_index in 0..environment_count {
            let reward = reward_values[environment_index];
            let collection_timestep = first_timestep.saturating_add(environment_index + 1);
            if let Some(entry) = episodes.record(
                environment_index,
                reward,
                dones[environment_index],
                truncateds[environment_index],
                collection_timestep,
            ) {
                completed_episodes.push(entry);
            }
        }

        let next_dones = Tensor::from_vec(
            dones
                .iter()
                .map(|&done| if done { 1.0f32 } else { 0.0 })
                .collect::<Vec<_>>(),
            environment_count,
            &self.replay_storage_config.storage_device(),
        )?;
        self.experience_replay.add(QLearningInsert {
            states: states.clone(),
            next_states: next_states.clone(),
            actions: actions.clone(),
            rewards: rewards.clone(),
            next_dones,
            truncateds: truncateds.to_vec(),
        })?;

        Ok(completed_episodes)
    }

    fn run_scheduled_updates<I>(
        &mut self,
        first_timestep: usize,
        environment_count: usize,
        logger: &mut dyn QLearningLogger<I>,
    ) -> Result<(), QAgentError<GE, SE>> {
        for timestep_offset in 1..=environment_count {
            let training_timestep = first_timestep.saturating_add(timestep_offset);
            if training_timestep % self.update_frequency == 0
                && training_timestep >= self.training_start
            {
                self.optimize(training_timestep, logger)?;
            }
            if training_timestep % self.target_update_interval == 0 {
                self.update_target_network();
            }
        }
        Ok(())
    }

    pub(crate) fn learn<I>(
        &mut self,
        env: &mut dyn MultiGym<I, Error = GE, SpaceError = SE>,
        num_timesteps: usize,
        logger: &mut dyn QLearningLogger<I>,
    ) -> Result<(), QAgentError<GE, SE>> {
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
            let transition_next_states = step_info.transition_next_states()?;
            let MultiGymStepInfo {
                states: reset_next_states,
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
                    states: &observations,
                    next_states: &transition_next_states,
                    actions: &actions,
                    rewards: &rewards,
                    dones: &dones,
                    truncateds: &truncateds,
                    first_timestep,
                },
                &mut episodes,
            )?;
            observations = reset_next_states;
            let collection_timestep = first_timestep.saturating_add(environment_count);
            let collection_entry = QCollectionLogEntry {
                collection_rewards,
                infos,
                epsilon: self.current_epsilon,
                collection_timestep,
                completed_episodes,
            };

            elapsed_timesteps += environment_count;
            self.run_scheduled_updates(first_timestep, environment_count, logger)?;
            logger.log_collection(&collection_entry);
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

/// Selects scalar actions `[batch]` for observations
/// `[batch, ...observation_shape]`; `forward` must return
/// `[selected_batch, action_count]`.
pub(crate) fn epsilon_greedy_actions(
    observation: &Tensor,
    epsilon: f64,
    action_space: &Discrete,
    rng: &mut impl Rng,
    forward: impl FnOnce(&Tensor) -> Result<Tensor, Error>,
) -> Result<Tensor, Error> {
    let batch_size = observation.shape().dims()[0];
    let device = observation.device();
    if epsilon == 0.0 {
        return forward(observation)?.argmax(1);
    }

    let action_count = action_space.get_possible_values() as u32;
    let mut actions = Vec::with_capacity(batch_size);
    let mut greedy_indices = Vec::with_capacity(batch_size);
    for index in 0..batch_size {
        if rng.random_bool(epsilon) {
            actions.push(rng.random_range(0..action_count));
        } else {
            actions.push(0);
            greedy_indices.push(index as u32);
        }
    }

    if greedy_indices.is_empty() {
        return Tensor::from_vec(actions, batch_size, device);
    }
    if greedy_indices.len() == batch_size {
        return forward(observation)?.argmax(1);
    }

    let greedy_count = greedy_indices.len();
    let greedy_indices = Tensor::from_vec(greedy_indices, greedy_count, device)?;
    let greedy_observations = observation.index_select(&greedy_indices, 0)?;
    let greedy_actions = forward(&greedy_observations)?.argmax(1)?;
    Tensor::from_vec(actions, batch_size, device)?.scatter(&greedy_indices, &greedy_actions, 0)
}

/// Gathers scalar `actions` containing one index per batch item from `q_values`
/// shaped `[batch, action_count]`, returning `[batch, 1]`.
pub(crate) fn selected_action_q_values(
    q_values: &Tensor,
    actions: &Tensor,
) -> Result<Tensor, Error> {
    let actions = actions.reshape(&[actions.shape().dims()[0], 1])?;
    q_values.gather(&actions, 1)
}

#[cfg(test)]
mod tests {
    use super::{
        QAgentError, QCollectionLogEntry, QLearningAgent, QLearningConfigurationError,
        QLearningConfigurationValidator, QLearningInsert, QLearningLogger, QLearningReplayStorage,
        QLearningTarget, epsilon_greedy_actions, selected_action_q_values, validate_epsilon,
    };
    use crate::{
        agents::{
            ReplayDeviceStrategy, ReplayStorageConfig,
            test_support::{CountingOptimizer, FixedEnv},
        },
        buffers::experience_replay::ReplayStorageError,
        gym::{Gym, MultiGym, ResetInfo, StepInfo, VectorizedGymError, VectorizedGymWrapper},
        models::MLP,
        objectives::bellman_targets,
        parameter_schedule::LinearSchedule,
        spaces::{BoxSpace, Discrete},
    };
    use candle_core::{DType, Device, Error, Tensor};
    use candle_nn::{VarBuilder, VarMap};
    use rand::{SeedableRng, rngs::StdRng};

    struct TestTarget;

    fn assert_device_native_epsilon_greedy(device: &Device) {
        let observations = Tensor::new(
            &[[0.0f32, 2.0, 1.0], [3.0, 1.0, 2.0], [0.0, 1.0, 4.0]],
            device,
        )
        .unwrap();
        let action_space = Discrete::new(3);
        let actions = epsilon_greedy_actions(
            &observations,
            0.0,
            &action_space,
            &mut StdRng::seed_from_u64(1),
            |input| {
                assert_eq!(input.dims(), &[3, 3]);
                Ok(input.clone())
            },
        )
        .unwrap();

        assert_eq!(actions.dtype(), DType::U32);
        assert_eq!(actions.to_vec1::<u32>().unwrap(), vec![1, 0, 2]);
    }

    #[test]
    fn epsilon_greedy_selects_actions_without_changing_the_batch_shape() {
        assert_device_native_epsilon_greedy(&Device::Cpu);
    }

    #[test]
    fn full_exploration_skips_the_q_network() {
        let observations = Tensor::zeros((16, 2), DType::F32, &Device::Cpu).unwrap();
        let actions = epsilon_greedy_actions(
            &observations,
            1.0,
            &Discrete::new(3),
            &mut StdRng::seed_from_u64(2),
            |_| panic!("full exploration must not evaluate the Q-network"),
        )
        .unwrap();

        assert_eq!(actions.dims(), &[16]);
        assert!(
            actions
                .to_vec1::<u32>()
                .unwrap()
                .into_iter()
                .all(|action| action < 3)
        );
    }

    #[test]
    fn mixed_exploration_forwards_only_greedy_observations() {
        let observations = Tensor::arange(0u32, 128, &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .reshape((128, 1))
            .unwrap();
        let mut forwarded = 0;
        let actions = epsilon_greedy_actions(
            &observations,
            0.5,
            &Discrete::new(2),
            &mut StdRng::seed_from_u64(3),
            |input| {
                forwarded = input.dim(0)?;
                Tensor::cat(&[input, &input.affine(-1.0, 0.0)?], 1)
            },
        )
        .unwrap();

        assert!(forwarded > 0 && forwarded < 128);
        assert_eq!(actions.dims(), &[128]);
        assert!(
            actions
                .to_vec1::<u32>()
                .unwrap()
                .into_iter()
                .all(|action| action < 2)
        );
    }

    impl QLearningTarget for TestTarget {
        fn requires_online_next_q_values() -> bool {
            false
        }

        /// Computes `[batch]` targets from reward/done vectors `[batch]` and Q
        /// values `[batch, action_count]`.
        fn target_q_values(
            rewards: &Tensor,
            next_dones: &Tensor,
            _online_next_q_values: Option<&Tensor>,
            target_next_q_values: &Tensor,
            gamma: f32,
        ) -> Result<Tensor, Error> {
            bellman_targets(
                rewards,
                next_dones,
                &target_next_q_values.max(1)?.detach(),
                f64::from(gamma),
            )
        }
    }

    struct NoopLogger;

    impl QLearningLogger for NoopLogger {
        fn log_update(&mut self, _entry: &super::QLogEntry) {}

        fn log_collection(&mut self, _entry: &QCollectionLogEntry) {}
    }

    fn replay_insert(states: &[u8], device: &Device) -> QLearningInsert {
        let count = states.len();
        QLearningInsert {
            states: Tensor::from_vec(states.to_vec(), (count, 1), device).unwrap(),
            next_states: Tensor::from_vec(
                states
                    .iter()
                    .map(|&state| state.saturating_add(10))
                    .collect::<Vec<_>>(),
                (count, 1),
                device,
            )
            .unwrap(),
            actions: Tensor::from_vec(vec![0u32; count], count, device).unwrap(),
            rewards: Tensor::from_vec(vec![1.0f32; count], count, device).unwrap(),
            next_dones: Tensor::zeros(count, DType::F32, device).unwrap(),
            truncateds: vec![false; count],
        }
    }

    #[test]
    fn preallocated_replay_wraps_and_gathered_batches_do_not_alias_storage() {
        use crate::buffers::experience_replay::ReplayStorage;

        let device = Device::Cpu;
        let mut storage = QLearningReplayStorage::new(4, &[1], DType::U8, device.clone()).unwrap();
        storage.initialize_environment_count(2).unwrap();
        storage.insert(0, replay_insert(&[1, 2], &device)).unwrap();

        let gathered_before_overwrite = storage.gather(&[0]).unwrap();
        storage.insert(2, replay_insert(&[3, 4], &device)).unwrap();
        storage.insert(0, replay_insert(&[5, 6], &device)).unwrap();

        assert_eq!(
            gathered_before_overwrite.states.to_vec2::<u8>().unwrap(),
            vec![vec![1]]
        );
        assert_eq!(
            storage
                .gather(&[0, 1, 2, 3])
                .unwrap()
                .states
                .to_vec2::<u8>()
                .unwrap(),
            vec![vec![5], vec![6], vec![15], vec![16]]
        );
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

    #[test]
    fn selected_action_q_values_uses_one_value_per_transition() {
        let device = Device::Cpu;
        let q_values =
            Tensor::from_vec(vec![1.0f32, 5.0, 2.0, 7.0, 3.0, 4.0], (2, 3), &device).unwrap();
        let actions = Tensor::from_vec(vec![1u32, 2], 2, &device).unwrap();

        let selected = selected_action_q_values(&q_values, &actions).unwrap();
        assert_eq!(
            selected.to_vec2::<f32>().unwrap(),
            vec![vec![5.0], vec![4.0]]
        );
    }

    fn q_network(var_map: &VarMap, device: &Device) -> MLP {
        MLP::builder()
            .input_size(4)
            .output_size(2)
            .vb(VarBuilder::from_varmap(var_map, DType::F32, device))
            .activation(Tensor::tanh)
            .hidden_layer_sizes(vec![2])
            .build()
            .unwrap()
    }

    struct EpisodeEnv {
        device: Device,
        steps: usize,
    }

    impl EpisodeEnv {
        fn new(device: Device) -> Self {
            Self { device, steps: 0 }
        }
    }

    impl Gym for EpisodeEnv {
        type Error = candle_core::Error;
        type SpaceError = candle_core::Error;

        /// Steps with one scalar discrete action shaped `[]`.
        fn step(&mut self, _action: Tensor) -> Result<StepInfo, Self::Error> {
            self.steps += 1;
            Ok(StepInfo {
                state: Tensor::zeros(&[4], DType::F32, &self.device)?,
                reward: 1.0,
                done: self.steps == 2,
                truncated: false,
                info: (),
            })
        }

        fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
            self.steps = 0;
            Ok(ResetInfo {
                state: Tensor::zeros(&[4], DType::F32, &self.device)?,
                info: (),
            })
        }

        fn observation_space(&self) -> Box<dyn crate::spaces::Space<Error = Self::SpaceError>> {
            Box::new(BoxSpace::new(
                Tensor::zeros(&[4], DType::F32, &self.device).unwrap(),
                Tensor::ones(&[4], DType::F32, &self.device).unwrap(),
            ))
        }

        fn action_space(&self) -> Box<dyn crate::spaces::Space<Error = Self::SpaceError>> {
            Box::new(Discrete::new(2))
        }
    }

    #[test]
    fn learn_initializes_replay_and_preserves_alignment_across_calls() {
        let device = Device::Cpu;
        let mut env: VectorizedGymWrapper<FixedEnv> =
            vec![FixedEnv::new(device.clone()), FixedEnv::new(device.clone())].into();
        let online_var_map = VarMap::new();
        let mut target_var_map = VarMap::new();
        let online_network = q_network(&online_var_map, &device);
        let target_network = q_network(&target_var_map, &device);
        let variable_name = online_var_map
            .data()
            .lock()
            .unwrap()
            .keys()
            .next()
            .unwrap()
            .clone();
        let online_variable = online_var_map.data().lock().unwrap()[&variable_name].clone();
        let target_variable = target_var_map.data().lock().unwrap()[&variable_name].clone();
        let mut agent: QLearningAgent<
            '_,
            CountingOptimizer,
            VectorizedGymError<candle_core::Error>,
            candle_core::Error,
            TestTarget,
        > = QLearningAgent::builder()
            .action_space(Discrete::new(2))
            .observation_space(env.observation_space())
            .online_q_network(Box::new(online_network))
            .target_q_network(Box::new(target_network))
            .online_vars(&online_var_map)
            .target_vars(&mut target_var_map)
            .optimizer(CountingOptimizer::with_learning_rate(1e-3))
            .epsilon_schedule(Box::new(LinearSchedule::new(1.0, 0.0)))
            .replay_capacity(8)
            .batch_size(1)
            .training_start(3)
            .update_frequency(2)
            .target_update_interval(4)
            .training_horizon(10)
            .replay_storage_config(ReplayStorageConfig::new(ReplayDeviceStrategy::OneDevice(
                device.clone(),
            )))
            .build()
            .unwrap();

        let mut logger = NoopLogger;
        agent.learn(&mut env, 2, &mut logger).unwrap();
        assert_eq!(agent.schedule_progress.elapsed_steps(), 2);
        assert_eq!(agent.current_epsilon, 1.0);
        assert_eq!(agent.optimizer.steps, 0);

        let changed = Tensor::full(5.0f32, online_variable.as_tensor().shape(), &device).unwrap();
        online_variable.set(&changed).unwrap();
        agent.learn(&mut env, 2, &mut logger).unwrap();

        assert_eq!(agent.schedule_progress.elapsed_steps(), 4);
        assert_eq!(agent.current_epsilon, 0.8);
        assert_eq!(agent.optimizer.steps, 1);
        assert_eq!(
            online_variable
                .as_tensor()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            target_variable
                .as_tensor()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );

        let replay_len = agent.experience_replay.len();
        let mut incompatible_env: VectorizedGymWrapper<FixedEnv> =
            vec![FixedEnv::new(device)].into();
        assert!(matches!(
            agent.learn(&mut incompatible_env, 1, &mut logger),
            Err(QAgentError::ReplayStorageError(
                ReplayStorageError::EnvironmentCountMismatch {
                    expected: 2,
                    actual: 1,
                }
            ))
        ));
        assert_eq!(agent.experience_replay.len(), replay_len);
    }

    #[test]
    fn collection_logs_report_fresh_rewards_and_completed_episodes() {
        let device = Device::Cpu;
        let mut env: VectorizedGymWrapper<EpisodeEnv> = vec![
            EpisodeEnv::new(device.clone()),
            EpisodeEnv::new(device.clone()),
        ]
        .into();
        let online_var_map = VarMap::new();
        let mut target_var_map = VarMap::new();
        let online_network = q_network(&online_var_map, &device);
        let target_network = q_network(&target_var_map, &device);
        let mut agent: QLearningAgent<
            '_,
            CountingOptimizer,
            VectorizedGymError<candle_core::Error>,
            candle_core::Error,
            TestTarget,
        > = QLearningAgent::builder()
            .action_space(Discrete::new(2))
            .observation_space(env.observation_space())
            .online_q_network(Box::new(online_network))
            .target_q_network(Box::new(target_network))
            .online_vars(&online_var_map)
            .target_vars(&mut target_var_map)
            .optimizer(CountingOptimizer::with_learning_rate(1e-3))
            .replay_capacity(8)
            .batch_size(1)
            .training_start(10)
            .update_frequency(1)
            .target_update_interval(4)
            .training_horizon(4)
            .replay_storage_config(ReplayStorageConfig::new(ReplayDeviceStrategy::OneDevice(
                device,
            )))
            .build()
            .unwrap();

        struct CollectionLogger {
            collection_rewards: Vec<Vec<f32>>,
            completed_episodes: Vec<(usize, f32, usize, bool, bool, usize)>,
        }

        impl QLearningLogger for CollectionLogger {
            fn log_update(&mut self, _entry: &super::QLogEntry) {
                panic!("training has not reached its warm-up");
            }

            fn log_collection(&mut self, entry: &QCollectionLogEntry) {
                self.collection_rewards
                    .push(entry.collection_rewards.to_vec1::<f32>().unwrap());
                self.completed_episodes
                    .extend(entry.completed_episodes.iter().map(|episode| {
                        (
                            episode.environment_index,
                            episode.episode_return,
                            episode.episode_length,
                            episode.terminated,
                            episode.truncated,
                            episode.collection_timestep,
                        )
                    }));
            }
        }

        let mut logger = CollectionLogger {
            collection_rewards: Vec::new(),
            completed_episodes: Vec::new(),
        };
        agent.learn(&mut env, 4, &mut logger).unwrap();

        assert_eq!(
            logger.collection_rewards,
            vec![vec![1.0, 1.0], vec![1.0, 1.0]]
        );
        assert_eq!(
            logger.completed_episodes,
            vec![(0, 2.0, 2, true, false, 3), (1, 2.0, 2, true, false, 4)]
        );
    }
}

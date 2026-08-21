use bon::bon;
use candle_core::{DType, Error, Tensor};
use candle_nn::{Optimizer, VarMap};
use std::{marker::PhantomData, ops::Deref};

use crate::{
    agents::ReplayDeviceStrategy,
    buffers::experience_replay::{
        ExperienceReplay, ExperienceReplayError, ReplayStorage, ReplayStorageError,
        replay_index_tensor,
    },
    gym::{MultiGym, MultiGymStepInfo},
    parameter_schedule::{LinearSchedule, ParameterSchedule, ScheduleProgress},
    spaces::{Discrete, Space},
    tensor_operations::tensor_has_nan,
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
}

struct QLearningReplayStorage {
    states: Tensor,
    next_states: Tensor,
    actions: Tensor,
    rewards: Tensor,
    next_dones: Tensor,
    capacity: usize,
    replay_dtype: DType,
    device: candle_core::Device,
}

impl QLearningReplayStorage {
    fn new(
        capacity: usize,
        observation_shape: &[usize],
        replay_dtype: DType,
        device: candle_core::Device,
    ) -> Result<Self, Error> {
        let mut state_shape = Vec::with_capacity(observation_shape.len() + 1);
        state_shape.push(capacity);
        state_shape.extend_from_slice(observation_shape);

        let states = Tensor::zeros(state_shape.as_slice(), replay_dtype, &device)?;
        let next_states = Tensor::zeros(state_shape.as_slice(), replay_dtype, &device)?;
        let actions = Tensor::zeros(capacity, DType::U32, &device)?;
        let rewards = Tensor::zeros(capacity, DType::F32, &device)?;
        let next_dones = Tensor::zeros(capacity, DType::F32, &device)?;

        Ok(Self {
            states,
            next_states,
            actions,
            rewards,
            next_dones,
            capacity,
            replay_dtype,
            device,
        })
    }

    /// Copies `[batch, ...]` into `[capacity, ...]` with ring wrapping.
    fn ring_write(
        &self,
        destination: &Tensor,
        source: &Tensor,
        start: usize,
    ) -> Result<(), ReplayStorageError> {
        let count = source.dim(0)?;
        if count > self.capacity {
            return Err(ReplayStorageError::InsertionExceedsCapacity {
                capacity: self.capacity,
                inserted: count,
            });
        }

        let first_count = count.min(self.capacity - start);
        if first_count != 0 {
            let first = source.narrow(0, 0, first_count)?.contiguous()?;
            destination.slice_set(&first, 0, start)?;
        }

        let second_count = count - first_count;
        if second_count != 0 {
            let second = source.narrow(0, first_count, second_count)?.contiguous()?;
            destination.slice_set(&second, 0, 0)?;
        }
        Ok(())
    }

    /// Converts a `[batch, ...]` tensor to the replay dtype and device.
    fn prepare(&self, tensor: &Tensor, dtype: DType) -> Result<Tensor, Error> {
        tensor
            .to_dtype(dtype)?
            .to_device(&self.device)?
            .contiguous()
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
        let count = transitions.states.dim(0)?;
        for (name, tensor) in [
            ("next states", &transitions.next_states),
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

        let states = self.prepare(&transitions.states, self.replay_dtype)?;
        let next_states = self.prepare(&transitions.next_states, self.replay_dtype)?;
        let actions = self.prepare(&transitions.actions, DType::U32)?;
        let rewards = self.prepare(&transitions.rewards, DType::F32)?;
        let next_dones = self.prepare(&transitions.next_dones, DType::F32)?;

        self.ring_write(&self.states, &states, start)?;
        self.ring_write(&self.next_states, &next_states, start)?;
        self.ring_write(&self.actions, &actions, start)?;
        self.ring_write(&self.rewards, &rewards, start)?;
        self.ring_write(&self.next_dones, &next_dones, start)?;
        Ok(count)
    }

    fn gather(&self, indices: &[usize]) -> Result<Self::Batch, Self::Error> {
        let indices = replay_index_tensor(indices, &self.device)?;

        Ok(QLearningBatch {
            states: self.states.index_select(&indices, 0)?.detach(),
            next_states: self.next_states.index_select(&indices, 0)?.detach(),
            actions: self.actions.index_select(&indices, 0)?.detach(),
            rewards: self.rewards.index_select(&indices, 0)?.detach(),
            next_dones: self.next_dones.index_select(&indices, 0)?.detach(),
        })
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
    device_strategy: ReplayDeviceStrategy,
    dtype: DType,
    optimization_steps: usize,
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
        device_strategy: ReplayDeviceStrategy,
        #[builder(default = DType::F32)] dtype: DType,
        #[builder(default = DType::F32)] replay_dtype: DType,
    ) -> Result<Self, QAgentError<GE, SE>> {
        assert!(
            dtype.is_float(),
            "Q-learning compute dtype must be floating-point"
        );
        assert!(
            replay_dtype.is_float() || replay_dtype == DType::U8,
            "Q-learning replay dtype must be floating-point or u8"
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

        let storage_device = device_strategy.storage_device();
        let replay_storage = QLearningReplayStorage::new(
            replay_capacity,
            &observation_space.shape(),
            replay_dtype,
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
            device_strategy,
            dtype,
            optimization_steps: 0,
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
            .to_device(&self.device_strategy.optimization_device())?
            .to_dtype(self.dtype)?;
        Ok(epsilon_greedy_actions(
            &observation,
            self.current_epsilon,
            &self.action_space,
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
        let optimization_device = self.device_strategy.optimization_device();
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
        if !tensor_has_nan(&loss)? {
            self.optimizer.backward_step(&loss)?;
        }
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
            &candle_core::Device::Cpu,
        )?;
        self.experience_replay.add(QLearningInsert {
            states: states.clone(),
            next_states: next_states.clone(),
            actions: actions.clone(),
            rewards: rewards.clone(),
            next_dones,
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
        let mut observations = env.reset().map_err(QAgentError::GymError)?;
        let environment_count = env.num_envs();
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
            // A vectorized collection batch spans several consecutive
            // timesteps. Emit its endpoint after any scheduled updates inside
            // that range so timestamp-ordered loggers never see an update
            // following the batch's later collection timestamp.
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
    forward: impl FnOnce(&Tensor) -> Result<Tensor, Error>,
) -> Result<Tensor, Error> {
    let batch_size = observation.shape().dims()[0];
    let explores = Tensor::rand(0.0f64, 1.0, &[batch_size], observation.device())?
        .to_vec1::<f64>()?
        .into_iter()
        .map(|value| value < epsilon)
        .collect::<Vec<_>>();

    let greedy_indices = explores
        .iter()
        .enumerate()
        .filter_map(|(index, &explore)| (!explore).then_some(index as u32))
        .collect::<Vec<_>>();
    let greedy_count = greedy_indices.len();
    let mut greedy_actions = if greedy_indices.is_empty() {
        None
    } else {
        let greedy_indices =
            Tensor::from_vec(greedy_indices, &[greedy_count], observation.device())?;
        let greedy_observations = observation.index_select(&greedy_indices, 0)?;
        Some(
            forward(&greedy_observations)?
                .argmax(1)?
                .chunk(greedy_count, 0)?
                .into_iter(),
        )
    };

    let mut actions = Vec::with_capacity(batch_size);
    for explore in explores {
        if explore {
            actions.push(action_space.sample(observation.device())?);
        } else {
            let action = greedy_actions
                .as_mut()
                .and_then(|actions| actions.next())
                .expect("greedy actions must exist for non-exploring environments");
            actions.push(action.squeeze(0)?);
        }
    }
    Tensor::stack(&actions, 0)
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
        QCollectionLogEntry, QLearningAgent, QLearningConfigurationError,
        QLearningConfigurationValidator, QLearningInsert, QLearningLogger, QLearningReplayStorage,
        QLearningTarget, selected_action_q_values, validate_epsilon,
    };
    use crate::{
        agents::{
            ReplayDeviceStrategy,
            test_support::{CountingOptimizer, FixedEnv},
        },
        gym::{Gym, MultiGym, ResetInfo, StepInfo, VectorizedGymError, VectorizedGymWrapper},
        models::MLP,
        objectives::bellman_targets,
        parameter_schedule::LinearSchedule,
        spaces::{BoxSpace, Discrete},
    };
    use candle_core::{DType, Device, Error, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    struct TestTarget;

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
        }
    }

    #[test]
    fn preallocated_replay_wraps_and_gathered_batches_do_not_alias_storage() {
        use crate::buffers::experience_replay::ReplayStorage;

        let device = Device::Cpu;
        let mut storage = QLearningReplayStorage::new(3, &[1], DType::U8, device.clone()).unwrap();
        storage.insert(0, replay_insert(&[1, 2], &device)).unwrap();

        let gathered_before_overwrite = storage.gather(&[0]).unwrap();
        storage.insert(2, replay_insert(&[3, 4], &device)).unwrap();

        assert_eq!(
            gathered_before_overwrite.states.to_vec2::<u8>().unwrap(),
            vec![vec![1]]
        );
        assert_eq!(
            storage
                .gather(&[0, 1, 2])
                .unwrap()
                .states
                .to_vec2::<u8>()
                .unwrap(),
            vec![vec![4], vec![2], vec![3]]
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
    fn cadence_and_epsilon_progress_continue_across_learn_calls() {
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
            .device_strategy(ReplayDeviceStrategy::OneDevice(device.clone()))
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
            .device_strategy(ReplayDeviceStrategy::OneDevice(device))
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

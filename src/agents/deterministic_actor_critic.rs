//! Shared implementation for deterministic off-policy actor-critic agents.
//!
//! [`ddpg::DDPGAgent`] and [`td3::TD3Agent`] are thin public wrappers around
//! the training loop in this module. Their algorithm-specific behavior is
//! supplied by an internal strategy.

use std::{fmt::Debug, marker::PhantomData, ops::Deref};

use bon::bon;
use candle_core::{DType, Tensor};
use candle_nn::{Optimizer, VarMap};

use super::{
    ReplayDeviceStrategy,
    sac::{SACCritic, SACCriticError},
};
use crate::{
    buffers::experience_replay::{
        AlignedObservationReplay, ExperienceReplay, ExperienceReplayError, ReplayStorage,
        ReplayStorageError, TensorReplayColumn, replay_index_tensor,
    },
    gym::{MultiGym, MultiGymStepInfo},
    objectives::bellman_targets,
    parameter_schedule::ScheduleProgress,
    spaces::{BoxSpace, Space},
};

pub mod ddpg;
pub mod td3;

/// A scalar state-action critic suitable for DDPG and TD3.
pub type DeterministicCritic<'a, O> = SACCritic<'a, O>;

/// Invalid deterministic actor-critic configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeterministicActorCriticConfigurationError {
    /// TD3 received an empty critic ensemble.
    NoCritics,
    /// Replay capacity was zero.
    ZeroReplayCapacity,
    /// Replay batch size was zero.
    ZeroBatchSize,
    /// Replay cannot retain one complete optimization batch.
    ReplayCapacityBelowBatchSize,
    /// The replay-update frequency was zero.
    ZeroUpdateFrequency,
    /// The global collection horizon was zero.
    ZeroTrainingHorizon,
    /// Gamma was non-finite or outside `0.0..=1.0`.
    InvalidGamma,
    /// Tau was non-finite or outside `0.0..=1.0`.
    InvalidTau,
    /// The collection exploration-noise deviation was invalid.
    InvalidExplorationNoise,
    /// The target-policy-noise deviation was invalid.
    InvalidTargetPolicyNoise,
    /// The target-policy-noise clip was invalid.
    InvalidTargetNoiseClip,
    /// The actor-update interval was zero.
    ZeroActorUpdateInterval,
    /// An algorithm received a different number of critics than it requires.
    IncorrectCriticCount {
        /// Number of critics required by the algorithm.
        expected: usize,
        /// Number of critics supplied to the builder.
        actual: usize,
    },
}

/// DDPG and TD3 construction or training failure.
#[derive(Debug)]
pub enum DeterministicActorCriticError<GE, SE>
where
    GE: Debug,
    SE: Debug,
{
    /// A Candle tensor operation failed.
    TensorError(candle_core::Error),
    ReplayStorageError(ReplayStorageError),
    /// Critic construction, aggregation, or optimization failed.
    CriticError(SACCriticError),
    /// An agent builder value violated its configuration contract.
    ConfigurationError(DeterministicActorCriticConfigurationError),
    /// The online and target actors contain different parameter names.
    ActorParameterMapMismatch {
        /// Parameter names found only in the online actor.
        online_only: Vec<String>,
        /// Parameter names found only in the target actor.
        target_only: Vec<String>,
    },
    /// The vectorized environment returned an error.
    GymError(GE),
    /// The action or observation space returned an error.
    SpaceError(SE),
}

/// Result type returned by DDPG and TD3 construction and training operations.
pub type DeterministicActorCriticResult<T, GE, SE> =
    Result<T, DeterministicActorCriticError<GE, SE>>;

impl<GE: Debug, SE: Debug> From<candle_core::Error> for DeterministicActorCriticError<GE, SE> {
    fn from(error: candle_core::Error) -> Self {
        Self::TensorError(error)
    }
}

impl<GE: Debug, SE: Debug> From<SACCriticError> for DeterministicActorCriticError<GE, SE> {
    fn from(error: SACCriticError) -> Self {
        Self::CriticError(error)
    }
}

impl<GE: Debug, SE: Debug> From<DeterministicActorCriticConfigurationError>
    for DeterministicActorCriticError<GE, SE>
{
    fn from(error: DeterministicActorCriticConfigurationError) -> Self {
        Self::ConfigurationError(error)
    }
}

/// Metrics from one replay update.
pub struct DeterministicActorCriticLogEntry {
    /// One scalar optimization loss per critic.
    pub critic_losses: Vec<Tensor>,
    /// Each critic's Q predictions for the sampled replay state-action pairs.
    /// Every tensor is shaped `[batch_size]`.
    pub critic_q_values: Vec<Tensor>,
    /// Scalar negative mean policy Q loss.
    ///
    /// This is `None` on updates where TD3 delays the actor.
    pub actor_loss: Option<Tensor>,
    /// Q values used by the actor objective, shaped `[batch_size]`.
    ///
    /// DDPG uses its sole critic. Canonical TD3 uses its first critic for the
    /// actor objective. A configured TD3 actor aggregation mode combines the
    /// critic ensemble instead. This is `None` on updates where TD3 delays the
    /// actor.
    pub policy_q_values: Option<Tensor>,
    /// Current policy actions shaped `[batch_size, ...action_shape]`.
    /// `None` on updates where TD3 delays the actor.
    pub policy_actions: Option<Tensor>,
    /// Replay actions shaped `[batch_size, ...action_shape]`.
    pub replay_actions: Tensor,
    /// Detached expected Q targets shared by all critics, shaped
    /// `[batch_size]`.
    pub bellman_targets: Tensor,
    /// Rewards from the sampled replay batch, shaped `[batch_size]`.
    pub replay_rewards: Tensor,
    /// Current actor optimizer learning rate.
    pub actor_learning_rate: f32,
    /// Current learning rate for each critic optimizer.
    pub critic_learning_rates: Vec<f32>,
    /// Standard deviation of the Gaussian collection-action noise.
    pub exploration_noise_standard_deviation: f64,
    /// Whether this replay update changed the actor and target networks.
    pub actor_updated: bool,
    /// Zero-based replay-optimization index.
    pub update_index: usize,
    /// Global collected-transition count that triggered this update.
    pub collection_timestep: usize,
}

/// Metrics from one vectorized environment step.
pub struct DeterministicActorCriticCollectionLogEntry<I = ()> {
    /// Newest reward from each inner environment, shaped `[environment_count]`.
    pub collection_rewards: Tensor,
    /// Typed metadata returned by the inner environments.
    pub infos: Vec<I>,
    /// Global collected-transition count after this vectorized step.
    pub collection_timestep: usize,
    /// Episodes that terminated or truncated on this vectorized step.
    pub completed_episodes: Vec<DeterministicActorCriticEpisodeLogEntry>,
    /// Number of transitions currently retained in replay.
    pub replay_len: usize,
}

/// Summary of an episode completed during collection.
pub struct DeterministicActorCriticEpisodeLogEntry {
    /// Index of the inner vectorized environment that completed the episode.
    pub environment_index: usize,
    /// Sum of rewards over the completed episode.
    pub episode_return: f32,
    /// Number of transitions in the completed episode.
    pub episode_length: usize,
    /// Whether the environment terminated the episode.
    pub terminated: bool,
    /// Whether the episode was truncated.
    pub truncated: bool,
    /// Global collected-transition count at the end of the episode.
    pub collection_timestep: usize,
}

struct EpisodeTracker {
    returns: Vec<f32>,
    lengths: Vec<usize>,
}

impl EpisodeTracker {
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
    ) -> Option<DeterministicActorCriticEpisodeLogEntry> {
        self.returns[environment_index] += reward;
        self.lengths[environment_index] += 1;
        if !terminated && !truncated {
            return None;
        }
        let entry = DeterministicActorCriticEpisodeLogEntry {
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

pub(crate) trait DeterministicActorCriticLogger<I = ()> {
    fn log_update(&mut self, entry: &DeterministicActorCriticLogEntry);
    fn log_collection(&mut self, entry: &DeterministicActorCriticCollectionLogEntry<I>);
}

pub(crate) trait DeterministicActorCriticStrategy {
    fn critic_count_requirement(&self) -> CriticCountRequirement;
    fn actor_update_interval(&self) -> usize;
    fn target_policy_noise(&self) -> f64;
    fn target_noise_clip(&self) -> f64;

    /// Combines the target critics' next-state Q estimates into one value per
    /// replay transition. DDPG uses its sole critic; TD3 applies its configured
    /// ensemble aggregation. Each input and the output are shaped
    /// `[batch_size]`.
    fn aggregate_target_values(&self, values: Vec<Tensor>) -> Result<Tensor, SACCriticError>;

    /// Combines the critics' current-policy Q estimates into the
    /// `[batch_size]` values used by the actor objective.
    fn aggregate_actor_values(&self, values: Vec<Tensor>) -> Result<Tensor, SACCriticError>;
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum CriticCountRequirement {
    Exact(usize),
    NonEmpty,
}

struct DeterministicBatch {
    states: Tensor,
    next_states: Tensor,
    actions: Tensor,
    rewards: Tensor,
    terminated: Tensor,
}

impl<GE: Debug, SE: Debug> From<ReplayStorageError> for DeterministicActorCriticError<GE, SE> {
    fn from(error: ReplayStorageError) -> Self {
        Self::ReplayStorageError(error)
    }
}

impl<GE: Debug, SE: Debug> From<ExperienceReplayError<ReplayStorageError>>
    for DeterministicActorCriticError<GE, SE>
{
    fn from(error: ExperienceReplayError<ReplayStorageError>) -> Self {
        Self::ReplayStorageError(error.into())
    }
}

struct DeterministicReplayInsert {
    states: Tensor,
    next_states: Tensor,
    actions: Tensor,
    rewards: Tensor,
    terminated: Tensor,
    truncateds: Vec<bool>,
}

struct DeterministicReplayStorage {
    observations: AlignedObservationReplay,
    actions: TensorReplayColumn,
    rewards: TensorReplayColumn,
    terminated: TensorReplayColumn,
    capacity: usize,
    device: candle_core::Device,
}

impl DeterministicReplayStorage {
    fn new(
        capacity: usize,
        state_shape: &[usize],
        action_shape: &[usize],
        dtype: DType,
        device: candle_core::Device,
    ) -> Result<Self, ReplayStorageError> {
        Ok(Self {
            observations: AlignedObservationReplay::new(capacity, state_shape, dtype, &device),
            actions: TensorReplayColumn::new(capacity, action_shape, dtype, &device)?,
            rewards: TensorReplayColumn::new(capacity, &[], dtype, &device)?,
            terminated: TensorReplayColumn::new(capacity, &[], dtype, &device)?,
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

impl ReplayStorage for DeterministicReplayStorage {
    type Insert = DeterministicReplayInsert;
    type Batch = DeterministicBatch;
    type Error = ReplayStorageError;

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn insert(&mut self, start: usize, transitions: Self::Insert) -> Result<usize, Self::Error> {
        let prepared = DeterministicReplayInsert {
            states: transitions.states,
            next_states: transitions.next_states,
            actions: self.actions.prepare(&transitions.actions)?,
            rewards: self.rewards.prepare(&transitions.rewards)?,
            terminated: self.terminated.prepare(&transitions.terminated)?,
            truncateds: transitions.truncateds,
        };
        let count = prepared.states.dim(0)?;
        for (name, tensor) in [
            ("actions", &prepared.actions),
            ("rewards", &prepared.rewards),
            ("terminated", &prepared.terminated),
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
            &prepared.states,
            &prepared.next_states,
            &prepared.truncateds,
        )?;
        for (column, tensor) in [
            (&self.actions, &prepared.actions),
            (&self.rewards, &prepared.rewards),
            (&self.terminated, &prepared.terminated),
        ] {
            column.write(start, tensor)?;
        }
        Ok(count)
    }

    fn gather(&self, indices: &[usize]) -> Result<Self::Batch, Self::Error> {
        let (states, next_states) = self.observations.gather(indices)?;
        let indices = replay_index_tensor(indices, &self.device)?;
        Ok(DeterministicBatch {
            states,
            next_states,
            actions: self.actions.gather(&indices)?,
            rewards: self.rewards.gather(&indices)?,
            terminated: self.terminated.gather(&indices)?,
        })
    }

    fn sampleable_len(&self, len: usize) -> usize {
        self.observations.sampleable_len(len)
    }

    fn sample_index(&self, index: usize, len: usize) -> usize {
        self.observations.sample_index(index, len)
    }
}

struct CollectedTransitions<'a> {
    states: &'a Tensor,
    next_states: &'a Tensor,
    actions: &'a Tensor,
    rewards: &'a Tensor,
    dones: &'a [bool],
    truncateds: &'a [bool],
    first_timestep: usize,
}

struct CriticUpdate {
    losses: Vec<Tensor>,
    q_values: Vec<Tensor>,
}

struct ActorUpdate {
    loss: Tensor,
    q_values: Tensor,
    actions: Tensor,
}

pub(crate) struct DeterministicActorCriticAgent<'a, AO, CO, GE, SE, S>
where
    AO: Optimizer,
    CO: Optimizer,
    GE: Debug,
    SE: Debug,
    S: DeterministicActorCriticStrategy,
{
    online_actor: Box<dyn candle_core::Module>,
    target_actor: Box<dyn candle_core::Module>,
    online_actor_vars: &'a VarMap,
    target_actor_vars: &'a mut VarMap,
    actor_optimizer: AO,
    critics: Vec<SACCritic<'a, CO>>,
    strategy: S,
    action_space: BoxSpace,
    observation_space: Box<dyn Space<Error = SE>>,
    replay: ExperienceReplay<DeterministicReplayInsert, DeterministicReplayStorage>,
    gamma: f64,
    tau: f64,
    exploration_noise: f64,
    training_start: usize,
    update_frequency: usize,
    schedule_progress: ScheduleProgress,
    device_strategy: ReplayDeviceStrategy,
    optimization_steps: usize,
    dtype: DType,
    _errors: PhantomData<GE>,
}

#[bon]
impl<'a, AO, CO, GE, SE, S> DeterministicActorCriticAgent<'a, AO, CO, GE, SE, S>
where
    AO: Optimizer,
    CO: Optimizer,
    GE: Debug,
    SE: Debug,
    S: DeterministicActorCriticStrategy,
{
    #[builder]
    pub(crate) fn new(
        online_actor: Box<dyn candle_core::Module>,
        target_actor: Box<dyn candle_core::Module>,
        online_actor_vars: &'a VarMap,
        target_actor_vars: &'a mut VarMap,
        actor_optimizer: AO,
        critics: Vec<SACCritic<'a, CO>>,
        strategy: S,
        action_space: BoxSpace,
        observation_space: Box<dyn Space<Error = SE>>,
        device_strategy: ReplayDeviceStrategy,
        #[builder(default = 0.99)] gamma: f64,
        #[builder(default = 0.005)] tau: f64,
        #[builder(default = 0.1)] exploration_noise: f64,
        #[builder(default = 1_000_000)] replay_capacity: usize,
        #[builder(default = 256)] batch_size: usize,
        #[builder(default = 1)] update_frequency: usize,
        #[builder(default = 1_000)] training_start: usize,
        training_horizon: usize,
        #[builder(default = DType::F32)] dtype: DType,
    ) -> DeterministicActorCriticResult<Self, GE, SE> {
        assert!(
            dtype.is_float(),
            "deterministic actor-critic compute dtype must be floating-point"
        );
        DeterministicActorCriticConfigurationValidator::validate_configuration()
            .critic_count_requirement(strategy.critic_count_requirement())
            .actual_critic_count(critics.len())
            .replay_capacity(replay_capacity)
            .batch_size(batch_size)
            .gamma(gamma)
            .tau(tau)
            .exploration_noise(exploration_noise)
            .target_policy_noise(strategy.target_policy_noise())
            .target_noise_clip(strategy.target_noise_clip())
            .actor_update_interval(strategy.actor_update_interval())
            .update_frequency(update_frequency)
            .training_horizon(training_horizon)
            .call()?;
        copy_actor_var_map(online_actor_vars, target_actor_vars, 1.0)?;
        let storage_device = device_strategy.storage_device();
        let observation_sample = observation_space
            .sample(&storage_device)
            .map_err(DeterministicActorCriticError::SpaceError)?;
        let action_sample = action_space.sample(&storage_device)?;
        let replay_storage = DeterministicReplayStorage::new(
            replay_capacity,
            observation_sample.dims(),
            action_sample.dims(),
            dtype,
            storage_device.clone(),
        )?;
        Ok(Self {
            online_actor,
            target_actor,
            online_actor_vars,
            target_actor_vars,
            actor_optimizer,
            critics,
            strategy,
            action_space,
            observation_space,
            replay: ExperienceReplay::with_storage(replay_storage, batch_size),
            gamma,
            tau,
            exploration_noise,
            training_start,
            update_frequency,
            schedule_progress: ScheduleProgress::new(training_horizon),
            device_strategy,
            optimization_steps: 0,
            dtype,
            _errors: PhantomData,
        })
    }
}

struct DeterministicActorCriticConfigurationValidator;

#[bon]
impl DeterministicActorCriticConfigurationValidator {
    #[builder]
    fn validate_configuration(
        critic_count_requirement: CriticCountRequirement,
        actual_critic_count: usize,
        replay_capacity: usize,
        batch_size: usize,
        gamma: f64,
        tau: f64,
        exploration_noise: f64,
        target_policy_noise: f64,
        target_noise_clip: f64,
        actor_update_interval: usize,
        update_frequency: usize,
        training_horizon: usize,
    ) -> Result<(), DeterministicActorCriticConfigurationError> {
        use DeterministicActorCriticConfigurationError as Error;

        let error = if matches!(critic_count_requirement, CriticCountRequirement::NonEmpty)
            && actual_critic_count == 0
        {
            Some(Error::NoCritics)
        } else if let CriticCountRequirement::Exact(expected) = critic_count_requirement
            && actual_critic_count != expected
        {
            Some(Error::IncorrectCriticCount {
                expected,
                actual: actual_critic_count,
            })
        } else if replay_capacity == 0 {
            Some(Error::ZeroReplayCapacity)
        } else if batch_size == 0 {
            Some(Error::ZeroBatchSize)
        } else if replay_capacity < batch_size {
            Some(Error::ReplayCapacityBelowBatchSize)
        } else if update_frequency == 0 {
            Some(Error::ZeroUpdateFrequency)
        } else if training_horizon == 0 {
            Some(Error::ZeroTrainingHorizon)
        } else if !gamma.is_finite() || !(0.0..=1.0).contains(&gamma) {
            Some(Error::InvalidGamma)
        } else if !tau.is_finite() || !(0.0..=1.0).contains(&tau) {
            Some(Error::InvalidTau)
        } else if !exploration_noise.is_finite() || exploration_noise < 0.0 {
            Some(Error::InvalidExplorationNoise)
        } else if !target_policy_noise.is_finite() || target_policy_noise < 0.0 {
            Some(Error::InvalidTargetPolicyNoise)
        } else if !target_noise_clip.is_finite() || target_noise_clip < 0.0 {
            Some(Error::InvalidTargetNoiseClip)
        } else if actor_update_interval == 0 {
            Some(Error::ZeroActorUpdateInterval)
        } else {
            None
        };
        error.map_or(Ok(()), Err)
    }
}

fn copy_actor_var_map<GE: Debug, SE: Debug>(
    online: &VarMap,
    target: &mut VarMap,
    tau: f64,
) -> Result<(), DeterministicActorCriticError<GE, SE>> {
    let online_values = online
        .data()
        .lock()
        .unwrap()
        .deref()
        .iter()
        .map(|(name, var)| (name.clone(), var.as_tensor().clone()))
        .collect::<Vec<_>>();
    let online_names = online_values
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<std::collections::HashSet<_>>();
    let target_names = target
        .data()
        .lock()
        .unwrap()
        .keys()
        .cloned()
        .collect::<std::collections::HashSet<_>>();
    if online_names != target_names {
        let mut online_only = online_names
            .difference(&target_names)
            .cloned()
            .collect::<Vec<_>>();
        online_only.sort();
        let mut target_only = target_names
            .difference(&online_names)
            .cloned()
            .collect::<Vec<_>>();
        target_only.sort();
        return Err(DeterministicActorCriticError::ActorParameterMapMismatch {
            online_only,
            target_only,
        });
    }
    for (name, online_value) in online_values {
        let value = if tau == 1.0 {
            online_value
        } else {
            let target_value = target
                .data()
                .lock()
                .unwrap()
                .get(&name)
                .expect("validated target actor parameter name")
                .as_tensor()
                .clone();
            ((online_value * tau)? + (target_value * (1.0 - tau))?)?
        };
        target.set_one(name, &value)?;
    }
    Ok(())
}

impl<'a, AO, CO, GE, SE, S> DeterministicActorCriticAgent<'a, AO, CO, GE, SE, S>
where
    AO: Optimizer,
    CO: Optimizer,
    GE: Debug,
    SE: Debug,
    S: DeterministicActorCriticStrategy,
{
    pub(crate) fn get_action_space(&self) -> &BoxSpace {
        &self.action_space
    }

    pub(crate) fn get_observation_space(&self) -> &dyn Space<Error = SE> {
        self.observation_space.as_ref()
    }

    /// Maps observations `[batch, ...observation_shape]` to deterministic
    /// actions `[batch, ...action_shape]`.
    pub(crate) fn act_deterministic(
        &self,
        observation: &Tensor,
    ) -> Result<Tensor, DeterministicActorCriticError<GE, SE>> {
        let observation = observation
            .to_device(&self.device_strategy.optimization_device())?
            .to_dtype(self.dtype)?;
        let actions = self.online_actor.forward(&observation)?;
        Ok(self.action_space.tensor_from_neurons(&actions)?)
    }

    /// Maps observations `[batch, ...observation_shape]` to exploration-noised
    /// actions `[batch, ...action_shape]`.
    pub(crate) fn act(
        &self,
        observation: &Tensor,
    ) -> Result<Tensor, DeterministicActorCriticError<GE, SE>> {
        let observation = observation
            .to_device(&self.device_strategy.optimization_device())?
            .to_dtype(self.dtype)?;
        let actions = self.online_actor.forward(&observation)?;
        let actions = add_gaussian_noise(&actions, self.exploration_noise, None)?;
        Ok(self.action_space.tensor_from_neurons(&actions)?)
    }

    fn random_actions(
        &self,
        batch_size: usize,
    ) -> Result<Tensor, DeterministicActorCriticError<GE, SE>> {
        let actions = (0..batch_size)
            .map(|_| {
                self.action_space
                    .sample(&self.device_strategy.optimization_device())
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Tensor::stack(&actions, 0)?.to_dtype(self.dtype)?)
    }

    fn sample_batch(
        &self,
    ) -> Result<Option<DeterministicBatch>, DeterministicActorCriticError<GE, SE>> {
        if self.replay.len() < self.replay.get_batch_size() {
            return Ok(None);
        }
        let mut batch = self.replay.sample()?;
        let optimization_device = self.device_strategy.optimization_device();
        for tensor in [
            &mut batch.states,
            &mut batch.next_states,
            &mut batch.actions,
            &mut batch.rewards,
            &mut batch.terminated,
        ] {
            *tensor = tensor.to_device(&optimization_device)?;
        }
        batch.actions = batch.actions.to_dtype(self.dtype)?;
        Ok(Some(batch))
    }

    fn bellman_targets(
        &self,
        batch: &DeterministicBatch,
    ) -> Result<Tensor, DeterministicActorCriticError<GE, SE>> {
        let target_actions = self.target_actor.forward(&batch.next_states)?.detach();
        let target_actions = add_gaussian_noise(
            &target_actions,
            self.strategy.target_policy_noise(),
            Some(self.strategy.target_noise_clip()),
        )?;
        let target_actions = self
            .action_space
            .tensor_from_neurons(&target_actions)?
            .detach();
        let target_values = self
            .critics
            .iter()
            .map(|critic| critic.target_replay_values(&batch.next_states, &target_actions))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let target_values = self
            .strategy
            .aggregate_target_values(target_values)?
            .detach();
        Ok(bellman_targets(
            &batch.rewards,
            &batch.terminated,
            &target_values,
            self.gamma,
        )?
        .detach())
    }

    /// Optimizes critics from a replay batch and targets shaped `[batch]`,
    /// returning one scalar loss and one `[batch]` Q tensor per critic.
    fn optimize_critics(
        &mut self,
        batch: &DeterministicBatch,
        targets: &Tensor,
    ) -> Result<CriticUpdate, DeterministicActorCriticError<GE, SE>> {
        let mut losses = Vec::with_capacity(self.critics.len());
        let mut q_values = Vec::with_capacity(self.critics.len());
        for critic in &mut self.critics {
            let predicted = critic.online_replay_values(&batch.states, &batch.actions)?;
            let loss = candle_nn::loss::mse(&predicted, targets)?;
            critic.optimizer_mut().backward_step(&loss)?;
            losses.push(loss);
            q_values.push(predicted);
        }
        Ok(CriticUpdate { losses, q_values })
    }

    /// Optimizes the actor from states `[batch, ...observation_shape]`,
    /// returning its scalar loss, `[batch]` Q values, and
    /// `[batch, ...action_shape]` actions.
    fn optimize_actor(
        &mut self,
        states: &Tensor,
    ) -> Result<ActorUpdate, DeterministicActorCriticError<GE, SE>> {
        let actions = self.online_actor.forward(states)?;
        let actions = self.action_space.tensor_from_neurons(&actions)?;
        let candidate_actions = actions.unsqueeze(1)?;
        let values = self
            .critics
            .iter()
            .map(|critic| {
                critic
                    .online_actor_values(states, &candidate_actions)?
                    .squeeze(1)
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        let values = self.strategy.aggregate_actor_values(values)?;
        let loss = values.mean_all()?.neg()?;
        self.actor_optimizer.backward_step(&loss)?;
        Ok(ActorUpdate {
            loss,
            q_values: values,
            actions,
        })
    }

    fn update_targets(&mut self) -> Result<(), DeterministicActorCriticError<GE, SE>> {
        copy_actor_var_map(self.online_actor_vars, self.target_actor_vars, self.tau)?;
        for critic in &mut self.critics {
            critic.polyak_update(self.tau)?;
        }
        Ok(())
    }

    fn optimize<I>(
        &mut self,
        collection_timestep: usize,
        logger: &mut dyn DeterministicActorCriticLogger<I>,
    ) -> Result<(), DeterministicActorCriticError<GE, SE>> {
        let Some(batch) = self.sample_batch()? else {
            return Ok(());
        };
        let targets = self.bellman_targets(&batch)?;
        let critic_update = self.optimize_critics(&batch, &targets)?;
        let actor_updated =
            (self.optimization_steps + 1).is_multiple_of(self.strategy.actor_update_interval());
        let actor_update = actor_updated
            .then(|| self.optimize_actor(&batch.states))
            .transpose()?;
        if actor_updated {
            self.update_targets()?;
        }
        let (actor_loss, policy_q_values, policy_actions) = match actor_update {
            Some(actor) => (Some(actor.loss), Some(actor.q_values), Some(actor.actions)),
            None => (None, None, None),
        };
        logger.log_update(&DeterministicActorCriticLogEntry {
            critic_losses: critic_update.losses,
            critic_q_values: critic_update.q_values,
            actor_loss,
            policy_q_values,
            policy_actions,
            replay_actions: batch.actions,
            bellman_targets: targets,
            replay_rewards: batch.rewards,
            actor_learning_rate: self.actor_optimizer.learning_rate() as f32,
            critic_learning_rates: self
                .critics
                .iter()
                .map(|critic| critic.optimizer().learning_rate() as f32)
                .collect(),
            exploration_noise_standard_deviation: self.exploration_noise,
            actor_updated,
            update_index: self.optimization_steps,
            collection_timestep,
        });
        self.optimization_steps += 1;
        Ok(())
    }

    fn store_transitions(
        &mut self,
        transitions: CollectedTransitions<'_>,
        episodes: &mut EpisodeTracker,
    ) -> Result<Vec<DeterministicActorCriticEpisodeLogEntry>, DeterministicActorCriticError<GE, SE>>
    {
        let CollectedTransitions {
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
            let collection_timestep = first_timestep + environment_index + 1;
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
        let storage_device = self.device_strategy.storage_device();
        self.replay.add(DeterministicReplayInsert {
            states: states.clone(),
            next_states: next_states.clone(),
            actions: actions.clone(),
            rewards: rewards.clone(),
            terminated: Tensor::from_vec(
                dones.iter().map(|&done| u8::from(done)).collect::<Vec<_>>(),
                environment_count,
                &storage_device,
            )?
            .to_dtype(self.dtype)?,
            truncateds: truncateds.to_vec(),
        })?;
        Ok(completed_episodes)
    }

    pub(crate) fn learn<I>(
        &mut self,
        env: &mut dyn MultiGym<I, Error = GE, SpaceError = SE>,
        num_timesteps: usize,
        logger: &mut dyn DeterministicActorCriticLogger<I>,
    ) -> Result<(), DeterministicActorCriticError<GE, SE>> {
        let mut elapsed_timesteps = 0;
        let environment_count = env.num_envs();
        self.replay
            .storage_mut()
            .initialize_environment_count(environment_count)?;
        let mut observations = env
            .reset()
            .map_err(DeterministicActorCriticError::GymError)?
            .to_dtype(self.dtype)?;
        let mut episodes = EpisodeTracker::new(environment_count);
        while elapsed_timesteps < num_timesteps {
            let first_timestep = self.schedule_progress.elapsed_steps();
            let actions = if first_timestep < self.training_start {
                self.random_actions(environment_count)?
            } else {
                self.act(&observations)?
            };
            let step = env
                .step(actions.clone())
                .map_err(DeterministicActorCriticError::GymError)?;
            let transition_next_states = step.transition_next_states()?.to_dtype(self.dtype)?;
            let MultiGymStepInfo {
                states: reset_next_states,
                rewards,
                infos,
                dones,
                truncateds,
                ..
            } = step;
            let collection_rewards = rewards.clone();
            self.schedule_progress.advance_steps(environment_count);
            let completed_episodes = self.store_transitions(
                CollectedTransitions {
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
            for offset in 1..=environment_count {
                let collection_timestep = first_timestep + offset;
                if collection_timestep >= self.training_start
                    && collection_timestep.is_multiple_of(self.update_frequency)
                {
                    self.optimize(collection_timestep, logger)?;
                }
            }
            elapsed_timesteps += environment_count;
            observations = reset_next_states.to_dtype(self.dtype)?;
            logger.log_collection(&DeterministicActorCriticCollectionLogEntry {
                collection_rewards,
                infos,
                collection_timestep: self.schedule_progress.elapsed_steps(),
                completed_episodes,
                replay_len: self.replay.len(),
            });
        }
        Ok(())
    }
}

/// Adds elementwise noise to arbitrarily shaped `values`, preserving its
/// shape.
fn add_gaussian_noise(
    values: &Tensor,
    standard_deviation: f64,
    clip: Option<f64>,
) -> candle_core::Result<Tensor> {
    if standard_deviation == 0.0 {
        return Ok(values.clone());
    }
    let mut noise = (Tensor::randn(0.0f32, 1.0, values.shape(), values.device())?
        .to_dtype(values.dtype())?
        * standard_deviation)?;
    if let Some(clip) = clip {
        noise = noise.clamp(-clip, clip)?;
    }
    values + noise
}

#[cfg(test)]
mod tests {
    use super::{
        CriticCountRequirement, DeterministicActorCriticConfigurationError as Error,
        DeterministicActorCriticConfigurationValidator, add_gaussian_noise,
    };
    use candle_core::{Device, Tensor};

    #[test]
    fn validates_shared_and_strategy_configuration() {
        assert_eq!(
            DeterministicActorCriticConfigurationValidator::validate_configuration()
                .critic_count_requirement(CriticCountRequirement::Exact(2))
                .actual_critic_count(1)
                .replay_capacity(10)
                .batch_size(2)
                .gamma(0.99)
                .tau(0.005)
                .exploration_noise(0.1)
                .target_policy_noise(0.2)
                .target_noise_clip(0.5)
                .actor_update_interval(2)
                .update_frequency(1)
                .training_horizon(10)
                .call(),
            Err(Error::IncorrectCriticCount {
                expected: 2,
                actual: 1,
            })
        );
        assert_eq!(
            DeterministicActorCriticConfigurationValidator::validate_configuration()
                .critic_count_requirement(CriticCountRequirement::NonEmpty)
                .actual_critic_count(0)
                .replay_capacity(10)
                .batch_size(2)
                .gamma(0.99)
                .tau(0.005)
                .exploration_noise(0.1)
                .target_policy_noise(0.0)
                .target_noise_clip(0.0)
                .actor_update_interval(1)
                .update_frequency(1)
                .training_horizon(10)
                .call(),
            Err(Error::NoCritics)
        );
        assert_eq!(
            DeterministicActorCriticConfigurationValidator::validate_configuration()
                .critic_count_requirement(CriticCountRequirement::Exact(1))
                .actual_critic_count(1)
                .replay_capacity(10)
                .batch_size(2)
                .gamma(0.99)
                .tau(0.005)
                .exploration_noise(-0.1)
                .target_policy_noise(0.0)
                .target_noise_clip(0.0)
                .actor_update_interval(1)
                .update_frequency(1)
                .training_horizon(10)
                .call(),
            Err(Error::InvalidExplorationNoise)
        );
        assert_eq!(
            DeterministicActorCriticConfigurationValidator::validate_configuration()
                .critic_count_requirement(CriticCountRequirement::NonEmpty)
                .actual_critic_count(3)
                .replay_capacity(10)
                .batch_size(2)
                .gamma(0.99)
                .tau(0.005)
                .exploration_noise(0.1)
                .target_policy_noise(0.0)
                .target_noise_clip(0.0)
                .actor_update_interval(1)
                .update_frequency(1)
                .training_horizon(10)
                .call(),
            Ok(())
        );
    }

    #[test]
    fn zero_noise_preserves_values() {
        let values = Tensor::new(&[[1.0f32, 2.0]], &Device::Cpu).unwrap();
        assert_eq!(
            add_gaussian_noise(&values, 0.0, Some(0.0))
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            vec![vec![1.0, 2.0]]
        );
    }
}

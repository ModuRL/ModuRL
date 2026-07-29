//! Shared implementation for deterministic off-policy actor-critic agents.
//!
//! [`ddpg::DDPGAgent`] and [`td3::TD3Agent`] are thin public wrappers around
//! the training loop in this module. Their algorithm-specific behavior is
//! supplied by an internal strategy.

use std::{fmt::Debug, marker::PhantomData, ops::Deref};

use bon::bon;
use candle_core::{IndexOp, Tensor};
use candle_nn::{Optimizer, VarMap};

use super::{
    ReplayDeviceStrategy,
    sac::{SACCritic, SACCriticError},
};
use crate::{
    buffers::{
        experience,
        experience_replay::{ExperienceReplay, ExperienceReplayError},
    },
    gym::{VectorizedGym, VectorizedStepInfo},
    objectives::bellman_targets,
    parameter_schedule::ScheduleProgress,
    spaces::{BoxSpace, Space},
    tensor_operations::tensor_has_nan,
};

pub mod ddpg;
pub mod td3;

/// A scalar state-action critic suitable for DDPG and TD3.
pub type DeterministicCritic<'a, O> = SACCritic<'a, O>;

/// Invalid deterministic actor-critic configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeterministicActorCriticConfigurationError {
    NoCritics,
    ZeroReplayCapacity,
    ZeroBatchSize,
    ReplayCapacityBelowBatchSize,
    ZeroUpdateFrequency,
    ZeroTrainingHorizon,
    InvalidGamma,
    InvalidTau,
    InvalidExplorationNoise,
    InvalidTargetPolicyNoise,
    InvalidTargetNoiseClip,
    ZeroActorUpdateInterval,
    IncorrectCriticCount { expected: usize, actual: usize },
}

/// DDPG and TD3 construction or training failure.
#[derive(Debug)]
pub enum DeterministicActorCriticError<GE, SE>
where
    GE: Debug,
    SE: Debug,
{
    TensorError(candle_core::Error),
    CriticError(SACCriticError),
    ConfigurationError(DeterministicActorCriticConfigurationError),
    ActorParameterMapMismatch {
        online_only: Vec<String>,
        target_only: Vec<String>,
    },
    GymError(GE),
    SpaceError(SE),
}

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
    /// `None` on updates where TD3 delays the actor.
    pub actor_loss: Option<Tensor>,
    /// Q values from the first critic for current policy actions, shaped
    /// `[batch_size]`. `None` on updates where TD3 delays the actor.
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
    pub actor_updated: bool,
    pub update_index: usize,
    pub collection_timestep: usize,
}

/// Metrics from one vectorized environment step.
pub struct DeterministicActorCriticCollectionLogEntry<I = ()> {
    pub collection_rewards: Tensor,
    pub infos: Vec<I>,
    pub collection_timestep: usize,
    pub completed_episodes: Vec<DeterministicActorCriticEpisodeLogEntry>,
    pub replay_len: usize,
}

/// Summary of an episode completed during collection.
pub struct DeterministicActorCriticEpisodeLogEntry {
    pub environment_index: usize,
    pub episode_return: f32,
    pub episode_length: usize,
    pub terminated: bool,
    pub truncated: bool,
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

#[derive(Clone)]
struct DeterministicExperience {
    state: Tensor,
    next_state: Tensor,
    action: Tensor,
    reward: f32,
    terminated: f32,
}

struct DeterministicBatch {
    states: Tensor,
    next_states: Tensor,
    actions: Tensor,
    rewards: Tensor,
    terminated: Tensor,
}

impl experience::Experience for DeterministicExperience {
    type Batch = DeterministicBatch;
    type Error = candle_core::Error;

    fn batch(experiences: &[Self]) -> Result<Self::Batch, Self::Error> {
        let device = experiences
            .first()
            .expect("cannot batch an empty deterministic replay sample")
            .state
            .device();
        Ok(DeterministicBatch {
            states: experience::stack_tensor_field(experiences, |item| item.state.clone())?,
            next_states: experience::stack_tensor_field(experiences, |item| {
                item.next_state.clone()
            })?,
            actions: experience::stack_tensor_field(experiences, |item| item.action.clone())?,
            rewards: Tensor::new(
                experiences
                    .iter()
                    .map(|item| item.reward)
                    .collect::<Vec<_>>(),
                device,
            )?,
            terminated: Tensor::new(
                experiences
                    .iter()
                    .map(|item| item.terminated)
                    .collect::<Vec<_>>(),
                device,
            )?,
        })
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
    replay: ExperienceReplay<DeterministicExperience>,
    gamma: f64,
    tau: f64,
    exploration_noise: f64,
    training_start: usize,
    update_frequency: usize,
    schedule_progress: ScheduleProgress,
    device_strategy: ReplayDeviceStrategy,
    optimization_steps: usize,
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
    ) -> DeterministicActorCriticResult<Self, GE, SE> {
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
            replay: ExperienceReplay::new(
                replay_capacity,
                batch_size,
                device_strategy.storage_device(),
            ),
            gamma,
            tau,
            exploration_noise,
            training_start,
            update_frequency,
            schedule_progress: ScheduleProgress::new(training_horizon),
            device_strategy,
            optimization_steps: 0,
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
        let actions = self.online_actor.forward(observation)?;
        Ok(self.action_space.tensor_from_neurons(&actions)?)
    }

    /// Maps observations `[batch, ...observation_shape]` to exploration-noised
    /// actions `[batch, ...action_shape]`.
    pub(crate) fn act(
        &self,
        observation: &Tensor,
    ) -> Result<Tensor, DeterministicActorCriticError<GE, SE>> {
        let actions = self.online_actor.forward(observation)?;
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
        Ok(Tensor::stack(&actions, 0)?)
    }

    fn sample_batch(
        &self,
    ) -> Result<Option<DeterministicBatch>, DeterministicActorCriticError<GE, SE>> {
        if self.replay.len() < self.replay.get_batch_size() {
            return Ok(None);
        }
        let mut batch = match self.replay.sample() {
            Ok(batch) => batch,
            Err(ExperienceReplayError::TensorError(error))
            | Err(ExperienceReplayError::ExperienceError(error)) => return Err(error.into()),
        };
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
            if !tensor_has_nan(&loss)? {
                critic.optimizer_mut().backward_step(&loss)?;
            }
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
        if !tensor_has_nan(&loss)? {
            self.actor_optimizer.backward_step(&loss)?;
        }
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
        let state_rows = states.chunk(environment_count, 0)?;
        let next_state_rows = next_states.chunk(environment_count, 0)?;
        let action_rows = actions.chunk(environment_count, 0)?;
        let reward_rows = rewards.chunk(environment_count, 0)?;
        let storage_device = self.device_strategy.storage_device();
        let mut completed_episodes = Vec::new();
        for environment_index in 0..environment_count {
            let reward = reward_rows[environment_index].i(0)?.to_scalar::<f32>()?;
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
            self.replay.add(DeterministicExperience {
                state: state_rows[environment_index]
                    .squeeze(0)?
                    .detach()
                    .to_device(&storage_device)?,
                next_state: next_state_rows[environment_index]
                    .squeeze(0)?
                    .detach()
                    .to_device(&storage_device)?,
                action: action_rows[environment_index]
                    .squeeze(0)?
                    .detach()
                    .to_device(&storage_device)?,
                reward,
                terminated: f32::from(dones[environment_index]),
            });
        }
        Ok(completed_episodes)
    }

    pub(crate) fn learn<I>(
        &mut self,
        env: &mut dyn VectorizedGym<I, Error = GE, SpaceError = SE>,
        num_timesteps: usize,
        logger: &mut dyn DeterministicActorCriticLogger<I>,
    ) -> Result<(), DeterministicActorCriticError<GE, SE>> {
        let mut elapsed_timesteps = 0;
        let mut observations = env
            .reset()
            .map_err(DeterministicActorCriticError::GymError)?;
        let environment_count = env.num_envs();
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
            let transition_next_states = step.transition_next_states()?;
            let VectorizedStepInfo {
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
            observations = reset_next_states;
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
    let mut noise =
        (Tensor::randn(0.0f32, 1.0, values.shape(), values.device())? * standard_deviation)?;
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

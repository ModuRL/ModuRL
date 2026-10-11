use bon::bon;
use burn::{module::Module, optim::ModuleOptimizer, tensor::Tensor};
use candle_core::{D, DType, IndexOp};
use candle_nn::{Optimizer, loss};
use std::marker::PhantomData;

use crate::{
    buffers::{
        experience::{self, ExperienceBatchError},
        rollout_buffer::RolloutBuffer,
    },
    gym::{MultiGym, MultiGymStepInfo},
    models::probabilistic_model::ProbabilisticPolicy,
    objectives::clipped_value_loss,
    parameter_schedule::{ConstantSchedule, ParameterSchedule, ScheduleProgress},
    sampling::shuffle_with_device_rng,
    spaces,
    tensor_operations::normalize_tensor,
    tensor_rank::NextRank,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PPOConfigurationError {
    #[error("batch size must be nonzero")]
    ZeroBatchSize,
    #[error("minibatch size must be nonzero")]
    ZeroMiniBatchSize,
    #[error("the number of optimization epochs must be nonzero")]
    ZeroEpochs,
    #[error("training horizon must be nonzero")]
    ZeroTrainingHorizon,
    #[error("gamma must be finite and in 0..=1")]
    InvalidGamma,
    #[error("GAE lambda must be finite and in 0..=1")]
    InvalidGaeLambda,
    #[error("value coefficient must be finite and nonnegative")]
    InvalidValueCoefficient,
    #[error("entropy coefficient must be finite and nonnegative")]
    InvalidEntropyCoefficient,
    #[error("gradient clip must be finite and positive")]
    InvalidGradientClip,
    #[error("learning rate must be finite and positive")]
    InvalidLearningRate,
    #[error("PPO requires a floating-point dtype")]
    InvalidDType,
    #[error("the environment count must be nonzero")]
    ZeroEnvironments,
    #[error("batch size must be divisible by the environment count")]
    BatchNotDivisibleByEnvironmentCount,
}

#[derive(Debug, thiserror::Error)]
pub enum PPOError<AE, GE, SE>
where
    AE: std::fmt::Debug,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    #[error("policy failed: {0}")]
    PolicyError(#[source] AE),
    #[error("gym failed: {0}")]
    GymError(#[source] GE),
    #[error("PPO tensor operation failed: {0}")]
    TensorError(#[source] candle_core::Error),
    #[error("space operation failed: {0}")]
    SpaceError(#[source] SE),
    #[error("invalid PPO configuration: {0}")]
    ConfigurationError(#[source] PPOConfigurationError),
    #[error("termination batches differ in length: {dones} dones and {truncateds} truncations")]
    MismatchedTerminationBatch { dones: usize, truncateds: usize },
    #[error("no prepared rollout is available")]
    MissingPreparedRollout,
    #[error("old value estimates are missing")]
    MissingOldValues,
}

impl<AE, GE, SE> From<PPOConfigurationError> for PPOError<AE, GE, SE>
where
    AE: std::fmt::Debug,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    fn from(error: PPOConfigurationError) -> Self {
        Self::ConfigurationError(error)
    }
}

impl<AE, GE, SE> From<candle_core::Error> for PPOError<AE, GE, SE>
where
    AE: std::fmt::Debug,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    fn from(err: candle_core::Error) -> Self {
        PPOError::TensorError(err)
    }
}

use super::Agent;

#[derive(Debug, Clone)]
struct PPOPreparedExperience {
    advantages: Tensor<2>,
    returns: Tensor<2>,
    /// V(s) under the rollout-collection network, for PPO2-style value-loss clipping.
    old_values: Option<Tensor<2>>,
}

#[derive(Debug, Clone)]
struct PPOExperience<O: NextRank = Tensor<2>, A: NextRank = Tensor<2>> {
    observations: O,
    next_observations: O,
    actions: A,
    rewards: Tensor<2>,
    training_rewards: Tensor<2>,
    next_dones: Vec<bool>,
    truncateds: Vec<bool>,
    log_probs: Tensor<2>,
    prepared: Option<PPOPreparedExperience>,
}

struct PPOCollectionAction {
    policy_actions: candle_core::Tensor,
    environment_actions: candle_core::Tensor,
    log_probs: candle_core::Tensor,
}

struct PPOPreparedRolloutBatch {
    advantages: Tensor<3>,
    returns: Tensor<3>,
    old_values: Option<Tensor<3>>,
}

struct PPORolloutBatch<O: NextRank, A: NextRank> {
    observations: O::Next,
    next_observations: O::Next,
    actions: A::Next,
    rewards: Tensor<3>,
    training_rewards: Tensor<3>,
    next_dones: Tensor<3, burn::tensor::Bool>,
    truncateds: Tensor<3, burn::tensor::Bool>,
    log_probs: Tensor<3>,
    prepared: Option<PPOPreparedRolloutBatch>,
}

impl PPOPreparedRolloutBatch {
    /// Stacks prepared Float statistics `[environment_count, 1]` into `[time, environment_count, 1]`.
    /// Each field must share shape, dtype, and device across time. Retains gradient connections.
    /// Returns `None` if no transition has prepared statistics; mixed preparation violates an internal invariant.
    fn batch<O: NextRank, A: NextRank>(
        experiences: &[PPOExperience<O, A>],
    ) -> Result<Option<Self>, ExperienceBatchError> {
        let prepared = experiences
            .iter()
            .map(|experience| experience.prepared.as_ref())
            .collect::<Option<Vec<_>>>();
        let Some(prepared) = prepared else {
            assert!(
                experiences
                    .iter()
                    .all(|experience| experience.prepared.is_none()),
                "PPO rollout preparation must be all-or-nothing"
            );
            return Ok(None);
        };

        let old_values = prepared
            .iter()
            .map(|prepared| prepared.old_values.clone())
            .collect::<Option<Vec<_>>>();

        Ok(Some(Self {
            advantages: experience::stack_tensor_field(&prepared, |item| item.advantages.clone())?,
            returns: experience::stack_tensor_field(&prepared, |item| item.returns.clone())?,
            old_values: old_values
                .map(|old_values| experience::stack_tensor_field(&old_values, Clone::clone))
                .transpose()?,
        }))
    }
}

#[bon]
impl<const O: usize, const A: usize> PPOExperience<Tensor<O>, Tensor<A>>
where
    Tensor<O>: NextRank,
    Tensor<A>: NextRank,
{
    /// Creates one vectorized rollout transition.
    ///
    /// Rank-O `observations` and `next_observations` are `[env_count, ...observation_shape]`.
    /// Rank-A `actions` is `[env_count, ...action_shape]`; `rewards`,
    /// `training_rewards`, and `log_probs` are `[env_count, 1]`. `rewards`
    /// preserves the environment values for logging, while `training_rewards`
    /// may contain bootstrap corrections for nonterminal truncations.
    /// Fields preserve their Float dtype, device, and gradient connections; collection detaches stored experience.
    /// PPO training data is added internally after rollout collection.
    #[builder]
    pub fn new(
        observations: Tensor<O>,
        next_observations: Tensor<O>,
        actions: Tensor<A>,
        rewards: Tensor<2>,
        training_rewards: Tensor<2>,
        next_dones: Vec<bool>,
        truncateds: Vec<bool>,
        log_probs: Tensor<2>,
    ) -> Self {
        Self {
            observations,
            next_observations,
            actions,
            rewards,
            training_rewards,
            next_dones,
            truncateds,
            log_probs,
            prepared: None,
        }
    }
}

impl<const O: usize, const A: usize, const BO: usize, const BA: usize> experience::Experience
    for PPOExperience<Tensor<O>, Tensor<A>>
where
    Tensor<O>: NextRank<Next = Tensor<BO>>,
    Tensor<A>: NextRank<Next = Tensor<BA>>,
{
    type Batch = PPORolloutBatch<Tensor<O>, Tensor<A>>;
    type Error = ExperienceBatchError;

    /// Adds a leading time axis to rank-O observations and rank-A actions, producing ranks BO = O + 1 and BA = A + 1.
    /// Scalar statistics and Bool flags become `[time, environment_count, 1]` without changing environment order.
    /// Each tensor field must share shape, dtype, and device across time. Preserves gradient connections.
    fn batch(experiences: &[Self]) -> Result<Self::Batch, Self::Error> {
        let first = experiences
            .first()
            .ok_or(ExperienceBatchError::EmptyBatch)?;
        Ok(PPORolloutBatch {
            observations: experience::stack_tensor_field(experiences, |experience| {
                experience.observations.clone()
            })?,
            next_observations: experience::stack_tensor_field(experiences, |experience| {
                experience.next_observations.clone()
            })?,
            actions: experience::stack_tensor_field(experiences, |experience| {
                experience.actions.clone()
            })?,
            rewards: experience::stack_tensor_field(experiences, |experience| {
                experience.rewards.clone()
            })?,
            training_rewards: experience::stack_tensor_field(experiences, |experience| {
                experience.training_rewards.clone()
            })?,
            next_dones: experience::stack_bool_field(
                experiences,
                |experience| &experience.next_dones,
                &first.observations.device(),
            )?,
            truncateds: experience::stack_bool_field(
                experiences,
                |experience| &experience.truncateds,
                &first.observations.device(),
            )?,
            log_probs: experience::stack_tensor_field(experiences, |experience| {
                experience.log_probs.clone()
            })?,
            prepared: PPOPreparedRolloutBatch::batch(experiences)?,
        })
    }
}

struct PPOLoggingInfo<'a, I> {
    logger: &'a mut dyn PPOLogger<I>,
    epoch: usize,
    timestep: usize,
}

impl<'a, I> PPOLoggingInfo<'a, I> {
    fn new(logger: &'a mut dyn PPOLogger<I>) -> Self {
        Self {
            logger,
            epoch: 0,
            timestep: 0,
        }
    }
}

pub struct PPOLogEntry {
    pub actor_loss: candle_core::Tensor,
    pub critic_loss: candle_core::Tensor,
    pub entropy: candle_core::Tensor,
    pub kl_divergence: candle_core::Tensor,
    pub explained_variance: candle_core::Tensor,
    pub rewards: candle_core::Tensor,
    pub epoch: usize,
    pub timestep: usize,
    pub returns: candle_core::Tensor,
    pub advantages: candle_core::Tensor,
}

pub struct PPOCollectionLogEntry<I = ()> {
    pub collection_rewards: candle_core::Tensor,
    pub infos: Vec<I>,
    pub collection_timestep: usize,
    pub completed_episodes: Vec<PPOEpisodeLogEntry>,
}

pub struct PPOEpisodeLogEntry {
    pub environment_index: usize,
    pub episode_return: f32,
    pub episode_length: usize,
    pub terminated: bool,
    pub truncated: bool,
    pub collection_timestep: usize,
}

struct PPOEpisodeTracker {
    returns: Vec<f32>,
    lengths: Vec<usize>,
}

impl PPOEpisodeTracker {
    fn new(environment_count: usize) -> Self {
        Self {
            returns: vec![0.0; environment_count],
            lengths: vec![0; environment_count],
        }
    }

    fn environment_count(&self) -> usize {
        self.returns.len()
    }

    fn clear(&mut self) {
        self.returns.clear();
        self.lengths.clear();
    }

    fn record(
        &mut self,
        environment_index: usize,
        reward: f32,
        terminated: bool,
        truncated: bool,
        collection_timestep: usize,
    ) -> Option<PPOEpisodeLogEntry> {
        self.returns[environment_index] += reward;
        self.lengths[environment_index] += 1;
        if !terminated && !truncated {
            return None;
        }

        let entry = PPOEpisodeLogEntry {
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

pub trait PPOLogger<I = ()> {
    fn log(&mut self, info: &PPOLogEntry);

    fn log_collection(&mut self, _info: &PPOCollectionLogEntry<I>) {}
}

#[derive(Clone)]
struct PPOLosses {
    actor_loss: candle_core::Tensor,
    critic_loss: candle_core::Tensor,
}

/// Averages every axis of rank-D Float `values` and returns population variance `[1]`.
/// Preserves the floating-point dtype, device, and gradient connections.
fn population_variance<const D: usize>(values: &Tensor<D>) -> Tensor<1> {
    let mean = values.clone().mean().unsqueeze::<D>();
    (values.clone() - mean).square().mean()
}

/// Returns explained variance `[1]` for matching rank-D Float statistics.
/// Inputs use `[sample_count, 1]` or `[time, env_count, 1]`, with matching dtype and device.
/// Averages every axis and retains gradient connections. Constant returns produce NaN.
fn compute_explained_variance<const D: usize>(
    returns: &Tensor<D>,
    rollout_values: &Tensor<D>,
) -> Tensor<1> {
    let return_variance = population_variance(returns);
    let residuals = returns.clone() - rollout_values.clone();
    let residual_variance = population_variance(&residuals);
    let explained_variance = 1.0f64 - residual_variance / return_variance.clone();
    explained_variance.mask_fill(return_variance.equal_elem(0.0), f64::NAN)
}

/// Selects an owned shared or separate actor-critic configuration.
/// `N` defaults to `S`, so `PPONetworkInfo::<_>::Shared(network)` fixes the unused variant's type.
pub enum PPONetworkInfo<S, N = S> {
    Shared(S),
    Separate(N),
}

/// Owns a shared model whose output feeds both actor and critic heads.
/// One optimizer updates all three models using one learning rate and schedule.
pub struct SharedPPONetwork<T: Module, A: Module, C: Module> {
    optimizer: ModuleOptimizer,
    learning_rate: f64,
    shared_network: T,
    critic_head: C,
    actor_head: A,
    lr_scheduler: Option<Box<dyn ParameterSchedule>>,
}

#[bon]
impl<T: Module, A: Module, C: Module> SharedPPONetwork<T, A, C> {
    /// Owns the models and optimizer, and enables training on the models.
    /// Models must share PPO's compute dtype and device. Rejects a nonpositive or non-finite learning rate.
    #[builder]
    pub fn new(
        #[builder(into)] optimizer: ModuleOptimizer,
        learning_rate: f64,
        shared_network: T,
        critic_head: C,
        actor_head: A,
        #[builder(with = |schedule: impl ParameterSchedule + 'static| Box::new(schedule))]
        lr_scheduler: Option<Box<dyn ParameterSchedule>>,
    ) -> Result<Self, PPOConfigurationError> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(PPOConfigurationError::InvalidLearningRate);
        }
        Ok(Self {
            optimizer,
            learning_rate,
            shared_network: shared_network.train(),
            critic_head: critic_head.train(),
            actor_head: actor_head.train(),
            lr_scheduler,
        })
    }
}

/// Owns separate actor and critic models with independently configurable optimizers, rates, and schedules.
/// `combined_loss` selects a shared gradient-clipping norm instead of independent actor and critic norms.
pub struct SeparatePPONetwork<A: Module, C: Module> {
    actor_optimizer: ModuleOptimizer,
    critic_optimizer: ModuleOptimizer,
    actor_learning_rate: f64,
    critic_learning_rate: f64,
    actor_network: A,
    critic_network: C,
    actor_lr_scheduler: Option<Box<dyn ParameterSchedule>>,
    critic_lr_scheduler: Option<Box<dyn ParameterSchedule>>,
    combined_loss: bool,
}

#[bon]
impl<A: Module, C: Module> SeparatePPONetwork<A, C> {
    /// Owns the models and optimizers, and enables training on both models.
    /// Models must share PPO's compute dtype and device. Rejects nonpositive or non-finite learning rates.
    #[builder]
    pub fn new(
        #[builder(into)] actor_optimizer: ModuleOptimizer,
        #[builder(into)] critic_optimizer: ModuleOptimizer,
        actor_learning_rate: f64,
        critic_learning_rate: f64,
        actor_network: A,
        critic_network: C,
        #[builder(with = |schedule: impl ParameterSchedule + 'static| Box::new(schedule))]
        actor_lr_scheduler: Option<Box<dyn ParameterSchedule>>,
        #[builder(with = |schedule: impl ParameterSchedule + 'static| Box::new(schedule))]
        critic_lr_scheduler: Option<Box<dyn ParameterSchedule>>,
        #[builder(default = false)] combined_loss: bool,
    ) -> Result<Self, PPOConfigurationError> {
        if !actor_learning_rate.is_finite()
            || actor_learning_rate <= 0.0
            || !critic_learning_rate.is_finite()
            || critic_learning_rate <= 0.0
        {
            return Err(PPOConfigurationError::InvalidLearningRate);
        }
        Ok(Self {
            actor_optimizer,
            critic_optimizer,
            actor_learning_rate,
            critic_learning_rate,
            actor_network: actor_network.train(),
            critic_network: critic_network.train(),
            actor_lr_scheduler,
            critic_lr_scheduler,
            combined_loss,
        })
    }
}

pub struct PPOAgent<'a, O1, O2, AE, GE, SE, I = ()>
where
    O1: Optimizer,
    O2: Optimizer,
    AE: std::fmt::Debug,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    clipped: bool,
    clip_value_loss: bool,
    gamma: f32,
    gae_lambda: f32,
    clip_range: Box<dyn ParameterSchedule>,
    normalize_advantage: bool,
    normalize_returns: bool,
    vf_coef: f32,
    ent_coef: f32,
    action_space: Box<dyn spaces::Space<Error = SE>>,
    network_info: PPONetworkInfo<O1, AE, O2>,
    rollout_buffer: RolloutBuffer<PPOExperience>,
    mini_batch_size: usize,
    num_epochs: usize,
    batch_size: usize,
    schedule_progress: ScheduleProgress,
    logging_info: Option<PPOLoggingInfo<'a, I>>,
    gradient_clip: f32,
    current_states: Option<candle_core::Tensor>,
    episode_tracker: PPOEpisodeTracker,
    dtype: DType,
    _phantom: PhantomData<GE>,
}

#[bon]
impl<'a, O1, O2, AE, GE, SE, I> PPOAgent<'a, O1, O2, AE, GE, SE, I>
where
    O1: Optimizer,
    O2: Optimizer,
    AE: std::fmt::Debug,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    #[builder]
    pub fn new(
        action_space: Box<dyn spaces::Space<Error = SE>>,
        network_info: PPONetworkInfo<O1, AE, O2>,
        #[builder(default = true)] clipped: bool,
        // PPO2/CleanRL-style value-loss clipping. Disabled by default because
        // its usefulness is task- and reward-scale-dependent.
        // research has shown it's often harmful
        #[builder(default = false)] clip_value_loss: bool,
        #[builder(default = 0.99)] gamma: f32,
        #[builder(default = 0.95)] gae_lambda: f32,
        #[builder(
            default = Box::new(ConstantSchedule::new(0.1)),
            with = |schedule: impl ParameterSchedule + 'static| Box::new(schedule)
        )]
        clip_range: Box<dyn ParameterSchedule>,
        #[builder(default = true)] normalize_advantage: bool,
        #[builder(default = false)] normalize_returns: bool,
        #[builder(default = 0.5)] vf_coef: f32,
        #[builder(default = 0.01)] ent_coef: f32,
        #[builder(default = 1024)] batch_size: usize,
        #[builder(default = 128)] mini_batch_size: usize,
        #[builder(default = 4)] num_epochs: usize,
        #[builder(default = 0.5)] gradient_clip: f32,
        training_horizon: usize,
        logging_info: Option<&'a mut dyn PPOLogger<I>>,
        device: candle_core::Device,
        #[builder(default = DType::F32)] dtype: DType,
    ) -> Result<Self, PPOConfigurationError> {
        if batch_size == 0 {
            return Err(PPOConfigurationError::ZeroBatchSize);
        }
        if mini_batch_size == 0 {
            return Err(PPOConfigurationError::ZeroMiniBatchSize);
        }
        if num_epochs == 0 {
            return Err(PPOConfigurationError::ZeroEpochs);
        }
        if training_horizon == 0 {
            return Err(PPOConfigurationError::ZeroTrainingHorizon);
        }
        if !gamma.is_finite() || !(0.0..=1.0).contains(&gamma) {
            return Err(PPOConfigurationError::InvalidGamma);
        }
        if !gae_lambda.is_finite() || !(0.0..=1.0).contains(&gae_lambda) {
            return Err(PPOConfigurationError::InvalidGaeLambda);
        }
        if !vf_coef.is_finite() || vf_coef < 0.0 {
            return Err(PPOConfigurationError::InvalidValueCoefficient);
        }
        if !ent_coef.is_finite() || ent_coef < 0.0 {
            return Err(PPOConfigurationError::InvalidEntropyCoefficient);
        }
        if !gradient_clip.is_finite() || gradient_clip <= 0.0 {
            return Err(PPOConfigurationError::InvalidGradientClip);
        }
        if !dtype.is_float() {
            return Err(PPOConfigurationError::InvalidDType);
        }

        Ok(Self {
            clipped,
            clip_value_loss,
            gamma,
            gae_lambda,
            clip_range,
            normalize_advantage,
            normalize_returns,
            vf_coef,
            ent_coef,
            network_info,
            action_space,
            rollout_buffer: RolloutBuffer::new(0, device), // placeholder, will set when we know num envs
            num_epochs,
            batch_size,
            schedule_progress: ScheduleProgress::new(training_horizon),
            mini_batch_size,
            logging_info: logging_info.map(PPOLoggingInfo::new),
            gradient_clip,
            current_states: None,
            episode_tracker: PPOEpisodeTracker::new(0),
            dtype,
            _phantom: PhantomData,
        })
    }
}

#[bon]
impl<'a, O1, O2, AE, GE, SE, I> PPOAgent<'a, O1, O2, AE, GE, SE, I>
where
    O1: Optimizer,
    O2: Optimizer,
    AE: std::fmt::Debug,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(name = "ppo.optimize", target = "modurl::performance", skip_all)
    )]
    fn optimize(&mut self) -> Result<(), PPOError<AE, GE, SE>> {
        let rollout_explained_variance = self.add_advantages_and_returns()?;

        let batch =
            <PPOExperience as experience::Experience>::batch(self.rollout_buffer.get_raw())?;
        let all_states = batch.observations.flatten(0, 1)?;
        let all_actions = batch.actions.flatten(0, 1)?;
        let all_log_probs = batch.log_probs.flatten(0, 1)?;
        let prepared = batch.prepared.ok_or(PPOError::MissingPreparedRollout)?;
        let all_advantages = prepared.advantages.flatten(0, 1)?;
        let all_returns = prepared.returns.flatten(0, 1)?;
        let all_rewards = batch.rewards.flatten(0, 1)?;
        let all_old_values = prepared
            .old_values
            .map(|old_values| old_values.flatten(0, 1))
            .transpose()?;

        let total_samples = all_states.dims()[0];
        let device = all_states.device();
        let clip_range = self.schedule_progress.parameter(&*self.clip_range) as f32;

        for epoch in 0..self.num_epochs {
            if let Some(logging_info) = &mut self.logging_info {
                logging_info.epoch = epoch;
            }

            let mut indices: Vec<u32> = (0..total_samples as u32).collect();
            shuffle_with_device_rng(&mut indices, device)?;
            let indices_tensor = candle_core::Tensor::from_vec(indices, &[total_samples], device)?;

            for start in (0..total_samples).step_by(self.mini_batch_size) {
                let end = (start + self.mini_batch_size).min(total_samples);
                let batch_size = end - start;

                let batch_indices = indices_tensor.narrow(0, start, batch_size)?;

                let batch_states = all_states.index_select(&batch_indices, 0)?;
                let batch_actions = all_actions.index_select(&batch_indices, 0)?;
                let batch_log_probs = all_log_probs.index_select(&batch_indices, 0)?;
                let batch_advantages = all_advantages.index_select(&batch_indices, 0)?;
                let batch_returns = all_returns.index_select(&batch_indices, 0)?;
                let batch_rewards = all_rewards.index_select(&batch_indices, 0)?;
                let batch_old_values = all_old_values
                    .as_ref()
                    .map(|old_values| old_values.index_select(&batch_indices, 0))
                    .transpose()?;

                let ppo_losses = self
                    .compute_loss()
                    .states(&batch_states)
                    .actions(&batch_actions)
                    .old_log_probs(batch_log_probs.detach())
                    .advantages(batch_advantages)
                    .returns(batch_returns)
                    .rewards(batch_rewards)
                    .maybe_old_values(batch_old_values.map(|old_values| old_values.detach()))
                    .explained_variance(rollout_explained_variance.clone())
                    .clip_range(clip_range)
                    .call()?;

                self.backpropagate_loss(ppo_losses.clone())?;
            }
        }

        self.rollout_buffer.clear();
        Ok(())
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            name = "ppo.prepare_rollout",
            target = "modurl::performance",
            level = "trace",
            skip_all
        )
    )]
    fn add_advantages_and_returns(&mut self) -> Result<candle_core::Tensor, PPOError<AE, GE, SE>> {
        let batch =
            <PPOExperience as experience::Experience>::batch(self.rollout_buffer.get_raw())?;
        let states = batch.observations;
        let (batch_size, env_count) = (states.dims()[0], states.dims()[1]);
        let states = states.flatten(0, 1)?;

        let latent_states = match self.network_info {
            PPONetworkInfo::Shared(ref mut shared_info) => {
                shared_info.shared_network.forward(&states)?
            }
            PPONetworkInfo::Separate(ref mut _separate_info) => states,
        };

        let values_tensor = self.critic_network_forward(&latent_states)?.detach();

        // Unflatten back to [batch_size, env_count, ...].
        let values_tensor = values_tensor.reshape((batch_size, env_count, 1))?;

        let next_states_tensor = batch.next_observations;
        let bootstrapped_states = next_states_tensor.i(next_states_tensor.shape().dims()[0] - 1)?; // shape [env_count, ...]
        let latent_bootstrapped_states = match self.network_info {
            PPONetworkInfo::Shared(ref mut shared_info) => {
                shared_info.shared_network.forward(&bootstrapped_states)?
            }
            PPONetworkInfo::Separate(ref mut _separate_info) => bootstrapped_states,
        };

        let bootstrapped_values = self
            .critic_network_forward(&latent_bootstrapped_states)?
            .detach(); // shape [env_count, 1]

        let advantages = self
            .compute_gae(
                &batch.training_rewards,
                &values_tensor,
                &batch.next_dones,
                &batch.truncateds,
                &bootstrapped_values,
            )?
            .detach();

        let returns = (&values_tensor + &advantages)?;
        let rollout_explained_variance = compute_explained_variance(&returns, &values_tensor)?;

        let experiences = self.rollout_buffer.get_raw_mut();

        for (i, experience) in experiences.iter_mut().enumerate() {
            // The detaches here should be redundant but just to be safe
            experience.prepared = Some(PPOPreparedExperience {
                advantages: advantages.i(i)?.clone().detach(),
                returns: returns.i(i)?.clone().detach(),
                // values_tensor comes from the pre-update network, i.e. the
                // collection-time values PPO2 clips the new values against.
                old_values: if self.clip_value_loss {
                    Some(values_tensor.i(i)?.clone().detach())
                } else {
                    None
                },
            });
        }

        Ok(rollout_explained_variance)
    }

    /// Computes advantages `[time, env_count, 1]` from rewards, values, done
    /// masks, and truncation masks with that same shape; `bootstrapped_values`
    /// is `[env_count, 1]`.
    fn compute_gae(
        &self,
        rewards: &candle_core::Tensor,
        values: &candle_core::Tensor,
        next_dones: &candle_core::Tensor,
        next_truncateds: &candle_core::Tensor,
        bootstrapped_values: &candle_core::Tensor,
    ) -> Result<candle_core::Tensor, candle_core::Error> {
        let device = values.device();
        let rewards = rewards.to_dtype(self.dtype)?;
        let next_dones = next_dones.to_dtype(self.dtype)?;
        let next_truncateds = next_truncateds.to_dtype(self.dtype)?;
        let bootstrapped_values = bootstrapped_values.to_dtype(self.dtype)?;
        let values = values.detach();
        let masks = ((1.0 - next_dones)? * (1.0 - next_truncateds)?)?;
        let mut next_value = bootstrapped_values.detach();
        let mut gae = candle_core::Tensor::zeros(next_value.shape(), self.dtype, device)?;
        let mut advantages = Vec::with_capacity(rewards.dim(0)?);
        // Environments are independent; only time has a recurrence.
        for time in (0..rewards.dim(0)?).rev() {
            let mask = masks.i(time)?;
            let value = values.i(time)?;
            let delta =
                (rewards.i(time)? + ((&next_value * &mask)? * self.gamma as f64)? - &value)?;
            gae = (delta + ((&gae * &mask)? * (self.gamma as f64 * self.gae_lambda as f64))?)?;
            advantages.push(gae.clone());
            next_value = value;
        }
        advantages.reverse();
        candle_core::Tensor::stack(&advantages, 0)
    }

    /// Forwards latent or raw `states` shaped `[batch, ...state_shape]` and
    /// returns critic values shaped `[batch, 1]`.
    fn critic_network_forward(
        &mut self,
        states: &candle_core::Tensor,
    ) -> Result<candle_core::Tensor, PPOError<AE, GE, SE>> {
        match self.network_info {
            PPONetworkInfo::Shared(ref mut shared_info) => {
                let values = shared_info.critic_head.forward(states)?;
                Ok(values)
            }
            PPONetworkInfo::Separate(ref mut separate_info) => {
                let values = separate_info.critic_network.forward(states)?;
                Ok(values)
            }
        }
    }

    /// Adds the bootstrap value for each truncated transition to its reward.
    /// GAE still stops at the truncation boundary, so the terminal value is
    /// included exactly once.
    /// `rewards` is `[environment_count, 1]`, `transition_next_states` is
    /// `[environment_count, ...state_shape]`, and both boolean slices have
    /// `environment_count` entries. A true termination always suppresses
    /// bootstrapping, even if the transition is also truncated.
    fn bootstrap_truncated_rewards(
        &mut self,
        rewards: &candle_core::Tensor,
        transition_next_states: &candle_core::Tensor,
        dones: &[bool],
        truncateds: &[bool],
    ) -> Result<candle_core::Tensor, PPOError<AE, GE, SE>> {
        if dones.len() != truncateds.len() {
            return Err(PPOError::MismatchedTerminationBatch {
                dones: dones.len(),
                truncateds: truncateds.len(),
            });
        }
        let truncated_indices = truncateds
            .iter()
            .zip(dones)
            .enumerate()
            .filter_map(|(index, (truncated, done))| (*truncated && !*done).then_some(index as u32))
            .collect::<Vec<_>>();
        if truncated_indices.is_empty() {
            return Ok(rewards.to_dtype(self.dtype)?);
        }

        let truncation_count = truncated_indices.len();
        let truncated_indices = candle_core::Tensor::from_vec(
            truncated_indices,
            truncation_count,
            transition_next_states.device(),
        )?;
        let truncated_next_states = transition_next_states.index_select(&truncated_indices, 0)?;

        let latent_states = match self.network_info {
            PPONetworkInfo::Shared(ref mut shared_info) => {
                shared_info.shared_network.forward(&truncated_next_states)?
            }
            PPONetworkInfo::Separate(_) => truncated_next_states,
        };
        let terminal_values = self.critic_network_forward(&latent_states)?.detach();
        let terminal_bootstrap = (terminal_values.to_dtype(self.dtype)? * self.gamma as f64)?;
        Ok(rewards
            .to_dtype(self.dtype)?
            .index_add(&truncated_indices, &terminal_bootstrap, 0)?)
    }

    /// Evaluates `actions` `[batch, ...action_shape]` for latent or raw
    /// `states` `[batch, ...state_shape]`, returning two `[batch, 1]` tensors.
    fn actor_network_log_prob_and_entropy(
        &mut self,
        states: &candle_core::Tensor,
        actions: &candle_core::Tensor,
    ) -> Result<(candle_core::Tensor, candle_core::Tensor), PPOError<AE, GE, SE>> {
        match self.network_info {
            PPONetworkInfo::Shared(ref mut shared_info) => {
                let (log_probs, entropy) = shared_info
                    .actor_head
                    .log_prob_and_entropy(states, actions)
                    .map_err(PPOError::PolicyError)?;
                Ok((log_probs, entropy))
            }
            PPONetworkInfo::Separate(ref mut separate_info) => {
                let (log_probs, entropy) = separate_info
                    .actor_network
                    .log_prob_and_entropy(states, actions)
                    .map_err(PPOError::PolicyError)?;
                Ok((log_probs, entropy))
            }
        }
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            name = "ppo.compute_loss",
            target = "modurl::performance",
            level = "trace",
            skip_all
        )
    )]
    #[builder]
    /// Computes losses from `states` `[batch, ...state_shape]`, `actions`
    /// `[batch, ...action_shape]`, and scalar statistics (`old_log_probs`,
    /// `advantages`, `returns`, and `rewards`) shaped `[batch, 1]`. `old_values`
    /// is present only when value-loss clipping is enabled.
    /// `explained_variance` is scalar `[]`.
    fn compute_loss(
        &mut self,
        states: &candle_core::Tensor,
        actions: &candle_core::Tensor,
        old_log_probs: candle_core::Tensor,
        advantages: candle_core::Tensor,
        returns: candle_core::Tensor,
        rewards: candle_core::Tensor,
        old_values: Option<candle_core::Tensor>,
        explained_variance: candle_core::Tensor,
        clip_range: f32,
    ) -> Result<PPOLosses, PPOError<AE, GE, SE>> {
        let advantages = if self.normalize_advantage {
            normalize_tensor(&advantages)?
        } else {
            advantages
        }
        .detach();

        // if the networks are shared, we need to extract the latent state
        // if it's separate we just say this is the state as is
        let latent_state = match self.network_info {
            PPONetworkInfo::Shared(ref mut shared_info) => {
                shared_info.shared_network.forward(states)?
            }
            PPONetworkInfo::Separate(ref mut _separate_info) => states.clone(),
        };

        let (log_probs, entropy) =
            self.actor_network_log_prob_and_entropy(&latent_state, actions)?;

        let log_ratio = (&log_probs - &old_log_probs)?;
        let ratio = log_ratio.exp()?;
        let approx_kl = ((&ratio - 1.0)? - &log_ratio)?;

        let actor_loss = match self.clipped {
            true => {
                let clipped_ratio = ratio.clamp(1.0 - clip_range, 1.0 + clip_range)?;

                let surrogate1 = (ratio.clone() * advantages.clone())?;
                let surrogate2 = (clipped_ratio.clone() * advantages.clone())?;
                let surrogate = surrogate1.minimum(&surrogate2)?;

                (-1.0 * surrogate.mean_all()?)?
            }
            false => {
                let surrogate = (ratio.clone() * &advantages)?;
                (-1.0 * surrogate.mean_all()?)?
            }
        };

        let values = self.critic_network_forward(&latent_state)?;

        let entropy_loss = entropy.mean_all()?;

        let final_actor_loss =
            (actor_loss.clone() - ((self.ent_coef as f64) * entropy_loss.clone()))?;

        let returns = returns.detach();
        let returns = if self.normalize_returns {
            normalize_tensor(&returns)?
        } else {
            returns
        }
        .detach();

        let critic_loss = if self.clip_value_loss {
            // PPO2/CleanRL value-loss clipping: bound how far the value estimate
            // may move from its rollout-time value within one update.
            let old_values = old_values.ok_or(PPOError::MissingOldValues)?;
            clipped_value_loss(&values, &returns, &old_values, clip_range as f64)?
        } else {
            loss::mse(&values, &returns)?
        };
        // Match SB3 PPO: value_loss = vf_coef * mean_squared_error.
        let final_critic_loss = ((self.vf_coef as f64) * critic_loss)?;

        if let Some(logging_info) = &mut self.logging_info {
            let log_entry = PPOLogEntry {
                actor_loss: final_actor_loss.clone(),
                critic_loss: final_critic_loss.clone(),
                entropy: entropy_loss,
                kl_divergence: approx_kl,
                explained_variance,
                rewards,
                epoch: logging_info.epoch,
                timestep: logging_info.timestep,
                returns,
                advantages,
            };
            logging_info.logger.log(&log_entry);
        }

        Ok(PPOLosses {
            actor_loss: final_actor_loss,
            critic_loss: final_critic_loss,
        })
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            name = "ppo.backpropagate",
            target = "modurl::performance",
            level = "trace",
            skip_all
        )
    )]
    fn backpropagate_loss(&mut self, losses: PPOLosses) -> Result<(), PPOError<AE, GE, SE>> {
        let actor_loss = losses.actor_loss;
        let critic_loss = losses.critic_loss;

        match self.network_info {
            PPONetworkInfo::Shared(ref mut shared_info) => {
                let total_loss = (&actor_loss + &critic_loss)?;
                let total_grad = &mut total_loss.backward()?;
                let _total_grad_norm = crate::tensor_operations::clip_gradients(
                    &total_loss,
                    total_grad,
                    self.gradient_clip,
                )?;
                shared_info.optimizer.step(total_grad)?;
            }
            PPONetworkInfo::Separate(ref mut separate_info) => match separate_info.combined_loss {
                true => {
                    let total_loss = (&actor_loss + &critic_loss)?;
                    let total_grad = &mut total_loss.backward()?;
                    let _total_grad_norm = crate::tensor_operations::clip_gradients(
                        &total_loss,
                        total_grad,
                        self.gradient_clip,
                    )?;
                    separate_info.actor_optimizer.step(total_grad)?;
                    separate_info.critic_optimizer.step(total_grad)?;
                }
                false => {
                    let actor_grad = &mut actor_loss.backward()?;
                    let _actor_grad_norm = crate::tensor_operations::clip_gradients(
                        &actor_loss,
                        actor_grad,
                        self.gradient_clip,
                    )?;
                    separate_info.actor_optimizer.step(actor_grad)?;

                    let critic_grad = &mut critic_loss.backward()?;
                    let _critic_grad_norm = crate::tensor_operations::clip_gradients(
                        &critic_loss,
                        critic_grad,
                        self.gradient_clip,
                    )?;
                    separate_info.critic_optimizer.step(critic_grad)?;
                }
            },
        }

        Ok(())
    }

    /// Samples latent actions `[batch, ...action_shape]` for latent or raw
    /// `states` `[batch, ...state_shape]`.
    fn act_neurons(
        &mut self,
        states: &candle_core::Tensor,
    ) -> Result<candle_core::Tensor, PPOError<AE, GE, SE>> {
        let network = match self.network_info {
            PPONetworkInfo::Shared(ref mut shared_info) => &shared_info.actor_head,
            PPONetworkInfo::Separate(ref mut separate_info) => &separate_info.actor_network,
        };

        network.sample(states).map_err(PPOError::PolicyError)
    }

    /// Samples policy actions for `states` shaped
    /// `[environment_count, ...observation_shape]`.
    fn sample_collection_action(
        &mut self,
        states: &candle_core::Tensor,
    ) -> Result<PPOCollectionAction, PPOError<AE, GE, SE>> {
        let latent_states = match self.network_info {
            PPONetworkInfo::Shared(ref mut shared_info) => {
                shared_info.shared_network.forward(states)?
            }
            PPONetworkInfo::Separate(_) => states.clone(),
        };
        let policy_actions = self.act_neurons(&latent_states)?;
        let environment_actions = self
            .action_space
            .tensor_from_neurons(&policy_actions)
            .map_err(PPOError::SpaceError)?;
        let (log_probs, _) =
            self.actor_network_log_prob_and_entropy(&latent_states, &policy_actions)?;
        Ok(PPOCollectionAction {
            policy_actions,
            environment_actions,
            // Do not retain the collection forward graph in the rollout buffer.
            log_probs: log_probs.detach(),
        })
    }

    fn update_learning_rates(&mut self) {
        match self.network_info {
            PPONetworkInfo::Shared(ref mut shared_info) => {
                if let Some(lr_scheduler) = &shared_info.lr_scheduler {
                    let new_lr = self.schedule_progress.parameter(&**lr_scheduler);
                    shared_info.optimizer.set_learning_rate(new_lr);
                }
            }
            PPONetworkInfo::Separate(ref mut separate_info) => {
                if let Some(actor_lr_scheduler) = &separate_info.actor_lr_scheduler {
                    let new_lr = self.schedule_progress.parameter(&**actor_lr_scheduler);
                    separate_info.actor_optimizer.set_learning_rate(new_lr);
                }
                if let Some(critic_lr_scheduler) = &separate_info.critic_lr_scheduler {
                    let new_lr = self.schedule_progress.parameter(&**critic_lr_scheduler);
                    separate_info.critic_optimizer.set_learning_rate(new_lr);
                }
            }
        }
    }

    /// Logs vectorized `rewards` shaped `[env_count, 1]` alongside one metadata
    /// and termination entry per environment.
    fn log_collection(
        &mut self,
        rewards: &candle_core::Tensor,
        infos: Vec<I>,
        dones: &[bool],
        truncateds: &[bool],
        first_collection_timestep: usize,
    ) -> Result<(), candle_core::Error> {
        let Some(logging_info) = &mut self.logging_info else {
            return Ok(());
        };

        let rewards_vec = rewards.flatten_all()?.to_vec1::<f32>()?;
        let mut completed_episodes = Vec::new();
        for (environment_index, reward) in rewards_vec.iter().copied().enumerate() {
            let collection_timestep =
                first_collection_timestep.saturating_add(environment_index + 1);
            if let Some(entry) = self.episode_tracker.record(
                environment_index,
                reward,
                dones[environment_index],
                truncateds[environment_index],
                collection_timestep,
            ) {
                completed_episodes.push(entry);
            }
        }

        logging_info.logger.log_collection(&PPOCollectionLogEntry {
            collection_rewards: rewards.clone(),
            infos,
            collection_timestep: first_collection_timestep.saturating_add(rewards_vec.len()),
            completed_episodes,
        });
        Ok(())
    }

    pub fn set_learning_rate(&mut self, lr: f64) {
        match self.network_info {
            PPONetworkInfo::Shared(ref mut shared_info) => {
                shared_info.optimizer.set_learning_rate(lr);
            }
            PPONetworkInfo::Separate(ref mut separate_info) => {
                separate_info.actor_optimizer.set_learning_rate(lr);
                separate_info.critic_optimizer.set_learning_rate(lr);
            }
        }
    }

    /// Selects the policy distribution's mode without sampling.
    ///
    /// This is intended for deterministic evaluation. PPO rollouts remain
    /// stochastic and continue to use [`Agent::act`]. `observation` is shaped
    /// `[batch, ...observation_shape]`; the returned environment action is
    /// `[batch, ...action_shape]`.
    pub fn act_deterministic(
        &mut self,
        observation: &candle_core::Tensor,
    ) -> Result<candle_core::Tensor, PPOError<AE, GE, SE>> {
        let observation = observation.to_dtype(self.dtype)?;
        let latent_states = match self.network_info {
            PPONetworkInfo::Shared(ref mut shared_info) => {
                shared_info.shared_network.forward(&observation)?
            }
            PPONetworkInfo::Separate(_) => observation,
        };
        let network = match self.network_info {
            PPONetworkInfo::Shared(ref mut shared_info) => &shared_info.actor_head,
            PPONetworkInfo::Separate(ref mut separate_info) => &separate_info.actor_network,
        };
        let neurons = network
            .mode(&latent_states)
            .map_err(PPOError::PolicyError)?;
        self.action_space
            .tensor_from_neurons(&neurons)
            .map_err(PPOError::SpaceError)
    }

    pub fn reset_current_states(&mut self) {
        self.current_states = None;
        self.episode_tracker.clear();
    }

    /// Prepares an empty rollout buffer and episode tracking, returning the
    /// observations shaped `[environment_count, ...observation_shape]` from
    /// which collection should resume.
    fn prepare_rollout_collection(
        &mut self,
        env: &mut dyn MultiGym<I, Error = GE, SpaceError = SE>,
    ) -> Result<candle_core::Tensor, PPOError<AE, GE, SE>> {
        let states = if let Some(states) = self.current_states.take() {
            states
        } else {
            env.reset().map_err(PPOError::GymError)?
        }
        .to_dtype(self.dtype)?;
        let environment_count = env.num_envs();
        if environment_count == 0 {
            return Err(PPOConfigurationError::ZeroEnvironments.into());
        }
        if !self.batch_size.is_multiple_of(environment_count) {
            return Err(PPOConfigurationError::BatchNotDivisibleByEnvironmentCount.into());
        }
        self.rollout_buffer = RolloutBuffer::new(
            self.mini_batch_size / environment_count,
            states.device().clone(),
        );
        if self.logging_info.is_some()
            && self.episode_tracker.environment_count() != environment_count
        {
            self.episode_tracker = PPOEpisodeTracker::new(environment_count);
        }
        Ok(states)
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            name = "ppo.collect_rollout",
            target = "modurl::performance",
            skip_all
        )
    )]
    /// Fills the rollout buffer, updating `next_states` shaped
    /// `[environment_count, ...observation_shape]` after each environment step.
    fn collect_rollout(
        &mut self,
        env: &mut dyn MultiGym<I, Error = GE, SpaceError = SE>,
        next_states: &mut candle_core::Tensor,
    ) -> Result<usize, PPOError<AE, GE, SE>> {
        let environment_count = env.num_envs();
        while self.rollout_buffer.len() * environment_count < self.batch_size {
            let states = next_states.clone();
            let collection_action = self.sample_collection_action(&states)?;
            let step_info = env
                .step(collection_action.environment_actions.clone())
                .map_err(PPOError::GymError)?;
            let training_next_states = step_info.transition_next_states()?.to_dtype(self.dtype)?;
            let MultiGymStepInfo {
                states: reset_or_next_states,
                rewards: collection_rewards,
                infos,
                dones: next_dones,
                truncateds,
                terminal_states: _,
            } = step_info;
            let first_collection_timestep = self
                .schedule_progress
                .elapsed_steps()
                .saturating_add(self.rollout_buffer.len() * environment_count);
            self.log_collection(
                &collection_rewards,
                infos,
                &next_dones,
                &truncateds,
                first_collection_timestep,
            )?;
            let training_rewards = self.bootstrap_truncated_rewards(
                &collection_rewards,
                &training_next_states,
                &next_dones,
                &truncateds,
            )?;
            self.rollout_buffer.add(
                PPOExperience::builder()
                    .observations(states)
                    .next_observations(training_next_states)
                    .actions(collection_action.policy_actions.detach())
                    .rewards(collection_rewards)
                    .training_rewards(training_rewards)
                    .next_dones(next_dones)
                    .truncateds(truncateds)
                    .log_probs(collection_action.log_probs)
                    .build(),
            );
            *next_states = reset_or_next_states.to_dtype(self.dtype)?;
        }

        Ok(self.rollout_buffer.len() * environment_count)
    }

    fn train_on_collected_rollout(
        &mut self,
        collected_timesteps: usize,
    ) -> Result<(), PPOError<AE, GE, SE>> {
        self.schedule_progress.advance_steps(collected_timesteps);
        self.update_learning_rates();
        if let Some(logging_info) = &mut self.logging_info {
            logging_info.timestep = self.schedule_progress.elapsed_steps();
        }
        self.optimize()
    }
}

impl<'a, O1, O2, AE, GE, SE, I> Agent<I> for PPOAgent<'a, O1, O2, AE, GE, SE, I>
where
    O1: Optimizer,
    O2: Optimizer,
    AE: std::fmt::Debug,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    type Error = PPOError<AE, GE, SE>;
    type GymError = GE;
    type SpaceError = SE;

    /// Selects environment actions `[batch, ...action_shape]` for observations
    /// `[batch, ...observation_shape]`.
    fn act(
        &mut self,
        observation: &candle_core::Tensor,
    ) -> Result<candle_core::Tensor, Self::Error> {
        let observation = observation.to_dtype(self.dtype)?;
        let latent_states = match self.network_info {
            PPONetworkInfo::Shared(ref mut shared_info) => {
                shared_info.shared_network.forward(&observation)?
            }
            PPONetworkInfo::Separate(ref mut _separate_info) => observation,
        };
        let neurons = self.act_neurons(&latent_states)?;
        let actions = self
            .action_space
            .tensor_from_neurons(&neurons)
            .map_err(PPOError::SpaceError)?;
        Ok(actions)
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            name = "ppo.learn",
            target = "modurl::performance",
            skip_all,
            fields(num_timesteps)
        )
    )]
    fn learn(
        &mut self,
        env: &mut dyn MultiGym<I, Error = Self::GymError, SpaceError = Self::SpaceError>,
        num_timesteps: usize,
    ) -> Result<(), PPOError<AE, GE, SE>> {
        let mut elapsed_timesteps = 0;
        let mut next_states = self.prepare_rollout_collection(env)?;

        while elapsed_timesteps < num_timesteps {
            let collected_timesteps = self.collect_rollout(env, &mut next_states)?;
            elapsed_timesteps += collected_timesteps;
            self.train_on_collected_rollout(collected_timesteps)?;
        }
        self.current_states = Some(next_states);
        Ok(())
    }
}

#[cfg(test)]
mod rollout_batch_tests {
    use burn::tensor::{DType, Device, Tensor};

    use super::{PPOExperience, PPOPreparedExperience};
    use crate::buffers::experience::{Experience, ExperienceBatchError};

    /// Creates Float rank-3 observations `[2, 1, 2]`, rank-2 actions `[2, 1]`, and scalar statistics `[2, 1]`.
    /// Uses F64 on the supplied device; observations retain gradients, while training rewards include distinct bootstrap corrections.
    fn transition(value: f64, device: &Device) -> PPOExperience<Tensor<3>, Tensor<2>> {
        let observations = Tensor::from_data(
            [[[value, value + 1.0]], [[value + 2.0, value + 3.0]]],
            (device, DType::F64),
        )
        .require_grad();
        let rewards = Tensor::from_data([[value + 2.0], [value + 3.0]], (device, DType::F64));
        PPOExperience::builder()
            .observations(observations.clone())
            .next_observations(observations + 10.0)
            .actions(Tensor::from_data(
                [[value], [value + 1.0]],
                (device, DType::F64),
            ))
            .rewards(rewards.clone())
            .training_rewards(rewards + 100.0)
            .next_dones(vec![true, false])
            .truncateds(vec![false, true])
            .log_probs(Tensor::zeros([2, 1], (device, DType::F64)))
            .build()
    }

    #[test]
    fn rollout_batch_preserves_axes_statistics_dtype_and_gradients() {
        let device = Device::flex().autodiff();
        let mut experiences = [transition(1.0, &device), transition(10.0, &device)];
        for item in &mut experiences {
            item.prepared = Some(PPOPreparedExperience {
                advantages: item.rewards.clone(),
                returns: item.training_rewards.clone(),
                old_values: Some(item.log_probs.clone()),
            });
        }
        let batch = PPOExperience::batch(&experiences).unwrap();
        let observations: Tensor<4> = batch.observations;
        let actions: Tensor<3> = batch.actions;
        assert_eq!(observations.dims(), [2, 2, 1, 2]);
        assert_eq!(batch.next_observations.dims(), [2, 2, 1, 2]);
        assert_eq!(actions.dims(), [2, 2, 1]);
        assert_eq!(observations.dtype(), DType::F64);
        assert_eq!(observations.device(), device);
        assert_eq!(batch.rewards.dims(), [2, 2, 1]);
        assert_eq!(batch.rewards.dtype(), DType::F64);
        assert_eq!(
            batch.rewards.into_data().try_to_vec::<f64>().unwrap(),
            [3.0, 4.0, 12.0, 13.0]
        );
        assert_eq!(
            batch
                .training_rewards
                .into_data()
                .try_to_vec::<f64>()
                .unwrap(),
            [103.0, 104.0, 112.0, 113.0]
        );
        assert_eq!(batch.next_dones.dims(), [2, 2, 1]);
        assert_eq!(
            batch.next_dones.into_data().try_to_vec::<bool>().unwrap(),
            [true, false, true, false]
        );
        assert_eq!(
            batch.truncateds.into_data().try_to_vec::<bool>().unwrap(),
            [false, true, false, true]
        );
        assert_eq!(batch.log_probs.dims(), [2, 2, 1]);
        let prepared = batch.prepared.unwrap();
        assert_eq!(prepared.advantages.dims(), [2, 2, 1]);
        assert_eq!(prepared.returns.dims(), [2, 2, 1]);
        assert_eq!(prepared.old_values.unwrap().dims(), [2, 2, 1]);

        let gradients = observations.sum().backward();
        for item in experiences {
            assert_eq!(
                item.observations
                    .grad(&gradients)
                    .unwrap()
                    .into_data()
                    .try_to_vec::<f64>()
                    .unwrap(),
                [1.0; 4]
            );
        }
    }

    #[test]
    fn rollout_batch_reports_empty_and_mismatched_columns() {
        assert!(matches!(
            <PPOExperience as Experience>::batch(&[]),
            Err(ExperienceBatchError::EmptyBatch)
        ));
        let device = Device::flex().autodiff();
        let mut experiences = [transition(1.0, &device), transition(10.0, &device)];
        experiences[1].rewards = experiences[1].rewards.clone().cast(DType::F32);
        assert!(matches!(
            PPOExperience::batch(&experiences),
            Err(ExperienceBatchError::DTypeMismatch { index: 1, .. })
        ));
    }

    #[test]
    fn rollout_batch_preserves_optional_prepared_values() {
        let device = Device::flex().autodiff();
        let mut experiences = [transition(1.0, &device), transition(10.0, &device)];
        assert!(
            PPOExperience::batch(&experiences)
                .unwrap()
                .prepared
                .is_none()
        );
        for item in &mut experiences {
            item.prepared = Some(PPOPreparedExperience {
                advantages: item.rewards.clone(),
                returns: item.training_rewards.clone(),
                old_values: None,
            });
        }
        assert!(
            PPOExperience::batch(&experiences)
                .unwrap()
                .prepared
                .unwrap()
                .old_values
                .is_none()
        );
    }
}

#[cfg(test)]
mod network_configuration_tests {
    use burn::{
        module::Module,
        nn::{Initializer, Linear, LinearConfig, Relu},
        optim::{AdamConfig, GradientsParams, SgdConfig},
        tensor::Device,
    };

    use super::{PPOConfigurationError, PPONetworkInfo, SeparatePPONetwork, SharedPPONetwork};

    fn layer(device: &Device) -> Linear {
        LinearConfig::new(2, 2)
            .with_initializer(Initializer::Constant { value: 0.5 })
            .init(device)
            .valid()
    }

    #[test]
    fn shared_builder_owns_models_and_enables_parameter_gradients() {
        let device = Device::flex().autodiff();
        let network = SharedPPONetwork::builder()
            .shared_network(layer(&device))
            .actor_head(layer(&device))
            .critic_head(layer(&device))
            .optimizer(SgdConfig::new().init())
            .learning_rate(0.001)
            .lr_scheduler(|_| 0.0005)
            .build()
            .unwrap();
        let network = PPONetworkInfo::<_>::Shared(network);
        let PPONetworkInfo::Shared(mut network) = network else {
            panic!("expected a shared network");
        };

        assert!(network.shared_network.weight.val().is_require_grad());
        assert!(network.actor_head.weight.val().is_require_grad());
        assert!(network.critic_head.weight.val().is_require_grad());
        assert_eq!(network.learning_rate, 0.001);
        assert_eq!(network.lr_scheduler.as_ref().unwrap().value(0.5), 0.0005);

        let models = (
            network.shared_network,
            network.actor_head,
            network.critic_head,
        );
        let loss =
            models.0.weight.val().sum() + models.1.weight.val().sum() + models.2.weight.val().sum();
        let gradients = GradientsParams::from_grads(loss.backward(), &models);
        let models = network
            .optimizer
            .step(network.learning_rate, models, gradients);
        for model in [models.0, models.1, models.2] {
            assert!(
                (model.weight.val().into_data().try_to_vec::<f32>().unwrap()[0] - 0.499).abs()
                    < 1e-6
            );
        }
    }

    #[test]
    fn separate_builder_preserves_independent_rates_and_combined_loss() {
        let device = Device::flex().autodiff();
        let network = SeparatePPONetwork::builder()
            .actor_network(layer(&device))
            .critic_network(layer(&device))
            .actor_optimizer(AdamConfig::new().init())
            .critic_optimizer(SgdConfig::new().init())
            .actor_learning_rate(0.001)
            .critic_learning_rate(0.002)
            .actor_lr_scheduler(|_| 0.0005)
            .critic_lr_scheduler(|_| 0.0007)
            .combined_loss(true)
            .build()
            .unwrap();
        let network = PPONetworkInfo::<_>::Separate(network);
        let PPONetworkInfo::Separate(mut network) = network else {
            panic!("expected a separate network");
        };

        assert!(network.actor_network.weight.val().is_require_grad());
        assert!(network.critic_network.weight.val().is_require_grad());
        assert_eq!(network.actor_learning_rate, 0.001);
        assert_eq!(network.critic_learning_rate, 0.002);
        assert_eq!(
            network.actor_lr_scheduler.as_ref().unwrap().value(0.5),
            0.0005
        );
        assert_eq!(
            network.critic_lr_scheduler.as_ref().unwrap().value(0.5),
            0.0007
        );
        assert!(network.combined_loss);

        let actor_gradients = GradientsParams::from_grads(
            network.actor_network.weight.val().sum().backward(),
            &network.actor_network,
        );
        let critic_gradients = GradientsParams::from_grads(
            network.critic_network.weight.val().sum().backward(),
            &network.critic_network,
        );
        let actor = network.actor_optimizer.step(
            network.actor_learning_rate,
            network.actor_network,
            actor_gradients,
        );
        let critic = network.critic_optimizer.step(
            network.critic_learning_rate,
            network.critic_network,
            critic_gradients,
        );
        assert!(
            (actor.weight.val().into_data().try_to_vec::<f32>().unwrap()[0] - 0.499).abs() < 1e-6
        );
        assert!(
            (critic.weight.val().into_data().try_to_vec::<f32>().unwrap()[0] - 0.498).abs() < 1e-6
        );
    }

    #[test]
    fn network_builders_reject_invalid_learning_rates() {
        for rate in [0.0, -0.001, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let shared = SharedPPONetwork::builder()
                .shared_network(Relu)
                .actor_head(Relu)
                .critic_head(Relu)
                .optimizer(SgdConfig::new().init())
                .learning_rate(rate)
                .build();
            assert!(matches!(
                shared,
                Err(PPOConfigurationError::InvalidLearningRate)
            ));

            for (actor_rate, critic_rate) in [(rate, 0.001), (0.001, rate)] {
                let separate = SeparatePPONetwork::builder()
                    .actor_network(Relu)
                    .critic_network(Relu)
                    .actor_optimizer(SgdConfig::new().init())
                    .critic_optimizer(SgdConfig::new().init())
                    .actor_learning_rate(actor_rate)
                    .critic_learning_rate(critic_rate)
                    .build();
                assert!(matches!(
                    separate,
                    Err(PPOConfigurationError::InvalidLearningRate)
                ));
            }
        }
    }
}

#[cfg(test)]
mod schedule_tests {
    use std::collections::HashSet;

    use candle_core::{DType, Device, Tensor, TensorId, Var};
    use candle_nn::{Optimizer, VarBuilder, VarMap};

    use super::*;
    use crate::{
        agents::{
            Agent,
            test_support::{CountingOptimizer, FixedEnv},
        },
        distributions::{CategoricalDistribution, GaussianDistribution},
        gym::{Gym, MultiGym, ResetInfo, StepInfo, VectorizedGymWrapper},
        models::{
            MLP,
            probabilistic_model::{ProbabilisticPolicyModel, ProbabilisticPolicyModelError},
        },
        parameter_schedule::LinearSchedule,
        spaces::{BoxSpace, Discrete, Space},
    };

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    struct NoopLogger;

    impl PPOLogger for NoopLogger {
        fn log(&mut self, _info: &PPOLogEntry) {}
    }

    #[test]
    fn candle_elementwise_minimum_and_maximum_split_tied_gradients() {
        /// Checks scalar `loss` shaped `[1]` made from tied scalar variables.
        fn assert_tied_gradients(loss: Tensor, left: &Var, right: &Var) {
            let gradients = loss.sum_all().unwrap().backward().unwrap();
            for variable in [left, right] {
                assert_eq!(
                    gradients
                        .get(variable.as_tensor())
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap(),
                    vec![0.5]
                );
            }
        }

        for maximum in [false, true] {
            let left = Var::from_vec(vec![1.0_f32], 1, &Device::Cpu).unwrap();
            let right = Var::from_vec(vec![1.0_f32], 1, &Device::Cpu).unwrap();
            let loss = if maximum {
                left.as_tensor().maximum(right.as_tensor()).unwrap()
            } else {
                left.as_tensor().minimum(right.as_tensor()).unwrap()
            };
            assert_tied_gradients(loss, &left, &right);
        }
    }

    struct TwoStepTestGym {
        device: Device,
        steps: usize,
        truncates: bool,
    }

    impl TwoStepTestGym {
        fn new(device: Device, truncates: bool) -> Self {
            Self {
                device,
                steps: 0,
                truncates,
            }
        }
    }

    impl Gym<usize> for TwoStepTestGym {
        type Error = candle_core::Error;
        type SpaceError = candle_core::Error;

        /// Steps with one scalar discrete action shaped `[]`.
        fn step(&mut self, _action: Tensor) -> Result<StepInfo<usize>, Self::Error> {
            self.steps += 1;
            Ok(StepInfo {
                state: Tensor::zeros(4, DType::F32, &self.device)?,
                reward: 1.0,
                done: self.steps == 2 && !self.truncates,
                truncated: self.steps == 2 && self.truncates,
                info: self.steps,
            })
        }

        fn reset(&mut self) -> Result<ResetInfo<usize>, Self::Error> {
            self.steps = 0;
            Ok(ResetInfo {
                state: Tensor::zeros(4, DType::F32, &self.device)?,
                info: 0,
            })
        }

        fn observation_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
            Box::new(BoxSpace::new_with_universal_bounds(
                vec![4],
                -1.0,
                1.0,
                &self.device,
            ))
        }

        fn action_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
            Box::new(Discrete::new(2))
        }
    }

    #[test]
    fn explained_variance_uses_population_variance() {
        use burn::tensor::Tensor;

        let device = burn::tensor::Device::flex();
        let returns = Tensor::<2>::from_data([[1.0f32], [2.0], [3.0]], &device);
        let rollout_values = Tensor::<2>::from_data([[1.0f32], [2.0], [2.0]], &device);

        let actual = compute_explained_variance(&returns, &rollout_values)
            .try_into_scalar::<f32>()
            .unwrap();

        assert!((actual - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn explained_variance_is_nan_for_constant_returns() {
        use burn::tensor::Tensor;

        let device = burn::tensor::Device::flex();
        let returns = Tensor::<2>::from_data([[1.0f32], [1.0], [1.0]], &device);
        let rollout_values = Tensor::<2>::from_data([[0.0f32], [1.0], [2.0]], &device);

        let actual = compute_explained_variance(&returns, &rollout_values)
            .try_into_scalar::<f32>()
            .unwrap();

        assert!(actual.is_nan());

        assert!(
            compute_explained_variance(&returns, &returns)
                .try_into_scalar::<f32>()
                .unwrap()
                .is_nan()
        );
    }

    #[test]
    fn population_variance_preserves_f64_device_and_gradients() {
        use burn::tensor::Tensor;

        let device = burn::tensor::Device::flex().autodiff();
        let values = Tensor::<3>::from_data(
            [[[1.0f64], [2.0], [3.0]]],
            (&device, burn::tensor::DType::F64),
        )
        .require_grad();
        let variance = population_variance(&values);
        assert_eq!(variance.dims(), [1]);
        assert_eq!(variance.dtype(), burn::tensor::DType::F64);
        assert_eq!(variance.device(), device);
        assert_close(
            variance.clone().try_into_scalar::<f64>().unwrap(),
            2.0 / 3.0,
        );

        let gradients = variance.backward();
        let gradient = values
            .grad(&gradients)
            .unwrap()
            .into_data()
            .try_to_vec::<f64>()
            .unwrap();
        for (actual, expected) in gradient.into_iter().zip([-2.0 / 3.0, 0.0, 2.0 / 3.0]) {
            assert_close(actual, expected);
        }
    }

    #[test]
    fn simultaneous_termination_and_truncation_does_not_bootstrap() {
        struct UnitCritic;

        impl candle_core::Module for UnitCritic {
            /// Returns one value prediction per input row, shaped `[batch, 1]`.
            fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
                Tensor::ones((input.dim(0)?, 1), input.dtype(), input.device())
            }
        }

        let device = Device::Cpu;
        let actor_vars = VarMap::new();
        let actor = MLP::builder()
            .input_size(4)
            .output_size(2)
            .vb(VarBuilder::from_varmap(&actor_vars, DType::F32, &device))
            .build()
            .unwrap();
        let networks = PPONetworkInfo::Separate(
            SeparatePPONetwork::builder()
                .actor_optimizer(CountingOptimizer::with_learning_rate(1e-3))
                .critic_optimizer(CountingOptimizer::with_learning_rate(1e-3))
                .actor_network(ProbabilisticPolicyModel::<CategoricalDistribution>::new(
                    actor,
                ))
                .critic_network(UnitCritic)
                .build(),
        );
        let mut agent: PPOAgent<
            '_,
            CountingOptimizer,
            CountingOptimizer,
            ProbabilisticPolicyModelError<crate::distributions::CategoricalDistributionError>,
            candle_core::Error,
            candle_core::Error,
        > = PPOAgent::builder()
            .action_space(Box::new(Discrete::new(2)))
            .network_info(networks)
            .batch_size(2)
            .mini_batch_size(2)
            .training_horizon(2)
            .device(device.clone())
            .build()
            .unwrap();
        let rewards = Tensor::from_vec(vec![1.0_f32, 2.0], (2, 1), &device).unwrap();
        let next_states = Tensor::zeros((2, 4), DType::F32, &device).unwrap();

        let actual = agent
            .bootstrap_truncated_rewards(&rewards, &next_states, &[true, false], &[true, true])
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert_eq!(actual, vec![1.0, 2.99]);
    }

    #[test]
    fn collection_logs_report_fresh_rewards_and_completed_episodes() {
        let device = Device::Cpu;
        let mut env: VectorizedGymWrapper<TwoStepTestGym, usize> = vec![
            TwoStepTestGym::new(device.clone(), false),
            TwoStepTestGym::new(device.clone(), true),
        ]
        .into();
        let actor_vars = VarMap::new();
        let critic_vars = VarMap::new();
        let actor = MLP::builder()
            .input_size(4)
            .output_size(2)
            .vb(VarBuilder::from_varmap(&actor_vars, DType::F32, &device))
            .hidden_layer_sizes(vec![2])
            .build()
            .unwrap();
        let critic = MLP::builder()
            .input_size(4)
            .output_size(1)
            .vb(VarBuilder::from_varmap(&critic_vars, DType::F32, &device))
            .hidden_layer_sizes(vec![2])
            .build()
            .unwrap();
        let networks = PPONetworkInfo::Separate(
            SeparatePPONetwork::builder()
                .actor_optimizer(CountingOptimizer::with_learning_rate(1e-3))
                .critic_optimizer(CountingOptimizer::with_learning_rate(1e-3))
                .actor_network(ProbabilisticPolicyModel::<CategoricalDistribution>::new(
                    actor,
                ))
                .critic_network(critic)
                .build(),
        );

        struct CollectionLogger {
            collection_rewards: Vec<Vec<f32>>,
            optimization_rewards: Vec<Vec<f32>>,
            infos: Vec<Vec<usize>>,
            completed_episodes: Vec<(usize, f32, usize, bool, bool, usize)>,
        }

        impl PPOLogger<usize> for CollectionLogger {
            fn log(&mut self, entry: &PPOLogEntry) {
                self.optimization_rewards.push(
                    entry
                        .rewards
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap(),
                );
            }

            fn log_collection(&mut self, entry: &PPOCollectionLogEntry<usize>) {
                self.collection_rewards.push(
                    entry
                        .collection_rewards
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap(),
                );
                self.infos.push(entry.infos.clone());
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
            optimization_rewards: Vec::new(),
            infos: Vec::new(),
            completed_episodes: Vec::new(),
        };
        let mut agent = PPOAgent::builder()
            .action_space(env.action_space())
            .network_info(networks)
            .batch_size(2)
            .mini_batch_size(2)
            .num_epochs(1)
            .training_horizon(4)
            .logging_info(&mut logger)
            .device(device)
            .build()
            .unwrap();

        agent.learn(&mut env, 2).unwrap();
        agent.learn(&mut env, 2).unwrap();

        assert_eq!(
            logger.collection_rewards,
            vec![vec![1.0, 1.0], vec![1.0, 1.0]]
        );
        assert_eq!(
            logger.optimization_rewards,
            vec![vec![1.0, 1.0], vec![1.0, 1.0]]
        );
        assert_eq!(logger.infos, vec![vec![1, 1], vec![2, 2]]);
        assert_eq!(
            logger.completed_episodes,
            vec![(0, 2.0, 2, true, false, 3), (1, 2.0, 2, false, true, 4)]
        );
    }

    #[test]
    fn shared_agent_runs_collection_training_actions_and_state_reset() {
        let device = Device::Cpu;
        let mut env: VectorizedGymWrapper<FixedEnv> =
            vec![FixedEnv::new(device.clone()), FixedEnv::new(device.clone())].into();
        let shared_vars = VarMap::new();
        let actor_vars = VarMap::new();
        let critic_vars = VarMap::new();
        let shared_network = MLP::builder()
            .input_size(4)
            .output_size(4)
            .vb(VarBuilder::from_varmap(&shared_vars, DType::F32, &device))
            .hidden_layer_sizes(vec![4])
            .build()
            .unwrap();
        let actor_head = MLP::builder()
            .input_size(4)
            .output_size(2)
            .vb(VarBuilder::from_varmap(&actor_vars, DType::F32, &device))
            .hidden_layer_sizes(vec![2])
            .build()
            .unwrap();
        let critic_head = MLP::builder()
            .input_size(4)
            .output_size(1)
            .vb(VarBuilder::from_varmap(&critic_vars, DType::F32, &device))
            .hidden_layer_sizes(vec![2])
            .build()
            .unwrap();
        let network_info: PPONetworkInfo<CountingOptimizer, _, FakeOptimizer> =
            PPONetworkInfo::Shared(
                SharedPPONetwork::builder()
                    .optimizer(CountingOptimizer::with_learning_rate(1e-3))
                    .shared_network(shared_network)
                    .actor_head(ProbabilisticPolicyModel::<CategoricalDistribution>::new(
                        actor_head,
                    ))
                    .critic_head(critic_head)
                    .lr_scheduler(LinearSchedule::new(1e-3, 1e-4))
                    .build(),
            );
        let mut logger = NoopLogger;
        let mut agent = PPOAgent::builder()
            .action_space(env.action_space())
            .network_info(network_info)
            .batch_size(2)
            .mini_batch_size(2)
            .num_epochs(1)
            .training_horizon(10)
            .logging_info(&mut logger)
            .device(device.clone())
            .build()
            .unwrap();

        agent.learn(&mut env, 2).unwrap();
        let observations = agent.current_states.as_ref().unwrap().clone();
        assert_eq!(agent.act(&observations).unwrap().dims(), &[2, 1]);
        assert_eq!(
            agent.act_deterministic(&observations).unwrap().dims(),
            &[2, 1]
        );
        assert_eq!(agent.episode_tracker.environment_count(), 2);
        let PPONetworkInfo::Shared(network) = &agent.network_info else {
            panic!("expected shared PPO network");
        };
        assert_eq!(network.optimizer.steps, 1);
        assert_close(network.optimizer.learning_rate(), 0.00082);

        agent.set_learning_rate(0.25);
        let PPONetworkInfo::Shared(network) = &agent.network_info else {
            panic!("expected shared PPO network");
        };
        assert_close(network.optimizer.learning_rate(), 0.25);

        agent.reset_current_states();
        assert!(agent.current_states.is_none());
        assert_eq!(agent.episode_tracker.environment_count(), 0);
    }

    #[test]
    fn separate_ppo_losses_only_reach_their_parameters() {
        fn parameter_ids(vars: &[Var]) -> HashSet<TensorId> {
            vars.iter().map(|var| var.as_tensor().id()).collect()
        }

        fn gradient_ids(gradients: &candle_core::backprop::GradStore) -> HashSet<TensorId> {
            gradients.get_ids().copied().collect()
        }

        let device = Device::Cpu;
        let actor_var_map = VarMap::new();
        let critic_var_map = VarMap::new();
        let actor_network = MLP::builder()
            .input_size(4)
            .output_size(6)
            .vb(VarBuilder::from_varmap(&actor_var_map, DType::F32, &device))
            .activation(Tensor::tanh)
            .hidden_layer_sizes(vec![8])
            .build()
            .unwrap();
        let critic_network = MLP::builder()
            .input_size(4)
            .output_size(1)
            .vb(VarBuilder::from_varmap(
                &critic_var_map,
                DType::F32,
                &device,
            ))
            .activation(Tensor::tanh)
            .hidden_layer_sizes(vec![8])
            .build()
            .unwrap();
        let actor_parameters = actor_var_map.all_vars();
        let critic_parameters = critic_var_map.all_vars();
        let actor_ids = parameter_ids(&actor_parameters);
        let critic_ids = parameter_ids(&critic_parameters);
        let expected_combined_ids = actor_ids
            .union(&critic_ids)
            .copied()
            .collect::<HashSet<_>>();

        let network_info = PPONetworkInfo::Separate(
            SeparatePPONetwork::builder()
                .actor_optimizer(CountingOptimizer::with_learning_rate(3e-4))
                .critic_optimizer(CountingOptimizer::with_learning_rate(3e-4))
                .actor_network(ProbabilisticPolicyModel::<GaussianDistribution>::new(
                    actor_network,
                ))
                .critic_network(critic_network)
                .combined_loss(true)
                .build(),
        );
        let mut agent: PPOAgent<
            '_,
            CountingOptimizer,
            CountingOptimizer,
            ProbabilisticPolicyModelError<crate::distributions::GaussianDistributionError>,
            candle_core::Error,
            candle_core::Error,
        > = PPOAgent::builder()
            .action_space(Box::new(BoxSpace::new_with_universal_bounds(
                vec![3],
                -1.0,
                1.0,
                &device,
            )))
            .network_info(network_info)
            .batch_size(4)
            .mini_batch_size(4)
            .ent_coef(0.001)
            .clip_value_loss(true)
            .training_horizon(4)
            .device(device.clone())
            .build()
            .unwrap();

        // Model rollout data as variables, then detach every field at the same
        // boundaries used by optimize(). None of these IDs may survive in the
        // loss gradient store.
        let rollout_states = Var::from_vec(
            vec![
                0.1f32, 0.2, 0.3, 0.4, -0.1, 0.3, -0.2, 0.5, 0.7, -0.4, 0.2, -0.3, -0.5, -0.2, 0.6,
                0.1,
            ],
            (4, 4),
            &device,
        )
        .unwrap();
        let rollout_actions = Var::from_vec(vec![0.1f32; 12], (4, 3), &device).unwrap();
        let rollout_log_probs = Var::from_vec(vec![-1.0f32; 4], 4, &device).unwrap();
        let rollout_advantages = Var::from_vec(vec![1.0f32, -1.0, 0.5, -0.5], 4, &device).unwrap();
        let rollout_returns = Var::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], 4, &device).unwrap();
        let rollout_rewards = Var::from_vec(vec![0.1f32; 4], 4, &device).unwrap();
        let rollout_old_values = Var::from_vec(vec![0.0f32; 4], 4, &device).unwrap();
        let rollout_ids = parameter_ids(&[
            rollout_states.clone(),
            rollout_actions.clone(),
            rollout_log_probs.clone(),
            rollout_advantages.clone(),
            rollout_returns.clone(),
            rollout_rewards.clone(),
            rollout_old_values.clone(),
        ]);

        let losses = agent
            .compute_loss()
            .states(&rollout_states.detach())
            .actions(&rollout_actions.detach())
            .old_log_probs(rollout_log_probs.detach())
            .advantages(rollout_advantages.detach())
            .returns(rollout_returns.detach())
            .rewards(rollout_rewards.detach())
            .old_values(rollout_old_values.detach())
            .explained_variance(Tensor::new(0.0f32, &device).unwrap())
            .clip_range(0.2)
            .call()
            .unwrap();

        let actor_gradients = losses.actor_loss.backward().unwrap();
        let critic_gradients = losses.critic_loss.backward().unwrap();
        let combined_gradients = (&losses.actor_loss + &losses.critic_loss)
            .unwrap()
            .backward()
            .unwrap();

        let actor_gradient_ids = gradient_ids(&actor_gradients);
        let critic_gradient_ids = gradient_ids(&critic_gradients);
        let combined_gradient_ids = gradient_ids(&combined_gradients);

        // Candle also retains local gradients for some non-variable operands.
        // Optimizers only query registered variable IDs, so audit the variable
        // boundaries rather than requiring the GradStore to contain only
        // parameter entries.
        assert!(actor_ids.is_subset(&actor_gradient_ids));
        assert!(actor_gradient_ids.is_disjoint(&critic_ids));
        assert!(actor_gradient_ids.is_disjoint(&rollout_ids));
        assert!(critic_ids.is_subset(&critic_gradient_ids));
        assert!(critic_gradient_ids.is_disjoint(&actor_ids));
        assert!(critic_gradient_ids.is_disjoint(&rollout_ids));
        assert!(expected_combined_ids.is_subset(&combined_gradient_ids));
        assert!(combined_gradient_ids.is_disjoint(&rollout_ids));
    }

    #[test]
    fn schedules_continue_across_learn_calls() {
        let device = Device::Cpu;
        let mut env: VectorizedGymWrapper<FixedEnv> =
            vec![FixedEnv::new(device.clone()), FixedEnv::new(device.clone())].into();
        let actor_var_map = VarMap::new();
        let critic_var_map = VarMap::new();
        let actor_network = MLP::builder()
            .input_size(4)
            .output_size(2)
            .vb(VarBuilder::from_varmap(&actor_var_map, DType::F32, &device))
            .activation(Tensor::tanh)
            .hidden_layer_sizes(vec![2])
            .build()
            .unwrap();
        let critic_network = MLP::builder()
            .input_size(4)
            .output_size(1)
            .vb(VarBuilder::from_varmap(
                &critic_var_map,
                DType::F32,
                &device,
            ))
            .activation(Tensor::tanh)
            .hidden_layer_sizes(vec![2])
            .build()
            .unwrap();
        let network_info = PPONetworkInfo::Separate(
            SeparatePPONetwork::builder()
                .actor_optimizer(CountingOptimizer::with_learning_rate(1e-3))
                .critic_optimizer(CountingOptimizer::with_learning_rate(1e-3))
                .actor_network(ProbabilisticPolicyModel::<CategoricalDistribution>::new(
                    actor_network,
                ))
                .critic_network(critic_network)
                .actor_lr_scheduler(LinearSchedule::new(1e-3, 1e-4))
                .critic_lr_scheduler(LinearSchedule::new(1e-3, 1e-4))
                .build(),
        );
        let mut agent = PPOAgent::builder()
            .action_space(env.action_space())
            .network_info(network_info)
            .batch_size(2)
            .mini_batch_size(2)
            .num_epochs(1)
            .clip_range(LinearSchedule::new(0.2, 0.1))
            .training_horizon(10)
            .device(device.clone())
            .build()
            .unwrap();

        agent.learn(&mut env, 2).unwrap();
        assert_eq!(agent.schedule_progress.elapsed_steps(), 2);
        assert_close(agent.schedule_progress.parameter(&*agent.clip_range), 0.18);
        let PPONetworkInfo::Separate(network) = &agent.network_info else {
            panic!("expected separate PPO network");
        };
        assert_close(network.actor_optimizer.learning_rate(), 0.00082);
        assert_close(network.critic_optimizer.learning_rate(), 0.00082);
        assert_eq!(network.actor_optimizer.steps, 1);
        assert_eq!(network.critic_optimizer.steps, 1);
        agent.learn(&mut env, 2).unwrap();

        assert_eq!(agent.schedule_progress.elapsed_steps(), 4);
        assert_close(agent.schedule_progress.parameter(&*agent.clip_range), 0.16);
        let PPONetworkInfo::Separate(network) = &agent.network_info else {
            panic!("expected separate PPO network");
        };
        assert_close(network.actor_optimizer.learning_rate(), 0.00064);
        assert_close(network.critic_optimizer.learning_rate(), 0.00064);
        assert_eq!(network.actor_optimizer.steps, 2);
        assert_eq!(network.critic_optimizer.steps, 2);

        let observations = Tensor::zeros((2, 4), DType::F32, &device).unwrap();
        assert_eq!(agent.act(&observations).unwrap().dims(), &[2, 1]);
        assert_eq!(
            agent.act_deterministic(&observations).unwrap().dims(),
            &[2, 1]
        );
        agent.set_learning_rate(0.25);
        let PPONetworkInfo::Separate(network) = &agent.network_info else {
            panic!("expected separate PPO network");
        };
        assert_close(network.actor_optimizer.learning_rate(), 0.25);
        assert_close(network.critic_optimizer.learning_rate(), 0.25);
    }
    #[test]
    fn gae_matches_scalar_reference_across_environments_and_episode_boundaries() {
        let device = Device::Cpu;
        let mut env: VectorizedGymWrapper<TwoStepTestGym, usize> = (0..3)
            .map(|_| TwoStepTestGym::new(device.clone(), false))
            .collect::<Vec<_>>()
            .into();
        let actor_vars = VarMap::new();
        let critic_vars = VarMap::new();
        let actor = MLP::builder()
            .input_size(4)
            .output_size(2)
            .vb(VarBuilder::from_varmap(&actor_vars, DType::F32, &device))
            .hidden_layer_sizes(vec![2])
            .build()
            .unwrap();
        let critic = MLP::builder()
            .input_size(4)
            .output_size(1)
            .vb(VarBuilder::from_varmap(&critic_vars, DType::F32, &device))
            .hidden_layer_sizes(vec![2])
            .build()
            .unwrap();
        let networks = PPONetworkInfo::Separate(
            SeparatePPONetwork::builder()
                .actor_optimizer(CountingOptimizer::with_learning_rate(1e-3))
                .critic_optimizer(CountingOptimizer::with_learning_rate(1e-3))
                .actor_network(ProbabilisticPolicyModel::<CategoricalDistribution>::new(
                    actor,
                ))
                .critic_network(critic)
                .build(),
        );
        let mut agent = PPOAgent::builder()
            .action_space(env.action_space())
            .network_info(networks)
            .batch_size(6)
            .mini_batch_size(6)
            .num_epochs(1)
            .training_horizon(6)
            .gamma(0.9)
            .gae_lambda(0.8)
            .device(device.clone())
            .build()
            .unwrap();
        agent.learn(&mut env, 0).unwrap();

        let rewards = [[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let values = [[0.2_f32, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]];
        let dones = [[0.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let truncateds = [[0.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let bootstrap = [1.1_f32, 1.2, 1.3];
        let flat = |rows: [[f32; 3]; 3]| rows.into_iter().flatten().collect::<Vec<_>>();
        let reward_tensor = Tensor::from_vec(flat(rewards), (3, 3, 1), &device).unwrap();
        let value_tensor = Tensor::from_vec(flat(values), (3, 3, 1), &device).unwrap();
        let done_tensor = Tensor::from_vec(flat(dones), (3, 3, 1), &device).unwrap();
        let truncated_tensor = Tensor::from_vec(flat(truncateds), (3, 3, 1), &device).unwrap();
        let bootstrap_tensor = Tensor::from_vec(bootstrap.to_vec(), (3, 1), &device).unwrap();

        let actual = agent
            .compute_gae(
                &reward_tensor,
                &value_tensor,
                &done_tensor,
                &truncated_tensor,
                &bootstrap_tensor,
            )
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        for env_idx in 0..3 {
            let mut next_value = bootstrap[env_idx];
            let mut gae = 0.0_f32;
            for time in (0..3).rev() {
                let mask = (1.0 - dones[time][env_idx]) * (1.0 - truncateds[time][env_idx]);
                let value = values[time][env_idx];
                let delta = rewards[time][env_idx] + 0.9 * next_value * mask - value;
                gae = delta + 0.9 * 0.8 * gae * mask;
                assert!(
                    (actual[time][env_idx][0] - gae).abs() < 1e-5,
                    "time {time}, env {env_idx}: expected {gae}, got {}",
                    actual[time][env_idx][0]
                );
                next_value = value;
            }
        }
    }
}

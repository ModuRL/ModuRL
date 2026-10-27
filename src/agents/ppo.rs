use crate::tensor_rank::TensorBatchTypes;
mod network;

pub use network::*;

#[cfg(test)]
mod tests;

use bon::bon;
use burn::tensor::{
    Bool, DType, Int, Tensor, TensorData, TensorReadError,
    kind::{Autodiff, Basic},
};
use std::marker::PhantomData;

use crate::{
    buffers::experience::{self, Experience, ExperienceBatchError},
    gym::{MultiGym, MultiGymStepInfo},
    objectives::clipped_value_loss,
    parameter_schedule::{ConstantSchedule, ParameterSchedule, ScheduleProgress},
    sampling::{SamplingError, shuffle_with_device_rng},
    spaces::{ActionMap, ObservationSpace},
    tensor_operations::{clamp_with_boundary_gradients, normalize_tensor},
    tensor_rank::{NextRank, PrevRank},
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
    #[error("critic dtype {critic:?} must match environment reward dtype {reward:?}")]
    CriticRewardDTypeMismatch { critic: DType, reward: DType },
    #[error("network failed: {0}")]
    NetworkError(#[source] AE),
    #[error("gym failed: {0}")]
    GymError(#[source] GE),
    #[error("reading PPO tensor data failed: {0}")]
    TensorError(#[from] TensorReadError),
    #[error("rollout batching failed: {0}")]
    ExperienceBatch(#[from] ExperienceBatchError),
    #[error("minibatch sampling failed: {0}")]
    Sampling(#[from] SamplingError),
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
    values: Tensor<2>,
    prepared: Option<PPOPreparedExperience>,
}

struct PPOCollectionAction<AS: TensorBatchTypes, const L: usize> {
    policy_actions: Tensor<L>,
    environment_actions: AS::Batch,
    log_probs: Tensor<2>,
    values: Tensor<2>,
}

struct PPOPreparedRolloutBatch {
    advantages: Tensor<3>,
    returns: Tensor<3>,
    old_values: Option<Tensor<3>>,
}

struct PPORolloutBatch<O: NextRank, A: NextRank> {
    observations: O::Next,
    actions: A::Next,
    rewards: Tensor<3>,
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
impl<O: NextRank, A: NextRank> PPOExperience<O, A> {
    /// Creates one vectorized rollout transition.
    ///
    /// `observations` and `next_observations` are `[env_count, ...observation_shape]`.
    /// `actions` is `[env_count, ...action_shape]`; `rewards`, `log_probs`, and detached collection `values` are `[env_count, 1]`.
    /// `rewards` preserves the environment values for logging.
    /// Training rewards [env_count, 1] include terminal-value corrections for nonterminal truncations and retain the reward dtype.
    /// Observations preserve their native kind and dtype. Other fields preserve their Float dtype and device; collection detaches stored experience.
    /// PPO training data is added internally after rollout collection.
    #[builder]
    fn new(
        observations: O,
        next_observations: O,
        actions: A,
        rewards: Tensor<2>,
        training_rewards: Tensor<2>,
        next_dones: Vec<bool>,
        truncateds: Vec<bool>,
        log_probs: Tensor<2>,
        values: Tensor<2>,
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
            values,
            prepared: None,
        }
    }
}

impl<const O: usize, const A: usize, const BO: usize, const BA: usize, K: Basic>
    experience::Experience for PPOExperience<Tensor<O, K>, Tensor<A>>
where
    Tensor<O, K>: NextRank<Next = Tensor<BO, K>>,
    Tensor<A>: NextRank<Next = Tensor<BA>>,
{
    type Batch = PPORolloutBatch<Tensor<O, K>, Tensor<A>>;
    type Error = ExperienceBatchError;

    /// Adds a leading time axis to rank-O observations and rank-A actions, producing tensors with one additional axis.
    /// Training statistics become `[time, environment_count, 1]` without changing environment order.
    /// Each tensor field must share shape, dtype, and device across time. Preserves gradient connections.
    fn batch(experiences: &[Self]) -> Result<Self::Batch, Self::Error> {
        Ok(PPORolloutBatch {
            observations: experience::stack_tensor_field(experiences, |experience| {
                experience.observations.clone()
            })?,
            actions: experience::stack_tensor_field(experiences, |experience| {
                experience.actions.clone()
            })?,
            rewards: experience::stack_tensor_field(experiences, |experience| {
                experience.rewards.clone()
            })?,
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

/// Losses, mean entropy, and explained variance use Float `[1]` tensors.
/// Minibatch rewards, returns, advantages, and KL estimates use `[batch_size, 1]`.
/// Rewards retain the environment dtype. Returns, advantages, and explained variance share the environment reward dtype.
/// Policy statistics and actor loss use the policy's output dtype. All tensors use the model device.
pub struct PPOLogEntry {
    pub actor_loss: Tensor<1>,
    pub critic_loss: Tensor<1>,
    pub entropy: Tensor<1>,
    pub kl_divergence: Tensor<2>,
    pub explained_variance: Tensor<1>,
    pub rewards: Tensor<2>,
    pub epoch: usize,
    pub timestep: usize,
    pub returns: Tensor<2>,
    pub advantages: Tensor<2>,
}

/// Raw Float rewards `[environment_count, 1]` and metadata in environment-slot order.
pub struct PPOCollectionLogEntry<I = ()> {
    pub collection_rewards: Tensor<2>,
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
    actor_loss: Tensor<1>,
    critic_loss: Tensor<1>,
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

/// PPO with owned Burn models and explicit observation and action spaces.
/// The model group contains native observation, latent, and policy-action tensor types, each including the batch axis.
pub struct PPOAgent<'a, G: PPOModelTypes, OS, AS, GE, I = (), const A: usize = 2> {
    clipped: bool,
    clip_value_loss: bool,
    gamma: f32,
    gae_lambda: f32,
    clip_range: Box<dyn ParameterSchedule>,
    normalize_advantage: bool,
    normalize_returns: bool,
    vf_coef: f32,
    ent_coef: f32,
    observation_space: OS,
    action_space: AS,
    network_info: G::Network,
    rollout_buffer: Vec<PPOExperience<G::Observation, G::Action>>,
    mini_batch_size: usize,
    num_epochs: usize,
    batch_size: usize,
    schedule_progress: ScheduleProgress,
    logging_info: Option<PPOLoggingInfo<'a, I>>,
    gradient_clip: f32,
    current_observations: Option<G::Observation>,
    episode_tracker: PPOEpisodeTracker,
    _phantom: PhantomData<GE>,
}

impl<G: PPOModelTypes, OS, AS, GE, I, const A: usize> PPOAgent<'_, G, OS, AS, GE, I, A> {
    /// Returns the owned network configuration for model inspection and checkpoint snapshots.
    pub fn network(&self) -> &G::Network {
        &self.network_info
    }

    pub fn get_observation_space(&self) -> &OS {
        &self.observation_space
    }

    pub fn get_action_space(&self) -> &AS {
        &self.action_space
    }

    pub fn reset_current_observations(&mut self) {
        self.current_observations = None;
        self.episode_tracker.clear();
    }
}

#[bon]
impl<'a, G: PPOModelTypes, OS, AS, GE, I, const A: usize> PPOAgent<'a, G, OS, AS, GE, I, A> {
    /// Builds PPO for rank-O observation batches and rank-A environment-action batches.
    /// Preserves observation kind and dtype. Policy inputs have rank H; original policy actions have rank L.
    /// Shared trunks produce Float features for Float-input heads. Separate models receive observations directly and require H = O.
    /// Models own input conversions and must share a device. Critic outputs must match reward dtype; tensor kinds must agree at compile time.
    #[builder]
    pub fn new<N, const O: usize, const H: usize, const L: usize>(
        observation_space: OS,
        action_space: AS,
        network_info: N,
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
    ) -> Result<Self, PPOConfigurationError>
    where
        N: PPONetwork<O, H, L, InputKind = OS::Kind, ModelGroup = G>,
        G: PPOModelTypes<Network = N, Observation = OS::Batch, Action = Tensor<L>>,
        OS: ObservationSpace<O>,
        AS: ActionMap<L, A>,
        AS::Error: std::fmt::Debug,
        Tensor<O, OS::Kind>: NextRank,
        Tensor<L>: NextRank,
        GE: std::fmt::Debug,
    {
        const {
            assert!(
                O >= 2 && A >= 2,
                "observation and environment-action ranks require batch and item axes"
            );
        }
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
            observation_space,
            action_space,
            rollout_buffer: Vec::new(),
            num_epochs,
            batch_size,
            schedule_progress: ScheduleProgress::new(training_horizon),
            mini_batch_size,
            logging_info: logging_info.map(PPOLoggingInfo::new),
            gradient_clip,
            current_observations: None,
            episode_tracker: PPOEpisodeTracker::new(0),
            _phantom: PhantomData,
        })
    }
}

#[bon]
impl<'a, N, OS, AS, GE, I, const O: usize, const H: usize, const L: usize, const A: usize>
    PPOAgent<
        'a,
        PPOModelGroup<N, Tensor<O, OS::Kind>, Tensor<H, N::LatentKind>, Tensor<L>>,
        OS,
        AS,
        GE,
        I,
        A,
    >
where
    N: PPONetwork<O, H, L, InputKind = OS::Kind>,
    OS: ObservationSpace<O>,
    AS: ActionMap<L, A>,
    AS::Error: std::fmt::Debug,
    OS::Kind: Autodiff,
    Tensor<O, OS::Kind>: PrevRank + NextRank,
    Tensor<L>: NextRank,
    GE: std::fmt::Debug,
{
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(name = "ppo.optimize", target = "modurl::performance", skip_all)
    )]
    /// Updates training models from prepared rollout statistics and explained variance `[1]`.
    /// Rollout tensors preserve their native observation kind, reward dtype, and device.
    fn optimize<const BO: usize, const BL: usize>(
        &mut self,
        rollout_explained_variance: Tensor<1>,
    ) -> Result<(), PPOError<N::Error, GE, AS::Error>>
    where
        Tensor<O, OS::Kind>: NextRank<Next = Tensor<BO, OS::Kind>>,
        Tensor<L>: NextRank<Next = Tensor<BL>>,
    {
        let batch = PPOExperience::<OS::Batch, Tensor<L>>::batch(&self.rollout_buffer)?;
        let all_observations = batch.observations.flatten::<O>(0, 1);
        let all_actions = batch.actions.flatten::<L>(0, 1);
        let all_log_probs = batch.log_probs.flatten::<2>(0, 1);
        let prepared = batch.prepared.ok_or(PPOError::MissingPreparedRollout)?;
        let all_advantages = prepared.advantages.flatten::<2>(0, 1);
        let all_returns = prepared.returns.flatten::<2>(0, 1);
        let all_rewards = batch.rewards.flatten::<2>(0, 1);
        let all_old_values = prepared
            .old_values
            .map(|old_values| old_values.flatten::<2>(0, 1));

        let total_samples = all_observations.dims()[0];
        let device = all_observations.device();
        let clip_range = self.schedule_progress.parameter(&*self.clip_range) as f32;

        for epoch in 0..self.num_epochs {
            if let Some(logging_info) = &mut self.logging_info {
                logging_info.epoch = epoch;
            }

            let mut indices: Vec<i64> = (0..total_samples as i64).collect();
            shuffle_with_device_rng(&mut indices, &device)?;
            let indices_tensor = Tensor::<1, Int>::from_data(
                TensorData::new(indices, [total_samples]),
                (&device, DType::I64),
            );

            for start in (0..total_samples).step_by(self.mini_batch_size) {
                let end = (start + self.mini_batch_size).min(total_samples);
                let batch_indices = indices_tensor.clone().slice_dim(0, start..end);

                let batch_observations = all_observations.clone().select(0, batch_indices.clone());
                let batch_actions = all_actions.clone().select(0, batch_indices.clone());
                let batch_log_probs = all_log_probs.clone().select(0, batch_indices.clone());
                let batch_advantages = all_advantages.clone().select(0, batch_indices.clone());
                let batch_returns = all_returns.clone().select(0, batch_indices.clone());
                let batch_rewards = all_rewards.clone().select(0, batch_indices.clone());
                let batch_old_values = all_old_values
                    .as_ref()
                    .map(|old_values| old_values.clone().select(0, batch_indices));

                let ppo_losses = self
                    .compute_loss()
                    .observations(batch_observations)
                    .actions(batch_actions)
                    .old_log_probs(batch_log_probs.detach())
                    .advantages(batch_advantages)
                    .returns(batch_returns)
                    .rewards(batch_rewards)
                    .maybe_old_values(batch_old_values.map(|old_values| old_values.detach()))
                    .explained_variance(rollout_explained_variance.clone())
                    .clip_range(clip_range)
                    .call()?;

                self.backpropagate_loss(ppo_losses.clone())?;
                device.flush().map_err(TensorReadError::from)?;
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
    /// Returns explained variance `[1]` and prepares detached statistics `[time, env_count, 1]`.
    /// Reuses stored collection values and evaluates only the final next-observation batch without autodiff.
    fn add_advantages_and_returns(
        &mut self,
        validation: &N,
    ) -> Result<Tensor<1>, PPOError<N::Error, GE, AS::Error>> {
        let experiences = &self.rollout_buffer;
        let values_tensor: Tensor<3> =
            experience::stack_tensor_field(experiences, |experience| experience.values.clone())?;
        let final_experience = experiences.last().ok_or(ExperienceBatchError::EmptyBatch)?;
        let values_tensor =
            values_tensor.reshape([experiences.len(), final_experience.rewards.dims()[0], 1]);
        let bootstrapped_observations = final_experience.next_observations.clone();
        let latent_bootstrapped_observations = validation
            .forward_observations(bootstrapped_observations.without_autodiff())
            .map_err(PPOError::NetworkError)?;

        let bootstrapped_values = self
            .critic_network_forward(
                validation,
                &latent_bootstrapped_observations,
                final_experience.rewards.dtype(),
            )?
            .detach(); // shape [env_count, 1]

        let training_rewards = experience::stack_tensor_field(experiences, |experience| {
            experience.training_rewards.clone()
        })?;
        let next_dones = experience::stack_bool_field(
            experiences,
            |experience| &experience.next_dones,
            &values_tensor.device(),
        )?;
        let truncateds = experience::stack_bool_field(
            experiences,
            |experience| &experience.truncateds,
            &values_tensor.device(),
        )?;
        let advantages = self
            .compute_gae(
                &training_rewards,
                &values_tensor,
                &next_dones,
                &truncateds,
                &bootstrapped_values,
            )
            .detach();

        let returns = (values_tensor.clone() + advantages.clone()).detach();
        let rollout_explained_variance = compute_explained_variance(&returns, &values_tensor);

        let experiences = &mut self.rollout_buffer;

        for (i, experience) in experiences.iter_mut().enumerate() {
            experience.prepared = Some(PPOPreparedExperience {
                advantages: advantages
                    .clone()
                    .slice_dim(0, i..i + 1)
                    .squeeze_dim(0)
                    .detach(),
                returns: returns
                    .clone()
                    .slice_dim(0, i..i + 1)
                    .squeeze_dim(0)
                    .detach(),
                // Value-loss clipping compares current values with fixed collection values.
                old_values: if self.clip_value_loss {
                    Some(experience.values.clone())
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
    /// Float inputs share the environment reward dtype and device. Bool masks are converted to that dtype.
    /// Float inputs share a device. Detaches critic values and bootstrap values; rewards include corrections for truncated transitions.
    fn compute_gae(
        &self,
        rewards: &Tensor<3>,
        values: &Tensor<3>,
        next_dones: &Tensor<3, Bool>,
        next_truncateds: &Tensor<3, Bool>,
        bootstrapped_values: &Tensor<2>,
    ) -> Tensor<3> {
        let device = values.device();
        let dtype = values.dtype();
        let rewards = rewards.clone();
        let next_dones = next_dones.clone().float().cast(dtype);
        let next_truncateds = next_truncateds.clone().float().cast(dtype);
        let bootstrapped_values = bootstrapped_values.clone();
        let values = values.clone().detach();
        let masks = (1.0f64 - next_dones) * (1.0f64 - next_truncateds);
        let mut next_value = bootstrapped_values.detach();
        let mut gae = Tensor::zeros(next_value.dims(), (&device, dtype));
        let mut advantages = Vec::with_capacity(rewards.dims()[0]);
        // Environments are independent; only time has a recurrence.
        for time in (0..rewards.dims()[0]).rev() {
            let mask = masks
                .clone()
                .slice_dim(0, time..time + 1)
                .squeeze_dim::<2>(0);
            let value = values
                .clone()
                .slice_dim(0, time..time + 1)
                .squeeze_dim::<2>(0);
            let reward = rewards
                .clone()
                .slice_dim(0, time..time + 1)
                .squeeze_dim::<2>(0);
            let delta = reward + next_value * mask.clone() * f64::from(self.gamma) - value.clone();
            gae = delta + gae * mask * (f64::from(self.gamma) * f64::from(self.gae_lambda));
            advantages.push(gae.clone());
            next_value = value;
        }
        advantages.reverse();
        Tensor::stack(advantages, 0)
    }

    /// Maps rank-O observations `[batch, ...observation_shape]` to rank-H model inputs, preserving the batch axis.
    /// Shared models compute latent features; separate models preserve every axis, dtype, and device because O = H.
    fn latent_observations(
        &self,
        observations: OS::Batch,
    ) -> Result<Tensor<H, N::LatentKind>, N::Error> {
        self.network_info.forward_observations(observations)
    }

    /// Forwards latent or raw `observations` shaped `[batch, ...model_input_shape]` of rank H and
    /// returns critic values shaped `[batch, 1]`.
    /// Accepts the policy's input kind and rejects outputs that differ from the environment reward dtype; all tensors share a device.
    fn critic_network_forward(
        &self,
        network: &N,
        observations: &Tensor<H, N::LatentKind>,
        reward_dtype: DType,
    ) -> Result<Tensor<2>, PPOError<N::Error, GE, AS::Error>> {
        let values = network
            .forward_values(observations.clone())
            .map_err(PPOError::NetworkError)?;
        Self::check_critic_dtype(&values, reward_dtype)?;
        Ok(values)
    }

    /// Checks critic values `[batch, 1]` against the environment reward dtype without changing the tensor.
    fn check_critic_dtype(
        values: &Tensor<2>,
        reward_dtype: DType,
    ) -> Result<(), PPOError<N::Error, GE, AS::Error>> {
        if values.dtype() != reward_dtype {
            return Err(PPOError::CriticRewardDTypeMismatch {
                critic: values.dtype(),
                reward: reward_dtype,
            });
        }
        Ok(())
    }

    /// Adds discounted terminal values to rewards [environment_count, 1] for nonterminal truncated rows.
    /// Rank-O terminal observations are [environment_count, ...observation_shape]; both flag slices have environment_count entries.
    /// Preserves reward dtype and device, rejecting critic dtype mismatches. Terminal values are detached from the model.
    /// True termination suppresses bootstrapping, including transitions that also truncate.
    fn bootstrap_truncated_rewards(
        &mut self,
        validation: &N,
        rewards: Tensor<2>,
        transition_next_observations: &OS::Batch,
        dones: &[bool],
        truncateds: &[bool],
    ) -> Result<Tensor<2>, PPOError<N::Error, GE, AS::Error>> {
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
            .filter_map(|(index, (truncated, done))| (*truncated && !*done).then_some(index as i64))
            .collect::<Vec<_>>();
        if truncated_indices.is_empty() {
            return Ok(rewards);
        }

        let truncation_count = truncated_indices.len();
        let truncated_indices = Tensor::<1, Int>::from_data(
            TensorData::new(truncated_indices, [truncation_count]),
            (&transition_next_observations.device(), DType::I64),
        );
        let truncated_next_observations = transition_next_observations
            .clone()
            .select(0, truncated_indices.clone());
        let latent_observations = validation
            .forward_observations(truncated_next_observations.without_autodiff())
            .map_err(PPOError::NetworkError)?;
        let terminal_values = self
            .critic_network_forward(validation, &latent_observations, rewards.dtype())?
            .detach();
        Ok(rewards.select_assign(
            0,
            truncated_indices,
            terminal_values * f64::from(self.gamma),
            burn::tensor::IndexingUpdateOp::Add,
        ))
    }

    /// Evaluates rank-L `actions` `[batch, ...action_shape]` for latent or raw
    /// rank-H `observations` `[batch, ...model_input_shape]`, returning log probabilities and entropy `[batch, 1]`.
    /// Inputs share the model's Float dtype and device; evaluation retains parameter gradients.
    fn actor_network_log_prob_and_entropy(
        &mut self,
        observations: &Tensor<H, N::LatentKind>,
        actions: &Tensor<L>,
    ) -> Result<(Tensor<2>, Tensor<2>), N::Error> {
        self.network_info
            .log_prob_and_entropy(observations.clone(), actions.clone())
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
    /// Computes losses from `observations` `[batch, ...observation_shape]`, `actions`
    /// `[batch, ...action_shape]`, and scalar statistics (`old_log_probs`,
    /// `advantages`, `returns`, and `rewards`) shaped `[batch, 1]`. `old_values`
    /// is present only when value-loss clipping is enabled.
    /// `explained_variance` is `[1]`. Rewards, returns, and advantages share the environment reward dtype.
    /// Advantages are cast to the policy dtype for the surrogate.
    /// Policy actions and log probabilities retain their policy dtype. All tensors share a device.
    /// Detaches rollout estimates while retaining gradients through current model predictions.
    fn compute_loss(
        &mut self,
        observations: OS::Batch,
        actions: Tensor<L>,
        old_log_probs: Tensor<2>,
        advantages: Tensor<2>,
        returns: Tensor<2>,
        rewards: Tensor<2>,
        old_values: Option<Tensor<2>>,
        explained_variance: Tensor<1>,
        clip_range: f32,
    ) -> Result<PPOLosses, PPOError<N::Error, GE, AS::Error>> {
        let advantages = if self.normalize_advantage {
            normalize_tensor(&advantages)
        } else {
            advantages
        }
        .detach();

        let latent_observations = self
            .latent_observations(observations)
            .map_err(PPOError::NetworkError)?;

        let (log_probs, entropy) = self
            .actor_network_log_prob_and_entropy(&latent_observations, &actions)
            .map_err(PPOError::NetworkError)?;

        let log_ratio = log_probs - old_log_probs.detach();
        let ratio = log_ratio.clone().exp();
        let policy_advantages = advantages.clone().cast(ratio.dtype());
        let approx_kl = ratio.clone() - 1.0 - log_ratio;

        let actor_loss = match self.clipped {
            true => {
                let clipped_ratio = clamp_with_boundary_gradients(
                    ratio.clone(),
                    1.0 - f64::from(clip_range),
                    1.0 + f64::from(clip_range),
                );

                let surrogate1 = ratio.clone() * policy_advantages.clone();
                let surrogate2 = clipped_ratio * policy_advantages.clone();
                let ties = surrogate1.clone().equal(surrogate2.clone());
                let surrogate = surrogate1
                    .clone()
                    .min_pair(surrogate2.clone())
                    .mask_where(ties, (surrogate1 + surrogate2) * 0.5);

                -surrogate.mean()
            }
            false => {
                let surrogate = ratio * policy_advantages;
                -surrogate.mean()
            }
        };

        let values =
            self.critic_network_forward(&self.network_info, &latent_observations, rewards.dtype())?;

        let entropy_loss = entropy.mean();

        let final_actor_loss = actor_loss - entropy_loss.clone() * f64::from(self.ent_coef);

        let returns = returns.detach();
        let returns = if self.normalize_returns {
            normalize_tensor(&returns)
        } else {
            returns
        }
        .detach();

        let critic_loss = if self.clip_value_loss {
            // PPO2/CleanRL value-loss clipping: bound how far the value estimate
            // may move from its rollout-time value within one update.
            let old_values = old_values.ok_or(PPOError::MissingOldValues)?;
            clipped_value_loss(
                values,
                returns.clone(),
                old_values.detach(),
                f64::from(clip_range),
            )
        } else {
            (values - returns.clone()).square().mean()
        };
        // Match SB3 PPO: value_loss = vf_coef * mean_squared_error.
        let final_critic_loss = critic_loss * f64::from(self.vf_coef);

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
    /// Updates model parameters from actor and critic losses `[1]`.
    /// Shared training clips one norm across all models; separate training uses one combined norm or two independent norms.
    fn backpropagate_loss(
        &mut self,
        losses: PPOLosses,
    ) -> Result<(), PPOError<N::Error, GE, AS::Error>> {
        self.network_info.backpropagate_loss(
            losses.actor_loss,
            losses.critic_loss,
            self.gradient_clip,
        )?;
        Ok(())
    }

    /// Samples rank-L policy actions `[batch, ...action_shape]` from rank-H
    /// model observations `[batch, ...model_input_shape]`, preserving the batch axis. The policy defines action dtype.
    fn act_neurons(
        &mut self,
        observations: &Tensor<H, N::LatentKind>,
    ) -> Result<Tensor<L>, PPOError<N::Error, GE, AS::Error>> {
        self.network_info
            .sample(observations.clone())
            .map_err(PPOError::NetworkError)
    }

    /// Samples rank-L policy actions and maps them to rank-A environment actions from native rank-O observations.
    /// All tensors preserve the leading environment axis; log probabilities and critic values are detached `[environment_count, 1]`.
    fn sample_collection_action(
        &mut self,
        validation: &N,
        observations: &OS::Batch,
    ) -> Result<PPOCollectionAction<AS, L>, PPOError<N::Error, GE, AS::Error>> {
        let latent_observations = validation
            .forward_observations(observations.clone().without_autodiff())
            .map_err(PPOError::NetworkError)?;
        let policy_actions = validation
            .sample(latent_observations.clone())
            .map_err(PPOError::NetworkError)?;
        let environment_actions = self
            .action_space
            .tensor_from_neurons(policy_actions.clone())
            .map_err(PPOError::SpaceError)?;
        let values = validation
            .forward_values(latent_observations.clone())
            .map_err(PPOError::NetworkError)?;
        let (log_probs, _) = validation
            .log_prob_and_entropy(latent_observations, policy_actions.clone())
            .map_err(PPOError::NetworkError)?;
        Ok(PPOCollectionAction {
            policy_actions,
            environment_actions,
            // Do not retain the collection forward graph in the rollout buffer.
            log_probs: log_probs.detach(),
            values: values.detach(),
        })
    }

    fn update_learning_rates(&mut self) {
        self.network_info
            .update_learning_rates(&self.schedule_progress);
    }

    /// Logs vectorized `rewards` shaped `[env_count, 1]` alongside one metadata
    /// and termination entry per environment.
    fn log_collection(
        &mut self,
        rewards: &Tensor<2>,
        infos: Vec<I>,
        dones: &[bool],
        truncateds: &[bool],
        first_collection_timestep: usize,
    ) -> Result<(), TensorReadError> {
        let Some(logging_info) = &mut self.logging_info else {
            return Ok(());
        };

        let rewards_vec = rewards
            .clone()
            .try_into_data_as::<f32>()?
            .try_to_vec::<f32>()
            .map_err(TensorReadError::from)?;
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
        self.network_info.set_learning_rate(lr);
    }

    /// Selects the policy distribution's mode without sampling.
    ///
    /// This is intended for deterministic evaluation. PPO rollouts remain
    /// stochastic and continue to use [`Agent::act`]. `observation` is shaped
    /// `[batch, ...observation_shape]`; the returned environment action is
    /// `[batch, ...action_shape]`.
    pub fn act_deterministic(
        &mut self,
        observation: &OS::Batch,
    ) -> Result<AS::Batch, PPOError<N::Error, GE, AS::Error>> {
        let latent_observations = self
            .latent_observations(observation.clone())
            .map_err(PPOError::NetworkError)?;
        let neurons = self
            .network_info
            .mode(latent_observations)
            .map_err(PPOError::NetworkError)?;
        self.action_space
            .tensor_from_neurons(neurons)
            .map_err(PPOError::SpaceError)
    }

    /// Prepares an empty rollout buffer and episode tracking, returning the
    /// observations shaped `[environment_count, ...observation_shape]` from
    /// which collection should resume.
    fn prepare_rollout_collection(
        &mut self,
        env: &mut dyn MultiGym<I, O, A, Error = GE, ObservationSpace = OS, ActionSpace = AS>,
    ) -> Result<OS::Batch, PPOError<N::Error, GE, AS::Error>> {
        let observations = if let Some(observations) = self.current_observations.take() {
            observations
        } else {
            env.reset().map_err(PPOError::GymError)?
        };
        let environment_count = env.num_envs();
        if environment_count == 0 {
            return Err(PPOConfigurationError::ZeroEnvironments.into());
        }
        if !self.batch_size.is_multiple_of(environment_count) {
            return Err(PPOConfigurationError::BatchNotDivisibleByEnvironmentCount.into());
        }
        self.rollout_buffer.clear();
        if self.logging_info.is_some()
            && self.episode_tracker.environment_count() != environment_count
        {
            self.episode_tracker = PPOEpisodeTracker::new(environment_count);
        }
        Ok(observations)
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            name = "ppo.collect_rollout",
            target = "modurl::performance",
            skip_all
        )
    )]
    /// Fills the rollout buffer, updating `next_observations` shaped
    /// `[environment_count, ...observation_shape]` after each environment step.
    /// Uses the supplied validation view without changing the training models, including on errors.
    fn collect_rollout<const U: usize>(
        &mut self,
        validation: &N,
        env: &mut dyn MultiGym<I, O, A, Error = GE, ObservationSpace = OS, ActionSpace = AS>,
        next_observations: &mut OS::Batch,
    ) -> Result<usize, PPOError<N::Error, GE, AS::Error>>
    where
        OS: ObservationSpace<O, Item = Tensor<U, <OS as TensorBatchTypes>::Kind>>,
        Tensor<U, OS::Kind>: NextRank<Next = Tensor<O, OS::Kind>>,
    {
        let environment_count = env.num_envs();
        while self.rollout_buffer.len() * environment_count < self.batch_size {
            let observations = next_observations.clone();
            let collection_action = self.sample_collection_action(validation, &observations)?;
            let step_info = env
                .step(collection_action.environment_actions.clone())
                .map_err(PPOError::GymError)?;
            let training_next_observations = step_info.transition_next_observations();
            let MultiGymStepInfo {
                observations: reset_or_next_observations,
                rewards: collection_rewards,
                infos,
                dones: next_dones,
                truncateds,
                terminal_observations: _,
            } = step_info;
            Self::check_critic_dtype(&collection_action.values, collection_rewards.dtype())?;
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
                validation,
                collection_rewards.clone().detach(),
                &training_next_observations,
                &next_dones,
                &truncateds,
            )?;
            self.rollout_buffer.push(
                PPOExperience::builder()
                    .observations(observations.without_autodiff())
                    .next_observations(training_next_observations.without_autodiff())
                    .actions(collection_action.policy_actions.detach())
                    .rewards(collection_rewards.detach())
                    .training_rewards(training_rewards)
                    .next_dones(next_dones)
                    .truncateds(truncateds)
                    .log_probs(collection_action.log_probs)
                    .values(collection_action.values)
                    .build(),
            );
            *next_observations = reset_or_next_observations;
        }

        Ok(self.rollout_buffer.len() * environment_count)
    }

    /// Advances collection progress and trains with explained variance `[1]` from detached rollout values.
    fn train_on_collected_rollout<const BO: usize, const BL: usize>(
        &mut self,
        collected_timesteps: usize,
        rollout_explained_variance: Tensor<1>,
    ) -> Result<(), PPOError<N::Error, GE, AS::Error>>
    where
        Tensor<O, OS::Kind>: NextRank<Next = Tensor<BO, OS::Kind>>,
        Tensor<L>: NextRank<Next = Tensor<BL>>,
    {
        self.schedule_progress.advance_steps(collected_timesteps);
        self.update_learning_rates();
        if let Some(logging_info) = &mut self.logging_info {
            logging_info.timestep = self.schedule_progress.elapsed_steps();
        }
        self.optimize(rollout_explained_variance)
    }
}

impl<
    'a,
    N,
    OS,
    AS,
    GE,
    I,
    const O: usize,
    const H: usize,
    const L: usize,
    const A: usize,
    const BO: usize,
    const BL: usize,
    const U: usize,
> Agent<I, O, A>
    for PPOAgent<
        'a,
        PPOModelGroup<N, Tensor<O, OS::Kind>, Tensor<H, N::LatentKind>, Tensor<L>>,
        OS,
        AS,
        GE,
        I,
        A,
    >
where
    N: PPONetwork<O, H, L, InputKind = OS::Kind>,
    OS: ObservationSpace<O, Item = Tensor<U, <OS as crate::tensor_rank::TensorBatchTypes>::Kind>>,
    AS: ActionMap<L, A>,
    AS::Error: std::fmt::Debug,
    OS::Kind: Autodiff,
    Tensor<O, OS::Kind>: PrevRank + NextRank,
    Tensor<L>: NextRank,
    Tensor<O, OS::Kind>: NextRank<Next = Tensor<BO, OS::Kind>>,
    Tensor<L>: NextRank<Next = Tensor<BL>>,
    Tensor<U, OS::Kind>: NextRank<Next = Tensor<O, OS::Kind>>,
    GE: std::fmt::Debug,
{
    type Error = PPOError<N::Error, GE, AS::Error>;
    type GymError = GE;
    type SpaceError = AS::Error;
    type ObservationSpace = OS;
    type ActionSpace = AS;

    /// Selects environment actions `[batch, ...action_shape]` for observations
    /// `[batch, ...observation_shape]`.
    fn act(&mut self, observation: &OS::Batch) -> Result<AS::Batch, Self::Error> {
        let latent_observations = self
            .latent_observations(observation.clone())
            .map_err(PPOError::NetworkError)?;
        let neurons = self.act_neurons(&latent_observations)?;
        let actions = self
            .action_space
            .tensor_from_neurons(neurons)
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
    /// Trains with rank-O observation batches `[num_envs, ...observation_shape]`.
    /// Terminal observations omit the batch axis and preserve the environment observation kind, dtype, and device.
    fn learn(
        &mut self,
        env: &mut dyn MultiGym<I, O, A, Error = GE, ObservationSpace = OS, ActionSpace = AS>,
        num_timesteps: usize,
    ) -> Result<(), PPOError<N::Error, GE, AS::Error>> {
        let mut elapsed_timesteps = 0;
        let mut next_observations = self.prepare_rollout_collection(env)?;

        while elapsed_timesteps < num_timesteps {
            let (collected_timesteps, rollout_explained_variance) = {
                let validation = self.network_info.valid();
                let collected_timesteps =
                    self.collect_rollout(&validation, env, &mut next_observations)?;
                let rollout_explained_variance = self.add_advantages_and_returns(&validation)?;
                (collected_timesteps, rollout_explained_variance)
            };
            elapsed_timesteps += collected_timesteps;
            self.train_on_collected_rollout(collected_timesteps, rollout_explained_variance)?;
        }
        self.current_observations = Some(next_observations);
        Ok(())
    }
}

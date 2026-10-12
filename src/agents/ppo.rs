use bon::bon;
use burn::{
    module::{Module, ModuleVisitor, Param},
    optim::{GradientsParams, ModuleOptimizer},
    tensor::{
        Bool, Cast, DType, Device, Float, FloatDType, Int, Tensor, TensorData, TensorReadError,
        kind::Basic,
    },
};
use std::marker::PhantomData;

use crate::{
    buffers::{
        experience::{self, Experience, ExperienceBatchError},
        rollout_buffer::RolloutBuffer,
    },
    gym::{MultiGym, MultiGymStepInfo},
    models::{Forward, probabilistic_model::ProbabilisticPolicy},
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
    #[error("PPO requires a floating-point dtype")]
    InvalidDType,
    #[error("the environment count must be nonzero")]
    ZeroEnvironments,
    #[error("batch size must be divisible by the environment count")]
    BatchNotDivisibleByEnvironmentCount,
    #[error(
        "separate networks require observation rank {observation_rank} to match model input rank {model_rank}"
    )]
    SeparateObservationRank {
        observation_rank: usize,
        model_rank: usize,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum PPOError<AE, GE, SE>
where
    AE: std::fmt::Debug,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
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
    prepared: Option<PPOPreparedExperience>,
}

struct PPOCollectionAction<const L: usize, const A: usize, K: Basic> {
    policy_actions: Tensor<L>,
    environment_actions: Tensor<A, K>,
    log_probs: Tensor<2>,
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
    next_dones: Tensor<3, Bool>,
    truncateds: Tensor<3, Bool>,
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

/// Losses, mean entropy, and explained variance use Float `[1]` tensors.
/// Minibatch rewards, returns, advantages, and KL estimates use `[batch_size, 1]`.
/// Rewards retain the environment dtype. Other tensors use PPO's compute dtype and model device.
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

struct PPOGradientSelection<'a> {
    gradients: &'a GradientsParams,
    selected: GradientsParams,
}

impl ModuleVisitor for PPOGradientSelection<'_> {
    /// Copies a parameter's rank-D gradient into the actor's store, preserving shape, dtype, and device.
    /// Keeps the combined store intact so shared parameter IDs remain available to the critic optimizer.
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        if let Some(gradient) = self.gradients.get::<D>(param.id) {
            self.selected.register(param.id, gradient);
        }
    }
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

/// Supplies the unused shared-model type for separate actor and critic configurations.
/// Contains no parameters. Executing this placeholder violates the separate-network contract.
#[derive(Module, Debug)]
pub struct FakeModule;

impl<const I: usize, const O: usize> Forward<I, O> for FakeModule {
    type Error = std::convert::Infallible;

    /// Rejects execution with any rank-I Float input instead of producing a rank-O output.
    /// Separate configurations never execute the shared-model placeholder.
    fn forward(&self, _input: Tensor<I>) -> Result<Tensor<O>, Self::Error> {
        panic!("FakeModule must not execute a shared model");
    }
}

/// Selects an owned shared or separate actor-critic configuration.
/// `T` defaults to a parameter-free placeholder because separate networks have no shared model.
pub enum PPONetworkInfo<A: Module, C: Module, T: Module = FakeModule> {
    Shared(SharedPPONetwork<T, A, C>),
    Separate(SeparatePPONetwork<A, C>),
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

/// Groups the owned network configuration and native execution tensor types without adding execution methods.
pub trait PPOModelTypes {
    type Network;
    type Observation: NextRank + PrevRank;
    type Action: NextRank;
    type Error: std::fmt::Debug;
}

/// A type-only group for the actor, critic, shared model, and native observation, latent, and policy-action tensors.
/// PPO stores the actual models and tensors directly; this group contains no runtime values.
pub struct PPOModelGroup<A, C, T = FakeModule, O = Tensor<2>, H = O, L = Tensor<2>> {
    types: PhantomData<(A, C, T, O, H, L)>,
}

#[derive(Debug, thiserror::Error)]
pub enum PPONetworkError<AE: std::fmt::Debug, CE: std::fmt::Debug, TE: std::fmt::Debug> {
    #[error("policy failed: {0}")]
    Policy(#[source] AE),
    #[error("critic failed: {0}")]
    Critic(#[source] CE),
    #[error("shared model failed: {0}")]
    Shared(#[source] TE),
}

impl<A, C, T, const O: usize, const H: usize, const L: usize> PPOModelTypes
    for PPOModelGroup<A, C, T, Tensor<O>, Tensor<H>, Tensor<L>>
where
    A: Module + ProbabilisticPolicy<H, L, ObservationKind = Float>,
    C: Module + Forward<H, 2>,
    T: Module + Forward<O, H>,
    A::Error: std::fmt::Debug,
    C::Error: std::fmt::Debug,
    T::Error: std::fmt::Debug,
    Tensor<O>: NextRank + PrevRank,
    Tensor<L>: NextRank,
{
    type Network = PPONetworkInfo<A, C, T>;
    type Observation = Tensor<O>;
    type Action = Tensor<L>;
    type Error = PPONetworkError<A::Error, C::Error, T::Error>;
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
    rollout_buffer: RolloutBuffer<PPOExperience<G::Observation, G::Action>>,
    mini_batch_size: usize,
    num_epochs: usize,
    batch_size: usize,
    schedule_progress: ScheduleProgress,
    logging_info: Option<PPOLoggingInfo<'a, I>>,
    gradient_clip: f32,
    current_observations: Option<G::Observation>,
    episode_tracker: PPOEpisodeTracker,
    dtype: DType,
    _phantom: PhantomData<GE>,
}

#[bon]
impl<
    'a,
    P,
    C,
    T,
    OS,
    AS,
    GE,
    I,
    K: Basic,
    NE: std::fmt::Debug,
    const O: usize,
    const H: usize,
    const L: usize,
    const A: usize,
    const BO: usize,
    const BL: usize,
> PPOAgent<'a, PPOModelGroup<P, C, T, Tensor<O>, Tensor<H>, Tensor<L>>, OS, AS, GE, I, A>
where
    P: Module + ProbabilisticPolicy<H, L, ObservationKind = Float>,
    C: Module + Forward<H, 2>,
    T: Module + Forward<O, H>,
    P::Error: std::fmt::Debug,
    C::Error: std::fmt::Debug,
    T::Error: std::fmt::Debug,
    NE: From<PPONetworkError<P::Error, C::Error, T::Error>>,
    PPOModelGroup<P, C, T, Tensor<O>, Tensor<H>, Tensor<L>>: PPOModelTypes<
            Network = PPONetworkInfo<P, C, T>,
            Observation = Tensor<O>,
            Action = Tensor<L>,
            Error = NE,
        >,
    OS: ObservationSpace<O, Kind = K>,
    AS: ActionMap<L, A>,
    AS::Error: std::fmt::Debug,
    FloatDType: Cast<O, K, OutputKind = Float>,
    Tensor<O, K>: PrevRank,
    Tensor<O>: PrevRank + NextRank<Next = Tensor<BO>>,
    Tensor<L>: NextRank<Next = Tensor<BL>>,
    GE: std::fmt::Debug,
{
    /// Builds PPO for rank-O observation batches and rank-A environment-action batches.
    /// Casts observations to the compute dtype. Policy inputs have rank H; original policy actions have rank L.
    /// Shared models may change observation rank. Separate models receive observations directly and require H = O.
    /// Models and their outputs must share the compute dtype and device. Construction rejects invalid configuration.
    #[builder]
    pub fn new(
        observation_space: OS,
        action_space: AS,
        network_info: PPONetworkInfo<P, C, T>,
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
        device: Device,
        #[builder(default = DType::F32)] dtype: DType,
    ) -> Result<Self, PPOConfigurationError> {
        const {
            assert!(
                O >= 2 && A >= 2,
                "observation and environment-action ranks require batch and item axes"
            );
        }
        if const { O != H } && matches!(network_info, PPONetworkInfo::Separate(_)) {
            return Err(PPOConfigurationError::SeparateObservationRank {
                observation_rank: O,
                model_rank: H,
            });
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
            observation_space,
            action_space,
            rollout_buffer: RolloutBuffer::new(0, device),
            num_epochs,
            batch_size,
            schedule_progress: ScheduleProgress::new(training_horizon),
            mini_batch_size,
            logging_info: logging_info.map(PPOLoggingInfo::new),
            gradient_clip,
            current_observations: None,
            episode_tracker: PPOEpisodeTracker::new(0),
            dtype,
            _phantom: PhantomData,
        })
    }
}

#[bon]
impl<
    'a,
    P,
    C,
    T,
    OS,
    AS,
    GE,
    I,
    K: Basic,
    NE: std::fmt::Debug,
    const O: usize,
    const H: usize,
    const L: usize,
    const A: usize,
    const BO: usize,
    const BL: usize,
    const U: usize,
> PPOAgent<'a, PPOModelGroup<P, C, T, Tensor<O>, Tensor<H>, Tensor<L>>, OS, AS, GE, I, A>
where
    P: Module + ProbabilisticPolicy<H, L, ObservationKind = Float>,
    C: Module + Forward<H, 2>,
    T: Module + Forward<O, H>,
    P::Error: std::fmt::Debug,
    C::Error: std::fmt::Debug,
    T::Error: std::fmt::Debug,
    NE: From<PPONetworkError<P::Error, C::Error, T::Error>>,
    PPOModelGroup<P, C, T, Tensor<O>, Tensor<H>, Tensor<L>>: PPOModelTypes<
            Network = PPONetworkInfo<P, C, T>,
            Observation = Tensor<O>,
            Action = Tensor<L>,
            Error = NE,
        >,
    OS: ObservationSpace<O, Kind = K>,
    AS: ActionMap<L, A>,
    AS::Error: std::fmt::Debug,
    FloatDType: Cast<O, K, OutputKind = Float>,
    Tensor<O, K>: PrevRank<Prev = Tensor<U, K>>,
    Tensor<U, K>: NextRank<Next = Tensor<O, K>>,
    Tensor<O>: PrevRank + NextRank<Next = Tensor<BO>>,
    Tensor<L>: NextRank<Next = Tensor<BL>>,
    GE: std::fmt::Debug,
{
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(name = "ppo.optimize", target = "modurl::performance", skip_all)
    )]
    fn optimize(&mut self) -> Result<(), PPOError<NE, GE, AS::Error>> {
        let rollout_explained_variance = self.add_advantages_and_returns()?;

        let batch = PPOExperience::<Tensor<O>, Tensor<L>>::batch(self.rollout_buffer.get_raw())?;
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
    /// Temporarily combines time and environment axes for rank-O observation evaluation, then restores both axes.
    fn add_advantages_and_returns(&mut self) -> Result<Tensor<1>, PPOError<NE, GE, AS::Error>> {
        let batch = PPOExperience::<Tensor<O>, Tensor<L>>::batch(self.rollout_buffer.get_raw())?;
        let observations = batch.observations;
        let (batch_size, env_count) = (observations.dims()[0], observations.dims()[1]);
        let observations = observations.flatten::<O>(0, 1);
        let latent_observations = self.latent_observations(observations)?;

        let values_tensor = self.critic_network_forward(&latent_observations)?.detach();

        // Unflatten back to [batch_size, env_count, ...].
        let values_tensor = values_tensor.reshape([batch_size, env_count, 1]);

        let next_observations_tensor = batch.next_observations;
        let bootstrapped_observations = next_observations_tensor
            .slice_dim(0, batch_size - 1..batch_size)
            .squeeze_dim::<O>(0);
        let latent_bootstrapped_observations =
            self.latent_observations(bootstrapped_observations)?;

        let bootstrapped_values = self
            .critic_network_forward(&latent_bootstrapped_observations)?
            .detach(); // shape [env_count, 1]

        let advantages = self
            .compute_gae(
                &batch.training_rewards,
                &values_tensor,
                &batch.next_dones,
                &batch.truncateds,
                &bootstrapped_values,
            )
            .detach();

        let returns = (values_tensor.clone() + advantages.clone()).detach();
        let rollout_explained_variance = compute_explained_variance(&returns, &values_tensor);

        let experiences = self.rollout_buffer.get_raw_mut();

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
                // values_tensor comes from the pre-update network, i.e. the
                // collection-time values PPO2 clips the new values against.
                old_values: if self.clip_value_loss {
                    Some(
                        values_tensor
                            .clone()
                            .slice_dim(0, i..i + 1)
                            .squeeze_dim(0)
                            .detach(),
                    )
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
    /// Masks use Bool tensors. Float inputs share dtype and device; rewards and bootstrap values are cast to the compute dtype.
    /// Detaches critic values and bootstrap values. Collection already includes terminal-value corrections in truncation rewards.
    fn compute_gae(
        &self,
        rewards: &Tensor<3>,
        values: &Tensor<3>,
        next_dones: &Tensor<3, Bool>,
        next_truncateds: &Tensor<3, Bool>,
        bootstrapped_values: &Tensor<2>,
    ) -> Tensor<3> {
        let device = values.device();
        let rewards = rewards.clone().cast(self.dtype);
        let next_dones = next_dones.clone().float().cast(self.dtype);
        let next_truncateds = next_truncateds.clone().float().cast(self.dtype);
        let bootstrapped_values = bootstrapped_values.clone().cast(self.dtype);
        let values = values.clone().detach();
        let masks = (1.0f64 - next_dones) * (1.0f64 - next_truncateds);
        let mut next_value = bootstrapped_values.detach();
        let mut gae = Tensor::zeros(next_value.dims(), (&device, self.dtype));
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
        observations: Tensor<O>,
    ) -> Result<Tensor<H>, PPOError<NE, GE, AS::Error>> {
        match &self.network_info {
            PPONetworkInfo::Shared(network) => network
                .shared_network
                .forward(observations)
                .map_err(|error| PPOError::NetworkError(NE::from(PPONetworkError::Shared(error)))),
            PPONetworkInfo::Separate(_) => {
                if const { O == H } {
                    // Copy the identical axis sizes into the model's rank-H array without adding or removing axes.
                    let mut shape = [0; H];
                    shape.copy_from_slice(&observations.dims());
                    Ok(observations.reshape(shape))
                } else {
                    unreachable!("separate observation rank is checked during construction")
                }
            }
        }
    }

    /// Forwards latent or raw `observations` shaped `[batch, ...model_input_shape]` of rank H and
    /// returns critic values shaped `[batch, 1]`.
    fn critic_network_forward(
        &mut self,
        observations: &Tensor<H>,
    ) -> Result<Tensor<2>, PPOError<NE, GE, AS::Error>> {
        match self.network_info {
            PPONetworkInfo::Shared(ref mut shared_info) => {
                let values = shared_info
                    .critic_head
                    .forward(observations.clone())
                    .map_err(|error| {
                        PPOError::NetworkError(NE::from(PPONetworkError::Critic(error)))
                    })?;
                Ok(values)
            }
            PPONetworkInfo::Separate(ref mut separate_info) => {
                let values = separate_info
                    .critic_network
                    .forward(observations.clone())
                    .map_err(|error| {
                        PPOError::NetworkError(NE::from(PPONetworkError::Critic(error)))
                    })?;
                Ok(values)
            }
        }
    }

    /// Adds the bootstrap value for each truncated transition to its reward.
    /// GAE still stops at the truncation boundary, so the terminal value is
    /// included exactly once.
    /// `rewards` is `[environment_count, 1]`, rank-O `transition_next_observations` is
    /// `[environment_count, ...observation_shape]`, and both boolean slices have
    /// `environment_count` entries. A true termination always suppresses
    /// bootstrapping, even if the transition is also truncated.
    fn bootstrap_truncated_rewards(
        &mut self,
        rewards: &Tensor<2>,
        transition_next_observations: &Tensor<O>,
        dones: &[bool],
        truncateds: &[bool],
    ) -> Result<Tensor<2>, PPOError<NE, GE, AS::Error>> {
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
            return Ok(rewards.clone().cast(self.dtype));
        }

        let truncation_count = truncated_indices.len();
        let truncated_indices = Tensor::<1, Int>::from_data(
            TensorData::new(truncated_indices, [truncation_count]),
            (&transition_next_observations.device(), DType::I64),
        );
        let truncated_next_observations = transition_next_observations
            .clone()
            .select(0, truncated_indices.clone());
        let latent_observations = self.latent_observations(truncated_next_observations)?;
        let terminal_values = self.critic_network_forward(&latent_observations)?.detach();
        let terminal_bootstrap = terminal_values.cast(self.dtype) * f64::from(self.gamma);
        Ok(rewards.clone().cast(self.dtype).select_assign(
            0,
            truncated_indices,
            terminal_bootstrap,
            burn::tensor::IndexingUpdateOp::Add,
        ))
    }

    /// Evaluates rank-L `actions` `[batch, ...action_shape]` for latent or raw
    /// rank-H `observations` `[batch, ...model_input_shape]`, returning log probabilities and entropy `[batch, 1]`.
    /// Inputs share the model's Float dtype and device; evaluation retains parameter gradients.
    fn actor_network_log_prob_and_entropy(
        &mut self,
        observations: &Tensor<H>,
        actions: &Tensor<L>,
    ) -> Result<(Tensor<2>, Tensor<2>), PPOError<NE, GE, AS::Error>> {
        match self.network_info {
            PPONetworkInfo::Shared(ref mut shared_info) => {
                let (log_probs, entropy) = shared_info
                    .actor_head
                    .log_prob_and_entropy(observations.clone(), actions.clone())
                    .map_err(|error| {
                        PPOError::NetworkError(NE::from(PPONetworkError::Policy(error)))
                    })?;
                Ok((log_probs, entropy))
            }
            PPONetworkInfo::Separate(ref mut separate_info) => {
                let (log_probs, entropy) = separate_info
                    .actor_network
                    .log_prob_and_entropy(observations.clone(), actions.clone())
                    .map_err(|error| {
                        PPOError::NetworkError(NE::from(PPONetworkError::Policy(error)))
                    })?;
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
    /// Computes losses from `observations` `[batch, ...observation_shape]`, `actions`
    /// `[batch, ...action_shape]`, and scalar statistics (`old_log_probs`,
    /// `advantages`, `returns`, and `rewards`) shaped `[batch, 1]`. `old_values`
    /// is present only when value-loss clipping is enabled.
    /// `explained_variance` is `[1]`. Rewards retain the environment dtype; other Float inputs share the compute dtype and model device.
    /// Detaches rollout estimates while retaining gradients through current model predictions.
    fn compute_loss(
        &mut self,
        observations: Tensor<O>,
        actions: Tensor<L>,
        old_log_probs: Tensor<2>,
        advantages: Tensor<2>,
        returns: Tensor<2>,
        rewards: Tensor<2>,
        old_values: Option<Tensor<2>>,
        explained_variance: Tensor<1>,
        clip_range: f32,
    ) -> Result<PPOLosses, PPOError<NE, GE, AS::Error>> {
        let advantages = if self.normalize_advantage {
            normalize_tensor(&advantages)
        } else {
            advantages
        }
        .detach();

        let latent_observations = self.latent_observations(observations)?;

        let (log_probs, entropy) =
            self.actor_network_log_prob_and_entropy(&latent_observations, &actions)?;

        let log_ratio = log_probs - old_log_probs.detach();
        let ratio = log_ratio.clone().exp();
        let approx_kl = ratio.clone() - 1.0 - log_ratio;

        let actor_loss = match self.clipped {
            true => {
                let clipped_ratio = clamp_with_boundary_gradients(
                    ratio.clone(),
                    1.0 - f64::from(clip_range),
                    1.0 + f64::from(clip_range),
                );

                let surrogate1 = ratio.clone() * advantages.clone();
                let surrogate2 = clipped_ratio * advantages.clone();
                let ties = surrogate1.clone().equal(surrogate2.clone());
                let surrogate = surrogate1
                    .clone()
                    .min_pair(surrogate2.clone())
                    .mask_where(ties, (surrogate1 + surrogate2) * 0.5);

                -surrogate.mean()
            }
            false => {
                let surrogate = ratio * advantages.clone();
                -surrogate.mean()
            }
        };

        let values = self.critic_network_forward(&latent_observations)?;

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
    fn backpropagate_loss(&mut self, losses: PPOLosses) -> Result<(), PPOError<NE, GE, AS::Error>> {
        let actor_loss = losses.actor_loss;
        let critic_loss = losses.critic_loss;

        match self.network_info {
            PPONetworkInfo::Shared(ref mut shared_info) => {
                let total_loss = actor_loss + critic_loss;
                let models = (
                    shared_info.shared_network.clone(),
                    shared_info.actor_head.clone(),
                    shared_info.critic_head.clone(),
                );
                let mut total_grad = GradientsParams::from_grads(total_loss.backward(), &models);
                let _total_grad_norm = crate::tensor_operations::clip_gradients(
                    &models,
                    &mut total_grad,
                    self.gradient_clip,
                )?;
                let (trunk, actor, critic) =
                    shared_info
                        .optimizer
                        .step(shared_info.learning_rate, models, total_grad);
                shared_info.shared_network = trunk;
                shared_info.actor_head = actor;
                shared_info.critic_head = critic;
            }
            PPONetworkInfo::Separate(ref mut separate_info) => match separate_info.combined_loss {
                true => {
                    let total_loss = actor_loss + critic_loss;
                    let models = (
                        separate_info.actor_network.clone(),
                        separate_info.critic_network.clone(),
                    );
                    let mut total_grad =
                        GradientsParams::from_grads(total_loss.backward(), &models);
                    let _total_grad_norm = crate::tensor_operations::clip_gradients(
                        &models,
                        &mut total_grad,
                        self.gradient_clip,
                    )?;
                    let mut actor_gradients = PPOGradientSelection {
                        gradients: &total_grad,
                        selected: GradientsParams::new(),
                    };
                    separate_info.actor_network.visit(&mut actor_gradients);
                    separate_info.actor_network = separate_info.actor_optimizer.step(
                        separate_info.actor_learning_rate,
                        separate_info.actor_network.clone(),
                        actor_gradients.selected,
                    );
                    separate_info.critic_network = separate_info.critic_optimizer.step(
                        separate_info.critic_learning_rate,
                        separate_info.critic_network.clone(),
                        total_grad,
                    );
                }
                false => {
                    let mut actor_grad = GradientsParams::from_grads(
                        actor_loss.backward(),
                        &separate_info.actor_network,
                    );
                    let _actor_grad_norm = crate::tensor_operations::clip_gradients(
                        &separate_info.actor_network,
                        &mut actor_grad,
                        self.gradient_clip,
                    )?;
                    separate_info.actor_network = separate_info.actor_optimizer.step(
                        separate_info.actor_learning_rate,
                        separate_info.actor_network.clone(),
                        actor_grad,
                    );

                    let mut critic_grad = GradientsParams::from_grads(
                        critic_loss.backward(),
                        &separate_info.critic_network,
                    );
                    let _critic_grad_norm = crate::tensor_operations::clip_gradients(
                        &separate_info.critic_network,
                        &mut critic_grad,
                        self.gradient_clip,
                    )?;
                    separate_info.critic_network = separate_info.critic_optimizer.step(
                        separate_info.critic_learning_rate,
                        separate_info.critic_network.clone(),
                        critic_grad,
                    );
                }
            },
        }

        Ok(())
    }

    /// Samples rank-L policy actions `[batch, ...action_shape]` from rank-H
    /// model observations `[batch, ...model_input_shape]`, preserving the batch axis and compute dtype.
    fn act_neurons(
        &mut self,
        observations: &Tensor<H>,
    ) -> Result<Tensor<L>, PPOError<NE, GE, AS::Error>> {
        let network = match self.network_info {
            PPONetworkInfo::Shared(ref mut shared_info) => &shared_info.actor_head,
            PPONetworkInfo::Separate(ref mut separate_info) => &separate_info.actor_network,
        };

        network
            .sample(observations.clone())
            .map_err(|error| PPOError::NetworkError(NE::from(PPONetworkError::Policy(error))))
    }

    /// Samples rank-L policy actions and maps them to rank-A environment actions from rank-O Float observations.
    /// All tensors preserve the leading environment axis; log probabilities are detached `[environment_count, 1]`.
    fn sample_collection_action(
        &mut self,
        observations: &Tensor<O>,
    ) -> Result<PPOCollectionAction<L, A, AS::Kind>, PPOError<NE, GE, AS::Error>> {
        let latent_observations = self.latent_observations(observations.clone())?;
        let policy_actions = self.act_neurons(&latent_observations)?;
        let environment_actions = self
            .action_space
            .tensor_from_neurons(policy_actions.clone())
            .map_err(PPOError::SpaceError)?;
        let (log_probs, _) =
            self.actor_network_log_prob_and_entropy(&latent_observations, &policy_actions)?;
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
                    shared_info.learning_rate = new_lr;
                }
            }
            PPONetworkInfo::Separate(ref mut separate_info) => {
                if let Some(actor_lr_scheduler) = &separate_info.actor_lr_scheduler {
                    let new_lr = self.schedule_progress.parameter(&**actor_lr_scheduler);
                    separate_info.actor_learning_rate = new_lr;
                }
                if let Some(critic_lr_scheduler) = &separate_info.critic_lr_scheduler {
                    let new_lr = self.schedule_progress.parameter(&**critic_lr_scheduler);
                    separate_info.critic_learning_rate = new_lr;
                }
            }
        }
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
        match self.network_info {
            PPONetworkInfo::Shared(ref mut shared_info) => {
                shared_info.learning_rate = lr;
            }
            PPONetworkInfo::Separate(ref mut separate_info) => {
                separate_info.actor_learning_rate = lr;
                separate_info.critic_learning_rate = lr;
            }
        }
    }

    pub fn get_observation_space(&self) -> &OS {
        &self.observation_space
    }

    pub fn get_action_space(&self) -> &AS {
        &self.action_space
    }

    /// Selects the policy distribution's mode without sampling.
    ///
    /// This is intended for deterministic evaluation. PPO rollouts remain
    /// stochastic and continue to use [`Agent::act`]. `observation` is shaped
    /// `[batch, ...observation_shape]`; the returned environment action is
    /// `[batch, ...action_shape]`.
    pub fn act_deterministic(
        &mut self,
        observation: &Tensor<O, K>,
    ) -> Result<Tensor<A, AS::Kind>, PPOError<NE, GE, AS::Error>> {
        let observation = FloatDType::cast(observation.clone(), FloatDType::from(self.dtype));
        let latent_observations = self.latent_observations(observation)?;
        let network = match self.network_info {
            PPONetworkInfo::Shared(ref mut shared_info) => &shared_info.actor_head,
            PPONetworkInfo::Separate(ref mut separate_info) => &separate_info.actor_network,
        };
        let neurons = network
            .mode(latent_observations)
            .map_err(|error| PPOError::NetworkError(NE::from(PPONetworkError::Policy(error))))?;
        self.action_space
            .tensor_from_neurons(neurons)
            .map_err(PPOError::SpaceError)
    }

    pub fn reset_current_observations(&mut self) {
        self.current_observations = None;
        self.episode_tracker.clear();
    }

    /// Prepares an empty rollout buffer and episode tracking, returning the
    /// observations shaped `[environment_count, ...observation_shape]` from
    /// which collection should resume.
    fn prepare_rollout_collection(
        &mut self,
        env: &mut dyn MultiGym<I, O, A, Error = GE, ObservationSpace = OS, ActionSpace = AS>,
    ) -> Result<Tensor<O>, PPOError<NE, GE, AS::Error>> {
        let observations = if let Some(observations) = self.current_observations.take() {
            observations
        } else {
            FloatDType::cast(
                env.reset().map_err(PPOError::GymError)?,
                FloatDType::from(self.dtype),
            )
        }
        .cast(self.dtype);
        let environment_count = env.num_envs();
        if environment_count == 0 {
            return Err(PPOConfigurationError::ZeroEnvironments.into());
        }
        if !self.batch_size.is_multiple_of(environment_count) {
            return Err(PPOConfigurationError::BatchNotDivisibleByEnvironmentCount.into());
        }
        self.rollout_buffer = RolloutBuffer::new(
            self.mini_batch_size / environment_count,
            observations.device(),
        );
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
    fn collect_rollout(
        &mut self,
        env: &mut dyn MultiGym<I, O, A, Error = GE, ObservationSpace = OS, ActionSpace = AS>,
        next_observations: &mut Tensor<O>,
    ) -> Result<usize, PPOError<NE, GE, AS::Error>> {
        let environment_count = env.num_envs();
        while self.rollout_buffer.len() * environment_count < self.batch_size {
            let observations = next_observations.clone();
            let collection_action = self.sample_collection_action(&observations)?;
            let step_info = env
                .step(collection_action.environment_actions.clone())
                .map_err(PPOError::GymError)?;
            let training_next_observations = FloatDType::cast(
                step_info.transition_next_observations(),
                FloatDType::from(self.dtype),
            );
            let MultiGymStepInfo {
                observations: reset_or_next_observations,
                rewards: collection_rewards,
                infos,
                dones: next_dones,
                truncateds,
                terminal_observations: _,
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
                &training_next_observations,
                &next_dones,
                &truncateds,
            )?;
            self.rollout_buffer.add(
                PPOExperience::builder()
                    .observations(observations.detach())
                    .next_observations(training_next_observations.detach())
                    .actions(collection_action.policy_actions.detach())
                    .rewards(collection_rewards.detach())
                    .training_rewards(training_rewards.detach())
                    .next_dones(next_dones)
                    .truncateds(truncateds)
                    .log_probs(collection_action.log_probs)
                    .build(),
            );
            *next_observations =
                FloatDType::cast(reset_or_next_observations, FloatDType::from(self.dtype));
        }

        Ok(self.rollout_buffer.len() * environment_count)
    }

    fn train_on_collected_rollout(
        &mut self,
        collected_timesteps: usize,
    ) -> Result<(), PPOError<NE, GE, AS::Error>> {
        self.schedule_progress.advance_steps(collected_timesteps);
        self.update_learning_rates();
        if let Some(logging_info) = &mut self.logging_info {
            logging_info.timestep = self.schedule_progress.elapsed_steps();
        }
        self.optimize()
    }
}

impl<
    'a,
    P,
    C,
    T,
    OS,
    AS,
    GE,
    I,
    K: Basic,
    NE: std::fmt::Debug,
    const O: usize,
    const H: usize,
    const L: usize,
    const A: usize,
    const BO: usize,
    const BL: usize,
    const U: usize,
> Agent<I, O, A>
    for PPOAgent<'a, PPOModelGroup<P, C, T, Tensor<O>, Tensor<H>, Tensor<L>>, OS, AS, GE, I, A>
where
    P: Module + ProbabilisticPolicy<H, L, ObservationKind = Float>,
    C: Module + Forward<H, 2>,
    T: Module + Forward<O, H>,
    P::Error: std::fmt::Debug,
    C::Error: std::fmt::Debug,
    T::Error: std::fmt::Debug,
    PPOModelGroup<P, C, T, Tensor<O>, Tensor<H>, Tensor<L>>: PPOModelTypes<
            Network = PPONetworkInfo<P, C, T>,
            Observation = Tensor<O>,
            Action = Tensor<L>,
            Error = NE,
        >,
    OS: ObservationSpace<O, Kind = K>,
    AS: ActionMap<L, A>,
    AS::Error: std::fmt::Debug,
    FloatDType: Cast<O, K, OutputKind = Float>,
    Tensor<O, K>: PrevRank<Prev = Tensor<U, K>>,
    Tensor<U, K>: NextRank<Next = Tensor<O, K>>,
    Tensor<O>: PrevRank + NextRank<Next = Tensor<BO>>,
    Tensor<L>: NextRank<Next = Tensor<BL>>,
    NE: From<PPONetworkError<P::Error, C::Error, T::Error>>,
    GE: std::fmt::Debug,
{
    type Error = PPOError<NE, GE, AS::Error>;
    type GymError = GE;
    type SpaceError = AS::Error;
    type ObservationSpace = OS;
    type ActionSpace = AS;

    /// Selects environment actions `[batch, ...action_shape]` for observations
    /// `[batch, ...observation_shape]`.
    fn act(&mut self, observation: &Tensor<O, K>) -> Result<Tensor<A, AS::Kind>, Self::Error> {
        let observation = FloatDType::cast(observation.clone(), FloatDType::from(self.dtype));
        let latent_observations = self.latent_observations(observation)?;
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
    fn learn(
        &mut self,
        env: &mut dyn MultiGym<I, O, A, Error = GE, ObservationSpace = OS, ActionSpace = AS>,
        num_timesteps: usize,
    ) -> Result<(), PPOError<NE, GE, AS::Error>> {
        let mut elapsed_timesteps = 0;
        let mut next_observations = self.prepare_rollout_collection(env)?;

        while elapsed_timesteps < num_timesteps {
            let collected_timesteps = self.collect_rollout(env, &mut next_observations)?;
            elapsed_timesteps += collected_timesteps;
            self.train_on_collected_rollout(collected_timesteps)?;
        }
        self.current_observations = Some(next_observations);
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

    use super::{
        FakeModule, PPOConfigurationError, PPONetworkInfo, SeparatePPONetwork, SharedPPONetwork,
    };

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
        let network = PPONetworkInfo::Shared(network);
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
        let network: PPONetworkInfo<Linear, Linear> = PPONetworkInfo::Separate(network);
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
        assert_eq!(FakeModule.num_params(), 0);

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

    use burn::{
        module::ParamId,
        nn::{Initializer, Tanh},
        optim::SgdConfig,
    };
    use std::convert::Infallible;

    use super::*;
    use crate::{
        agents::Agent,
        distributions::{CategoricalDistribution, GaussianDistribution},
        gym::{Gym, MultiGym, ResetInfo, StepInfo, VectorizedGymWrapper},
        models::{MLP, probabilistic_model::ProbabilisticPolicyModel},
        parameter_schedule::LinearSchedule,
        spaces::{BoxSpace, Discrete},
    };

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    fn mlp(input: usize, output: usize, device: &Device) -> MLP {
        MLP::builder()
            .input_size(input)
            .output_size(output)
            .hidden_layer_sizes(vec![2])
            .activation(Tanh)
            .hidden_initializer(Initializer::Constant { value: 0.1 })
            .output_initializer(Initializer::Constant { value: 0.1 })
            .bias_initializer(Initializer::Constant { value: 0.0 })
            .options((device, DType::F32))
            .build()
            .unwrap()
    }

    #[derive(Default)]
    struct NoopLogger {
        updates: usize,
    }

    impl<I> PPOLogger<I> for NoopLogger {
        fn log(&mut self, _info: &PPOLogEntry) {
            self.updates += 1;
        }
    }

    #[test]
    fn shared_model_can_change_observation_rank() {
        #[derive(Module, Debug)]
        struct FlattenModel;

        impl Forward<3, 2> for FlattenModel {
            type Error = Infallible;

            /// Combines matrix item axes in [batch_size, 2, 2] into features [batch_size, 4].
            /// Preserves Float dtype, device, and gradient connections.
            fn forward(&self, input: Tensor<3>) -> Result<Tensor<2>, Self::Error> {
                Ok(input.flatten(1, 2))
            }
        }

        struct MatrixGym {
            device: Device,
        }

        impl Gym<(), 3> for MatrixGym {
            type Error = Infallible;
            type ObservationSpace = BoxSpace<3>;
            type ActionSpace = Discrete;

            /// Accepts an Int scalar [1] and returns unbatched Float observations [2, 2] on the fixture device.
            fn step(
                &mut self,
                _action: Tensor<1, Int>,
            ) -> Result<StepInfo<(), Tensor<2>>, Self::Error> {
                Ok(StepInfo {
                    observation: Tensor::ones([2, 2], (&self.device, DType::F32)),
                    reward: 1.0,
                    done: false,
                    truncated: false,
                    info: (),
                })
            }

            /// Returns Float observations [2, 2] on the fixture device.
            fn reset(&mut self) -> Result<ResetInfo<(), Tensor<2>>, Self::Error> {
                Ok(ResetInfo {
                    observation: Tensor::ones([2, 2], (&self.device, DType::F32)),
                    info: (),
                })
            }

            fn observation_space(&self) -> Self::ObservationSpace {
                BoxSpace::new_unbounded([1, 2, 2], &self.device)
            }

            fn action_space(&self) -> Self::ActionSpace {
                Discrete::new(2)
            }
        }

        let device = Device::flex().autodiff();
        let mut env: VectorizedGymWrapper<MatrixGym, (), 3> = MatrixGym {
            device: device.clone(),
        }
        .into();
        let actor = ProbabilisticPolicyModel::<(
            MLP,
            CategoricalDistribution,
            (Tensor<2>, Tensor<2>, Tensor<2>),
        )>::new(mlp(4, 2, &device));
        let networks = PPONetworkInfo::Shared(
            SharedPPONetwork::builder()
                .shared_network(FlattenModel)
                .actor_head(actor)
                .critic_head(mlp(4, 1, &device))
                .optimizer(SgdConfig::new().init())
                .learning_rate(1e-3)
                .build()
                .unwrap(),
        );
        let mut logger = NoopLogger::default();
        let mut agent = PPOAgent::builder()
            .observation_space(env.observation_space())
            .action_space(env.action_space())
            .network_info(networks)
            .batch_size(2)
            .mini_batch_size(2)
            .num_epochs(1)
            .training_horizon(2)
            .logging_info(&mut logger)
            .device(device.clone())
            .build()
            .unwrap();
        agent.learn(&mut env, 2).unwrap();
        assert_eq!(
            agent.current_observations.as_ref().unwrap().dims(),
            [1, 2, 2]
        );
        assert_eq!(
            agent
                .act_deterministic(&Tensor::ones([1, 2, 2], (&device, DType::F32)))
                .unwrap()
                .dims(),
            [1, 1]
        );
        assert_eq!(logger.updates, 1);
    }

    #[test]
    fn integer_and_boolean_observations_use_the_compute_dtype() {
        struct BoolSpace;

        impl ObservationSpace<2> for BoolSpace {
            type Kind = Bool;

            /// Accepts Bool observations [batch_size, 1] without changing their dtype, device, or axes.
            fn contains(&self, values: &Tensor<2, Bool>) -> bool {
                values.dims()[1] == 1
            }

            fn shape(&self) -> Vec<usize> {
                vec![1]
            }
        }
        let device = Device::flex().autodiff();
        let actor = ProbabilisticPolicyModel::<(
            MLP,
            CategoricalDistribution,
            (Tensor<2>, Tensor<2>, Tensor<2>),
        )>::new(mlp(1, 2, &device));
        let networks = PPONetworkInfo::<_, _>::Separate(
            SeparatePPONetwork::builder()
                .actor_network(actor)
                .critic_network(mlp(1, 1, &device))
                .actor_optimizer(SgdConfig::new().init())
                .critic_optimizer(SgdConfig::new().init())
                .actor_learning_rate(1e-3)
                .critic_learning_rate(1e-3)
                .build()
                .unwrap(),
        );
        let mut boolean_agent = PPOAgent::<_, _, _, Infallible>::builder()
            .observation_space(BoolSpace)
            .action_space(Discrete::new(2))
            .network_info(networks)
            .training_horizon(1)
            .device(device.clone())
            .build()
            .unwrap();
        assert_eq!(
            boolean_agent
                .act_deterministic(&Tensor::from_data([[false], [true]], &device))
                .unwrap()
                .dims(),
            [2, 1]
        );

        let actor = ProbabilisticPolicyModel::<(
            MLP,
            CategoricalDistribution,
            (Tensor<2>, Tensor<2>, Tensor<2>),
        )>::new(mlp(1, 2, &device));
        let networks = PPONetworkInfo::<_, _>::Separate(
            SeparatePPONetwork::builder()
                .actor_network(actor)
                .critic_network(mlp(1, 1, &device))
                .actor_optimizer(SgdConfig::new().init())
                .critic_optimizer(SgdConfig::new().init())
                .actor_learning_rate(1e-3)
                .critic_learning_rate(1e-3)
                .build()
                .unwrap(),
        );
        let mut integer_agent = PPOAgent::<_, _, _, Infallible>::builder()
            .observation_space(Discrete::new(2))
            .action_space(Discrete::new(2))
            .network_info(networks)
            .training_horizon(1)
            .device(device.clone())
            .build()
            .unwrap();
        assert_eq!(
            integer_agent
                .act_deterministic(&Tensor::from_data([[0i64], [1]], (&device, DType::I64)))
                .unwrap()
                .dims(),
            [2, 1]
        );
    }

    #[test]
    fn f64_models_train_with_f32_environment_observations() {
        let device = Device::flex().autodiff();
        let mut env: VectorizedGymWrapper<TwoStepTestGym, usize> =
            TwoStepTestGym::new(device.clone(), false).into();
        let model = |output| {
            MLP::builder()
                .input_size(4)
                .output_size(output)
                .hidden_layer_sizes(vec![2])
                .hidden_initializer(Initializer::Constant { value: 0.1 })
                .output_initializer(Initializer::Constant { value: 0.1 })
                .bias_initializer(Initializer::Constant { value: 0.0 })
                .options((&device, DType::F64))
                .build()
                .unwrap()
        };
        let actor = ProbabilisticPolicyModel::<(
            MLP,
            CategoricalDistribution,
            (Tensor<2>, Tensor<2>, Tensor<2>),
        )>::new(model(2));
        let networks = PPONetworkInfo::<_, _>::Separate(
            SeparatePPONetwork::builder()
                .actor_network(actor)
                .critic_network(model(1))
                .actor_optimizer(SgdConfig::new().init())
                .critic_optimizer(SgdConfig::new().init())
                .actor_learning_rate(1e-3)
                .critic_learning_rate(1e-3)
                .build()
                .unwrap(),
        );
        struct DTypeLogger {
            updates: usize,
        }

        impl PPOLogger<usize> for DTypeLogger {
            fn log(&mut self, entry: &PPOLogEntry) {
                for tensor in [
                    &entry.actor_loss,
                    &entry.critic_loss,
                    &entry.entropy,
                    &entry.explained_variance,
                ] {
                    assert_eq!(tensor.dtype(), DType::F64);
                }
                assert_eq!(entry.returns.dtype(), DType::F64);
                assert_eq!(entry.advantages.dtype(), DType::F64);
                assert_eq!(entry.rewards.dtype(), DType::F32);
                self.updates += 1;
            }
        }
        let mut logger = DTypeLogger { updates: 0 };
        let mut agent = PPOAgent::builder()
            .observation_space(env.observation_space())
            .action_space(env.action_space())
            .network_info(networks)
            .dtype(DType::F64)
            .batch_size(2)
            .mini_batch_size(2)
            .num_epochs(1)
            .training_horizon(2)
            .logging_info(&mut logger)
            .device(device)
            .build()
            .unwrap();
        agent.learn(&mut env, 2).unwrap();
        assert_eq!(
            agent.current_observations.as_ref().unwrap().dtype(),
            DType::F64
        );
        assert_eq!(logger.updates, 1);
    }

    #[test]
    fn separate_updates_both_models_with_combined_or_independent_losses() {
        struct Parameters(Vec<f32>);

        impl ModuleVisitor for Parameters {
            /// Reads each rank-D Float parameter to the host in visit order without changing the model.
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
                self.0
                    .extend(param.val().into_data().try_to_vec::<f32>().unwrap());
            }
        }
        let device = Device::flex().autodiff();
        let mut updates = Vec::new();
        for combined in [false, true] {
            let actor = ProbabilisticPolicyModel::<(
                MLP,
                GaussianDistribution,
                (Tensor<2>, Tensor<2>, Tensor<2>),
            )>::new(mlp(4, 6, &device));
            let critic = mlp(4, 1, &device);
            let mut before_actor = Parameters(Vec::new());
            let mut before_critic = Parameters(Vec::new());
            actor.visit(&mut before_actor);
            critic.visit(&mut before_critic);
            let networks = PPONetworkInfo::<_, _>::Separate(
                SeparatePPONetwork::builder()
                    .actor_network(actor)
                    .critic_network(critic)
                    .actor_optimizer(SgdConfig::new().init())
                    .critic_optimizer(SgdConfig::new().init())
                    .actor_learning_rate(0.01)
                    .critic_learning_rate(0.02)
                    .combined_loss(combined)
                    .build()
                    .unwrap(),
            );
            let mut agent = PPOAgent::<_, _, _, Infallible>::builder()
                .observation_space(BoxSpace::new_unbounded([1, 4], &device))
                .action_space(BoxSpace::new_with_universal_bounds(
                    [1, 3],
                    -1.0,
                    1.0,
                    &device,
                ))
                .network_info(networks)
                .normalize_advantage(false)
                .gradient_clip(100.0)
                .training_horizon(1)
                .device(device.clone())
                .build()
                .unwrap();
            let losses = agent
                .compute_loss()
                .observations(Tensor::ones([2, 4], (&device, DType::F32)))
                .actions(Tensor::full([2, 3], 0.1, (&device, DType::F32)))
                .old_log_probs(Tensor::zeros([2, 1], (&device, DType::F32)))
                .advantages(Tensor::ones([2, 1], (&device, DType::F32)))
                .returns(Tensor::ones([2, 1], (&device, DType::F32)))
                .rewards(Tensor::ones([2, 1], (&device, DType::F32)))
                .explained_variance(Tensor::zeros([1], (&device, DType::F32)))
                .clip_range(0.2)
                .call()
                .unwrap();
            agent.backpropagate_loss(losses).unwrap();
            let PPONetworkInfo::Separate(network) = &agent.network_info else {
                panic!("expected separate network");
            };
            let mut after_actor = Parameters(Vec::new());
            let mut after_critic = Parameters(Vec::new());
            network.actor_network.visit(&mut after_actor);
            network.critic_network.visit(&mut after_critic);
            assert_ne!(before_actor.0, after_actor.0);
            assert_ne!(before_critic.0, after_critic.0);
            updates.push((after_actor.0, after_critic.0));
        }
        assert_eq!(updates[0], updates[1]);
    }

    #[test]
    fn seeded_collection_and_training_are_repeatable() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        let mut runs = Vec::new();
        for _ in 0..2 {
            let mut env: VectorizedGymWrapper<TwoStepTestGym, usize> =
                TwoStepTestGym::new(device.clone(), false).into();
            let actor = ProbabilisticPolicyModel::<(
                MLP,
                CategoricalDistribution,
                (Tensor<2>, Tensor<2>, Tensor<2>),
            )>::new(mlp(4, 2, &device));
            let networks = PPONetworkInfo::<_, _>::Separate(
                SeparatePPONetwork::builder()
                    .actor_network(actor)
                    .critic_network(mlp(4, 1, &device))
                    .actor_optimizer(SgdConfig::new().init())
                    .critic_optimizer(SgdConfig::new().init())
                    .actor_learning_rate(1e-3)
                    .critic_learning_rate(1e-3)
                    .build()
                    .unwrap(),
            );
            struct Logger(Vec<(f32, f32)>);

            impl PPOLogger<usize> for Logger {
                fn log(&mut self, entry: &PPOLogEntry) {
                    self.0.push((
                        entry.actor_loss.clone().try_into_scalar::<f32>().unwrap(),
                        entry.critic_loss.clone().try_into_scalar::<f32>().unwrap(),
                    ));
                }
            }
            let mut logger = Logger(Vec::new());
            let mut agent = PPOAgent::builder()
                .observation_space(env.observation_space())
                .action_space(env.action_space())
                .network_info(networks)
                .batch_size(4)
                .mini_batch_size(2)
                .num_epochs(2)
                .training_horizon(8)
                .logging_info(&mut logger)
                .device(device.clone())
                .build()
                .unwrap();
            device.seed(42);
            agent.learn(&mut env, 8).unwrap();
            let evaluation = agent
                .act_deterministic(&Tensor::zeros([2, 4], (&device, DType::F32)))
                .unwrap()
                .into_data()
                .try_to_vec::<i32>()
                .unwrap();
            runs.push((logger.0, evaluation));
        }
        assert_eq!(runs[0], runs[1]);
    }

    #[test]
    fn truncation_uses_terminal_observations_and_preserves_reset_observations() {
        #[derive(Module, Debug)]
        struct ObservationCritic {
            weight: Param<Tensor<1>>,
        }

        impl Forward<2, 2> for ObservationCritic {
            type Error = Infallible;

            /// Returns the first feature times the learned weight, mapping [batch_size, 4] to [batch_size, 1].
            /// Preserves Float dtype, device, and parameter gradients.
            fn forward(&self, input: Tensor<2>) -> Result<Tensor<2>, Self::Error> {
                Ok(input.slice_dim(1, 0..1) * self.weight.val().unsqueeze::<2>())
            }
        }
        struct TruncatedGym {
            device: Device,
        }

        impl Gym for TruncatedGym {
            type Error = Infallible;
            type ObservationSpace = BoxSpace;
            type ActionSpace = Discrete;

            /// Accepts one Int action [1] and returns terminal Float observations [4] filled with seven.
            fn step(&mut self, _action: Tensor<1, Int>) -> Result<StepInfo, Self::Error> {
                Ok(StepInfo {
                    observation: Tensor::full([4], 7.0, (&self.device, DType::F32)),
                    reward: 1.0,
                    done: false,
                    truncated: true,
                    info: (),
                })
            }

            /// Returns reset Float observations [4] filled with zero on the fixture device.
            fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
                Ok(ResetInfo {
                    observation: Tensor::zeros([4], (&self.device, DType::F32)),
                    info: (),
                })
            }

            fn observation_space(&self) -> Self::ObservationSpace {
                BoxSpace::new_unbounded([1, 4], &self.device)
            }

            fn action_space(&self) -> Self::ActionSpace {
                Discrete::new(2)
            }
        }
        let device = Device::flex().autodiff();
        let mut env: VectorizedGymWrapper<TruncatedGym> = TruncatedGym {
            device: device.clone(),
        }
        .into();
        let actor = ProbabilisticPolicyModel::<(
            MLP,
            CategoricalDistribution,
            (Tensor<2>, Tensor<2>, Tensor<2>),
        )>::new(mlp(4, 2, &device));
        let critic = ObservationCritic {
            weight: Param::from_tensor(Tensor::ones([1], (&device, DType::F32))),
        };
        let networks = PPONetworkInfo::<_, _>::Separate(
            SeparatePPONetwork::builder()
                .actor_network(actor)
                .critic_network(critic)
                .actor_optimizer(SgdConfig::new().init())
                .critic_optimizer(SgdConfig::new().init())
                .actor_learning_rate(1e-3)
                .critic_learning_rate(1e-3)
                .build()
                .unwrap(),
        );
        let mut agent = PPOAgent::builder()
            .observation_space(env.observation_space())
            .action_space(env.action_space())
            .network_info(networks)
            .batch_size(1)
            .mini_batch_size(1)
            .num_epochs(1)
            .gamma(0.9)
            .training_horizon(1)
            .device(device)
            .build()
            .unwrap();
        let mut next_observations = agent.prepare_rollout_collection(&mut env).unwrap();
        agent
            .collect_rollout(&mut env, &mut next_observations)
            .unwrap();
        assert_eq!(
            next_observations.into_data().try_to_vec::<f32>().unwrap(),
            [0.0; 4]
        );
        let item = &agent.rollout_buffer.get_raw()[0];
        assert_eq!(
            item.next_observations
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            [7.0; 4]
        );
        assert_eq!(
            item.rewards
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            [1.0]
        );
        assert!(
            (item
                .training_rewards
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap()[0]
                - 7.3)
                .abs()
                < 1e-6
        );
        agent.add_advantages_and_returns().unwrap();
        let returns = agent.rollout_buffer.get_raw()[0]
            .prepared
            .as_ref()
            .unwrap()
            .returns
            .clone();
        assert!((returns.into_data().try_to_vec::<f32>().unwrap()[0] - 7.3).abs() < 1e-6);
    }

    #[test]
    fn surrogate_minimum_splits_tied_gradients() {
        let device = Device::flex().autodiff();
        let left = Tensor::<2>::from_floats([[1.0]], &device).require_grad();
        let right = Tensor::<2>::from_floats([[1.0]], &device).require_grad();
        let ties = left.clone().equal(right.clone());
        let loss = left
            .clone()
            .min_pair(right.clone())
            .mask_where(ties, (left.clone() + right.clone()) * 0.5)
            .mean();
        let gradients = loss.backward();
        for operand in [left, right] {
            assert_eq!(
                operand
                    .grad(&gradients)
                    .unwrap()
                    .into_data()
                    .try_to_vec::<f32>()
                    .unwrap(),
                [0.5]
            );
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
        type Error = Infallible;
        type ObservationSpace = BoxSpace;
        type ActionSpace = Discrete;

        /// Steps with one scalar Int action `[1]`, returning Float observations `[4]` on the fixture device.
        fn step(&mut self, _action: Tensor<1, Int>) -> Result<StepInfo<usize>, Self::Error> {
            self.steps += 1;
            Ok(StepInfo {
                observation: Tensor::zeros([4], (&self.device, DType::F32)),
                reward: 1.0,
                done: self.steps == 2 && !self.truncates,
                truncated: self.steps == 2 && self.truncates,
                info: self.steps,
            })
        }

        /// Returns Float observations `[4]` on the fixture device.
        fn reset(&mut self) -> Result<ResetInfo<usize>, Self::Error> {
            self.steps = 0;
            Ok(ResetInfo {
                observation: Tensor::zeros([4], (&self.device, DType::F32)),
                info: 0,
            })
        }

        fn observation_space(&self) -> Self::ObservationSpace {
            BoxSpace::new_with_universal_bounds([1, 4], -1.0, 1.0, &self.device)
        }

        fn action_space(&self) -> Self::ActionSpace {
            Discrete::new(2)
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
        #[derive(Module, Debug)]
        struct UnitCritic;

        impl Forward<2, 2> for UnitCritic {
            type Error = Infallible;

            /// Returns Float values [batch_size, 1] for observations [batch_size, 4], preserving dtype and device.
            fn forward(&self, input: Tensor<2>) -> Result<Tensor<2>, Self::Error> {
                Ok(Tensor::ones(
                    [input.dims()[0], 1],
                    (&input.device(), input.dtype()),
                ))
            }
        }
        let device = Device::flex().autodiff();
        let actor = ProbabilisticPolicyModel::<(
            MLP,
            CategoricalDistribution,
            (Tensor<2>, Tensor<2>, Tensor<2>),
        )>::new(mlp(4, 2, &device));
        let networks = PPONetworkInfo::<_, _>::Separate(
            SeparatePPONetwork::builder()
                .actor_optimizer(SgdConfig::new().init())
                .critic_optimizer(SgdConfig::new().init())
                .actor_learning_rate(1e-3)
                .critic_learning_rate(1e-3)
                .actor_network(actor)
                .critic_network(UnitCritic)
                .build()
                .unwrap(),
        );
        let mut agent = PPOAgent::<_, _, _, Infallible>::builder()
            .observation_space(BoxSpace::new_unbounded([1, 4], &device))
            .action_space(Discrete::new(2))
            .network_info(networks)
            .batch_size(2)
            .mini_batch_size(2)
            .training_horizon(2)
            .device(device.clone())
            .build()
            .unwrap();
        let rewards = Tensor::from_floats([[1.0], [2.0]], &device);
        let observations = Tensor::zeros([2, 4], (&device, DType::F32));
        let actual = agent
            .bootstrap_truncated_rewards(&rewards, &observations, &[true, false], &[true, true])
            .unwrap()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(actual, [1.0, 2.99]);
    }

    #[test]
    fn collection_logs_report_fresh_rewards_and_completed_episodes() {
        let device = Device::flex().autodiff();
        let mut env: VectorizedGymWrapper<TwoStepTestGym, usize> = vec![
            TwoStepTestGym::new(device.clone(), false),
            TwoStepTestGym::new(device.clone(), true),
        ]
        .try_into()
        .unwrap();
        let actor = ProbabilisticPolicyModel::<(
            MLP,
            CategoricalDistribution,
            (Tensor<2>, Tensor<2>, Tensor<2>),
        )>::new(mlp(4, 2, &device));
        let networks = PPONetworkInfo::<_, _>::Separate(
            SeparatePPONetwork::builder()
                .actor_optimizer(SgdConfig::new().init())
                .critic_optimizer(SgdConfig::new().init())
                .actor_learning_rate(1e-3)
                .critic_learning_rate(1e-3)
                .actor_network(actor)
                .critic_network(mlp(4, 1, &device))
                .build()
                .unwrap(),
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
                        .clone()
                        .into_data()
                        .try_to_vec::<f32>()
                        .unwrap(),
                );
            }

            fn log_collection(&mut self, entry: &PPOCollectionLogEntry<usize>) {
                self.collection_rewards.push(
                    entry
                        .collection_rewards
                        .clone()
                        .into_data()
                        .try_to_vec::<f32>()
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
            .observation_space(env.observation_space())
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
        assert_eq!(logger.collection_rewards, [vec![1.0, 1.0], vec![1.0, 1.0]]);
        assert_eq!(
            logger.optimization_rewards,
            [vec![1.0, 1.0], vec![1.0, 1.0]]
        );
        assert_eq!(logger.infos, [vec![1, 1], vec![2, 2]]);
        assert_eq!(
            logger.completed_episodes,
            [(0, 2.0, 2, true, false, 3), (1, 2.0, 2, false, true, 4)]
        );
    }

    #[test]
    fn shared_agent_runs_collection_training_actions_and_observation_reset() {
        let device = Device::flex().autodiff();
        let mut env: VectorizedGymWrapper<TwoStepTestGym, usize> = vec![
            TwoStepTestGym::new(device.clone(), false),
            TwoStepTestGym::new(device.clone(), false),
        ]
        .try_into()
        .unwrap();
        let actor = ProbabilisticPolicyModel::<(
            MLP,
            CategoricalDistribution,
            (Tensor<2>, Tensor<2>, Tensor<2>),
        )>::new(mlp(4, 2, &device));
        let networks = PPONetworkInfo::Shared(
            SharedPPONetwork::builder()
                .shared_network(mlp(4, 4, &device))
                .actor_head(actor)
                .critic_head(mlp(4, 1, &device))
                .optimizer(SgdConfig::new().init())
                .learning_rate(1e-3)
                .lr_scheduler(LinearSchedule::new(1e-3, 1e-4))
                .build()
                .unwrap(),
        );
        let mut logger = NoopLogger::default();
        let mut agent = PPOAgent::builder()
            .observation_space(env.observation_space())
            .action_space(env.action_space())
            .network_info(networks)
            .batch_size(2)
            .mini_batch_size(2)
            .num_epochs(1)
            .training_horizon(10)
            .logging_info(&mut logger)
            .device(device.clone())
            .build()
            .unwrap();
        agent.learn(&mut env, 2).unwrap();
        let observations = agent.current_observations.as_ref().unwrap().clone();
        assert_eq!(agent.act(&observations).unwrap().dims(), [2, 1]);
        assert_eq!(
            agent.act_deterministic(&observations).unwrap().dims(),
            [2, 1]
        );
        assert_eq!(agent.episode_tracker.environment_count(), 2);
        let PPONetworkInfo::Shared(network) = &agent.network_info else {
            panic!("expected shared network");
        };
        assert_close(network.learning_rate, 0.00082);
        agent.set_learning_rate(0.25);
        let PPONetworkInfo::Shared(network) = &agent.network_info else {
            panic!("expected shared network");
        };
        assert_close(network.learning_rate, 0.25);
        agent.reset_current_observations();
        assert!(agent.current_observations.is_none());
        assert_eq!(agent.episode_tracker.environment_count(), 0);
        assert_eq!(logger.updates, 1);
    }

    #[test]
    fn separate_ppo_losses_only_reach_their_parameters() {
        struct ParameterIds(HashSet<ParamId>);

        impl ModuleVisitor for ParameterIds {
            /// Records rank-D parameter IDs without reading their tensor values.
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
                self.0.insert(param.id);
            }
        }
        struct GradientIds<'a> {
            gradients: &'a GradientsParams,
            ids: HashSet<ParamId>,
        }

        impl ModuleVisitor for GradientIds<'_> {
            /// Records IDs with rank-D gradients, preserving the gradient store.
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
                if self.gradients.get::<D>(param.id).is_some() {
                    self.ids.insert(param.id);
                }
            }
        }
        let device = Device::flex().autodiff();
        let actor = ProbabilisticPolicyModel::<(
            MLP,
            GaussianDistribution,
            (Tensor<2>, Tensor<2>, Tensor<2>),
        )>::new(mlp(4, 6, &device));
        let critic = mlp(4, 1, &device);
        let mut actor_ids = ParameterIds(HashSet::new());
        let mut critic_ids = ParameterIds(HashSet::new());
        actor.visit(&mut actor_ids);
        critic.visit(&mut critic_ids);
        let models = (actor.clone(), critic.clone());
        let networks = PPONetworkInfo::<_, _>::Separate(
            SeparatePPONetwork::builder()
                .actor_network(actor)
                .critic_network(critic)
                .actor_optimizer(SgdConfig::new().init())
                .critic_optimizer(SgdConfig::new().init())
                .actor_learning_rate(3e-4)
                .critic_learning_rate(3e-4)
                .combined_loss(true)
                .build()
                .unwrap(),
        );
        let mut agent = PPOAgent::<_, _, _, Infallible>::builder()
            .observation_space(BoxSpace::new_unbounded([1, 4], &device))
            .action_space(BoxSpace::new_with_universal_bounds(
                [1, 3],
                -1.0,
                1.0,
                &device,
            ))
            .network_info(networks)
            .batch_size(4)
            .mini_batch_size(4)
            .ent_coef(0.001)
            .clip_value_loss(true)
            .training_horizon(4)
            .device(device.clone())
            .build()
            .unwrap();

        let observations = Tensor::<2>::from_floats(
            [
                [0.1, 0.2, 0.3, 0.4],
                [-0.1, 0.3, -0.2, 0.5],
                [0.7, -0.4, 0.2, -0.3],
                [-0.5, -0.2, 0.6, 0.1],
            ],
            &device,
        )
        .require_grad();
        let actions = Tensor::<2>::full([4, 3], 0.1, (&device, DType::F32)).require_grad();
        let log_probs = Tensor::<2>::full([4, 1], -1.0, (&device, DType::F32)).require_grad();
        let advantages =
            Tensor::<2>::from_floats([[1.0], [-1.0], [0.5], [-0.5]], &device).require_grad();
        let returns =
            Tensor::<2>::from_floats([[1.0], [2.0], [3.0], [4.0]], &device).require_grad();
        let rewards = Tensor::<2>::full([4, 1], 0.1, (&device, DType::F32)).require_grad();
        let old_values = Tensor::<2>::zeros([4, 1], (&device, DType::F32)).require_grad();
        for branch in 0..3 {
            let losses = agent
                .compute_loss()
                .observations(observations.clone().detach())
                .actions(actions.clone().detach())
                .old_log_probs(log_probs.clone().detach())
                .advantages(advantages.clone().detach())
                .returns(returns.clone().detach())
                .rewards(rewards.clone().detach())
                .old_values(old_values.clone().detach())
                .explained_variance(Tensor::zeros([1], (&device, DType::F32)))
                .clip_range(0.2)
                .call()
                .unwrap();
            let gradients = match branch {
                0 => losses.actor_loss.backward(),
                1 => losses.critic_loss.backward(),
                _ => (losses.actor_loss + losses.critic_loss).backward(),
            };
            assert!(observations.grad(&gradients).is_none());
            assert!(actions.grad(&gradients).is_none());
            for tensor in [&log_probs, &advantages, &returns, &rewards, &old_values] {
                assert!(tensor.grad(&gradients).is_none());
            }
            let gradients = GradientsParams::from_grads(gradients, &models);
            let mut ids = GradientIds {
                gradients: &gradients,
                ids: HashSet::new(),
            };
            models.visit(&mut ids);
            match branch {
                0 => {
                    assert!(actor_ids.0.is_subset(&ids.ids));
                    assert!(critic_ids.0.is_disjoint(&ids.ids));
                }
                1 => {
                    assert!(critic_ids.0.is_subset(&ids.ids));
                    assert!(actor_ids.0.is_disjoint(&ids.ids));
                }
                _ => {
                    assert!(actor_ids.0.is_subset(&ids.ids));
                    assert!(critic_ids.0.is_subset(&ids.ids));
                }
            }
        }
    }

    #[test]
    fn schedules_continue_across_learn_calls() {
        let device = Device::flex().autodiff();
        let mut env: VectorizedGymWrapper<TwoStepTestGym, usize> = vec![
            TwoStepTestGym::new(device.clone(), false),
            TwoStepTestGym::new(device.clone(), false),
        ]
        .try_into()
        .unwrap();
        let actor = ProbabilisticPolicyModel::<(
            MLP,
            CategoricalDistribution,
            (Tensor<2>, Tensor<2>, Tensor<2>),
        )>::new(mlp(4, 2, &device));
        let networks = PPONetworkInfo::<_, _>::Separate(
            SeparatePPONetwork::builder()
                .actor_optimizer(SgdConfig::new().init())
                .critic_optimizer(SgdConfig::new().init())
                .actor_learning_rate(1e-3)
                .critic_learning_rate(1e-3)
                .actor_network(actor)
                .critic_network(mlp(4, 1, &device))
                .actor_lr_scheduler(LinearSchedule::new(1e-3, 1e-4))
                .critic_lr_scheduler(LinearSchedule::new(1e-3, 1e-4))
                .build()
                .unwrap(),
        );
        let mut logger = NoopLogger::default();
        let mut agent = PPOAgent::builder()
            .observation_space(env.observation_space())
            .action_space(env.action_space())
            .network_info(networks)
            .batch_size(2)
            .mini_batch_size(2)
            .num_epochs(1)
            .clip_range(LinearSchedule::new(0.2, 0.1))
            .training_horizon(10)
            .logging_info(&mut logger)
            .device(device.clone())
            .build()
            .unwrap();
        for (steps, rate, clip) in [(2, 0.00082, 0.18), (4, 0.00064, 0.16)] {
            agent.learn(&mut env, 2).unwrap();
            assert_eq!(agent.schedule_progress.elapsed_steps(), steps);
            assert_close(agent.schedule_progress.parameter(&*agent.clip_range), clip);
            let PPONetworkInfo::Separate(network) = &agent.network_info else {
                panic!("expected separate network");
            };
            assert_close(network.actor_learning_rate, rate);
            assert_close(network.critic_learning_rate, rate);
        }
        let observations = Tensor::zeros([2, 4], (&device, DType::F32));
        assert_eq!(agent.act(&observations).unwrap().dims(), [2, 1]);
        assert_eq!(
            agent.act_deterministic(&observations).unwrap().dims(),
            [2, 1]
        );
        agent.set_learning_rate(0.25);
        let PPONetworkInfo::Separate(network) = &agent.network_info else {
            panic!("expected separate network");
        };
        assert_close(network.actor_learning_rate, 0.25);
        assert_close(network.critic_learning_rate, 0.25);
        assert_eq!(logger.updates, 2);
    }
    #[test]
    fn gae_matches_scalar_reference_across_environments_and_episode_boundaries() {
        let device = Device::flex().autodiff();
        let mut env: VectorizedGymWrapper<TwoStepTestGym, usize> = (0..3)
            .map(|_| TwoStepTestGym::new(device.clone(), false))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let actor = ProbabilisticPolicyModel::<(
            MLP,
            CategoricalDistribution,
            (Tensor<2>, Tensor<2>, Tensor<2>),
        )>::new(mlp(4, 2, &device));
        let networks = PPONetworkInfo::<_, _>::Separate(
            SeparatePPONetwork::builder()
                .actor_network(actor)
                .critic_network(mlp(4, 1, &device))
                .actor_optimizer(SgdConfig::new().init())
                .critic_optimizer(SgdConfig::new().init())
                .actor_learning_rate(1e-3)
                .critic_learning_rate(1e-3)
                .build()
                .unwrap(),
        );
        let mut agent = PPOAgent::builder()
            .observation_space(env.observation_space())
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
        let dones = [
            [false, false, false],
            [true, false, false],
            [false, true, false],
        ];
        let truncateds = [
            [false, false, false],
            [false, true, false],
            [false, false, true],
        ];
        let bootstrap = [1.1_f32, 1.2, 1.3];
        let reward_tensor = Tensor::from_data(
            TensorData::new(rewards.into_iter().flatten().collect::<Vec<_>>(), [3, 3, 1]),
            &device,
        );
        let value_tensor = Tensor::from_data(
            TensorData::new(values.into_iter().flatten().collect::<Vec<_>>(), [3, 3, 1]),
            &device,
        );
        let done_tensor = Tensor::from_data(
            TensorData::new(dones.into_iter().flatten().collect::<Vec<_>>(), [3, 3, 1]),
            &device,
        );
        let truncated_tensor = Tensor::from_data(
            TensorData::new(
                truncateds.into_iter().flatten().collect::<Vec<_>>(),
                [3, 3, 1],
            ),
            &device,
        );
        let bootstrap_tensor =
            Tensor::from_data(TensorData::new(bootstrap.to_vec(), [3, 1]), &device);
        let actual = agent
            .compute_gae(
                &reward_tensor,
                &value_tensor,
                &done_tensor,
                &truncated_tensor,
                &bootstrap_tensor,
            )
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        for env_idx in 0..3 {
            let mut next_value = bootstrap[env_idx];
            let mut gae = 0.0_f32;
            for time in (0..3).rev() {
                let mask = if dones[time][env_idx] || truncateds[time][env_idx] {
                    0.0
                } else {
                    1.0
                };
                let value = values[time][env_idx];
                let delta = rewards[time][env_idx] + 0.9 * next_value * mask - value;
                gae = delta + 0.9 * 0.8 * gae * mask;
                assert!((actual[time * 3 + env_idx] - gae).abs() < 1e-5);
                next_value = value;
            }
        }
    }
}

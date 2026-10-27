//! Shared collection and training for deterministic off-policy actor-critic agents.

use super::{
    ReplayStorageConfig,
    action_value_critic::{
        ActionValueCritic, ActionValueCriticError, ActionValueCriticNetwork,
        interpolate_target_model,
    },
};
use crate::{
    buffers::experience_replay::{
        ExperienceReplayError, ReplayStorageError, validate_replay_alignment,
    },
    gym::{MultiGym, MultiGymStepInfo},
    models::Forward,
    objectives::bellman_targets,
    parameter_schedule::ScheduleProgress,
    spaces::{ActionMap, ObservationSpace},
    tensor_operations::transfer_bool_tensor,
    tensor_rank::NextRank,
};
use bon::bon;
use burn::{
    module::Module,
    optim::{GradientsParams, ModuleOptimizer},
    tensor::{
        DType, Device, Distribution, Float, Tensor, TensorData, TensorReadError, kind::Autodiff,
    },
};
use std::{fmt::Debug, marker::PhantomData};

pub mod ddpg;
mod replay;
pub mod td3;

pub use replay::DeterministicReplayError;
use replay::{
    DeterministicBatch, DeterministicReplay, DeterministicReplayInsert, DeterministicReplayStorage,
};

#[cfg(test)]
mod tests;

/// Invalid deterministic actor-critic configuration.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum DeterministicActorCriticConfigurationError {
    /// TD3 received an empty critic ensemble.
    #[error("at least one critic is required")]
    NoCritics,
    /// Replay capacity was zero.
    #[error("replay capacity must be nonzero")]
    ZeroReplayCapacity,
    /// Replay batch size was zero.
    #[error("batch size must be nonzero")]
    ZeroBatchSize,
    /// Replay cannot retain one complete optimization batch.
    #[error("replay capacity must be at least the batch size")]
    ReplayCapacityBelowBatchSize,
    /// The replay-update frequency was zero.
    #[error("update frequency must be nonzero")]
    ZeroUpdateFrequency,
    /// The global collection horizon was zero.
    #[error("training horizon must be nonzero")]
    ZeroTrainingHorizon,
    /// Gamma was non-finite or outside `0.0..=1.0`.
    #[error("gamma must be finite and in 0..=1")]
    InvalidGamma,
    /// Tau was non-finite or outside `0.0..=1.0`.
    #[error("tau must be finite and in 0..=1")]
    InvalidTau,
    /// The collection exploration-noise deviation was invalid.
    #[error("exploration noise must be finite and nonnegative")]
    InvalidExplorationNoise,
    /// The target-policy-noise deviation was invalid.
    #[error("target policy noise must be finite and nonnegative")]
    InvalidTargetPolicyNoise,
    /// The target-policy-noise clip was invalid.
    #[error("target noise clip must be finite and nonnegative")]
    InvalidTargetNoiseClip,
    /// The actor-update interval was zero.
    #[error("actor update interval must be nonzero")]
    ZeroActorUpdateInterval,
    /// An algorithm received a different number of critics than it requires.
    #[error("incorrect critic count: expected {expected}, got {actual}")]
    IncorrectCriticCount {
        /// Number of critics required by the algorithm.
        expected: usize,
        /// Number of critics supplied to the builder.
        actual: usize,
    },
    #[error("actor learning rate must be finite and positive")]
    InvalidActorLearningRate,
    #[error("compute dtype must be floating-point, got {0:?}")]
    InvalidDType(DType),
    #[error("actor and critic parameters must share one device")]
    ModelDevice,
    #[error("{field} space has {actual} item axes, expected {expected}")]
    SpaceRank {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("{field} item dimensions must be nonzero")]
    ZeroItemDimension { field: &'static str },
}

/// Actor-critic construction, component, or training failure.
#[derive(Debug, thiserror::Error)]
pub enum DeterministicActorCriticError<AE, GE, SE, CE = std::convert::Infallible> {
    #[error("actor model failed: {0}")]
    ActorError(#[source] AE),
    #[error("critic model failed: {0}")]
    CriticError(#[source] CE),
    #[error("critic operation failed: {0}")]
    CriticOperationError(#[from] ActionValueCriticError),
    #[error("actor-critic tensor read failed: {0}")]
    TensorError(#[from] TensorReadError),
    #[error("replay storage failed: {0}")]
    ReplayStorageError(#[from] DeterministicReplayError),
    #[error("invalid actor-critic configuration: {0}")]
    ConfigurationError(#[from] DeterministicActorCriticConfigurationError),
    #[error("gym failed: {0}")]
    GymError(#[source] GE),
    #[error("space operation failed: {0}")]
    SpaceError(#[source] SE),
    #[error("actor-critic {field} shape differs: expected {expected:?}, got {actual:?}")]
    Shape {
        field: &'static str,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("actor-critic {field} dtype differs: expected {expected:?}, got {actual:?}")]
    DType {
        field: &'static str,
        expected: DType,
        actual: DType,
    },
    #[error("actor-critic {field} uses an incompatible device")]
    Device { field: &'static str },
    #[error("observation batches must be nonempty")]
    EmptyBatch,
}

impl<AE, GE, SE, CE> From<ReplayStorageError> for DeterministicActorCriticError<AE, GE, SE, CE> {
    fn from(error: ReplayStorageError) -> Self {
        Self::ReplayStorageError(error.into())
    }
}

impl<AE, GE, SE, CE> From<ExperienceReplayError<DeterministicReplayError>>
    for DeterministicActorCriticError<AE, GE, SE, CE>
{
    fn from(error: ExperienceReplayError<DeterministicReplayError>) -> Self {
        match error {
            ExperienceReplayError::ExperienceError(error) => error.into(),
            ExperienceReplayError::InsertionExceedsCapacity { capacity, inserted } => {
                ReplayStorageError::InsertionExceedsCapacity { capacity, inserted }.into()
            }
        }
    }
}

/// Metrics from one replay update.
pub struct DeterministicActorCriticLogEntry<const A: usize = 2> {
    /// One detached optimization loss [1] per critic.
    pub critic_losses: Vec<Tensor<1>>,
    /// Each critic's Q predictions for the sampled replay observation-action pairs.
    /// Every tensor is shaped `[batch_size, 1]`.
    pub critic_q_values: Vec<Tensor<2>>,
    /// Detached negative mean policy Q loss [1].
    ///
    /// This is `None` on updates where TD3 delays the actor.
    pub actor_loss: Option<Tensor<1>>,
    /// Q values used by the actor objective, shaped `[batch_size, 1]`.
    ///
    /// DDPG uses its sole critic. Canonical TD3 uses its first critic for the
    /// actor objective. A configured TD3 actor aggregation mode combines the
    /// critic ensemble instead. This is `None` on updates where TD3 delays the
    /// actor.
    pub policy_q_values: Option<Tensor<2>>,
    /// Current policy actions shaped `[batch_size, ...action_shape]`.
    /// `None` on updates where TD3 delays the actor.
    pub policy_actions: Option<Tensor<A>>,
    /// Replay actions shaped `[batch_size, ...action_shape]`.
    pub replay_actions: Tensor<A>,
    /// Detached expected Q targets shared by all critics, shaped
    /// `[batch_size, 1]`.
    pub bellman_targets: Tensor<2>,
    /// Rewards from the sampled replay batch, shaped `[batch_size, 1]`.
    pub replay_rewards: Tensor<2>,
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
    /// Newest reward from each inner environment, shaped `[environment_count, 1]`.
    pub collection_rewards: Tensor<2>,
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

pub(crate) trait DeterministicActorCriticLogger<I = (), const A: usize = 2> {
    fn log_update(&mut self, entry: &DeterministicActorCriticLogEntry<A>);

    fn log_collection(&mut self, entry: &DeterministicActorCriticCollectionLogEntry<I>);
}

pub(crate) trait DeterministicActorCriticStrategy {
    fn critic_count_requirement(&self) -> CriticCountRequirement;

    fn actor_update_interval(&self) -> usize;

    fn target_policy_noise(&self) -> f64;

    fn target_noise_clip(&self) -> f64;

    /// Combines the target critics' next-observation Q estimates into one value per
    /// replay transition. DDPG uses its sole critic; TD3 applies its configured
    /// ensemble aggregation. Each input and the output are shaped
    /// `[batch_size, 1]`.
    fn aggregate_target_values(
        &self,
        values: Vec<Tensor<2>>,
    ) -> Result<Tensor<2>, ActionValueCriticError>;

    /// Combines the critics' current-policy Q estimates into the
    /// `[batch_size, 1]` values used by the actor objective.
    fn aggregate_actor_values(
        &self,
        values: Vec<Tensor<2>>,
    ) -> Result<Tensor<2>, ActionValueCriticError>;
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum CriticCountRequirement {
    Exact(usize),
    NonEmpty,
}

struct DeterministicActorCriticConfigurationValidator;

#[bon]
impl DeterministicActorCriticConfigurationValidator {
    #[builder]
    fn validate_configuration(
        actor_learning_rate: f64,
        dtype: DType,
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
        let error = if !actor_learning_rate.is_finite() || actor_learning_rate <= 0.0 {
            Some(DeterministicActorCriticConfigurationError::InvalidActorLearningRate)
        } else if !dtype.is_float() {
            Some(DeterministicActorCriticConfigurationError::InvalidDType(
                dtype,
            ))
        } else if actual_critic_count == 0 {
            Some(DeterministicActorCriticConfigurationError::NoCritics)
        } else if let CriticCountRequirement::Exact(expected) = critic_count_requirement
            && actual_critic_count != expected
        {
            Some(
                DeterministicActorCriticConfigurationError::IncorrectCriticCount {
                    expected,
                    actual: actual_critic_count,
                },
            )
        } else if replay_capacity == 0 {
            Some(DeterministicActorCriticConfigurationError::ZeroReplayCapacity)
        } else if batch_size == 0 {
            Some(DeterministicActorCriticConfigurationError::ZeroBatchSize)
        } else if replay_capacity < batch_size {
            Some(DeterministicActorCriticConfigurationError::ReplayCapacityBelowBatchSize)
        } else if update_frequency == 0 {
            Some(DeterministicActorCriticConfigurationError::ZeroUpdateFrequency)
        } else if training_horizon == 0 {
            Some(DeterministicActorCriticConfigurationError::ZeroTrainingHorizon)
        } else if !gamma.is_finite() || !(0.0..=1.0).contains(&gamma) {
            Some(DeterministicActorCriticConfigurationError::InvalidGamma)
        } else if !tau.is_finite() || !(0.0..=1.0).contains(&tau) {
            Some(DeterministicActorCriticConfigurationError::InvalidTau)
        } else if !exploration_noise.is_finite() || exploration_noise < 0.0 {
            Some(DeterministicActorCriticConfigurationError::InvalidExplorationNoise)
        } else if !target_policy_noise.is_finite() || target_policy_noise < 0.0 {
            Some(DeterministicActorCriticConfigurationError::InvalidTargetPolicyNoise)
        } else if !target_noise_clip.is_finite() || target_noise_clip < 0.0 {
            Some(DeterministicActorCriticConfigurationError::InvalidTargetNoiseClip)
        } else if actor_update_interval == 0 {
            Some(DeterministicActorCriticConfigurationError::ZeroActorUpdateInterval)
        } else {
            None
        };
        error.map_or(Ok(()), Err)
    }
}

struct CollectedTransitions<'a, const O: usize, const A: usize, K: Autodiff> {
    observations: &'a Tensor<O, K>,
    next_observations: &'a Tensor<O, K>,
    actions: &'a Tensor<A>,
    rewards: &'a Tensor<2>,
    dones: &'a [bool],
    truncateds: &'a [bool],
    first_timestep: usize,
}

struct CriticUpdate {
    losses: Vec<Tensor<1>>,
    q_values: Vec<Tensor<2>>,
}

struct ActorUpdate<const A: usize> {
    loss: Tensor<1>,
    q_values: Tensor<2>,
    actions: Tensor<A>,
}

/// Owns deterministic actor and critic models for native rank-O observation batches and Float rank-A action batches.
/// Scalar observations/actions use [batch_size, 1]. Critic candidates add one axis after batch, with C = A + 1.
/// Models own observation preprocessing. Replay preserves observation dtype and transfers detached samples to the optimization device.
/// Groups owned models and native execution tensors without adding runtime storage or execution methods.
pub trait DeterministicActorCriticModelTypes {
    type Actor: Module;
    type Critic: Module;
    type Observation;
    type Action;
    type Candidates;
    type Replay;
}

/// Selects actor, critic, observation, action, and candidate types; all actual values remain in the agent.
pub struct DeterministicActorCriticModelGroup<M, N, O = Tensor<2>, A = Tensor<2>, C = Tensor<3>> {
    models: PhantomData<(M, N)>,
    tensors: PhantomData<(O, A, C)>,
}

impl<M: Module, N: Module, K: Autodiff, const O: usize, const A: usize, const C: usize>
    DeterministicActorCriticModelTypes
    for DeterministicActorCriticModelGroup<M, N, Tensor<O, K>, Tensor<A>, Tensor<C>>
{
    type Actor = M;
    type Critic = N;
    type Observation = Tensor<O, K>;
    type Action = Tensor<A>;
    type Candidates = Tensor<C>;
    type Replay = DeterministicReplay<O, A, K>;
}

/// Infers the model group from the concrete actor, critic, and native tensor types supplied to the builder.
pub trait DeterministicActorCriticModelFor<N: Module, O, A, C>: Module {
    type ModelGroup: DeterministicActorCriticModelTypes<Actor = Self, Critic = N>;
}

impl<M: Module, N: Module, K: Autodiff, const O: usize, const A: usize, const C: usize>
    DeterministicActorCriticModelFor<N, Tensor<O, K>, Tensor<A>, Tensor<C>> for M
{
    type ModelGroup = DeterministicActorCriticModelGroup<M, N, Tensor<O, K>, Tensor<A>, Tensor<C>>;
}

/// Stores owned models and replay through their associated types; spaces and action rank remain explicit.
pub(crate) struct DeterministicActorCriticAgent<
    G: DeterministicActorCriticModelTypes,
    OS,
    AS,
    S,
    GE,
    const A: usize = 2,
> {
    online_actor: G::Actor,
    target_actor: G::Actor,
    actor_optimizer: ModuleOptimizer,
    actor_learning_rate: f64,
    critics: Vec<ActionValueCritic<G::Critic>>,
    strategy: S,
    action_space: AS,
    observation_space: OS,
    replay: Option<G::Replay>,
    replay_capacity: usize,
    batch_size: usize,
    environment_count: Option<usize>,
    observation_dtype: Option<DType>,
    gamma: f64,
    tau: f64,
    exploration_noise: f64,
    training_start: usize,
    update_frequency: usize,
    schedule_progress: ScheduleProgress,
    replay_storage_config: ReplayStorageConfig,
    optimization_steps: usize,
    dtype: DType,
    _errors: PhantomData<GE>,
}

#[bon]
impl<G: DeterministicActorCriticModelTypes, OS, AS, S, GE, const A: usize>
    DeterministicActorCriticAgent<G, OS, AS, S, GE, A>
{
    /// Builds native observations [batch_size, ...observation_shape] into Float policy and environment actions of rank A.
    /// Policy outputs must match action_space.policy_shape(); environment actions match action_space.shape().
    /// Parameters must use the optimization device. Actions, rewards, and Q values use dtype; observations retain their native dtype.
    /// Clones a detached target actor and allocates replay from the first collected batch's observation dtype.
    /// Returns typed errors for invalid configuration, space shapes, or parameter devices.
    #[builder]
    pub(crate) fn new<M, N, AE: Debug, CE: Debug, SE: Debug, const O: usize, const C: usize>(
        online_actor: M,
        #[builder(into)] actor_optimizer: ModuleOptimizer,
        actor_learning_rate: f64,
        critics: Vec<ActionValueCritic<N>>,
        strategy: S,
        action_space: AS,
        observation_space: OS,
        replay_device: Option<Device>,
        #[builder(default = 0.99)] gamma: f64,
        #[builder(default = 0.005)] tau: f64,
        #[builder(default = 0.1)] exploration_noise: f64,
        #[builder(default = 1_000_000)] replay_capacity: usize,
        #[builder(default = 256)] batch_size: usize,
        #[builder(default = 1)] update_frequency: usize,
        #[builder(default = 1_000)] training_start: usize,
        training_horizon: usize,
        #[builder(default = DType::F32)] dtype: DType,
    ) -> Result<Self, DeterministicActorCriticConfigurationError>
    where
        OS: ObservationSpace<O>,
        OS::Kind: Autodiff,
        AS: ActionMap<A, A, Kind = Float, Error = SE>,
        Tensor<A>: NextRank<Next = Tensor<C>>,
        M: Module + Forward<O, A, InputKind = OS::Kind, OutputKind = Float, Error = AE>,
        N: ActionValueCriticNetwork<
                O,
                A,
                C,
                ObservationKind = OS::Kind,
                ActionKind = Float,
                Error = CE,
            >,
        S: DeterministicActorCriticStrategy,
        GE: Debug,
        M: DeterministicActorCriticModelFor<
                N,
                Tensor<O, OS::Kind>,
                Tensor<A>,
                Tensor<C>,
                ModelGroup = G,
            >,
        G: DeterministicActorCriticModelTypes<Actor = M, Critic = N>,
    {
        const {
            assert!(
                O >= 2 && A >= 2 && C == A + 1,
                "observations and actions require batch and item axes; candidates add one action axis"
            );
        }
        DeterministicActorCriticConfigurationValidator::validate_configuration()
            .actor_learning_rate(actor_learning_rate)
            .dtype(dtype)
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
        space_storage_shape::<O>("observations", observation_space.shape(), replay_capacity)?;
        space_storage_shape::<A>("actions", action_space.shape(), replay_capacity)?;
        space_storage_shape::<A>(
            "policy actions",
            action_space.policy_shape(),
            replay_capacity,
        )?;
        let replay_storage_config = ReplayStorageConfig::from_models(
            online_actor.devices().into_iter().chain(
                critics
                    .iter()
                    .flat_map(|critic| critic.online_network().devices()),
            ),
            replay_device,
        )
        .ok_or(DeterministicActorCriticConfigurationError::ModelDevice)?;
        let online_actor = online_actor.train();
        let target_actor = online_actor.clone().valid();
        Ok(Self {
            online_actor,
            target_actor,
            actor_optimizer,
            actor_learning_rate,
            critics,
            strategy,
            action_space,
            observation_space,
            replay: None,
            replay_capacity,
            batch_size,
            environment_count: None,
            observation_dtype: None,
            gamma,
            tau,
            exploration_noise,
            training_start,
            update_frequency,
            schedule_progress: ScheduleProgress::new(training_horizon),
            replay_storage_config,
            optimization_steps: 0,
            dtype,
            _errors: PhantomData,
        })
    }
}

impl<
    M,
    N,
    OS,
    AS,
    S,
    GE,
    AE: Debug,
    CE: Debug,
    SE: Debug,
    const O: usize,
    const A: usize,
    const C: usize,
>
    DeterministicActorCriticAgent<
        DeterministicActorCriticModelGroup<M, N, Tensor<O, OS::Kind>, Tensor<A>, Tensor<C>>,
        OS,
        AS,
        S,
        GE,
        A,
    >
where
    OS: ObservationSpace<O>,
    OS::Kind: Autodiff,
    AS: ActionMap<A, A, Kind = Float, Error = SE>,
    Tensor<A>: NextRank<Next = Tensor<C>>,
    M: Module + Forward<O, A, InputKind = OS::Kind, OutputKind = Float, Error = AE>,
    N: ActionValueCriticNetwork<
            O,
            A,
            C,
            ObservationKind = OS::Kind,
            ActionKind = Float,
            Error = CE,
        >,
    S: DeterministicActorCriticStrategy,
    GE: Debug,
{
    pub(crate) fn get_action_space(&self) -> &AS {
        &self.action_space
    }

    pub(crate) fn get_observation_space(&self) -> &OS {
        &self.observation_space
    }

    /// Checks native observations [batch_size, ...observation_shape] without casting or repairing axes.
    /// Requires a nonempty batch and the established replay dtype once replay storage exists.
    fn validate_observations(
        &self,
        observations: &Tensor<O, OS::Kind>,
    ) -> Result<(), DeterministicActorCriticError<AE, GE, SE, CE>> {
        let count = observations.dims()[0];
        if count == 0 {
            return Err(DeterministicActorCriticError::EmptyBatch);
        }
        let expected =
            space_storage_shape::<O>("observations", self.observation_space.shape(), count)?;
        if observations.dims() != expected {
            return Err(DeterministicActorCriticError::Shape {
                field: "observations",
                expected: expected.to_vec(),
                actual: observations.dims().to_vec(),
            });
        }
        if let Some(expected) = self.observation_dtype
            && observations.dtype() != expected
        {
            return Err(DeterministicActorCriticError::DType {
                field: "observations",
                expected,
                actual: observations.dtype(),
            });
        }
        Ok(())
    }

    /// Checks rank-D Float tensors with the required shape, compute dtype, and optimization device; preserves all axes and gradients.
    fn validate_float_tensor<const D: usize>(
        &self,
        field: &'static str,
        values: &Tensor<D>,
        expected: [usize; D],
    ) -> Result<(), DeterministicActorCriticError<AE, GE, SE, CE>> {
        if values.dims() != expected {
            return Err(DeterministicActorCriticError::Shape {
                field,
                expected: expected.to_vec(),
                actual: values.dims().to_vec(),
            });
        }
        if values.dtype() != self.dtype {
            return Err(DeterministicActorCriticError::DType {
                field,
                expected: self.dtype,
                actual: values.dtype(),
            });
        }
        if values.device() != self.replay_storage_config.optimization_device() {
            return Err(DeterministicActorCriticError::Device { field });
        }
        Ok(())
    }

    /// Maps Float policy actions [batch_size, ...policy_shape] to Float environment actions [batch_size, ...action_shape].
    /// Preserves dtype, device, batch axis, and policy gradients through the action map's clipping behavior.
    fn decode_actions(
        &self,
        latent: Tensor<A>,
        count: usize,
    ) -> Result<Tensor<A>, DeterministicActorCriticError<AE, GE, SE, CE>> {
        let expected =
            space_storage_shape::<A>("policy actions", self.action_space.policy_shape(), count)?;
        self.validate_float_tensor("policy actions", &latent, expected)?;
        let actions = self
            .action_space
            .tensor_from_neurons(latent)
            .map_err(DeterministicActorCriticError::SpaceError)?;
        let expected = space_storage_shape::<A>("actions", self.action_space.shape(), count)?;
        self.validate_float_tensor("actions", &actions, expected)?;
        Ok(actions)
    }

    /// Maps native observations [batch_size, ...observation_shape] to deterministic Float actions [batch_size, ...action_shape].
    /// Transfers observations to the optimization device without casting; preserves actor gradients through the action map.
    pub(crate) fn act_deterministic(
        &self,
        observations: &Tensor<O, OS::Kind>,
    ) -> Result<Tensor<A>, DeterministicActorCriticError<AE, GE, SE, CE>> {
        self.validate_observations(observations)?;
        let latent = self
            .online_actor
            .forward(
                observations
                    .clone()
                    .to_device(&self.replay_storage_config.optimization_device()),
            )
            .map_err(DeterministicActorCriticError::ActorError)?;
        self.decode_actions(latent, observations.dims()[0])
    }

    /// Maps native observations [batch_size, ...observation_shape] to exploration-noised Float actions [batch_size, ...action_shape].
    /// Adds Gaussian noise to policy outputs before action-space clipping; preserves dtype, device, and actor gradients.
    pub(crate) fn act(
        &self,
        observations: &Tensor<O, OS::Kind>,
    ) -> Result<Tensor<A>, DeterministicActorCriticError<AE, GE, SE, CE>> {
        self.validate_observations(observations)?;
        let latent = self
            .online_actor
            .forward(
                observations
                    .clone()
                    .to_device(&self.replay_storage_config.optimization_device()),
            )
            .map_err(DeterministicActorCriticError::ActorError)?;
        self.decode_actions(
            add_gaussian_noise(&latent, self.exploration_noise, None),
            observations.dims()[0],
        )
    }

    /// Samples Float environment actions [batch_size, ...action_shape] with the existing device RNG and casts to compute dtype.
    fn random_actions(
        &self,
        count: usize,
    ) -> Result<Tensor<A>, DeterministicActorCriticError<AE, GE, SE, CE>> {
        let actions = self
            .action_space
            .sample_batch(count, &self.replay_storage_config.optimization_device())
            .map_err(DeterministicActorCriticError::SpaceError)?
            .cast(self.dtype);
        let expected = space_storage_shape::<A>("actions", self.action_space.shape(), count)?;
        self.validate_float_tensor("random actions", &actions, expected)?;
        Ok(actions)
    }

    /// Samples detached observations [batch_size, ...observation_shape], actions [batch_size, ...action_shape], and scalars [batch_size, 1].
    /// Preserves native observation dtype and transfers every field to the optimization device.
    fn sample_batch(
        &self,
    ) -> Result<DeterministicBatch<O, A, OS::Kind>, DeterministicActorCriticError<AE, GE, SE, CE>>
    {
        let replay = self
            .replay
            .as_ref()
            .ok_or(ReplayStorageError::EnvironmentCountNotSet)?;
        let batch = replay.sample()?;
        let device = self.replay_storage_config.optimization_device();
        Ok(DeterministicBatch {
            observations: batch.observations.to_device(&device),
            next_observations: batch.next_observations.to_device(&device),
            actions: batch.actions.to_device(&device),
            rewards: batch.rewards.to_device(&device),
            terminated: transfer_bool_tensor(batch.terminated, &device),
        })
    }

    /// Computes detached targets [batch_size, 1] from replay rewards, Bool termination flags, and native next observations.
    /// Smooths target policy actions before clipping. Truncation alone retains bootstrapping; all Float statistics use compute dtype.
    fn bellman_targets(
        &self,
        batch: &DeterministicBatch<O, A, OS::Kind>,
    ) -> Result<Tensor<2>, DeterministicActorCriticError<AE, GE, SE, CE>> {
        let count = batch.rewards.dims()[0];
        let latent = self
            .target_actor
            .forward(batch.next_observations.clone())
            .map_err(DeterministicActorCriticError::ActorError)?
            .detach();
        let actions = self
            .decode_actions(
                add_gaussian_noise(
                    &latent,
                    self.strategy.target_policy_noise(),
                    Some(self.strategy.target_noise_clip()),
                ),
                count,
            )?
            .detach();
        let values = self
            .critics
            .iter()
            .map(|critic| {
                critic
                    .target_replay_values::<O, A, C>(&batch.next_observations, &actions)
                    .map_err(DeterministicActorCriticError::CriticError)
            })
            .collect::<Result<Vec<_>, _>>()?;
        for value in &values {
            self.validate_float_tensor("target critic values", value, [count, 1])?;
        }
        let values = self.strategy.aggregate_target_values(values)?;
        self.validate_float_tensor("aggregated target values", &values, [count, 1])?;
        Ok(bellman_targets(
            batch.rewards.clone(),
            batch.terminated.clone(),
            values.detach(),
            self.gamma,
        )
        .detach())
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            name = "deterministic_actor_critic.optimize_critics",
            target = "modurl::performance",
            level = "trace",
            skip_all
        )
    )]
    /// Optimizes critics from native replay batches and detached targets [batch_size, 1].
    /// Uses full mean squared error and returns detached losses [1] and predictions [batch_size, 1].
    fn optimize_critics(
        &mut self,
        batch: &DeterministicBatch<O, A, OS::Kind>,
        targets: &Tensor<2>,
    ) -> Result<CriticUpdate, DeterministicActorCriticError<AE, GE, SE, CE>> {
        let mut predictions = Vec::with_capacity(self.critics.len());
        for critic in &self.critics {
            let predicted = critic
                .online_replay_values::<O, A, C>(&batch.observations, &batch.actions)
                .map_err(DeterministicActorCriticError::CriticError)?;
            self.validate_float_tensor("critic values", &predicted, targets.dims())?;
            predictions.push(predicted);
        }
        let mut losses = Vec::with_capacity(self.critics.len());
        let mut q_values = Vec::with_capacity(self.critics.len());
        for (critic, predicted) in self.critics.iter_mut().zip(predictions) {
            let loss = (predicted.clone() - targets.clone().detach())
                .square()
                .mean();
            critic.backward_step(&loss)?;
            losses.push(loss.detach());
            q_values.push(predicted.detach());
        }
        Ok(CriticUpdate { losses, q_values })
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            name = "deterministic_actor_critic.optimize_actor",
            target = "modurl::performance",
            level = "trace",
            skip_all
        )
    )]
    /// Optimizes the actor from native observations [batch_size, ...observation_shape].
    /// Adds one candidate axis to Float actions, then removes it from critic values [batch_size, 1, 1].
    /// Keeps action gradients and excludes critic-parameter gradients. Returns detached loss [1], values [batch_size, 1], and actions of rank A.
    fn optimize_actor(
        &mut self,
        observations: &Tensor<O, OS::Kind>,
    ) -> Result<ActorUpdate<A>, DeterministicActorCriticError<AE, GE, SE, CE>> {
        let count = observations.dims()[0];
        let latent = self
            .online_actor
            .forward(observations.clone())
            .map_err(DeterministicActorCriticError::ActorError)?;
        let actions = self.decode_actions(latent, count)?;
        let candidates = actions.clone().unsqueeze_dim::<C>(1);
        let mut values = Vec::with_capacity(self.critics.len());
        for critic in &self.critics {
            let value = critic
                .online_actor_values::<O, A, C>(observations, &candidates)
                .map_err(DeterministicActorCriticError::CriticError)?;
            self.validate_float_tensor("actor critic values", &value, [count, 1, 1])?;
            values.push(value.squeeze_dim::<2>(1));
        }
        let values = self.strategy.aggregate_actor_values(values)?;
        self.validate_float_tensor("aggregated actor values", &values, [count, 1])?;
        let loss = -values.clone().mean();
        let gradients = GradientsParams::from_grads(loss.clone().backward(), &self.online_actor);
        self.online_actor = self.actor_optimizer.step(
            self.actor_learning_rate,
            self.online_actor.clone(),
            gradients,
        );
        Ok(ActorUpdate {
            loss: loss.detach(),
            q_values: values.detach(),
            actions: actions.detach(),
        })
    }

    fn update_targets(&mut self) -> Result<(), DeterministicActorCriticError<AE, GE, SE, CE>> {
        self.target_actor =
            interpolate_target_model(&self.online_actor, self.target_actor.clone(), self.tau);
        for critic in &mut self.critics {
            critic.polyak_update(self.tau)?;
        }
        Ok(())
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            name = "deterministic_actor_critic.optimize",
            target = "modurl::performance",
            skip_all,
            fields(collection_timestep)
        )
    )]
    fn optimize<I>(
        &mut self,
        collection_timestep: usize,
        logger: &mut dyn DeterministicActorCriticLogger<I, A>,
    ) -> Result<(), DeterministicActorCriticError<AE, GE, SE, CE>> {
        if !self
            .replay
            .as_ref()
            .is_some_and(DeterministicReplay::can_sample)
        {
            return Ok(());
        }
        let batch = self.sample_batch()?;
        let targets = self.bellman_targets(&batch)?;
        let critic_update = self.optimize_critics(&batch, &targets)?;
        let actor_updated =
            (self.optimization_steps + 1).is_multiple_of(self.strategy.actor_update_interval());
        let actor_update = actor_updated
            .then(|| self.optimize_actor(&batch.observations))
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
            actor_learning_rate: self.actor_learning_rate as f32,
            critic_learning_rates: self
                .critics
                .iter()
                .map(|critic| critic.learning_rate() as f32)
                .collect(),
            exploration_noise_standard_deviation: self.exploration_noise,
            actor_updated,
            update_index: self.optimization_steps,
            collection_timestep,
        });
        self.optimization_steps += 1;
        Ok(())
    }

    /// Stores native observations [num_envs, ...observation_shape], Float actions [num_envs, ...action_shape], and rewards [num_envs, 1].
    /// Establishes replay dtype from the first batch, detaches experience, and keeps termination distinct from truncation.
    /// Reads rewards to the host for episode summaries after successful replay insertion.
    fn store_transitions(
        &mut self,
        transitions: CollectedTransitions<'_, O, A, OS::Kind>,
        episodes: &mut EpisodeTracker,
    ) -> Result<
        Vec<DeterministicActorCriticEpisodeLogEntry>,
        DeterministicActorCriticError<AE, GE, SE, CE>,
    > {
        let count = transitions.dones.len();
        let reward_values = transitions
            .rewards
            .clone()
            .try_into_data_as::<f32>()?
            .try_to_vec::<f32>()
            .map_err(TensorReadError::from)?;
        if self.replay.is_none() {
            let storage = DeterministicReplayStorage::builder()
                .observation_shape(space_storage_shape::<O>(
                    "observations",
                    self.observation_space.shape(),
                    self.replay_capacity,
                )?)
                .action_shape(space_storage_shape::<A>(
                    "actions",
                    self.action_space.shape(),
                    self.replay_capacity,
                )?)
                .observation_dtype(transitions.observations.dtype())
                .environment_count(count)
                .dtype(self.dtype)
                .device(self.replay_storage_config.storage_device())
                .build()?;
            self.replay = Some(DeterministicReplay::new(storage, self.batch_size));
        }
        let replay = self
            .replay
            .as_mut()
            .ok_or(ReplayStorageError::EnvironmentCountNotSet)?;
        replay.insert(DeterministicReplayInsert {
            observations: transitions.observations.clone(),
            next_observations: transitions.next_observations.clone(),
            actions: transitions.actions.clone(),
            rewards: transitions.rewards.clone(),
            terminated: Tensor::from_data(
                TensorData::new(transitions.dones.to_vec(), [count, 1]),
                &self.replay_storage_config.storage_device(),
            ),
            truncateds: transitions.truncateds.to_vec(),
        })?;
        self.observation_dtype = Some(transitions.observations.dtype());
        let mut completed = Vec::new();
        for (index, reward) in reward_values.into_iter().enumerate() {
            if let Some(entry) = episodes.record(
                index,
                reward,
                transitions.dones[index],
                transitions.truncateds[index],
                transitions.first_timestep + index + 1,
            ) {
                completed.push(entry);
            }
        }
        Ok(completed)
    }

    /// Trains with native observations [num_envs, ...observation_shape] and Float actions [num_envs, ...action_shape].
    /// Terminal observations have rank U = O - 1. Preserves terminal targets, update cadence, and schedule progress across calls.
    /// Resets collection on each call; validates replay alignment and metadata before writing experience.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            name = "deterministic_actor_critic.learn",
            target = "modurl::performance",
            skip_all,
            fields(num_timesteps)
        )
    )]
    pub(crate) fn learn<I, K: Autodiff, const U: usize>(
        &mut self,
        env: &mut dyn MultiGym<I, O, A, Error = GE, ObservationSpace = OS, ActionSpace = AS>,
        num_timesteps: usize,
        logger: &mut dyn DeterministicActorCriticLogger<I, A>,
    ) -> Result<(), DeterministicActorCriticError<AE, GE, SE, CE>>
    where
        OS: ObservationSpace<O, Kind = K, Item = Tensor<U, K>>,
        Tensor<U, K>: NextRank<Next = Tensor<O, K>>,
    {
        const {
            assert!(
                O == U + 1,
                "terminal observation rank must omit the batch axis"
            );
        }
        let count = env.num_envs();
        validate_replay_alignment(self.replay_capacity, count)?;
        if let Some(expected) = self.environment_count
            && expected != count
        {
            return Err(ReplayStorageError::EnvironmentCountMismatch {
                expected,
                actual: count,
            }
            .into());
        }
        if let Some(replay) = &mut self.replay {
            replay.validate_environment_count(count)?;
        }
        let mut observations = env
            .reset()
            .map_err(DeterministicActorCriticError::GymError)?;
        self.validate_observations(&observations)?;
        if observations.dims()[0] != count {
            return Err(DeterministicActorCriticError::Shape {
                field: "environment observations",
                expected: vec![count],
                actual: vec![observations.dims()[0]],
            });
        }
        self.environment_count = Some(count);
        let mut episodes = EpisodeTracker::new(count);
        let mut elapsed = 0;
        while elapsed < num_timesteps {
            let first_timestep = self.schedule_progress.elapsed_steps();
            let actions = if first_timestep < self.training_start {
                self.random_actions(count)?
            } else {
                self.act(&observations)?
            }
            .without_autodiff();
            let step = env
                .step(actions.clone())
                .map_err(DeterministicActorCriticError::GymError)?;
            for (field, actual) in [
                ("done flags", step.dones.len()),
                ("truncation flags", step.truncateds.len()),
                ("terminal observations", step.terminal_observations.len()),
                ("environment infos", step.infos.len()),
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
            self.validate_observations(&step.observations)?;
            let expected =
                space_storage_shape::<O>("observations", self.observation_space.shape(), count)?;
            if step.observations.dims() != expected {
                return Err(DeterministicActorCriticError::Shape {
                    field: "environment observations",
                    expected: expected.to_vec(),
                    actual: step.observations.dims().to_vec(),
                });
            }
            if step.rewards.dims() != [count, 1] {
                return Err(DeterministicActorCriticError::Shape {
                    field: "environment rewards",
                    expected: vec![count, 1],
                    actual: step.rewards.dims().to_vec(),
                });
            }
            if step.observations.dtype() != observations.dtype() {
                return Err(DeterministicActorCriticError::DType {
                    field: "environment observations",
                    expected: observations.dtype(),
                    actual: step.observations.dtype(),
                });
            }
            for terminal in step.terminal_observations.iter().flatten() {
                let expected = self.observation_space.shape();
                if terminal.dims().to_vec() != expected {
                    return Err(DeterministicActorCriticError::Shape {
                        field: "terminal observations",
                        expected,
                        actual: terminal.dims().to_vec(),
                    });
                }
                if terminal.dtype() != step.observations.dtype() {
                    return Err(DeterministicActorCriticError::DType {
                        field: "terminal observations",
                        expected: step.observations.dtype(),
                        actual: terminal.dtype(),
                    });
                }
                if terminal.device() != step.observations.device() {
                    return Err(DeterministicActorCriticError::Device {
                        field: "terminal observations",
                    });
                }
            }
            let next_observations = step.transition_next_observations();
            let MultiGymStepInfo {
                observations: reset_observations,
                rewards,
                infos,
                dones,
                truncateds,
                ..
            } = step;
            let completed_episodes = self.store_transitions(
                CollectedTransitions {
                    observations: &observations,
                    next_observations: &next_observations,
                    actions: &actions,
                    rewards: &rewards,
                    dones: &dones,
                    truncateds: &truncateds,
                    first_timestep,
                },
                &mut episodes,
            )?;
            self.schedule_progress.advance_steps(count);
            for offset in 1..=count {
                let timestep = first_timestep + offset;
                if timestep >= self.training_start && timestep.is_multiple_of(self.update_frequency)
                {
                    self.optimize(timestep, logger)?;
                }
            }
            elapsed += count;
            observations = reset_observations;
            logger.log_collection(&DeterministicActorCriticCollectionLogEntry {
                collection_rewards: rewards.without_autodiff(),
                infos,
                collection_timestep: self.schedule_progress.elapsed_steps(),
                completed_episodes,
                replay_len: self.replay.as_ref().map_or(0, DeterministicReplay::len),
            });
        }
        Ok(())
    }
}

/// Builds rank-R storage shape [capacity, ...item_shape] from exactly R - 1 nonzero item axes.
fn space_storage_shape<const R: usize>(
    field: &'static str,
    item_shape: Vec<usize>,
    capacity: usize,
) -> Result<[usize; R], DeterministicActorCriticConfigurationError> {
    const {
        assert!(R >= 2, "storage requires capacity and item axes");
    }
    if item_shape.len() != R - 1 {
        return Err(DeterministicActorCriticConfigurationError::SpaceRank {
            field,
            expected: R - 1,
            actual: item_shape.len(),
        });
    }
    if item_shape.contains(&0) {
        return Err(DeterministicActorCriticConfigurationError::ZeroItemDimension { field });
    }
    let mut shape = [capacity; R];
    shape[1..].copy_from_slice(&item_shape);
    Ok(shape)
}

/// Adds elementwise Gaussian noise to rank-D Float values, preserving shape, dtype, device, and input gradients.
/// Samples F32 noise with the existing device RNG, casts to the input dtype, then optionally clips the noise.
fn add_gaussian_noise<const D: usize>(
    values: &Tensor<D>,
    standard_deviation: f64,
    clip: Option<f64>,
) -> Tensor<D> {
    if standard_deviation == 0.0 {
        return values.clone();
    }
    let mut noise = Tensor::<D>::random(
        values.dims(),
        Distribution::Normal(0.0, 1.0),
        (&values.device(), DType::F32),
    )
    .cast(values.dtype())
        * standard_deviation;
    if let Some(clip) = clip {
        noise = noise.clamp(-clip, clip);
    }
    values.clone() + noise
}

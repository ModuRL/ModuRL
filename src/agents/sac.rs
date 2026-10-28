//! Generalized Soft Actor-Critic with independent critic optimizers and exact or sampled policy expectations.

mod replay;

pub use replay::SACReplayError;

use bon::bon;
use burn::tensor::kind::TensorKind;
use burn::{
    module::{Module, Param},
    optim::{GradientsParams, ModuleOptimizer},
    tensor::{Bool, Cast, DType, Device, Tensor, TensorData, TensorReadError, kind::Autodiff},
};
use std::{fmt::Debug, marker::PhantomData, num::NonZeroUsize};

use self::replay::{SACOptimizationBatch, SACReplay, SACReplayInsert, SACReplayStorage};
use super::{
    Agent, ReplayStorageConfig,
    action_value_critic::{
        ActionValueCritic, ActionValueCriticAggregationMode, ActionValueCriticError,
        ActionValueCriticNetwork, aggregate_critic_values,
    },
};
use crate::{
    buffers::experience_replay::{
        ExperienceReplayError, ReplayStorageError, validate_replay_alignment,
    },
    gym::{MultiGym, MultiGymStepInfo},
    models::probabilistic_model::ExpectationPolicy,
    objectives::{bellman_targets, clipped_value_loss},
    parameter_schedule::{ParameterSchedule, ScheduleProgress},
    spaces::{ActionMap, ObservationSpace},
    tensor_operations::transfer_bool_tensor,
    tensor_rank::{NextRank, PrevRank},
};

#[derive(Debug, thiserror::Error)]
pub enum SACConfigurationError {
    #[error("at least one critic is required")]
    NoCritics,
    #[error("replay capacity must be nonzero")]
    ZeroReplayCapacity,
    #[error("batch size must be nonzero")]
    ZeroBatchSize,
    #[error("replay capacity must be at least the batch size")]
    ReplayCapacityBelowBatchSize,
    #[error("gamma must be finite and in 0..=1")]
    InvalidGamma,
    #[error("tau must be finite and in 0..=1")]
    InvalidTau,
    #[error("alpha must be finite and nonnegative")]
    InvalidAlpha,
    #[error("log alpha must be finite")]
    InvalidLogAlpha,
    #[error("log alpha must have shape [1]")]
    InvalidLogAlphaShape,
    #[error("learning rate must be finite and positive")]
    InvalidLearningRate,
    #[error("the target entropy schedule must return finite values")]
    InvalidTargetEntropySchedule,
    #[error("entropy change penalty must be finite and nonnegative")]
    InvalidEntropyChangePenalty,
    #[error("Q-value clip must be finite and positive")]
    InvalidQValueClip,
    #[error("training horizon must be nonzero")]
    ZeroTrainingHorizon,
    #[error("compute dtype must be floating-point, got {0:?}")]
    InvalidDType(DType),
    #[error("space {field} has {actual} event axes, expected {expected}")]
    SpaceRank {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("space event axes must be nonempty")]
    EmptySpace,
    #[error("log alpha dtype differs: expected {expected:?}, got {actual:?}")]
    LogAlphaDType { expected: DType, actual: DType },
    #[error("model parameters must share one device")]
    ModelDevice,
    #[error("reading entropy configuration failed: {0}")]
    Tensor(#[from] TensorReadError),
}

#[derive(Debug, thiserror::Error)]
pub enum SACError<PE, GE, SE, CE = std::convert::Infallible> {
    #[error("policy failed: {0}")]
    PolicyError(#[source] PE),
    #[error("gym failed: {0}")]
    GymError(#[source] GE),
    #[error("space operation failed: {0}")]
    SpaceError(#[source] SE),
    #[error("SAC tensor read failed: {0}")]
    TensorError(#[from] TensorReadError),
    #[error("replay storage failed: {0}")]
    ReplayStorageError(#[from] SACReplayError),
    #[error("critic network failed: {0}")]
    CriticError(#[source] CE),
    #[error("critic operation failed: {0}")]
    CriticOperationError(#[from] ActionValueCriticError),
    #[error("invalid SAC configuration: {0}")]
    ConfigurationError(#[from] SACConfigurationError),
    #[error("SAC {field} shape differs: expected {expected:?}, got {actual:?}")]
    Shape {
        field: &'static str,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("SAC {field} dtype differs: expected {expected:?}, got {actual:?}")]
    DType {
        field: &'static str,
        expected: DType,
        actual: DType,
    },
    #[error("SAC {field} uses an incompatible device")]
    Device { field: &'static str },
    #[error("SAC observation batches must be nonempty")]
    EmptyBatch,
}

impl<PE, GE, SE, CE> From<ExperienceReplayError<SACReplayError>> for SACError<PE, GE, SE, CE> {
    fn from(error: ExperienceReplayError<SACReplayError>) -> Self {
        match error {
            ExperienceReplayError::ExperienceError(error) => error.into(),
            ExperienceReplayError::InsertionExceedsCapacity { capacity, inserted } => {
                SACReplayError::Storage(ReplayStorageError::InsertionExceedsCapacity {
                    capacity,
                    inserted,
                })
                .into()
            }
        }
    }
}

/// Configures a fixed entropy coefficient or a separately optimized scalar log coefficient.
pub struct SACEntropyConfiguration {
    kind: SACEntropyConfigurationKind,
}

enum SACEntropyConfigurationKind {
    Automatic {
        log_alpha: Param<Tensor<1>>,
        optimizer: ModuleOptimizer,
        learning_rate: f64,
        target_entropy_schedule: Option<Box<dyn ParameterSchedule>>,
    },
    Fixed {
        alpha: f64,
    },
}

impl SACEntropyConfiguration {
    /// Owns Float log alpha `[1]` and its optimizer. Enables its gradients while preserving dtype and device.
    /// Rejects a nonscalar layout, non-finite log alpha, or a nonpositive or non-finite learning rate.
    pub fn automatic(
        log_alpha: Param<Tensor<1>>,
        optimizer: impl Into<ModuleOptimizer>,
        learning_rate: f64,
    ) -> Result<Self, SACConfigurationError> {
        if log_alpha.val().dims() != [1] {
            return Err(SACConfigurationError::InvalidLogAlphaShape);
        }
        if !log_alpha.val().try_into_scalar::<f64>()?.is_finite() {
            return Err(SACConfigurationError::InvalidLogAlpha);
        }
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(SACConfigurationError::InvalidLearningRate);
        }

        Ok(Self {
            kind: SACEntropyConfigurationKind::Automatic {
                log_alpha: log_alpha.train(),
                optimizer: optimizer.into(),
                learning_rate,
                target_entropy_schedule: None,
            },
        })
    }

    /// Owns Float log alpha `[1]`, its optimizer, and a target-entropy schedule over collected transitions.
    /// Preserves dtype and device; rejects invalid automatic settings or non-finite schedule endpoints.
    pub fn automatic_with_target_schedule(
        log_alpha: Param<Tensor<1>>,
        optimizer: impl Into<ModuleOptimizer>,
        learning_rate: f64,
        target_entropy_schedule: impl ParameterSchedule + 'static,
    ) -> Result<Self, SACConfigurationError> {
        if !target_entropy_schedule.value(0.0).is_finite()
            || !target_entropy_schedule.value(1.0).is_finite()
        {
            return Err(SACConfigurationError::InvalidTargetEntropySchedule);
        }

        let mut configuration = Self::automatic(log_alpha, optimizer, learning_rate)?;
        if let SACEntropyConfigurationKind::Automatic {
            target_entropy_schedule: schedule,
            ..
        } = &mut configuration.kind
        {
            *schedule = Some(Box::new(target_entropy_schedule));
        }

        Ok(configuration)
    }

    /// Uses a constant entropy coefficient. Rejects a negative or non-finite alpha.
    pub fn fixed(alpha: f64) -> Result<Self, SACConfigurationError> {
        if !alpha.is_finite() || alpha < 0.0 {
            return Err(SACConfigurationError::InvalidAlpha);
        }

        Ok(Self {
            kind: SACEntropyConfigurationKind::Fixed { alpha },
        })
    }

    /// Returns detached Float alpha `[1]` on the reference tensor's device and in its dtype.
    /// The rank-D reference keeps its layout; this scalar broadcasts over batch, candidate, and event axes.
    fn alpha_tensor<const D: usize>(&self, like: &Tensor<D>) -> Tensor<1> {
        match &self.kind {
            SACEntropyConfigurationKind::Automatic { log_alpha, .. } => log_alpha
                .val()
                .detach()
                .exp()
                .to_device(&like.device())
                .cast(like.dtype()),
            SACEntropyConfigurationKind::Fixed { alpha } => {
                Tensor::full([1], *alpha, (&like.device(), like.dtype()))
            }
        }
    }
}

/// Returns Float loss `[1]` from log alpha `[1]` and expected log probabilities `[batch_size, 1]`.
/// Requires matching dtype and device. Detaches the entropy constraint; only log alpha receives gradients.
fn sac_alpha_loss(
    log_alpha: &Tensor<1>,
    expected_log_probability: &Tensor<2>,
    target_entropy: f64,
) -> Tensor<1> {
    -(log_alpha.clone().reshape([1, 1])
        * (expected_log_probability.clone().detach() + target_entropy))
        .mean()
}

/// Returns Float entropy-drift loss `[1]` from three matching tensors `[batch_size, 1]`.
/// Preserves current-entropy gradients. Collection entropy and weights are fixed; dtype and device must match.
fn sac_entropy_change_loss(
    current_entropy: &Tensor<2>,
    collection_entropy: &Tensor<2>,
    weights: &Tensor<2>,
    coefficient: f64,
) -> Tensor<1> {
    ((current_entropy.clone() - collection_entropy.clone().detach()).square()
        * weights.clone().detach())
    .sum()
        / weights.clone().detach().sum().clamp_min(1.0)
        * coefficient
}

struct SACActorUpdate {
    loss: Tensor<1>,
    entropy_change_loss: Option<Tensor<1>>,
    log_probabilities: Tensor<3>,
    weights: Tensor<3>,
    q_values: Tensor<3>,
}

struct SACTemperatureUpdate {
    loss: Option<Tensor<1>>,
    target_entropy: Option<f64>,
}

struct SACCollectionEntropy {
    values: Tensor<2>,
    change_weight: f32,
}

/// Optimization losses and alpha have shape `[1]`; candidate statistics use `[batch_size, candidate_count, 1]`.
/// Bellman targets and replay rewards use `[batch_size, 1]`. All tensors use the compute dtype and optimization device and are detached.
pub struct SACLogEntry {
    pub critic_losses: Vec<Tensor<1>>,
    pub actor_loss: Tensor<1>,
    pub alpha_loss: Option<Tensor<1>>,
    pub entropy_change_loss: Option<Tensor<1>>,
    pub target_entropy: Option<f64>,
    pub alpha: Tensor<1>,
    pub bellman_targets: Tensor<2>,
    pub policy_log_probabilities: Tensor<3>,
    pub policy_weights: Tensor<3>,
    pub policy_q_values: Tensor<3>,
    pub replay_rewards: Tensor<2>,
    pub update_index: usize,
    pub collection_timestep: usize,
}

/// Collection rewards `[environment_count, 1]` preserve the environment reward dtype and device without gradients.
pub struct SACCollectionLogEntry<I = ()> {
    pub collection_rewards: Tensor<2>,
    pub infos: Vec<I>,
    pub collection_timestep: usize,
    pub completed_episodes: Vec<SACEpisodeLogEntry>,
    pub replay_len: usize,
}

#[derive(Debug, Clone)]
pub struct SACEpisodeLogEntry {
    pub environment_index: usize,
    pub episode_return: f32,
    pub episode_length: usize,
    pub collection_timestep: usize,
    pub terminated: bool,
    pub truncated: bool,
}

pub trait SACLogger<I = ()> {
    fn log_update(&mut self, entry: &SACLogEntry);

    fn log_collection(&mut self, entry: &SACCollectionLogEntry<I>);
}

impl<I> SACLogger<I> for Option<&mut dyn SACLogger<I>> {
    fn log_update(&mut self, entry: &SACLogEntry) {
        if let Some(logger) = self {
            logger.log_update(entry);
        }
    }

    fn log_collection(&mut self, entry: &SACCollectionLogEntry<I>) {
        if let Some(logger) = self {
            logger.log_collection(entry);
        }
    }
}

struct SACEpisodeTracker {
    returns: Vec<f32>,
    lengths: Vec<usize>,
}

impl SACEpisodeTracker {
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
    ) -> Option<SACEpisodeLogEntry> {
        self.returns[environment_index] += reward;
        self.lengths[environment_index] += 1;
        if !terminated && !truncated {
            return None;
        }

        Some(SACEpisodeLogEntry {
            environment_index,
            episode_return: std::mem::take(&mut self.returns[environment_index]),
            episode_length: std::mem::take(&mut self.lengths[environment_index]),
            collection_timestep,
            terminated,
            truncated,
        })
    }
}

/// Owns a policy, independent critics, entropy tuning, and replay for Soft Actor-Critic.
/// O and A include environment batch axes; L is the sampled policy rank and C is the candidate rank, with C = A + 1.
/// Observations retain their native kind and dtype. Models perform input conversions; dtype configures floating-point training statistics.
/// Replay storage is allocated for the first collected transition and retained across learn calls.
/// Groups the owned policy, critic model, native execution tensors, and replay type without adding execution methods.
pub trait SACModelTypes {
    type Policy: Module;
    type Critic: Module;
    type Observation;
    type Action;
    type PolicyAction;
    type Candidates;
    type Replay;
}

/// Selects policy, critic, and native tensor types; the agent stores all actual models and tensors directly.
pub struct SACModelGroup<P, N, O = Tensor<2>, A = Tensor<2>, L = Tensor<2>, C = Tensor<3>> {
    models: PhantomData<(P, N)>,
    tensors: PhantomData<(O, A, L, C)>,
}

impl<
    P: Module,
    N: Module,
    K: Autodiff,
    H: Autodiff,
    const O: usize,
    const A: usize,
    const L: usize,
    const C: usize,
> SACModelTypes for SACModelGroup<P, N, Tensor<O, K>, Tensor<A, H>, Tensor<L>, Tensor<C, H>>
where
    DType: Cast<A, H, OutputKind = H>,
{
    type Policy = P;
    type Critic = N;
    type Observation = Tensor<O, K>;
    type Action = Tensor<A, H>;
    type PolicyAction = Tensor<L>;
    type Candidates = Tensor<C, H>;
    type Replay = SACReplay<O, A, K, H>;
}

/// Infers a SAC model group from a concrete policy, critic, and native execution tensors supplied to the builder.
pub trait SACPolicyModelTypes<N: Module, O, A, L, C>: Module {
    type ModelGroup: SACModelTypes<Policy = Self, Critic = N>;
}

impl<
    P: Module,
    N: Module,
    K: Autodiff,
    H: Autodiff,
    const O: usize,
    const A: usize,
    const L: usize,
    const C: usize,
> SACPolicyModelTypes<N, Tensor<O, K>, Tensor<A, H>, Tensor<L>, Tensor<C, H>> for P
where
    DType: Cast<A, H, OutputKind = H>,
{
    type ModelGroup = SACModelGroup<P, N, Tensor<O, K>, Tensor<A, H>, Tensor<L>, Tensor<C, H>>;
}

/// Owns SAC models through an associated-type group; environment spaces and action rank remain explicit.
/// Observation, policy-action, and candidate ranks are inferred by the implementation from the native tensor types.
pub struct SACAgent<'a, G: SACModelTypes, OS, AS, GE, I = (), const A: usize = 2> {
    policy: G::Policy,
    actor_optimizer: ModuleOptimizer,
    actor_learning_rate: f64,
    critics: Vec<ActionValueCritic<G::Critic>>,
    entropy_configuration: SACEntropyConfiguration,
    aggregation_mode: ActionValueCriticAggregationMode,
    action_space: AS,
    observation_space: OS,
    replay: Option<G::Replay>,
    replay_capacity: usize,
    batch_size: usize,
    gamma: f64,
    tau: f64,
    training_start: usize,
    samples: NonZeroUsize,
    entropy_change_penalty: Option<f64>,
    q_value_clip: Option<f64>,
    schedule_progress: ScheduleProgress,
    replay_storage_config: ReplayStorageConfig,
    optimization_steps: usize,
    logger: Option<&'a mut dyn SACLogger<I>>,
    dtype: DType,
    _errors: PhantomData<GE>,
}

#[bon]
impl<'a, G: SACModelTypes, OS, AS, GE, I, const A: usize> SACAgent<'a, G, OS, AS, GE, I, A> {
    /// Builds SAC for observations `[batch_size, ...observation_shape]` and environment actions `[batch_size, ...action_shape]`.
    /// Preserves observation kind and dtype; policy and critic models own input conversions. Candidate actions add one axis after batch.
    /// Models and automatic log alpha must use the optimization device. Critic outputs must use dtype; replay transfers detached tensors to its storage device.
    /// Rejects invalid sizes, ranks, dtypes, devices, learning rates, and parameter ranges.
    #[builder]
    pub fn new<
        P,
        N,
        PE: Debug,
        CE: Debug,
        SE: Debug,
        const O: usize,
        const L: usize,
        const C: usize,
    >(
        policy: P,
        #[builder(into)] actor_optimizer: ModuleOptimizer,
        actor_learning_rate: f64,
        critics: Vec<ActionValueCritic<N>>,
        entropy_configuration: SACEntropyConfiguration,
        action_space: AS,
        observation_space: OS,
        replay_device: Option<Device>,
        logger: Option<&'a mut dyn SACLogger<I>>,
        aggregation_mode: Option<ActionValueCriticAggregationMode>,
        entropy_change_penalty: Option<f64>,
        q_value_clip: Option<f64>,
        #[builder(default = 0.99)] gamma: f64,
        #[builder(default = 0.005)] tau: f64,
        #[builder(default = 1_000_000)] replay_capacity: usize,
        #[builder(default = 256)] batch_size: usize,
        #[builder(default = 1_000)] training_start: usize,
        #[builder(default = NonZeroUsize::MIN)] samples: NonZeroUsize,
        training_horizon: Option<usize>,
        #[builder(default = DType::F32)] dtype: DType,
    ) -> Result<Self, SACConfigurationError>
    where
        OS: ObservationSpace<O>,
        AS: ActionMap<L, A, Error = SE>,
        OS::Kind: Autodiff,
        AS::Kind: Autodiff,
        Tensor<A, AS::Kind>: NextRank<Next = Tensor<C, AS::Kind>>,
        DType: Cast<A, AS::Kind, OutputKind = AS::Kind>,
        P: Module
            + ExpectationPolicy<
                O,
                L,
                C,
                ObservationKind = OS::Kind,
                CandidateKind = AS::Kind,
                Error = PE,
            >,
        N: ActionValueCriticNetwork<
                O,
                A,
                C,
                ObservationKind = OS::Kind,
                ActionKind = AS::Kind,
                Error = CE,
            >,
        GE: Debug,
        P: SACPolicyModelTypes<
                N,
                Tensor<O, OS::Kind>,
                Tensor<A, AS::Kind>,
                Tensor<L>,
                Tensor<C, AS::Kind>,
                ModelGroup = G,
            >,
        G: SACModelTypes<Policy = P, Critic = N>,
    {
        const {
            assert!(
                O >= 2 && A >= 2 && L >= 2 && C == A + 1,
                "SAC requires batch and event axes; candidates add one axis to actions"
            );
        }

        validate_sac_configuration()
            .critic_count(critics.len())
            .replay_capacity(replay_capacity)
            .batch_size(batch_size)
            .gamma(gamma)
            .tau(tau)
            .actor_learning_rate(actor_learning_rate)
            .dtype(dtype)
            .maybe_entropy_change_penalty(entropy_change_penalty)
            .maybe_q_value_clip(q_value_clip)
            .maybe_training_horizon(training_horizon)
            .call()?;

        let replay_storage_config = ReplayStorageConfig::from_models(
            policy.devices().into_iter().chain(
                critics
                    .iter()
                    .flat_map(|critic| critic.online_network().devices()),
            ),
            replay_device,
        )
        .ok_or(SACConfigurationError::ModelDevice)?;
        let optimization_device = replay_storage_config.optimization_device();

        if let SACEntropyConfigurationKind::Automatic { log_alpha, .. } =
            &entropy_configuration.kind
        {
            if log_alpha.val().device() != optimization_device {
                return Err(SACConfigurationError::ModelDevice);
            }
            if log_alpha.val().dtype() != dtype {
                return Err(SACConfigurationError::LogAlphaDType {
                    expected: dtype,
                    actual: log_alpha.val().dtype(),
                });
            }
        }

        space_storage_shape::<O>("observations", observation_space.shape(), replay_capacity)?;
        space_storage_shape::<A>("actions", action_space.shape(), replay_capacity)?;
        space_storage_shape::<L>(
            "policy actions",
            action_space.policy_shape(),
            replay_capacity,
        )?;

        Ok(Self {
            policy: policy.train(),
            actor_optimizer,
            actor_learning_rate,
            critics,
            entropy_configuration,
            aggregation_mode: aggregation_mode.unwrap_or(ActionValueCriticAggregationMode::Min),
            action_space,
            observation_space,
            replay: None,
            replay_capacity,
            batch_size,
            gamma,
            tau,
            training_start,
            samples,
            entropy_change_penalty,
            q_value_clip,
            schedule_progress: ScheduleProgress::new(training_horizon),
            replay_storage_config,
            optimization_steps: 0,
            logger,
            dtype,
            _errors: PhantomData,
        })
    }
}

impl<
    'a,
    P,
    N,
    OS,
    AS,
    GE,
    I,
    PE: Debug,
    CE: Debug,
    SE: Debug,
    const O: usize,
    const A: usize,
    const L: usize,
    const C: usize,
>
    SACAgent<
        'a,
        SACModelGroup<
            P,
            N,
            Tensor<O, OS::Kind>,
            Tensor<A, AS::Kind>,
            Tensor<L>,
            Tensor<C, AS::Kind>,
        >,
        OS,
        AS,
        GE,
        I,
        A,
    >
where
    OS: ObservationSpace<O>,
    AS: ActionMap<L, A, Error = SE>,
    OS::Kind: Autodiff,
    AS::Kind: Autodiff,
    Tensor<A, AS::Kind>: NextRank<Next = Tensor<C, AS::Kind>>,
    DType: Cast<A, AS::Kind, OutputKind = AS::Kind>,
    P: Module
        + ExpectationPolicy<O, L, C, ObservationKind = OS::Kind, CandidateKind = AS::Kind, Error = PE>,
    N: ActionValueCriticNetwork<
            O,
            A,
            C,
            ObservationKind = OS::Kind,
            ActionKind = AS::Kind,
            Error = CE,
        >,
    GE: Debug,
{
    pub fn get_action_space(&self) -> &AS {
        &self.action_space
    }

    pub fn get_observation_space(&self) -> &OS {
        &self.observation_space
    }

    /// Returns the owned policy for parameter inspection and checkpoint snapshots.
    pub fn policy(&self) -> &P {
        &self.policy
    }

    /// Returns the independently optimized critic ensemble.
    pub fn critics(&self) -> &[ActionValueCritic<N>] {
        &self.critics
    }

    /// Selects modal environment actions `[batch_size, ...action_shape]` from observations `[batch_size, ...observation_shape]`.
    /// Preserves observation kind and dtype, transfers to the optimization device, and detaches returned actions. Models perform conversions.
    pub fn act_deterministic(
        &mut self,
        observations: &Tensor<O, OS::Kind>,
    ) -> Result<Tensor<A, AS::Kind>, SACError<PE, GE, SE, CE>> {
        self.validate_observations(observations)?;

        let latent = self
            .policy
            .mode(
                observations
                    .clone()
                    .to_device(&self.replay_storage_config.optimization_device()),
            )
            .map_err(SACError::PolicyError)?;

        self.decode_actions(latent, observations.dims()[0])
    }

    /// Validates rank-O observations `[batch_size, ...observation_shape]`, preserving native kind, dtype, and device.
    fn validate_observations(
        &self,
        observations: &Tensor<O, OS::Kind>,
    ) -> Result<(), SACError<PE, GE, SE, CE>> {
        let mut expected = vec![observations.dims()[0]];
        expected.extend(self.observation_space.shape());
        if observations.dims()[0] == 0 {
            return Err(SACError::EmptyBatch);
        }
        if observations.dims().to_vec() != expected {
            return Err(SACError::Shape {
                field: "observations",
                expected,
                actual: observations.dims().to_vec(),
            });
        }
        Ok(())
    }

    /// Converts policy samples `[batch_size, ...policy_shape]` of rank L to detached environment actions `[batch_size, ...action_shape]` of rank A.
    /// The action map defines conversions; validates batch, event axes, and optimization device.
    fn decode_actions(
        &self,
        latent: Tensor<L>,
        batch_size: usize,
    ) -> Result<Tensor<A, AS::Kind>, SACError<PE, GE, SE, CE>> {
        let mut policy_shape = vec![batch_size];
        policy_shape.extend(self.action_space.policy_shape());
        if latent.dims().to_vec() != policy_shape {
            return Err(SACError::Shape {
                field: "policy actions",
                expected: policy_shape,
                actual: latent.dims().to_vec(),
            });
        }

        let actions = self
            .action_space
            .tensor_from_neurons(latent)
            .map_err(SACError::SpaceError)?;
        let mut expected = vec![batch_size];
        expected.extend(self.action_space.shape());
        if actions.dims().to_vec() != expected {
            return Err(SACError::Shape {
                field: "actions",
                expected,
                actual: actions.dims().to_vec(),
            });
        }
        if actions.device() != self.replay_storage_config.optimization_device() {
            return Err(SACError::Device { field: "actions" });
        }

        Ok(actions.without_autodiff())
    }

    /// Samples detached environment actions `[batch_size, ...action_shape]` from rank-O observations `[batch_size, ...observation_shape]`.
    /// Preserves observation kind and dtype while transferring to the optimization device; models perform input conversions.
    fn stochastic_action(
        &self,
        observations: &Tensor<O, OS::Kind>,
    ) -> Result<Tensor<A, AS::Kind>, SACError<PE, GE, SE, CE>> {
        self.validate_observations(observations)?;

        let latent = self
            .policy
            .sample(
                observations
                    .clone()
                    .to_device(&self.replay_storage_config.optimization_device()),
            )
            .map_err(SACError::PolicyError)?;

        self.decode_actions(latent, observations.dims()[0])
    }

    /// Computes detached soft Bellman targets `[batch_size, 1]` from native replay observations, actions, rewards, and Bool termination flags.
    /// Reduces policy candidates after critic aggregation; truncation alone retains bootstrapping. All statistics use dtype and the optimization device.
    fn compute_bellman_targets(
        &self,
        batch: &SACOptimizationBatch<O, A, OS::Kind, AS::Kind>,
    ) -> Result<Tensor<2>, SACError<PE, GE, SE, CE>> {
        let terms = self
            .policy
            .expectation(batch.next_observations.clone(), self.samples)
            .map_err(SACError::PolicyError)?;
        let actions = terms.actions().clone().without_autodiff();
        let logs = terms.log_probabilities().clone().detach().cast(self.dtype);
        let weights = terms.weights().clone().detach().cast(self.dtype);

        let values = self
            .critics
            .iter()
            .map(|critic| {
                critic
                    .target_policy_values::<O, A, C>(&batch.next_observations, &actions)
                    .map_err(SACError::CriticError)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let values = aggregate_critic_values::<3, 4>(&values, self.aggregation_mode)?;
        validate_critic_dtype(&values, self.dtype)?;

        let alpha = self
            .entropy_configuration
            .alpha_tensor(&logs)
            .reshape([1, 1, 1]);
        let next_values = ((values - logs * alpha) * weights)
            .sum_dim(1)
            .squeeze_dim::<2>(1)
            .detach();

        Ok(bellman_targets(
            batch.rewards.clone(),
            batch.terminated.clone(),
            next_values,
            self.gamma,
        )
        .detach())
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            name = "sac.optimize_critics",
            target = "modurl::performance",
            level = "trace",
            skip_all
        )
    )]
    /// Optimizes critics against detached targets `[batch_size, 1]`, returning one detached Float loss `[1]` per critic.
    /// Preserves half mean-squared-error and optional clipping around target-network replay values; all statistics use dtype and the optimization device.
    fn optimize_critics(
        &mut self,
        batch: &SACOptimizationBatch<O, A, OS::Kind, AS::Kind>,
        targets: &Tensor<2>,
    ) -> Result<Vec<Tensor<1>>, SACError<PE, GE, SE, CE>> {
        let mut losses = Vec::with_capacity(self.critics.len());

        for critic in &mut self.critics {
            let predicted = critic
                .online_replay_values::<O, A, C>(&batch.observations, &batch.actions)
                .map_err(SACError::CriticError)?;
            validate_critic_dtype(&predicted, self.dtype)?;

            let loss = match self.q_value_clip {
                Some(epsilon) => {
                    let anchor = critic
                        .target_replay_values::<O, A, C>(&batch.observations, &batch.actions)
                        .map_err(SACError::CriticError)?;
                    validate_critic_dtype(&anchor, self.dtype)?;
                    clipped_value_loss(
                        predicted,
                        targets.clone().detach(),
                        anchor.detach(),
                        epsilon,
                    ) * 0.5
                }
                None => (predicted - targets.clone().detach()).square().mean() * 0.5,
            };

            critic.backward_step(&loss)?;
            losses.push(loss.detach());
        }

        Ok(losses)
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            name = "sac.optimize_actor",
            target = "modurl::performance",
            level = "trace",
            skip_all
        )
    )]
    /// Optimizes the policy from replay observations `[batch_size, ...observation_shape]` and fixed entropy metadata `[batch_size, 1]`.
    /// Reduces candidates `[batch_size, candidate_count, 1]` to loss `[1]`; preserves policy/action gradients and excludes critic-parameter gradients.
    fn optimize_actor(
        &mut self,
        batch: &SACOptimizationBatch<O, A, OS::Kind, AS::Kind>,
    ) -> Result<SACActorUpdate, SACError<PE, GE, SE, CE>> {
        let terms = self
            .policy
            .expectation(batch.observations.clone(), self.samples)
            .map_err(SACError::PolicyError)?;

        let values = self
            .critics
            .iter()
            .map(|critic| {
                critic
                    .online_actor_values::<O, A, C>(&batch.observations, terms.actions())
                    .map_err(SACError::CriticError)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let values = aggregate_critic_values::<3, 4>(&values, self.aggregation_mode)?;
        let logs = terms.log_probabilities().clone().cast(self.dtype);
        let weights = terms.weights().clone().cast(self.dtype);
        validate_critic_dtype(&values, self.dtype)?;

        let alpha = self
            .entropy_configuration
            .alpha_tensor(&logs)
            .reshape([1, 1, 1]);
        let base_loss = ((logs.clone() * alpha - values.clone()) * weights.clone())
            .sum_dim(1)
            .mean();

        let entropy = -(logs.clone() * weights.clone())
            .sum_dim(1)
            .squeeze_dim::<2>(1);
        let entropy_change_loss = self.entropy_change_penalty.map(|coefficient| {
            sac_entropy_change_loss(
                &entropy,
                &batch.collection_policy_entropies,
                &batch.entropy_change_weights,
                coefficient,
            )
        });
        let loss = entropy_change_loss
            .as_ref()
            .map_or(base_loss.clone(), |penalty| base_loss + penalty.clone());

        let gradients = GradientsParams::from_grads(loss.clone().backward(), &self.policy);
        self.policy =
            self.actor_optimizer
                .step(self.actor_learning_rate, self.policy.clone(), gradients);

        Ok(SACActorUpdate {
            loss: loss.detach(),
            entropy_change_loss: entropy_change_loss.map(Tensor::detach),
            log_probabilities: logs.detach(),
            weights: weights.detach(),
            q_values: values.detach(),
        })
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            name = "sac.optimize_temperature",
            target = "modurl::performance",
            level = "trace",
            skip_all
        )
    )]
    /// Optimizes log alpha `[1]` from detached candidate statistics `[batch_size, candidate_count, 1]` and native observations.
    /// Reduces candidates to the entropy constraint `[batch_size, 1]` and loss `[1]`; only log alpha receives gradients.
    fn optimize_temperature(
        &mut self,
        observations: &Tensor<O, OS::Kind>,
        log_probabilities: &Tensor<3>,
        weights: &Tensor<3>,
    ) -> Result<SACTemperatureUpdate, SACError<PE, GE, SE, CE>> {
        match &mut self.entropy_configuration.kind {
            SACEntropyConfigurationKind::Automatic {
                log_alpha,
                optimizer,
                learning_rate,
                target_entropy_schedule,
            } => {
                let expected = (log_probabilities.clone() * weights.clone())
                    .sum_dim(1)
                    .squeeze_dim::<2>(1)
                    .detach();

                let target = match target_entropy_schedule {
                    Some(schedule) => self.schedule_progress.parameter(schedule.as_ref()),
                    None => self
                        .policy
                        .default_target_entropy(observations.clone())
                        .map_err(SACError::PolicyError)?,
                };
                if !target.is_finite() {
                    return Err(SACConfigurationError::InvalidTargetEntropySchedule.into());
                }

                let loss = sac_alpha_loss(&log_alpha.val(), &expected, target);
                let gradients = GradientsParams::from_grads(loss.clone().backward(), log_alpha);
                *log_alpha = optimizer.step(*learning_rate, log_alpha.clone(), gradients);

                Ok(SACTemperatureUpdate {
                    loss: Some(loss.detach()),
                    target_entropy: Some(target),
                })
            }
            SACEntropyConfigurationKind::Fixed { .. } => Ok(SACTemperatureUpdate {
                loss: None,
                target_entropy: None,
            }),
        }
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            name = "sac.optimize",
            target = "modurl::performance",
            skip_all,
            fields(collection_timestep)
        )
    )]
    fn optimize(&mut self, collection_timestep: usize) -> Result<(), SACError<PE, GE, SE, CE>> {
        let Some(replay) = self.replay.as_ref() else {
            return Ok(());
        };
        if !replay.can_sample() {
            return Ok(());
        }

        let mut batch = replay.sample()?;
        let device = self.replay_storage_config.optimization_device();
        batch.observations = batch.observations.to_device(&device);
        batch.next_observations = batch.next_observations.to_device(&device);
        batch.actions = batch.actions.to_device(&device);
        batch.terminated = transfer_bool_tensor(batch.terminated, &device);
        batch.rewards = batch.rewards.to_device(&device);
        batch.collection_policy_entropies = batch.collection_policy_entropies.to_device(&device);
        batch.entropy_change_weights = batch.entropy_change_weights.to_device(&device);

        let targets = self.compute_bellman_targets(&batch)?;
        let critic_losses = self.optimize_critics(&batch, &targets)?;
        let actor = self.optimize_actor(&batch)?;
        let temperature = self.optimize_temperature(
            &batch.observations,
            &actor.log_probabilities,
            &actor.weights,
        )?;

        for critic in &mut self.critics {
            critic.polyak_update(self.tau)?;
        }

        let alpha = self
            .entropy_configuration
            .alpha_tensor(&actor.log_probabilities);

        self.logger.log_update(&SACLogEntry {
            critic_losses,
            actor_loss: actor.loss,
            alpha_loss: temperature.loss,
            entropy_change_loss: actor.entropy_change_loss,
            target_entropy: temperature.target_entropy,
            alpha,
            bellman_targets: targets,
            policy_log_probabilities: actor.log_probabilities,
            policy_weights: actor.weights,
            policy_q_values: actor.q_values,
            replay_rewards: batch.rewards,
            update_index: self.optimization_steps,
            collection_timestep,
        });

        self.optimization_steps += 1;
        Ok(())
    }

    /// Computes detached collection entropy `[environment_count, 1]` from rank-O native observation batches.
    /// Random warmup actions receive zero entropy and zero penalty weight. Statistics use compute dtype; observations retain native dtype and kind.
    fn collection_entropy_metadata(
        &self,
        observations: &Tensor<O, OS::Kind>,
        environment_count: usize,
        random_actions: bool,
    ) -> Result<SACCollectionEntropy, SACError<PE, GE, SE, CE>> {
        if self.entropy_change_penalty.is_none() || random_actions {
            return Ok(SACCollectionEntropy {
                values: Tensor::zeros(
                    [environment_count, 1],
                    (&self.replay_storage_config.storage_device(), self.dtype),
                ),
                change_weight: 0.0,
            });
        }

        let terms = self
            .policy
            .expectation(
                observations
                    .clone()
                    .to_device(&self.replay_storage_config.optimization_device()),
                self.samples,
            )
            .map_err(SACError::PolicyError)?;
        let values = -(terms.log_probabilities().clone().cast(self.dtype)
            * terms.weights().clone().cast(self.dtype))
        .sum_dim(1)
        .squeeze_dim::<2>(1)
        .detach();

        Ok(SACCollectionEntropy {
            values,
            change_weight: 1.0,
        })
    }

    /// Resets the environment and validates rank-O observations `[num_envs, ...observation_shape]` and replay alignment.
    /// Preserves native observation kind, dtype, and device. Each learn call starts a new collection sequence.
    fn prepare_replay_collection(
        &mut self,
        env: &mut dyn MultiGym<I, O, A, Error = GE, ObservationSpace = OS, ActionSpace = AS>,
    ) -> Result<Tensor<O, OS::Kind>, SACError<PE, GE, SE, CE>> {
        let count = env.num_envs();
        if let Some(replay) = self.replay.as_mut() {
            replay.validate_environment_count(count)?;
        } else {
            validate_replay_alignment(self.replay_capacity, count).map_err(SACReplayError::from)?;
        }

        let observations = env.reset().map_err(SACError::GymError)?;
        self.validate_observations(&observations)?;
        if observations.dims()[0] != count {
            return Err(SACError::Shape {
                field: "environment observations",
                expected: vec![count],
                actual: vec![observations.dims()[0]],
            });
        }

        Ok(observations)
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            name = "sac.collect_transitions",
            target = "modurl::performance",
            skip_all
        )
    )]
    /// Collects one environment batch into replay and replaces observations `[num_envs, ...observation_shape]` with the next collection batch.
    /// Preserves native observations and terminal targets. Returns detached collection rewards `[num_envs, 1]` and episode statistics.
    fn collect_transitions<K: Autodiff, const U: usize>(
        &mut self,
        env: &mut dyn MultiGym<I, O, A, Error = GE, ObservationSpace = OS, ActionSpace = AS>,
        observations: &mut Tensor<O, K>,
        episodes: &mut SACEpisodeTracker,
    ) -> Result<SACCollectionLogEntry<I>, SACError<PE, GE, SE, CE>>
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
        let first_timestep = self.schedule_progress.elapsed_steps();
        let random_actions = first_timestep < self.training_start;
        let entropy = self.collection_entropy_metadata(observations, count, random_actions)?;

        let actions = if random_actions {
            let actions = self
                .action_space
                .sample_batch(count, &self.replay_storage_config.optimization_device())
                .map_err(SACError::SpaceError)?;
            if const { matches!(AS::Kind::KIND, burn::tensor::kind::Kind::Float) } {
                <DType as Cast<A, AS::Kind>>::cast(actions, self.dtype).without_autodiff()
            } else {
                actions.without_autodiff()
            }
        } else {
            self.stochastic_action(observations)?
        };

        let step = env.step(actions.clone()).map_err(SACError::GymError)?;
        for (field, actual) in [
            ("done flags", step.dones.len()),
            ("truncation flags", step.truncateds.len()),
            ("terminal observations", step.terminal_observations.len()),
            ("environment infos", step.infos.len()),
        ] {
            if actual != count {
                return Err(
                    SACReplayError::Storage(ReplayStorageError::BatchLengthMismatch {
                        field,
                        expected: count,
                        actual,
                    })
                    .into(),
                );
            }
        }

        self.validate_observations(&step.observations)?;
        if step.observations.dims()[0] != count {
            let mut expected = vec![count];
            expected.extend(self.observation_space.shape());
            return Err(SACError::Shape {
                field: "environment observations",
                expected,
                actual: step.observations.dims().to_vec(),
            });
        }
        if step.rewards.dims() != [count, 1] {
            return Err(SACError::Shape {
                field: "environment rewards",
                expected: vec![count, 1],
                actual: step.rewards.dims().to_vec(),
            });
        }

        for terminal in step.terminal_observations.iter().flatten() {
            let expected = self.observation_space.shape();
            if terminal.dims().to_vec() != expected {
                return Err(SACError::Shape {
                    field: "terminal observations",
                    expected,
                    actual: terminal.dims().to_vec(),
                });
            }
            if terminal.dtype() != step.observations.dtype() {
                return Err(SACError::DType {
                    field: "terminal observations",
                    expected: step.observations.dtype(),
                    actual: terminal.dtype(),
                });
            }
            if terminal.device() != step.observations.device() {
                return Err(SACError::Device {
                    field: "terminal observations",
                });
            }
        }

        let transition_next_observations = step.transition_next_observations();
        let MultiGymStepInfo {
            observations: reset_observations,
            rewards,
            infos,
            dones,
            truncateds,
            ..
        } = step;

        let mut completed_episodes = Vec::new();
        if self.logger.is_some() {
            let reward_values = rewards
                .clone()
                .try_into_data_as::<f32>()?
                .try_to_vec::<f32>()
                .map_err(TensorReadError::from)?;
            for (index, reward) in reward_values.into_iter().enumerate() {
                if let Some(entry) = episodes.record(
                    index,
                    reward,
                    dones[index],
                    truncateds[index],
                    first_timestep + index + 1,
                ) {
                    completed_episodes.push(entry);
                }
            }
        }

        let storage_device = self.replay_storage_config.storage_device();
        if self.replay.is_none() {
            let storage = SACReplayStorage::builder()
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
                .observation_dtype(observations.dtype())
                .action_dtype(actions.dtype())
                .environment_count(count)
                .dtype(self.dtype)
                .device(storage_device.clone())
                .build()?;
            self.replay = Some(SACReplay::new(storage, self.batch_size));
        }

        self.replay
            .as_mut()
            .ok_or(ReplayStorageError::EnvironmentCountNotSet)
            .map_err(SACReplayError::from)?
            .insert(SACReplayInsert {
                observations: observations.clone(),
                next_observations: transition_next_observations,
                actions,
                rewards: rewards.clone(),
                terminated: Tensor::<2, Bool>::from_data(
                    TensorData::new(dones, [count, 1]),
                    &storage_device,
                ),
                collection_policy_entropies: entropy.values,
                entropy_change_weights: Tensor::full(
                    [count, 1],
                    entropy.change_weight,
                    (&storage_device, self.dtype),
                ),
                truncateds,
            })?;

        self.schedule_progress.advance_steps(count);
        *observations = reset_observations;

        Ok(SACCollectionLogEntry {
            collection_rewards: rewards.detach(),
            infos,
            collection_timestep: self.schedule_progress.elapsed_steps(),
            completed_episodes,
            replay_len: self.replay.as_ref().map_or(0, SACReplay::len),
        })
    }

    /// Attempts one replay update per collected transition after the training threshold.
    /// Uses cumulative collection timesteps and preserves critic, actor, temperature, and target update order.
    fn train_on_collected_transitions(
        &mut self,
        collected_timesteps: usize,
    ) -> Result<(), SACError<PE, GE, SE, CE>> {
        let first_timestep = self.schedule_progress.elapsed_steps() - collected_timesteps;
        for index in 0..collected_timesteps {
            let collection_timestep = first_timestep + index + 1;
            if collection_timestep >= self.training_start {
                self.optimize(collection_timestep)?;
            }
        }
        Ok(())
    }
}

impl<
    P,
    N,
    OS,
    AS,
    GE,
    I,
    K: Autodiff,
    const O: usize,
    const A: usize,
    const L: usize,
    const C: usize,
    const U: usize,
> Agent<I, O, A>
    for SACAgent<
        '_,
        SACModelGroup<P, N, Tensor<O, K>, Tensor<A, AS::Kind>, Tensor<L>, Tensor<C, AS::Kind>>,
        OS,
        AS,
        GE,
        I,
        A,
    >
where
    OS: ObservationSpace<O, Kind = K, Item = Tensor<U, K>>,
    AS: ActionMap<L, A>,
    AS::Kind: Autodiff,
    Tensor<A, AS::Kind>: NextRank<Next = Tensor<C, AS::Kind>>,
    DType: Cast<A, AS::Kind, OutputKind = AS::Kind>,
    Tensor<U, K>: NextRank<Next = Tensor<O, K>>,
    Tensor<O, K>: PrevRank<Prev = Tensor<U, K>>,
    P: Module + ExpectationPolicy<O, L, C, ObservationKind = K, CandidateKind = AS::Kind>,
    N: ActionValueCriticNetwork<O, A, C, ObservationKind = K, ActionKind = AS::Kind>,
    P::Error: Debug,
    N::Error: Debug,
    AS::Error: Debug,
    GE: Debug,
{
    type Error = SACError<P::Error, GE, AS::Error, N::Error>;
    type GymError = GE;
    type SpaceError = AS::Error;
    type ObservationSpace = OS;
    type ActionSpace = AS;

    /// Samples detached rank-A environment actions `[batch_size, ...action_shape]` from native rank-O observations `[batch_size, ...observation_shape]`.
    /// Preserves observation dtype and kind, transfers to the optimization device, and lets the model perform any input conversions.
    fn act(&mut self, observations: &Tensor<O, K>) -> Result<Tensor<A, AS::Kind>, Self::Error> {
        self.stochastic_action(observations)
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            name = "sac.learn",
            target = "modurl::performance",
            skip_all,
            fields(num_timesteps)
        )
    )]
    /// Trains with rank-O native observation batches, rank-A environment action batches, and rewards `[num_envs, 1]`.
    /// Unbatched terminal observations have rank U, with O = U + 1. Preserves terminal targets and native observation kind and dtype.
    /// Resets the environment for each call, retains replay, and attempts one update per eligible collected transition.
    fn learn(
        &mut self,
        env: &mut dyn MultiGym<I, O, A, Error = GE, ObservationSpace = OS, ActionSpace = AS>,
        num_timesteps: usize,
    ) -> Result<(), Self::Error> {
        let mut elapsed_timesteps = 0;
        let mut next_observations = self.prepare_replay_collection(env)?;
        self.schedule_progress.default_horizon(num_timesteps);
        let mut episodes = SACEpisodeTracker::new(env.num_envs());

        while elapsed_timesteps < num_timesteps {
            let collection =
                self.collect_transitions(env, &mut next_observations, &mut episodes)?;
            let collected_timesteps = collection.collection_rewards.dims()[0];
            elapsed_timesteps += collected_timesteps;
            self.train_on_collected_transitions(collected_timesteps)?;
            self.logger.log_collection(&collection);
        }
        Ok(())
    }
}

/// Validates SAC critic count, replay sizes, training dtype, and tuning ranges before allocating agent storage.
#[bon::builder]
fn validate_sac_configuration(
    critic_count: usize,
    replay_capacity: usize,
    batch_size: usize,
    gamma: f64,
    tau: f64,
    actor_learning_rate: f64,
    dtype: DType,
    entropy_change_penalty: Option<f64>,
    q_value_clip: Option<f64>,
    training_horizon: Option<usize>,
) -> Result<(), SACConfigurationError> {
    if critic_count == 0 {
        return Err(SACConfigurationError::NoCritics);
    }

    if replay_capacity == 0 {
        return Err(SACConfigurationError::ZeroReplayCapacity);
    }

    if batch_size == 0 {
        return Err(SACConfigurationError::ZeroBatchSize);
    }

    if replay_capacity < batch_size {
        return Err(SACConfigurationError::ReplayCapacityBelowBatchSize);
    }

    if !gamma.is_finite() || !(0.0..=1.0).contains(&gamma) {
        return Err(SACConfigurationError::InvalidGamma);
    }

    if !tau.is_finite() || !(0.0..=1.0).contains(&tau) {
        return Err(SACConfigurationError::InvalidTau);
    }

    if !actor_learning_rate.is_finite() || actor_learning_rate <= 0.0 {
        return Err(SACConfigurationError::InvalidLearningRate);
    }

    if !dtype.is_float() {
        return Err(SACConfigurationError::InvalidDType(dtype));
    }

    if entropy_change_penalty.is_some_and(|value| !value.is_finite() || value < 0.0) {
        return Err(SACConfigurationError::InvalidEntropyChangePenalty);
    }

    if q_value_clip.is_some_and(|value| !value.is_finite() || value <= 0.0) {
        return Err(SACConfigurationError::InvalidQValueClip);
    }

    if training_horizon == Some(0) {
        return Err(SACConfigurationError::ZeroTrainingHorizon);
    }

    Ok(())
}

/// Adds a capacity axis to an event shape, returning `[capacity, ...event_shape]` of rank R.
/// Validates the runtime event-axis count and nonempty event dimensions; performs no tensor conversion.
fn space_storage_shape<const R: usize>(
    field: &'static str,
    event_shape: Vec<usize>,
    capacity: usize,
) -> Result<[usize; R], SACConfigurationError> {
    const {
        assert!(R >= 2, "space storage needs capacity and event axes");
    }
    if event_shape.contains(&0) {
        return Err(SACConfigurationError::EmptySpace);
    }
    let actual = event_shape.len();
    let mut shape = vec![capacity];
    shape.extend(event_shape);
    shape
        .try_into()
        .map_err(|_| SACConfigurationError::SpaceRank {
            field,
            expected: R - 1,
            actual,
        })
}

/// Requires rank-D Float critic outputs to use the training dtype, preserving shape, device, and gradients.
fn validate_critic_dtype<PE, GE, SE, CE, const D: usize>(
    values: &Tensor<D>,
    dtype: DType,
) -> Result<(), SACError<PE, GE, SE, CE>> {
    if values.dtype() != dtype {
        return Err(SACError::DType {
            field: "critic values",
            expected: dtype,
            actual: values.dtype(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests;

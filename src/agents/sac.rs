//! Generalized Soft Actor-Critic.
//!
//! The canonical configuration uses two independently optimized critics and
//! [`SACCriticAggregationMode::Min`]. Other non-empty ensemble sizes and
//! aggregation modes are supported experimentally.

use std::{fmt::Debug, num::NonZeroUsize, ops::Deref};

use bon::bon;
use candle_core::{D, DType, Tensor, Var};
use candle_nn::{Optimizer, VarMap};

use super::{Agent, ReplayDeviceStrategy};
use crate::{
    buffers::experience_replay::{
        AlignedObservationReplay, ExperienceReplay, ExperienceReplayError, ReplayStorage,
        ReplayStorageError, TensorReplayColumn, replay_index_tensor,
    },
    gym::{MultiGym, MultiGymStepInfo},
    models::probabilistic_model::ExpectationPolicy,
    objectives::{bellman_targets, clipped_value_loss},
    parameter_schedule::{ParameterSchedule, ScheduleProgress},
    spaces::Space,
};

/// A Q network that can evaluate replay actions and policy candidates.
pub trait SACCriticNetwork {
    /// Evaluates replay `states` shaped `[batch, ...state_shape]` and `actions`
    /// shaped `[batch, ...action_shape]`, returning Q values shaped `[batch]`.
    fn replay_values(&self, states: &Tensor, actions: &Tensor) -> candle_core::Result<Tensor>;

    /// Evaluates `states` shaped `[batch, ...state_shape]` and candidates
    /// shaped `[batch, candidates, ...action_shape]`, returning
    /// `[batch, candidates]`.
    fn policy_values(
        &self,
        states: &Tensor,
        candidate_actions: &Tensor,
    ) -> candle_core::Result<Tensor>;

    /// Values used by the actor loss.
    ///
    /// Implementations should omit critic-parameter gradient paths while
    /// preserving gradients with respect to candidate actions when those
    /// actions are differentiable. `states` is `[batch, ...state_shape]`,
    /// `candidate_actions` is `[batch, candidates, ...action_shape]`, and the
    /// result is `[batch, candidates]`.
    fn actor_values(
        &self,
        states: &Tensor,
        candidate_actions: &Tensor,
    ) -> candle_core::Result<Tensor>;
}

/// Failures from SAC critic construction, aggregation, and optimization.
#[derive(Debug)]
pub enum SACCriticError {
    TensorError(candle_core::Error),
    InvalidPolyakCoefficient {
        tau: f64,
    },
    ParameterMapMismatch {
        online_only: Vec<String>,
        target_only: Vec<String>,
    },
    NoCriticValues,
}

impl From<candle_core::Error> for SACCriticError {
    fn from(error: candle_core::Error) -> Self {
        Self::TensorError(error)
    }
}

/// Adapts a scalar `Q(s, a)` module to [`SACCriticNetwork`].
///
/// Replay inputs are states `[batch, ...state_shape]` and actions
/// `[batch, ...action_shape]`. Candidate inputs add the candidate axis:
/// `[batch, candidates, ...action_shape]`.
pub struct ScalarStateActionCritic {
    module: Box<dyn candle_core::Module>,
}

impl ScalarStateActionCritic {
    pub fn new(module: impl candle_core::Module + 'static) -> Self {
        Self {
            module: Box::new(module),
        }
    }

    /// Flattens states `[batch, ...state_shape]` and candidates
    /// `[batch, candidates, ...action_shape]` into critic inputs
    /// `[batch * candidates, state_size + action_size]`.
    fn flattened_policy_inputs(
        states: &Tensor,
        candidate_actions: &Tensor,
    ) -> candle_core::Result<(Tensor, usize, usize)> {
        let batch_size = states.dim(0)?;
        if candidate_actions.rank() < 2 {
            return Err(candle_core::Error::DimOutOfRange {
                shape: candidate_actions.shape().clone(),
                dim: 1,
                op: "SAC critic candidate actions",
            });
        }
        if candidate_actions.dim(0)? != batch_size {
            let mut expected = candidate_actions.dims().to_vec();
            expected[0] = batch_size;
            return Err(candle_core::Error::UnexpectedShape {
                msg: "critic candidate-action batch size must match the state batch size".into(),
                expected: expected.into(),
                got: candidate_actions.shape().clone(),
            });
        }
        let candidate_count = candidate_actions.dim(1)?;
        if candidate_count == 0 {
            return Err(candle_core::Error::EmptyTensor {
                op: "SAC critic candidate axis",
            });
        }
        let state_width = states.elem_count() / batch_size;
        let states = states.reshape((batch_size, state_width))?;
        let action_width = candidate_actions.elem_count() / (batch_size * candidate_count);
        let actions = candidate_actions.reshape((batch_size, candidate_count, action_width))?;
        let states =
            states
                .unsqueeze(1)?
                .broadcast_as((batch_size, candidate_count, state_width))?;
        let inputs = Tensor::cat(&[states, actions], 2)?
            .reshape((batch_size * candidate_count, state_width + action_width))?;
        Ok((inputs, batch_size, candidate_count))
    }
}

impl SACCriticNetwork for ScalarStateActionCritic {
    /// Evaluates states `[batch, ...state_shape]` and replay actions
    /// `[batch, ...action_shape]`, returning `[batch]`.
    fn replay_values(&self, states: &Tensor, actions: &Tensor) -> candle_core::Result<Tensor> {
        let candidate_actions = actions.unsqueeze(1)?;
        self.policy_values(states, &candidate_actions)?.squeeze(1)
    }

    /// Evaluates states `[batch, ...state_shape]` and candidates
    /// `[batch, candidates, ...action_shape]`, returning `[batch, candidates]`.
    fn policy_values(
        &self,
        states: &Tensor,
        candidate_actions: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let (inputs, batch_size, candidate_count) =
            Self::flattened_policy_inputs(states, candidate_actions)?;
        self.module
            .forward(&inputs)?
            .reshape((batch_size, candidate_count))
    }

    /// Evaluates states `[batch, ...state_shape]` and candidates
    /// `[batch, candidates, ...action_shape]`, returning `[batch, candidates]`.
    fn actor_values(
        &self,
        states: &Tensor,
        candidate_actions: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let (inputs, batch_size, candidate_count) =
            Self::flattened_policy_inputs(states, candidate_actions)?;
        self.module
            .forward(&inputs)?
            .reshape((batch_size, candidate_count))
    }
}

/// Adapts a discrete vector-head module returning `[batch, action_count]`.
///
/// Replay actions contain one scalar index per batch item. Policy candidates
/// are indices shaped `[batch, candidates]`; all candidate methods return
/// `[batch, candidates]`.
pub struct DiscreteVectorHeadCritic {
    module: Box<dyn candle_core::Module>,
}

impl DiscreteVectorHeadCritic {
    pub fn new(module: impl candle_core::Module + 'static) -> Self {
        Self {
            module: Box::new(module),
        }
    }

    /// Evaluates states `[batch, ...state_shape]` as vector-head values
    /// `[batch, action_count]`.
    fn values(&self, states: &Tensor) -> candle_core::Result<Tensor> {
        self.module.forward(states)
    }

    /// Converts scalar replay actions containing one element per batch item to
    /// indices shaped `[batch]`.
    fn replay_indices(actions: &Tensor, batch_size: usize) -> candle_core::Result<Tensor> {
        if actions.elem_count() != batch_size {
            return Err(candle_core::Error::ShapeMismatch {
                buffer_size: actions.elem_count(),
                shape: batch_size.into(),
            });
        }
        actions.reshape(batch_size)?.to_dtype(DType::U32)
    }

    /// Gathers candidate indices `[batch, candidates]` from values
    /// `[batch, action_count]`, returning `[batch, candidates]`.
    fn candidate_values(values: &Tensor, actions: &Tensor) -> candle_core::Result<Tensor> {
        let (batch_size, _) = values.dims2()?;
        let (action_batch_size, _) = actions.dims2()?;
        if action_batch_size != batch_size {
            let mut expected = actions.dims().to_vec();
            expected[0] = batch_size;
            return Err(candle_core::Error::UnexpectedShape {
                msg: "critic candidate-action batch size must match the state batch size".into(),
                expected: expected.into(),
                got: actions.shape().clone(),
            });
        }
        values.gather(&actions.to_dtype(DType::U32)?.contiguous()?, 1)
    }
}

impl SACCriticNetwork for DiscreteVectorHeadCritic {
    /// Gathers replay action indices `[batch]` from state values
    /// `[batch, action_count]`, returning `[batch]`.
    fn replay_values(&self, states: &Tensor, actions: &Tensor) -> candle_core::Result<Tensor> {
        let values = self.values(states)?;
        let indices = Self::replay_indices(actions, values.dim(0)?)?.unsqueeze(1)?;
        values.gather(&indices, 1)?.squeeze(1)
    }

    /// Gathers candidates `[batch, candidates]` from state values
    /// `[batch, action_count]`, returning `[batch, candidates]`.
    fn policy_values(
        &self,
        states: &Tensor,
        candidate_actions: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let values = self.values(states)?;
        Self::candidate_values(&values, candidate_actions)
    }

    /// Gathers candidates `[batch, candidates]` from state values
    /// `[batch, action_count]`, returning `[batch, candidates]`.
    fn actor_values(
        &self,
        states: &Tensor,
        candidate_actions: &Tensor,
    ) -> candle_core::Result<Tensor> {
        self.policy_values(states, candidate_actions)
            .map(|values| values.detach())
    }
}

/// One independently optimized online/target critic pair.
pub struct SACCritic<'a, O>
where
    O: Optimizer,
{
    online_network: Box<dyn SACCriticNetwork>,
    target_network: Box<dyn SACCriticNetwork>,
    online_vars: &'a VarMap,
    target_vars: &'a mut VarMap,
    optimizer: O,
}

#[bon]
impl<'a, O> SACCritic<'a, O>
where
    O: Optimizer,
{
    #[builder]
    pub fn new(
        #[builder(with = |network: impl SACCriticNetwork + 'static| Box::new(network))]
        online_network: Box<dyn SACCriticNetwork>,
        #[builder(with = |network: impl SACCriticNetwork + 'static| Box::new(network))]
        target_network: Box<dyn SACCriticNetwork>,
        online_vars: &'a VarMap,
        target_vars: &'a mut VarMap,
        optimizer: O,
    ) -> Result<Self, SACCriticError> {
        let mut critic = Self {
            online_network,
            target_network,
            online_vars,
            target_vars,
            optimizer,
        };
        critic.hard_update()?;
        Ok(critic)
    }

    /// Evaluates online replay values for states `[batch, ...state_shape]` and
    /// actions `[batch, ...action_shape]`, returning `[batch]`.
    pub fn online_replay_values(
        &self,
        states: &Tensor,
        actions: &Tensor,
    ) -> candle_core::Result<Tensor> {
        self.online_network.replay_values(states, actions)
    }

    /// Evaluates online policy values for states `[batch, ...state_shape]` and
    /// candidates `[batch, candidates, ...action_shape]`, returning
    /// `[batch, candidates]`.
    pub fn online_policy_values(
        &self,
        states: &Tensor,
        actions: &Tensor,
    ) -> candle_core::Result<Tensor> {
        self.online_network.policy_values(states, actions)
    }

    /// Evaluates actor-loss values for states `[batch, ...state_shape]` and
    /// candidates `[batch, candidates, ...action_shape]`, returning
    /// `[batch, candidates]`.
    pub fn online_actor_values(
        &self,
        states: &Tensor,
        actions: &Tensor,
    ) -> candle_core::Result<Tensor> {
        self.online_network.actor_values(states, actions)
    }

    /// Evaluates detached target values for states `[batch, ...state_shape]`
    /// and candidates `[batch, candidates, ...action_shape]`, returning
    /// `[batch, candidates]`.
    pub fn target_policy_values(
        &self,
        states: &Tensor,
        actions: &Tensor,
    ) -> candle_core::Result<Tensor> {
        Ok(self.target_network.policy_values(states, actions)?.detach())
    }

    /// Evaluates detached target replay values for states
    /// `[batch, ...state_shape]` and actions `[batch, ...action_shape]`,
    /// returning `[batch]`.
    pub fn target_replay_values(
        &self,
        states: &Tensor,
        actions: &Tensor,
    ) -> candle_core::Result<Tensor> {
        Ok(self.target_network.replay_values(states, actions)?.detach())
    }

    pub fn optimizer(&self) -> &O {
        &self.optimizer
    }

    pub fn optimizer_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }

    pub fn hard_update(&mut self) -> Result<(), SACCriticError> {
        copy_var_map(self.online_vars, self.target_vars, 1.0)
    }

    pub fn polyak_update(&mut self, tau: f64) -> Result<(), SACCriticError> {
        if !tau.is_finite() || !(0.0..=1.0).contains(&tau) {
            return Err(SACCriticError::InvalidPolyakCoefficient { tau });
        }
        copy_var_map(self.online_vars, self.target_vars, tau)
    }
}

fn copy_var_map(online: &VarMap, target: &mut VarMap, tau: f64) -> Result<(), SACCriticError> {
    let online_values = online
        .data()
        .lock()
        .unwrap()
        .deref()
        .iter()
        .map(|(name, var)| (name.clone(), var.as_tensor().clone()))
        .collect::<Vec<_>>();
    let target_names = target
        .data()
        .lock()
        .unwrap()
        .keys()
        .cloned()
        .collect::<std::collections::HashSet<_>>();
    let online_names = online_values
        .iter()
        .map(|(name, _)| name.clone())
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
        return Err(SACCriticError::ParameterMapMismatch {
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
                .expect("validated target parameter name")
                .as_tensor()
                .clone();
            ((online_value * tau)? + (target_value * (1.0 - tau))?)?
        };
        target.set_one(name, &value)?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SACCriticAggregationMode {
    /// Uses the arithmetic mean of all critic values.
    Mean,
    /// Uses the middle critic value, or the mean of the two middle values.
    Median,
    /// Uses the lowest critic value.
    Min,
    /// Uses the highest critic value.
    Max,
}

/// Differentiably aggregates critic tensors `values[i]`, each shaped
/// `value_shape`, and returns one tensor shaped `value_shape`.
pub fn aggregate_critic_values(
    values: &[Tensor],
    mode: SACCriticAggregationMode,
) -> Result<Tensor, SACCriticError> {
    if values.is_empty() {
        return Err(SACCriticError::NoCriticValues);
    }
    let rank = values[0].rank();
    if let Some(value) = values
        .iter()
        .skip(1)
        .find(|value| value.dims() != values[0].dims())
    {
        return Err(SACCriticError::TensorError(
            candle_core::Error::ShapeMismatchBinaryOp {
                lhs: values[0].shape().clone(),
                rhs: value.shape().clone(),
                op: "SAC critic aggregation",
            },
        ));
    }
    let aggregated = match mode {
        SACCriticAggregationMode::Mean => Tensor::stack(values, rank)?.mean(rank)?,
        SACCriticAggregationMode::Min => values
            .iter()
            .skip(1)
            .try_fold(values[0].clone(), |current, value| current.minimum(value))?,
        SACCriticAggregationMode::Max => values
            .iter()
            .skip(1)
            .try_fold(values[0].clone(), |current, value| current.maximum(value))?,
        SACCriticAggregationMode::Median => {
            let stacked = Tensor::stack(values, rank)?;
            let (sorted, _) = stacked.sort_last_dim(true)?;
            let count = values.len();
            if count % 2 == 1 {
                sorted.narrow(rank, count / 2, 1)?.squeeze(rank)?
            } else {
                let lower = sorted.narrow(rank, count / 2 - 1, 1)?.squeeze(rank)?;
                let upper = sorted.narrow(rank, count / 2, 1)?.squeeze(rank)?;
                ((lower + upper)? * 0.5)?
            }
        }
    };
    Ok(aggregated)
}

/// Returns a scalar temperature loss from scalar `log_alpha` shaped `[]` and
/// `expected_log_probability` shaped `[batch]`.
pub fn sac_alpha_loss(
    log_alpha: &Tensor,
    expected_log_probability: &Tensor,
    target_entropy: f64,
) -> candle_core::Result<Tensor> {
    let constraint = (expected_log_probability.detach() + target_entropy)?.detach();
    log_alpha.broadcast_mul(&constraint)?.neg()?.mean_all()
}

/// Returns a scalar entropy-drift loss from `current_entropy`,
/// `collection_entropy`, and `weights`, all shaped `[batch]`.
pub fn sac_entropy_change_loss(
    current_entropy: &Tensor,
    collection_entropy: &Tensor,
    weights: &Tensor,
    coefficient: f64,
) -> candle_core::Result<Tensor> {
    let weighted_error = (current_entropy - collection_entropy)?
        .sqr()?
        .mul(weights)?;
    let normalizer = weights.sum_all()?.clamp(1.0, f64::INFINITY)?;
    weighted_error.sum_all()?.broadcast_div(&normalizer)? * coefficient
}

/// Entropy coefficient configuration.
pub struct SACEntropyConfiguration<O>
where
    O: Optimizer,
{
    kind: SACEntropyConfigurationKind<O>,
}

enum SACEntropyConfigurationKind<O>
where
    O: Optimizer,
{
    /// Optimizes `log_alpha` toward a policy or scheduled target entropy.
    Automatic {
        /// Scalar logarithm of the entropy coefficient.
        log_alpha: Var,
        /// Optimizer containing `log_alpha`.
        optimizer: O,
        /// Optional target entropy over training progress. `None` uses the
        /// policy's [`ExpectationPolicy::default_target_entropy`].
        target_entropy_schedule: Option<Box<dyn ParameterSchedule>>,
    },
    /// Keeps the entropy coefficient constant.
    Fixed {
        /// Finite, non-negative entropy coefficient.
        alpha: f64,
    },
}

impl<O> SACEntropyConfiguration<O>
where
    O: Optimizer,
{
    /// Creates an automatically tuned entropy coefficient.
    pub fn automatic(log_alpha: Var, optimizer: O) -> Self {
        Self {
            kind: SACEntropyConfigurationKind::Automatic {
                log_alpha,
                optimizer,
                target_entropy_schedule: None,
            },
        }
    }

    /// Creates an automatically tuned entropy coefficient with a scheduled
    /// target entropy.
    pub fn automatic_with_target_schedule(
        log_alpha: Var,
        optimizer: O,
        target_entropy_schedule: impl ParameterSchedule + 'static,
    ) -> Self {
        Self {
            kind: SACEntropyConfigurationKind::Automatic {
                log_alpha,
                optimizer,
                target_entropy_schedule: Some(Box::new(target_entropy_schedule)),
            },
        }
    }

    /// Creates a fixed entropy coefficient.
    pub fn fixed(alpha: f64) -> Self {
        Self {
            kind: SACEntropyConfigurationKind::Fixed { alpha },
        }
    }
}

/// Optional SAC stabilization techniques.
///
/// [`Self::stable_discrete`] selects the published discrete-SAC defaults:
/// mean critic aggregation and a `0.5` replay-to-current entropy penalty.
/// Q-value clipping remains opt-in because its useful scale depends on the
/// task's rewards.
#[derive(Default)]
pub struct SACStabilizationConfiguration {
    entropy_change_penalty: Option<f64>,
    aggregation_mode: Option<SACCriticAggregationMode>,
}

#[bon]
impl SACStabilizationConfiguration {
    #[builder]
    pub fn new(
        /// Coefficient for the replay-to-current entropy-change penalty.
        entropy_change_penalty: Option<f64>,
        /// Critic aggregation override associated with this preset.
        aggregation_mode: Option<SACCriticAggregationMode>,
    ) -> Self {
        Self {
            entropy_change_penalty,
            aggregation_mode,
        }
    }

    pub fn stable_discrete() -> Self {
        Self {
            entropy_change_penalty: Some(0.5),
            aggregation_mode: Some(SACCriticAggregationMode::Mean),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SACConfigurationError {
    NoCritics,
    ZeroReplayCapacity,
    ZeroBatchSize,
    ReplayCapacityBelowBatchSize,
    InvalidGamma,
    InvalidTau,
    InvalidAlpha,
    InvalidTargetEntropySchedule,
    InvalidEntropyChangePenalty,
    InvalidQValueClip,
    ZeroTrainingHorizon,
}

#[derive(Debug)]
pub enum SACError<PE, GE, SE>
where
    PE: Debug,
    GE: Debug,
    SE: Debug,
{
    PolicyError(PE),
    GymError(GE),
    SpaceError(SE),
    TensorError(candle_core::Error),
    ReplayStorageError(ReplayStorageError),
    CriticError(SACCriticError),
    ConfigurationError(SACConfigurationError),
}

impl<PE: Debug, GE: Debug, SE: Debug> From<candle_core::Error> for SACError<PE, GE, SE> {
    fn from(value: candle_core::Error) -> Self {
        Self::TensorError(value)
    }
}

impl<PE: Debug, GE: Debug, SE: Debug> From<ReplayStorageError> for SACError<PE, GE, SE> {
    fn from(error: ReplayStorageError) -> Self {
        Self::ReplayStorageError(error)
    }
}

impl<PE: Debug, GE: Debug, SE: Debug> From<ExperienceReplayError<ReplayStorageError>>
    for SACError<PE, GE, SE>
{
    fn from(error: ExperienceReplayError<ReplayStorageError>) -> Self {
        Self::ReplayStorageError(error.into())
    }
}

impl<PE: Debug, GE: Debug, SE: Debug> From<SACCriticError> for SACError<PE, GE, SE> {
    fn from(value: SACCriticError) -> Self {
        Self::CriticError(value)
    }
}

struct SACReplayInsert {
    states: Tensor,
    next_states: Tensor,
    actions: Tensor,
    rewards: Tensor,
    terminated: Tensor,
    collection_policy_entropies: Tensor,
    entropy_change_weights: Tensor,
    truncateds: Vec<bool>,
}

struct SACOptimizationBatch {
    states: Tensor,
    next_states: Tensor,
    actions: Tensor,
    rewards: Tensor,
    terminated: Tensor,
    collection_policy_entropies: Tensor,
    entropy_change_weights: Tensor,
}

struct SACReplayStorage {
    observations: AlignedObservationReplay,
    actions: TensorReplayColumn,
    rewards: TensorReplayColumn,
    terminated: TensorReplayColumn,
    collection_policy_entropies: TensorReplayColumn,
    entropy_change_weights: TensorReplayColumn,
    capacity: usize,
    device: candle_core::Device,
}

impl SACReplayStorage {
    fn new(
        capacity: usize,
        state_shape: &[usize],
        action_shape: &[usize],
        action_dtype: DType,
        dtype: DType,
        device: candle_core::Device,
    ) -> Result<Self, ReplayStorageError> {
        Ok(Self {
            observations: AlignedObservationReplay::new(capacity, state_shape, dtype, &device),
            actions: TensorReplayColumn::new(capacity, action_shape, action_dtype, &device)?,
            rewards: TensorReplayColumn::new(capacity, &[], dtype, &device)?,
            terminated: TensorReplayColumn::new(capacity, &[], dtype, &device)?,
            collection_policy_entropies: TensorReplayColumn::new(capacity, &[], dtype, &device)?,
            entropy_change_weights: TensorReplayColumn::new(capacity, &[], dtype, &device)?,
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

impl ReplayStorage for SACReplayStorage {
    type Insert = SACReplayInsert;
    type Batch = SACOptimizationBatch;
    type Error = ReplayStorageError;

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn insert(&mut self, start: usize, transitions: Self::Insert) -> Result<usize, Self::Error> {
        let prepared = SACReplayInsert {
            states: transitions.states,
            next_states: transitions.next_states,
            actions: self.actions.prepare(&transitions.actions)?,
            rewards: self.rewards.prepare(&transitions.rewards)?,
            terminated: self.terminated.prepare(&transitions.terminated)?,
            collection_policy_entropies: self
                .collection_policy_entropies
                .prepare(&transitions.collection_policy_entropies)?,
            entropy_change_weights: self
                .entropy_change_weights
                .prepare(&transitions.entropy_change_weights)?,
            truncateds: transitions.truncateds,
        };
        let count = prepared.states.dim(0)?;
        for (name, tensor) in [
            ("actions", &prepared.actions),
            ("rewards", &prepared.rewards),
            ("terminated", &prepared.terminated),
            (
                "collection policy entropies",
                &prepared.collection_policy_entropies,
            ),
            ("entropy change weights", &prepared.entropy_change_weights),
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
            (
                &self.collection_policy_entropies,
                &prepared.collection_policy_entropies,
            ),
            (
                &self.entropy_change_weights,
                &prepared.entropy_change_weights,
            ),
        ] {
            column.write(start, tensor)?;
        }
        Ok(count)
    }

    fn gather(&self, indices: &[usize]) -> Result<Self::Batch, Self::Error> {
        let (states, next_states) = self.observations.gather(indices)?;
        let indices = replay_index_tensor(indices, &self.device)?;
        Ok(SACOptimizationBatch {
            states,
            next_states,
            actions: self.actions.gather(&indices)?,
            rewards: self.rewards.gather(&indices)?,
            terminated: self.terminated.gather(&indices)?,
            collection_policy_entropies: self.collection_policy_entropies.gather(&indices)?,
            entropy_change_weights: self.entropy_change_weights.gather(&indices)?,
        })
    }

    fn sampleable_len(&self, len: usize) -> usize {
        self.observations.sampleable_len(len)
    }

    fn sample_index(&self, index: usize, len: usize) -> usize {
        self.observations.sample_index(index, len)
    }
}

struct SACActorUpdate {
    loss: Tensor,
    entropy_change_loss: Option<Tensor>,
    log_probabilities: Tensor,
    weights: Tensor,
    q_values: Tensor,
}

struct SACTemperatureUpdate {
    loss: Option<Tensor>,
    target_entropy: Option<f64>,
}

struct SACCollectionEntropy {
    values: Tensor,
    change_weight: f32,
}

struct SACCollectedTransitions<'batch> {
    states: &'batch Tensor,
    next_states: &'batch Tensor,
    actions: &'batch Tensor,
    rewards: &'batch Tensor,
    dones: &'batch [bool],
    truncateds: &'batch [bool],
    policy_entropies: &'batch Tensor,
    entropy_change_weight: f32,
    first_timestep: usize,
}

struct SACEpisodeTracker {
    returns: Vec<f32>,
    lengths: Vec<usize>,
}

pub struct SACLogEntry {
    /// One scalar optimization loss per critic.
    pub critic_losses: Vec<Tensor>,
    /// Scalar actor optimization loss.
    pub actor_loss: Tensor,
    /// Scalar automatic entropy-coefficient loss, or `None` for fixed entropy.
    pub alpha_loss: Option<Tensor>,
    /// Scalar replay-to-current entropy penalty when enabled.
    pub entropy_change_loss: Option<Tensor>,
    /// Current automatic target entropy, or `None` for fixed entropy.
    pub target_entropy: Option<f64>,
    /// Current scalar entropy coefficient.
    pub alpha: Tensor,
    /// The detached soft Bellman targets shared by all critics for this update.
    /// Shape: `[batch_size]`.
    pub bellman_targets: Tensor,
    /// Candidate log probabilities shaped `[batch_size, candidate_count]`.
    pub policy_log_probabilities: Tensor,
    /// Candidate expectation weights shaped `[batch_size, candidate_count]`.
    pub policy_weights: Tensor,
    /// Raw soft-Q critic values before the actor's current entropy term.
    /// Shape: `[batch_size, candidate_count]`.
    pub policy_q_values: Tensor,
    /// Rewards from the sampled replay batch, shaped `[batch_size]`.
    pub replay_rewards: Tensor,
    /// Zero-based number of the optimization step.
    pub update_index: usize,
    /// Total collected transitions when this update was triggered.
    pub collection_timestep: usize,
}

pub struct SACCollectionLogEntry<I = ()> {
    /// Latest reward for each inner environment, shaped `[environment_count]`.
    pub collection_rewards: Tensor,
    /// Latest typed step metadata in inner-environment order.
    pub infos: Vec<I>,
    /// Total transitions collected after this vectorized environment step.
    pub collection_timestep: usize,
    /// Episodes completed during this vectorized environment step.
    pub completed_episodes: Vec<SACEpisodeLogEntry>,
    /// Number of transitions currently held in replay.
    pub replay_len: usize,
}

pub struct SACEpisodeLogEntry {
    /// Index of the inner environment that completed.
    pub environment_index: usize,
    /// Sum of rewards collected during the episode.
    pub episode_return: f32,
    /// Number of environment steps in the episode.
    pub episode_length: usize,
    /// Whether the environment reached a terminal state.
    pub terminated: bool,
    /// Whether the episode was truncated.
    pub truncated: bool,
    /// Total collected transitions when the episode completed.
    pub collection_timestep: usize,
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

        let entry = SACEpisodeLogEntry {
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

pub trait SACLogger<I = ()> {
    /// Receives one sampled-replay update after `training_start`.
    fn log_update(&mut self, entry: &SACLogEntry);
    /// Receives the result of one vectorized environment step.
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

/// Generalized Soft Actor-Critic agent.
pub struct SACAgent<'a, AO, CO, EO, PE, GE, SE, I = ()>
where
    AO: Optimizer,
    CO: Optimizer,
    EO: Optimizer,
    PE: Debug,
    GE: Debug,
    SE: Debug,
{
    policy: Box<dyn ExpectationPolicy<Error = PE>>,
    actor_optimizer: AO,
    critics: Vec<SACCritic<'a, CO>>,
    entropy_configuration: SACEntropyConfiguration<EO>,
    aggregation_mode: SACCriticAggregationMode,
    action_space: Box<dyn Space<Error = SE>>,
    observation_space: Box<dyn Space<Error = SE>>,
    replay: ExperienceReplay<SACReplayInsert, SACReplayStorage>,
    gamma: f64,
    tau: f64,
    training_start: usize,
    samples: NonZeroUsize,
    entropy_change_penalty: Option<f64>,
    q_value_clip: Option<f64>,
    schedule_progress: ScheduleProgress,
    device_strategy: ReplayDeviceStrategy,
    optimization_steps: usize,
    logger: Option<&'a mut dyn SACLogger<I>>,
    dtype: DType,
    _errors: std::marker::PhantomData<(GE, fn() -> I)>,
}

#[bon]
impl<'a, AO, CO, EO, PE, GE, SE, I> SACAgent<'a, AO, CO, EO, PE, GE, SE, I>
where
    AO: Optimizer,
    CO: Optimizer,
    EO: Optimizer,
    PE: Debug,
    GE: Debug,
    SE: Debug,
{
    #[builder]
    #[allow(
        clippy::type_complexity,
        reason = "Bon's generated typestate contains every SAC agent generic parameter"
    )]
    pub fn new(
        #[builder(with = |policy: impl ExpectationPolicy<Error = PE> + 'static| Box::new(policy))]
        policy: Box<dyn ExpectationPolicy<Error = PE>>,
        actor_optimizer: AO,
        critics: Vec<SACCritic<'a, CO>>,
        entropy_configuration: SACEntropyConfiguration<EO>,
        action_space: Box<dyn Space<Error = SE>>,
        observation_space: Box<dyn Space<Error = SE>>,
        device_strategy: ReplayDeviceStrategy,
        /// Optional update and collection metric sink.
        logger: Option<&'a mut dyn SACLogger<I>>,
        /// Explicit critic aggregation mode. This takes precedence over the
        /// stabilization configuration and defaults to `Min`.
        aggregation_mode: Option<SACCriticAggregationMode>,
        /// Optional grouped defaults for stabilization techniques.
        #[builder(default)]
        stabilization_configuration: SACStabilizationConfiguration,
        /// Enables PPO-style critic value clipping around target-network Q.
        q_value_clip: Option<f64>,
        #[builder(default = 0.99)] gamma: f64,
        #[builder(default = 0.005)] tau: f64,
        #[builder(default = 1_000_000)] replay_capacity: usize,
        #[builder(default = 256)] batch_size: usize,
        /// Number of random-action transitions collected before optimization.
        #[builder(default = 1_000)]
        training_start: usize,
        /// Candidate count used for sampled action expectations. Exact
        /// categorical expectations ignore this value.
        #[builder(default = NonZeroUsize::MIN)]
        samples: NonZeroUsize,
        /// Total collected transitions over which parameter schedules run.
        training_horizon: usize,
        #[builder(default = DType::F32)] dtype: DType,
    ) -> Result<Self, SACError<PE, GE, SE>> {
        assert!(dtype.is_float(), "SAC compute dtype must be floating-point");
        let aggregation_mode = aggregation_mode
            .or(stabilization_configuration.aggregation_mode)
            .unwrap_or(SACCriticAggregationMode::Min);
        SACConfigurationValidator::validate_configuration()
            .critic_count(critics.len())
            .replay_capacity(replay_capacity)
            .batch_size(batch_size)
            .gamma(gamma)
            .tau(tau)
            .entropy(&entropy_configuration)
            .maybe_entropy_change_penalty(stabilization_configuration.entropy_change_penalty)
            .maybe_q_value_clip(q_value_clip)
            .training_horizon(training_horizon)
            .call()?;
        let storage_device = device_strategy.storage_device();
        let observation_sample = observation_space
            .sample(&storage_device)
            .map_err(SACError::SpaceError)?;
        let action_sample = action_space
            .sample(&storage_device)
            .map_err(SACError::SpaceError)?;
        let replay_storage = SACReplayStorage::new(
            replay_capacity,
            observation_sample.dims(),
            action_sample.dims(),
            action_sample.dtype(),
            dtype,
            storage_device.clone(),
        )?;
        Ok(Self {
            policy,
            actor_optimizer,
            critics,
            entropy_configuration,
            aggregation_mode,
            action_space,
            observation_space,
            replay: ExperienceReplay::with_storage(replay_storage, batch_size),
            gamma,
            tau,
            training_start,
            samples,
            entropy_change_penalty: stabilization_configuration.entropy_change_penalty,
            q_value_clip,
            schedule_progress: ScheduleProgress::new(training_horizon),
            device_strategy,
            optimization_steps: 0,
            logger,
            dtype,
            _errors: std::marker::PhantomData,
        })
    }
}

struct SACConfigurationValidator;

#[bon]
impl SACConfigurationValidator {
    #[builder]
    fn validate_configuration<PE: Debug, GE: Debug, SE: Debug, O: Optimizer>(
        critic_count: usize,
        replay_capacity: usize,
        batch_size: usize,
        gamma: f64,
        tau: f64,
        entropy: &SACEntropyConfiguration<O>,
        entropy_change_penalty: Option<f64>,
        q_value_clip: Option<f64>,
        training_horizon: usize,
    ) -> Result<(), SACError<PE, GE, SE>> {
        let error = if critic_count == 0 {
            Some(SACConfigurationError::NoCritics)
        } else if replay_capacity == 0 {
            Some(SACConfigurationError::ZeroReplayCapacity)
        } else if batch_size == 0 {
            Some(SACConfigurationError::ZeroBatchSize)
        } else if replay_capacity < batch_size {
            Some(SACConfigurationError::ReplayCapacityBelowBatchSize)
        } else if !gamma.is_finite() || !(0.0..=1.0).contains(&gamma) {
            Some(SACConfigurationError::InvalidGamma)
        } else if !tau.is_finite() || !(0.0..=1.0).contains(&tau) {
            Some(SACConfigurationError::InvalidTau)
        } else if matches!(&entropy.kind, SACEntropyConfigurationKind::Fixed { alpha } if !alpha.is_finite() || *alpha < 0.0)
        {
            Some(SACConfigurationError::InvalidAlpha)
        } else if matches!(&entropy.kind, SACEntropyConfigurationKind::Automatic { target_entropy_schedule: Some(schedule), .. }
        if !schedule.value(0.0).is_finite() || !schedule.value(1.0).is_finite())
        {
            Some(SACConfigurationError::InvalidTargetEntropySchedule)
        } else if matches!(entropy_change_penalty, Some(value) if !value.is_finite() || value < 0.0)
        {
            Some(SACConfigurationError::InvalidEntropyChangePenalty)
        } else if matches!(q_value_clip, Some(value) if !value.is_finite() || value <= 0.0) {
            Some(SACConfigurationError::InvalidQValueClip)
        } else if training_horizon == 0 {
            Some(SACConfigurationError::ZeroTrainingHorizon)
        } else {
            None
        };
        match error {
            Some(error) => Err(SACError::ConfigurationError(error)),
            None => Ok(()),
        }
    }
}

impl<'a, AO, CO, EO, PE, GE, SE, I> SACAgent<'a, AO, CO, EO, PE, GE, SE, I>
where
    AO: Optimizer,
    CO: Optimizer,
    EO: Optimizer,
    PE: Debug,
    GE: Debug,
    SE: Debug,
{
    pub fn get_action_space(&self) -> &dyn Space<Error = SE> {
        self.action_space.as_ref()
    }

    pub fn get_observation_space(&self) -> &dyn Space<Error = SE> {
        self.observation_space.as_ref()
    }

    /// Selects actions for `observation` shaped
    /// `[batch, ...observation_shape]`, returning environment actions shaped
    /// `[batch, ...action_shape]`.
    pub fn act_deterministic(
        &mut self,
        observation: &Tensor,
    ) -> Result<Tensor, SACError<PE, GE, SE>> {
        let observation = observation
            .to_device(&self.device_strategy.optimization_device())?
            .to_dtype(self.dtype)?;
        let latent = self
            .policy
            .mode(&observation)
            .map_err(SACError::PolicyError)?;
        self.action_space
            .tensor_from_neurons(&latent)
            .map_err(SACError::SpaceError)
    }

    /// Samples environment actions `[batch, ...action_shape]` for observations
    /// `[batch, ...observation_shape]`.
    fn stochastic_action(&self, observation: &Tensor) -> Result<Tensor, SACError<PE, GE, SE>> {
        let observation = observation
            .to_device(&self.device_strategy.optimization_device())?
            .to_dtype(self.dtype)?;
        let latent = self
            .policy
            .sample(&observation)
            .map_err(SACError::PolicyError)?;
        self.action_space
            .tensor_from_neurons(&latent)
            .map_err(SACError::SpaceError)
    }

    fn random_actions(&self, batch_size: usize) -> Result<Tensor, SACError<PE, GE, SE>> {
        let actions = (0..batch_size)
            .map(|_| {
                self.action_space
                    .sample(&self.device_strategy.optimization_device())
                    .map_err(SACError::SpaceError)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let actions = Tensor::stack(&actions, 0)?;
        if actions.dtype().is_float() {
            Ok(actions.to_dtype(self.dtype)?)
        } else {
            Ok(actions)
        }
    }

    /// Returns scalar alpha shaped `[]` on the same device and with the same
    /// dtype as the arbitrarily shaped reference tensor `like`.
    fn alpha_tensor(&self, like: &Tensor) -> candle_core::Result<Tensor> {
        match &self.entropy_configuration.kind {
            SACEntropyConfigurationKind::Automatic { log_alpha, .. } => log_alpha
                .as_tensor()
                .exp()?
                .to_device(like.device())?
                .to_dtype(like.dtype()),
            SACEntropyConfigurationKind::Fixed { alpha } => {
                Tensor::new(*alpha, like.device())?.to_dtype(like.dtype())
            }
        }
    }

    fn sample_optimization_batch(
        &mut self,
    ) -> Result<Option<SACOptimizationBatch>, SACError<PE, GE, SE>> {
        if self.replay.len() < self.replay.get_batch_size() {
            return Ok(None);
        }
        let mut batch = self.replay.sample()?;
        let optimization_device = self.device_strategy.optimization_device();
        for element in [
            &mut batch.states,
            &mut batch.next_states,
            &mut batch.actions,
            &mut batch.rewards,
            &mut batch.terminated,
            &mut batch.collection_policy_entropies,
            &mut batch.entropy_change_weights,
        ] {
            *element = element.to_device(&optimization_device)?;
        }
        if batch.actions.dtype().is_float() {
            batch.actions = batch.actions.to_dtype(self.dtype)?;
        }
        Ok(Some(batch))
    }

    fn compute_bellman_targets(
        &self,
        batch: &SACOptimizationBatch,
    ) -> Result<Tensor, SACError<PE, GE, SE>> {
        // Soft target value over policy candidates j and target critics i:
        // V(s') = sum_j w_j * (aggregate_i Q_target_i(s', a'_j) - alpha * log pi(a'_j|s'))
        // y = r + gamma * (1 - terminated) * V(s')
        let next_terms = self
            .policy
            .expectation(&batch.next_states, self.samples)
            .map_err(SACError::PolicyError)?;
        let next_actions = next_terms.actions().detach();
        let next_log_probabilities = next_terms.log_probabilities().detach();
        let next_weights = next_terms.weights().detach();
        let target_values = self
            .critics
            .iter()
            .map(|critic| critic.target_policy_values(&batch.next_states, &next_actions))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let target_values = aggregate_critic_values(&target_values, self.aggregation_mode)?;
        let alpha = self.alpha_tensor(&next_log_probabilities)?.detach();
        let entropy_cost = next_log_probabilities.broadcast_mul(&alpha)?;
        let soft_values = (&target_values - entropy_cost)?
            .mul(&next_weights)?
            .sum(D::Minus1)?
            .detach();
        Ok(bellman_targets(
            &batch.rewards,
            &batch.terminated,
            &soft_values,
            self.gamma,
        )?)
    }

    /// Optimizes each critic against Bellman `targets` shaped `[batch]`.
    ///
    /// Returns one scalar loss tensor per critic.
    fn optimize_critics(
        &mut self,
        batch: &SACOptimizationBatch,
        targets: &Tensor,
    ) -> Result<Vec<Tensor>, SACError<PE, GE, SE>> {
        let targets = targets.detach();
        let mut critic_losses = Vec::with_capacity(self.critics.len());
        for critic in &mut self.critics {
            let predicted = critic.online_replay_values(&batch.states, &batch.actions)?;
            // Match SB3 SAC: each critic contributes
            // 0.5 * mean((Q_i(s, a) - y)^2).
            // With clipping, use the larger error after bounding Q_i's move
            // around its target-network value, as in PPO's value loss.
            let squared_error = match self.q_value_clip {
                Some(epsilon) => {
                    let anchor = critic
                        .target_replay_values(&batch.states, &batch.actions)?
                        .detach();
                    clipped_value_loss(&predicted, &targets, &anchor, epsilon)?
                }
                None => candle_nn::loss::mse(&predicted, &targets)?,
            };
            let loss = (squared_error * 0.5)?;
            critic.optimizer_mut().backward_step(&loss)?;
            critic_losses.push(loss);
        }
        Ok(critic_losses)
    }

    fn optimize_actor(
        &mut self,
        batch: &SACOptimizationBatch,
    ) -> Result<SACActorUpdate, SACError<PE, GE, SE>> {
        let policy_terms = self
            .policy
            .expectation(&batch.states, self.samples)
            .map_err(SACError::PolicyError)?;
        let policy_values = self
            .critics
            .iter()
            .map(|critic| critic.online_actor_values(&batch.states, policy_terms.actions()))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let policy_values = aggregate_critic_values(&policy_values, self.aggregation_mode)?;
        let alpha = self
            .alpha_tensor(policy_terms.log_probabilities())?
            .detach();
        // J_pi = mean_s sum_j w_j *
        //     (alpha * log pi(a_j|s) - aggregate_i Q_i(s, a_j)).
        let actor_objective =
            (policy_terms.log_probabilities().broadcast_mul(&alpha)? - &policy_values)?;
        let base_actor_loss = actor_objective
            .mul(policy_terms.weights())?
            .sum(D::Minus1)?
            .mean_all()?;
        // H_pi(s) = -sum_j w_j * log pi(a_j|s).
        let policy_entropies = policy_terms
            .log_probabilities()
            .mul(policy_terms.weights())?
            .sum(D::Minus1)?
            .neg()?;
        // Optional entropy-drift penalty:
        // L_H = coefficient * sum_b mask_b * (H_pi(s_b) - H_collection_b)^2
        //       / max(sum_b mask_b, 1).
        let entropy_change_loss = match self.entropy_change_penalty {
            Some(coefficient) => Some(sac_entropy_change_loss(
                &policy_entropies,
                &batch.collection_policy_entropies,
                &batch.entropy_change_weights,
                coefficient,
            )?),
            None => None,
        };
        let actor_loss = match &entropy_change_loss {
            Some(loss) => (&base_actor_loss + loss)?,
            None => base_actor_loss,
        };
        self.actor_optimizer.backward_step(&actor_loss)?;
        Ok(SACActorUpdate {
            loss: actor_loss,
            entropy_change_loss,
            log_probabilities: policy_terms.log_probabilities().clone(),
            weights: policy_terms.weights().clone(),
            q_values: policy_values,
        })
    }

    /// Optimizes automatic entropy temperature from states shaped
    /// `[batch, ...state_shape]`.
    fn optimize_temperature(
        &mut self,
        states: &Tensor,
        log_probabilities: &Tensor,
        weights: &Tensor,
    ) -> Result<SACTemperatureUpdate, SACError<PE, GE, SE>> {
        match &mut self.entropy_configuration.kind {
            SACEntropyConfigurationKind::Automatic {
                log_alpha,
                optimizer,
                target_entropy_schedule,
            } => {
                let expected_log_probability =
                    log_probabilities.mul(weights)?.sum(D::Minus1)?.detach();
                let target_entropy = match target_entropy_schedule {
                    Some(schedule) => self.schedule_progress.parameter(schedule.as_ref()),
                    None => self
                        .policy
                        .default_target_entropy(states)
                        .map_err(SACError::PolicyError)?,
                };
                if !target_entropy.is_finite() {
                    return Err(SACError::ConfigurationError(
                        SACConfigurationError::InvalidTargetEntropySchedule,
                    ));
                }
                // L_alpha = -mean(log(alpha) *
                //     (E_a[log pi(a|s)] + target_entropy)).
                // The parenthesized constraint is detached, so only alpha moves.
                let loss = sac_alpha_loss(
                    log_alpha.as_tensor(),
                    &expected_log_probability,
                    target_entropy,
                )?;
                optimizer.backward_step(&loss)?;
                Ok(SACTemperatureUpdate {
                    loss: Some(loss),
                    target_entropy: Some(target_entropy),
                })
            }
            SACEntropyConfigurationKind::Fixed { .. } => Ok(SACTemperatureUpdate {
                loss: None,
                target_entropy: None,
            }),
        }
    }

    fn update_target_critics(&mut self) -> Result<(), SACError<PE, GE, SE>> {
        // Polyak averaging: theta_target <- tau * theta_online
        //                              + (1 - tau) * theta_target.
        for critic in &mut self.critics {
            critic.polyak_update(self.tau)?;
        }
        Ok(())
    }

    /// Logs Bellman `targets` shaped `[batch]` and one scalar loss tensor per
    /// critic, along with the actor and temperature update tensors.
    fn log_optimization(
        &mut self,
        collection_timestep: usize,
        batch: SACOptimizationBatch,
        targets: Tensor,
        critic_losses: Vec<Tensor>,
        actor: SACActorUpdate,
        temperature: SACTemperatureUpdate,
    ) -> Result<(), SACError<PE, GE, SE>> {
        let alpha = self.alpha_tensor(&actor.log_probabilities)?.detach();
        let entry = SACLogEntry {
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
        };
        self.logger.log_update(&entry);
        Ok(())
    }

    fn optimize(&mut self, collection_timestep: usize) -> Result<(), SACError<PE, GE, SE>> {
        let Some(batch) = self.sample_optimization_batch()? else {
            return Ok(());
        };
        let targets = self.compute_bellman_targets(&batch)?;
        let critic_losses = self.optimize_critics(&batch, &targets)?;
        let actor = self.optimize_actor(&batch)?;
        let temperature =
            self.optimize_temperature(&batch.states, &actor.log_probabilities, &actor.weights)?;
        self.update_target_critics()?;
        self.log_optimization(
            collection_timestep,
            batch,
            targets,
            critic_losses,
            actor,
            temperature,
        )?;
        self.optimization_steps += 1;
        Ok(())
    }

    /// Computes collection-policy entropy metadata for observations shaped
    /// `[environment_count, ...observation_shape]`.
    fn collection_entropy_metadata(
        &self,
        observations: &Tensor,
        environment_count: usize,
        uses_random_actions: bool,
    ) -> Result<SACCollectionEntropy, SACError<PE, GE, SE>> {
        if self.entropy_change_penalty.is_none() || uses_random_actions {
            return Ok(SACCollectionEntropy {
                values: Tensor::zeros(
                    environment_count,
                    self.dtype,
                    &self.device_strategy.storage_device(),
                )?,
                change_weight: 0.0,
            });
        }

        let observations = observations
            .to_device(&self.device_strategy.optimization_device())?
            .to_dtype(self.dtype)?;
        let terms = self
            .policy
            .expectation(&observations, self.samples)
            .map_err(SACError::PolicyError)?;
        let values = terms
            .log_probabilities()
            .mul(terms.weights())?
            .sum(D::Minus1)?
            .neg()?
            .detach()
            .to_dtype(self.dtype)?;
        Ok(SACCollectionEntropy {
            values,
            change_weight: 1.0,
        })
    }

    /// Selects actions shaped `[environment_count, ...action_shape]` for
    /// observations shaped `[environment_count, ...observation_shape]`.
    fn select_collection_actions(
        &self,
        observations: &Tensor,
        environment_count: usize,
        uses_random_actions: bool,
    ) -> Result<Tensor, SACError<PE, GE, SE>> {
        if uses_random_actions {
            self.random_actions(environment_count)
        } else {
            self.stochastic_action(observations)
        }
    }

    fn store_vectorized_transitions(
        &mut self,
        transitions: SACCollectedTransitions<'_>,
        episodes: &mut SACEpisodeTracker,
    ) -> Result<Vec<SACEpisodeLogEntry>, SACError<PE, GE, SE>> {
        let SACCollectedTransitions {
            states,
            next_states,
            actions,
            rewards,
            dones,
            truncateds,
            policy_entropies,
            entropy_change_weight,
            first_timestep,
        } = transitions;
        let environment_count = dones.len();
        let reward_values = rewards
            .to_dtype(DType::F32)?
            .to_device(&candle_core::Device::Cpu)?
            .to_vec1::<f32>()?;
        let mut completed_episodes = Vec::new();

        for index in 0..environment_count {
            let reward = reward_values[index];
            let collection_timestep = first_timestep + index + 1;
            if let Some(entry) = episodes.record(
                index,
                reward,
                dones[index],
                truncateds[index],
                collection_timestep,
            ) {
                completed_episodes.push(entry);
            }
        }

        let storage_device = self.device_strategy.storage_device();
        self.replay.add(SACReplayInsert {
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
            collection_policy_entropies: policy_entropies.clone(),
            entropy_change_weights: Tensor::from_vec(
                vec![entropy_change_weight; environment_count],
                environment_count,
                &storage_device,
            )?
            .to_dtype(self.dtype)?,
            truncateds: truncateds.to_vec(),
        })?;

        Ok(completed_episodes)
    }

    fn learn_inner(
        &mut self,
        env: &mut dyn MultiGym<I, Error = GE, SpaceError = SE>,
        num_timesteps: usize,
    ) -> Result<(), SACError<PE, GE, SE>> {
        let mut elapsed_timesteps = 0usize;
        let environment_count = env.num_envs();
        self.replay
            .storage_mut()
            .initialize_environment_count(environment_count)?;
        let mut observations = env
            .reset()
            .map_err(SACError::GymError)?
            .to_dtype(self.dtype)?;
        let mut episodes = SACEpisodeTracker::new(environment_count);

        while elapsed_timesteps < num_timesteps {
            let first_timestep = self.schedule_progress.elapsed_steps();
            let uses_random_actions = first_timestep < self.training_start;
            let policy_observations = if uses_random_actions {
                observations.clone()
            } else {
                observations
                    .to_device(&self.device_strategy.optimization_device())?
                    .to_dtype(self.dtype)?
            };
            let entropy = self.collection_entropy_metadata(
                &policy_observations,
                environment_count,
                uses_random_actions,
            )?;
            let actions = self.select_collection_actions(
                &policy_observations,
                environment_count,
                uses_random_actions,
            )?;
            let step = env.step(actions.clone()).map_err(SACError::GymError)?;
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
            let completed_episodes = self.store_vectorized_transitions(
                SACCollectedTransitions {
                    states: &observations,
                    next_states: &transition_next_states,
                    actions: &actions,
                    rewards: &rewards,
                    dones: &dones,
                    truncateds: &truncateds,
                    policy_entropies: &entropy.values,
                    entropy_change_weight: entropy.change_weight,
                    first_timestep,
                },
                &mut episodes,
            )?;
            for index in 0..environment_count {
                let collection_timestep = first_timestep + index + 1;
                if collection_timestep >= self.training_start {
                    self.optimize(collection_timestep)?;
                }
            }

            elapsed_timesteps += environment_count;
            observations = reset_next_states.to_dtype(self.dtype)?;
            self.logger.log_collection(&SACCollectionLogEntry {
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

impl<'a, AO, CO, EO, PE, GE, SE, I> Agent<I> for SACAgent<'a, AO, CO, EO, PE, GE, SE, I>
where
    AO: Optimizer,
    CO: Optimizer,
    EO: Optimizer,
    PE: Debug,
    GE: Debug,
    SE: Debug,
{
    type Error = SACError<PE, GE, SE>;
    type GymError = GE;
    type SpaceError = SE;

    /// Selects environment actions `[batch, ...action_shape]` for observations
    /// `[batch, ...observation_shape]`.
    fn act(&mut self, observation: &Tensor) -> Result<Tensor, Self::Error> {
        self.stochastic_action(observation)
    }

    fn learn(
        &mut self,
        env: &mut dyn MultiGym<I, Error = GE, SpaceError = SE>,
        num_timesteps: usize,
    ) -> Result<(), Self::Error> {
        self.learn_inner(env, num_timesteps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::test_support::CountingOptimizer;
    use candle_core::{Device, Module};
    use candle_nn::{Init, VarBuilder};
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    struct SumModule;

    impl Module for SumModule {
        /// Reduces input `[batch, feature_count]` to `[batch, 1]`.
        fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
            input.sum(D::Minus1)?.unsqueeze(1)
        }
    }

    struct IdentityModule;

    impl Module for IdentityModule {
        /// Preserves the arbitrary input shape `[...]`.
        fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
            Ok(input.clone())
        }
    }

    struct DirectGaussianModule {
        outputs: Var,
    }

    impl Module for DirectGaussianModule {
        /// Maps input `[batch, ...state_shape]` to parameters `[batch, 2]`.
        fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
            self.outputs.as_tensor().broadcast_as((input.dim(0)?, 2))
        }
    }

    struct SharedCountingOptimizer {
        steps: Arc<AtomicUsize>,
        learning_rate: f64,
    }

    impl SharedCountingOptimizer {
        fn new_counter() -> (Self, Arc<AtomicUsize>) {
            let steps = Arc::new(AtomicUsize::new(0));
            (
                Self {
                    steps: steps.clone(),
                    learning_rate: 1e-3,
                },
                steps,
            )
        }
    }

    impl Optimizer for SharedCountingOptimizer {
        type Config = f64;

        fn new(_vars: Vec<Var>, learning_rate: Self::Config) -> candle_core::Result<Self> {
            Ok(Self {
                steps: Arc::new(AtomicUsize::new(0)),
                learning_rate,
            })
        }

        fn step(
            &mut self,
            _gradients: &candle_core::backprop::GradStore,
        ) -> candle_core::Result<()> {
            self.steps.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn learning_rate(&self) -> f64 {
            self.learning_rate
        }

        fn set_learning_rate(&mut self, learning_rate: f64) {
            self.learning_rate = learning_rate;
        }
    }

    #[derive(Default)]
    struct RecordingLogger {
        updates: usize,
        collections: usize,
        entropy_change_updates: usize,
        entropy_change_losses: Vec<f32>,
        target_entropies: Vec<f64>,
    }

    impl SACLogger for RecordingLogger {
        fn log_update(&mut self, entry: &SACLogEntry) {
            self.updates += 1;
            assert_eq!(entry.critic_losses.len(), 2);
            assert!(entry.alpha_loss.is_some());
            self.entropy_change_updates += usize::from(entry.entropy_change_loss.is_some());
            if let Some(loss) = &entry.entropy_change_loss {
                self.entropy_change_losses
                    .push(loss.to_scalar::<f32>().unwrap());
            }
            if let Some(target_entropy) = entry.target_entropy {
                self.target_entropies.push(target_entropy);
            }
        }

        fn log_collection(&mut self, _entry: &SACCollectionLogEntry) {
            self.collections += 1;
        }
    }

    fn tensor(values: &[f32], shape: impl Into<candle_core::Shape>) -> Tensor {
        Tensor::from_vec(values.to_vec(), shape, &Device::Cpu).unwrap()
    }

    #[test]
    fn all_aggregation_modes_support_arbitrary_critic_counts() {
        let values = vec![
            tensor(&[1.0, 8.0], (1, 2)),
            tensor(&[3.0, 4.0], (1, 2)),
            tensor(&[2.0, 6.0], (1, 2)),
        ];
        assert_eq!(
            aggregate_critic_values(&values, SACCriticAggregationMode::Min)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            vec![vec![1.0, 4.0]]
        );
        assert_eq!(
            aggregate_critic_values(&values, SACCriticAggregationMode::Max)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            vec![vec![3.0, 8.0]]
        );
        assert_eq!(
            aggregate_critic_values(&values, SACCriticAggregationMode::Median)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            vec![vec![2.0, 6.0]]
        );
        assert_eq!(
            aggregate_critic_values(&values, SACCriticAggregationMode::Mean)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            vec![vec![2.0, 6.0]]
        );
        assert!(matches!(
            aggregate_critic_values(&[], SACCriticAggregationMode::Min),
            Err(SACCriticError::NoCriticValues)
        ));
        assert!(matches!(
            aggregate_critic_values(
                &[tensor(&[1.0, 2.0], 2), tensor(&[1.0, 2.0], (1, 2))],
                SACCriticAggregationMode::Min
            ),
            Err(SACCriticError::TensorError(
                candle_core::Error::ShapeMismatchBinaryOp { .. }
            ))
        ));
    }

    #[test]
    fn even_median_averages_middle_values_and_handles_ties() {
        let values = vec![
            tensor(&[1.0, 2.0], (1, 2)),
            tensor(&[2.0, 2.0], (1, 2)),
            tensor(&[10.0, 2.0], (1, 2)),
            tensor(&[4.0, 2.0], (1, 2)),
        ];
        assert_eq!(
            aggregate_critic_values(&values, SACCriticAggregationMode::Median)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            vec![vec![3.0, 2.0]]
        );
    }

    #[test]
    fn aggregation_gradients_are_normalized_including_ties() {
        fn gradients(mode: SACCriticAggregationMode, values: &[f32]) -> Vec<f32> {
            let variables = values
                .iter()
                .map(|value| Var::from_vec(vec![*value], 1, &Device::Cpu).unwrap())
                .collect::<Vec<_>>();
            let tensors = variables
                .iter()
                .map(|variable| variable.as_tensor().clone())
                .collect::<Vec<_>>();
            let loss = aggregate_critic_values(&tensors, mode)
                .unwrap()
                .sum_all()
                .unwrap();
            let store = loss.backward().unwrap();
            variables
                .iter()
                .map(|variable| {
                    store
                        .get(variable.as_tensor())
                        .map(|gradient| gradient.to_vec1::<f32>().unwrap()[0])
                        .unwrap_or(0.0)
                })
                .collect()
        }

        assert_eq!(
            gradients(SACCriticAggregationMode::Mean, &[1.0, 2.0, 3.0]),
            vec![1.0 / 3.0; 3]
        );
        assert_eq!(
            gradients(SACCriticAggregationMode::Min, &[1.0, 2.0, 3.0]),
            vec![1.0, 0.0, 0.0]
        );
        assert_eq!(
            gradients(SACCriticAggregationMode::Max, &[1.0, 2.0, 3.0]),
            vec![0.0, 0.0, 1.0]
        );
        assert_eq!(
            gradients(SACCriticAggregationMode::Median, &[1.0, 2.0, 3.0]),
            vec![0.0, 1.0, 0.0]
        );
        let tied = gradients(SACCriticAggregationMode::Min, &[1.0, 1.0]);
        assert_eq!(tied.iter().sum::<f32>(), 1.0);
        assert_eq!(
            gradients(SACCriticAggregationMode::Median, &[1.0, 2.0]),
            vec![0.5, 0.5]
        );
    }

    #[test]
    fn stable_discrete_defaults_to_mean_and_entropy_change_penalty() {
        let configuration = SACStabilizationConfiguration::stable_discrete();
        assert_eq!(
            configuration.aggregation_mode,
            Some(SACCriticAggregationMode::Mean)
        );
        assert_eq!(configuration.entropy_change_penalty, Some(0.5));
    }

    #[test]
    fn scalar_adapter_broadcasts_states_across_candidates() {
        let critic = ScalarStateActionCritic::new(SumModule);
        let states = tensor(&[1.0, 2.0, 3.0, 4.0], (2, 2));
        let actions = tensor(&[10.0, 20.0, 30.0, 40.0], (2, 2, 1));
        assert_eq!(
            critic
                .policy_values(&states, &actions)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            vec![vec![13.0, 23.0], vec![37.0, 47.0]]
        );
        let replay_actions = tensor(&[10.0, 30.0], (2, 1));
        assert_eq!(
            critic
                .replay_values(&states, &replay_actions)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![13.0, 37.0]
        );
    }

    #[test]
    fn scalar_adapter_flattens_five_dimensional_action_events() {
        let critic = ScalarStateActionCritic::new(SumModule);
        let states = tensor(&[1.0, 2.0, 3.0, 4.0], (2, 2));
        let candidates = Tensor::ones(&[2, 2, 2, 2, 2, 2, 2], DType::F32, &Device::Cpu).unwrap();
        assert_eq!(
            critic
                .policy_values(&states, &candidates)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            vec![vec![35.0, 35.0], vec![39.0, 39.0]]
        );

        let replay_actions = Tensor::ones((2, 2, 2, 2, 2, 2), DType::F32, &Device::Cpu).unwrap();
        assert_eq!(
            critic
                .replay_values(&states, &replay_actions)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![35.0, 39.0]
        );
    }

    #[test]
    fn scalar_adapter_reports_candidate_rank_and_batch_errors_separately() {
        let critic = ScalarStateActionCritic::new(SumModule);
        let states = tensor(&[1.0, 2.0, 3.0, 4.0], (2, 2));

        let missing_candidate_axis = tensor(&[0.0, 1.0], 2);
        let error = critic
            .policy_values(&states, &missing_candidate_axis)
            .unwrap_err();
        assert!(matches!(
            error,
            candle_core::Error::DimOutOfRange { dim: 1, .. }
        ));

        let wrong_batch = tensor(&[0.0, 1.0], (1, 2, 1));
        let error = critic.policy_values(&states, &wrong_batch).unwrap_err();
        assert!(matches!(error, candle_core::Error::UnexpectedShape { .. }));

        let no_candidates = Tensor::zeros((2, 0, 1), DType::F32, &Device::Cpu).unwrap();
        let error = critic.policy_values(&states, &no_candidates).unwrap_err();
        assert!(matches!(error, candle_core::Error::EmptyTensor { .. }));
    }

    #[test]
    fn discrete_adapter_gathers_replay_and_exact_candidates() {
        let critic = DiscreteVectorHeadCritic::new(IdentityModule);
        let values = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3));
        let replay_actions = Tensor::from_vec(vec![2u32, 0], 2, &Device::Cpu).unwrap();
        assert_eq!(
            critic
                .replay_values(&values, &replay_actions)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![3.0, 4.0]
        );
        let candidates = Tensor::from_vec(vec![0u32, 1, 2, 0, 1, 2], (2, 3), &Device::Cpu).unwrap();
        assert_eq!(
            critic
                .policy_values(&values, &candidates)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            values.to_vec2::<f32>().unwrap()
        );
    }

    #[test]
    fn discrete_actor_gradients_reach_policy_but_not_critic_parameters() {
        use crate::{distributions::DifferentiableExpectation, models::MLP};

        let critic_vars = VarMap::new();
        let critic_module = MLP::builder()
            .input_size(2)
            .output_size(2)
            .vb(VarBuilder::from_varmap(
                &critic_vars,
                DType::F32,
                &Device::Cpu,
            ))
            .hidden_layer_sizes(vec![4])
            .build()
            .unwrap();
        let critic = DiscreteVectorHeadCritic::new(critic_module);
        let logits = Var::from_vec(vec![0.2f32, -0.1], (1, 2), &Device::Cpu).unwrap();
        let terms = crate::distributions::CategoricalDistribution
            .expectation(logits.as_tensor(), NonZeroUsize::MIN)
            .unwrap();
        let states = tensor(&[0.5, -0.25], (1, 2));
        let q_values = critic.actor_values(&states, terms.actions()).unwrap();
        let loss = q_values.mul(terms.weights()).unwrap().sum_all().unwrap();
        let gradients = loss.backward().unwrap();

        assert!(gradients.get(logits.as_tensor()).is_some());
        assert!(
            critic_vars
                .all_vars()
                .iter()
                .all(|parameter| gradients.get(parameter.as_tensor()).is_none())
        );
    }

    #[test]
    fn scalar_actor_forward_reaches_actions_and_unused_critic_gradients() {
        use crate::models::MLP;

        let critic_vars = VarMap::new();
        let critic_module = MLP::builder()
            .input_size(3)
            .output_size(1)
            .vb(VarBuilder::from_varmap(
                &critic_vars,
                DType::F32,
                &Device::Cpu,
            ))
            .hidden_layer_sizes(vec![4])
            .activation(Tensor::tanh)
            .build()
            .unwrap();
        let critic = ScalarStateActionCritic::new(critic_module);
        let actions = Var::from_vec(vec![0.25f32], (1, 1, 1), &Device::Cpu).unwrap();
        let states = tensor(&[0.5, -0.25], (1, 2));
        let loss = critic
            .actor_values(&states, actions.as_tensor())
            .unwrap()
            .sum_all()
            .unwrap();
        let gradients = loss.backward().unwrap();

        assert!(gradients.get(actions.as_tensor()).is_some());
        assert!(
            critic_vars
                .all_vars()
                .iter()
                .all(|parameter| gradients.get(parameter.as_tensor()).is_some())
        );
    }

    #[test]
    fn continuous_actor_reparameterization_reaches_actor_and_critic_but_not_detached_targets() {
        use crate::{
            distributions::{GaussianDistribution, TanhTransform, TransformedDistribution},
            models::{MLP, probabilistic_model::ProbabilisticPolicyModel},
        };

        let actor_outputs = Var::from_vec(vec![0.1f32, -0.5], (1, 2), &Device::Cpu).unwrap();
        let policy = ProbabilisticPolicyModel::<
            TransformedDistribution<GaussianDistribution, TanhTransform>,
        >::new(DirectGaussianModule {
            outputs: actor_outputs.clone(),
        });
        let states = tensor(&[0.25, -0.5, 0.75, 1.0], (4, 1));

        let critic_vars = VarMap::new();
        let critic_module = MLP::builder()
            .input_size(2)
            .output_size(1)
            .vb(VarBuilder::from_varmap(
                &critic_vars,
                DType::F32,
                &Device::Cpu,
            ))
            .hidden_layer_sizes(vec![4])
            .activation(Tensor::tanh)
            .build()
            .unwrap();
        let critic = ScalarStateActionCritic::new(critic_module);

        let terms = policy.expectation(&states, NonZeroUsize::MIN).unwrap();
        let q_values = critic.actor_values(&states, terms.actions()).unwrap();
        let loss = ((terms.log_probabilities() * 0.2).unwrap() - q_values)
            .unwrap()
            .mul(terms.weights())
            .unwrap()
            .mean_all()
            .unwrap();
        let gradients = loss.backward().unwrap();
        let actor_gradient = gradients
            .get(actor_outputs.as_tensor())
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert!(actor_gradient[0].iter().all(|value| value.abs() > 1e-6));
        assert!(
            critic_vars
                .all_vars()
                .iter()
                .all(|parameter| gradients.get(parameter.as_tensor()).is_some())
        );

        let target_terms = policy.expectation(&states, NonZeroUsize::MIN).unwrap();
        let detached_loss = target_terms.log_probabilities().detach().sum_all().unwrap();
        let detached_gradients = detached_loss.backward().unwrap();
        assert!(detached_gradients.get(actor_outputs.as_tensor()).is_none());
        let target_actions = target_terms.actions().detach();
        let detached_q = critic
            .policy_values(&states, &target_actions)
            .unwrap()
            .detach();
        assert!(
            detached_q
                .backward()
                .unwrap()
                .get(actor_outputs.as_tensor())
                .is_none()
        );
    }

    #[test]
    fn collection_preserves_truncated_terminal_state_and_bootstrap_mask() {
        use crate::{
            distributions::CategoricalDistribution,
            gym::{Gym, ResetInfo, StepInfo, VectorizedGymWrapper},
            models::probabilistic_model::ProbabilisticPolicyModel,
            spaces::{BoxSpace, Discrete, Space},
        };

        struct TruncatingEnv;

        impl Gym for TruncatingEnv {
            type Error = candle_core::Error;
            type SpaceError = candle_core::Error;

            /// Steps with one scalar discrete action shaped `[]`.
            fn step(&mut self, _action: Tensor) -> Result<StepInfo, Self::Error> {
                Ok(StepInfo {
                    state: Tensor::full(5.0f32, 4, &Device::Cpu)?,
                    reward: 1.0,
                    done: false,
                    truncated: true,
                    info: (),
                })
            }

            fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
                Ok(ResetInfo {
                    state: Tensor::zeros(4, DType::F32, &Device::Cpu)?,
                    info: (),
                })
            }

            fn observation_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
                Box::new(BoxSpace::new_with_universal_bounds(
                    vec![4],
                    -10.0,
                    10.0,
                    &Device::Cpu,
                ))
            }

            fn action_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
                Box::new(Discrete::new(2))
            }
        }

        let online_vars = VarMap::new();
        let mut target_vars = VarMap::new();
        let critic = SACCritic::builder()
            .online_network(DiscreteVectorHeadCritic::new(IdentityModule))
            .target_network(DiscreteVectorHeadCritic::new(IdentityModule))
            .online_vars(&online_vars)
            .target_vars(&mut target_vars)
            .optimizer(CountingOptimizer::with_learning_rate(1e-3))
            .build()
            .unwrap();
        let policy = ProbabilisticPolicyModel::<CategoricalDistribution>::new(IdentityModule);
        let mut agent = SACAgent::builder()
            .policy(policy)
            .actor_optimizer(CountingOptimizer::with_learning_rate(1e-3))
            .critics(vec![critic])
            .entropy_configuration(SACEntropyConfiguration::<CountingOptimizer>::fixed(0.0))
            .action_space(Box::new(Discrete::new(2)))
            .observation_space(Box::new(BoxSpace::new_with_universal_bounds(
                vec![4],
                -10.0,
                10.0,
                &Device::Cpu,
            )))
            .device_strategy(ReplayDeviceStrategy::OneDevice(Device::Cpu))
            .replay_capacity(2)
            .batch_size(1)
            .training_start(10)
            .training_horizon(1)
            .build()
            .unwrap();
        let mut env = VectorizedGymWrapper::new(vec![TruncatingEnv]);
        agent.learn(&mut env, 1).unwrap();

        let replay = agent.replay.sample().unwrap();
        assert_eq!(
            replay.next_states.to_vec2::<f32>().unwrap(),
            vec![vec![5.0, 5.0, 5.0, 5.0]]
        );
        assert_eq!(replay.terminated.to_vec1::<f32>().unwrap(), vec![0.0]);
        assert_eq!(
            replay.entropy_change_weights.to_vec1::<f32>().unwrap(),
            vec![0.0]
        );
    }

    #[test]
    fn alpha_gradient_moves_temperature_in_the_constraint_direction() {
        let log_alpha = Var::from_vec(vec![0.0f32], (), &Device::Cpu).unwrap();
        let too_random = Var::from_vec(vec![-2.0f32], 1, &Device::Cpu).unwrap();
        let loss = sac_alpha_loss(log_alpha.as_tensor(), too_random.as_tensor(), 1.0).unwrap();
        let gradients = loss.backward().unwrap();
        let gradient = gradients
            .get(log_alpha.as_tensor())
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(gradient > 0.0, "gradient descent should lower alpha");
        assert!(gradients.get(too_random.as_tensor()).is_none());

        let too_deterministic = tensor(&[-0.25], 1);
        let loss = sac_alpha_loss(log_alpha.as_tensor(), &too_deterministic, 1.0).unwrap();
        let gradient = loss
            .backward()
            .unwrap()
            .get(log_alpha.as_tensor())
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(gradient < 0.0, "gradient descent should raise alpha");
    }

    #[test]
    fn entropy_change_penalty_moves_toward_collection_entropy_and_honors_masks() {
        let current = Var::from_vec(vec![0.2f32], 1, &Device::Cpu).unwrap();
        let collection = tensor(&[0.8], 1);
        let enabled = tensor(&[1.0], 1);
        let loss =
            sac_entropy_change_loss(current.as_tensor(), &collection, &enabled, 0.5).unwrap();
        let gradient = loss
            .backward()
            .unwrap()
            .get(current.as_tensor())
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        assert!(gradient < 0.0, "gradient descent should raise entropy");

        let masked =
            sac_entropy_change_loss(current.as_tensor(), &collection, &tensor(&[0.0], 1), 0.5)
                .unwrap();
        assert_eq!(masked.to_scalar::<f32>().unwrap(), 0.0);
    }

    #[test]
    fn critic_hard_initialization_and_polyak_update_match_by_name() {
        let online = VarMap::new();
        let online_vb = VarBuilder::from_varmap(&online, DType::F32, &Device::Cpu);
        online_vb
            .get_with_hints(2, "weight", Init::Const(2.0))
            .unwrap();
        let online_weight = online.data().lock().unwrap()["weight"].clone();
        let mut target = VarMap::new();
        let target_vb = VarBuilder::from_varmap(&target, DType::F32, &Device::Cpu);
        let target_weight = target_vb
            .get_with_hints(2, "weight", Init::Const(-2.0))
            .unwrap();
        let mut critic = SACCritic::builder()
            .online_network(DiscreteVectorHeadCritic::new(IdentityModule))
            .target_network(DiscreteVectorHeadCritic::new(IdentityModule))
            .online_vars(&online)
            .target_vars(&mut target)
            .optimizer(CountingOptimizer::with_learning_rate(1e-3))
            .build()
            .unwrap();
        assert_eq!(target_weight.to_vec1::<f32>().unwrap(), vec![2.0, 2.0]);
        online_weight.set(&tensor(&[4.0, 6.0], 2)).unwrap();
        critic.polyak_update(0.5).unwrap();
        assert_eq!(target_weight.to_vec1::<f32>().unwrap(), vec![3.0, 4.0]);
        assert!(matches!(
            critic.polyak_update(f64::NAN),
            Err(SACCriticError::InvalidPolyakCoefficient { tau }) if tau.is_nan()
        ));
    }

    #[test]
    fn critic_rejects_mismatched_parameter_maps() {
        let online = VarMap::new();
        VarBuilder::from_varmap(&online, DType::F32, &Device::Cpu)
            .get_with_hints(1, "online", Init::Const(0.0))
            .unwrap();
        let mut target = VarMap::new();
        VarBuilder::from_varmap(&target, DType::F32, &Device::Cpu)
            .get_with_hints(1, "target", Init::Const(0.0))
            .unwrap();
        let result = SACCritic::builder()
            .online_network(DiscreteVectorHeadCritic::new(IdentityModule))
            .target_network(DiscreteVectorHeadCritic::new(IdentityModule))
            .online_vars(&online)
            .target_vars(&mut target)
            .optimizer(CountingOptimizer::with_learning_rate(1e-3))
            .build();
        assert!(matches!(
            result,
            Err(SACCriticError::ParameterMapMismatch {
                online_only,
                target_only,
            }) if online_only == ["online"] && target_only == ["target"]
        ));
    }

    #[test]
    fn discrete_agent_runs_collection_actor_critics_alpha_and_logging() {
        use crate::{
            agents::test_support::FixedEnv,
            distributions::CategoricalDistribution,
            gym::VectorizedGymWrapper,
            models::{MLP, probabilistic_model::ProbabilisticPolicyModel},
            spaces::{BoxSpace, Discrete},
        };

        let device = Device::Cpu;
        let actor_vars = VarMap::new();
        let actor = MLP::builder()
            .input_size(4)
            .output_size(2)
            .vb(VarBuilder::from_varmap(&actor_vars, DType::F32, &device))
            .hidden_layer_sizes(vec![4])
            .build()
            .unwrap();
        let policy = ProbabilisticPolicyModel::<CategoricalDistribution>::new(actor);
        let (actor_optimizer, actor_steps) = SharedCountingOptimizer::new_counter();

        let online_vars_1 = VarMap::new();
        let online_1 = MLP::builder()
            .input_size(4)
            .output_size(2)
            .vb(VarBuilder::from_varmap(&online_vars_1, DType::F32, &device))
            .hidden_layer_sizes(vec![4])
            .build()
            .unwrap();
        let mut target_vars_1 = VarMap::new();
        let target_1 = MLP::builder()
            .input_size(4)
            .output_size(2)
            .vb(VarBuilder::from_varmap(&target_vars_1, DType::F32, &device))
            .hidden_layer_sizes(vec![4])
            .build()
            .unwrap();
        let (critic_optimizer_1, critic_steps_1) = SharedCountingOptimizer::new_counter();
        let critic_1 = SACCritic::builder()
            .online_network(DiscreteVectorHeadCritic::new(online_1))
            .target_network(DiscreteVectorHeadCritic::new(target_1))
            .online_vars(&online_vars_1)
            .target_vars(&mut target_vars_1)
            .optimizer(critic_optimizer_1)
            .build()
            .unwrap();

        let online_vars_2 = VarMap::new();
        let online_2 = MLP::builder()
            .input_size(4)
            .output_size(2)
            .vb(VarBuilder::from_varmap(&online_vars_2, DType::F32, &device))
            .hidden_layer_sizes(vec![4])
            .build()
            .unwrap();
        let mut target_vars_2 = VarMap::new();
        let target_2 = MLP::builder()
            .input_size(4)
            .output_size(2)
            .vb(VarBuilder::from_varmap(&target_vars_2, DType::F32, &device))
            .hidden_layer_sizes(vec![4])
            .build()
            .unwrap();
        let (critic_optimizer_2, critic_steps_2) = SharedCountingOptimizer::new_counter();
        let critic_2 = SACCritic::builder()
            .online_network(DiscreteVectorHeadCritic::new(online_2))
            .target_network(DiscreteVectorHeadCritic::new(target_2))
            .online_vars(&online_vars_2)
            .target_vars(&mut target_vars_2)
            .optimizer(critic_optimizer_2)
            .build()
            .unwrap();

        let log_alpha = Var::from_vec(vec![0.0f32], (), &device).unwrap();
        let (alpha_optimizer, alpha_steps) = SharedCountingOptimizer::new_counter();
        let entropy = SACEntropyConfiguration::automatic_with_target_schedule(
            log_alpha,
            alpha_optimizer,
            crate::parameter_schedule::LinearSchedule::new(1.0, 0.0),
        );
        let mut logger = RecordingLogger::default();
        let mut agent = SACAgent::builder()
            .policy(policy)
            .actor_optimizer(actor_optimizer)
            .critics(vec![critic_1, critic_2])
            .entropy_configuration(entropy)
            .action_space(Box::new(Discrete::new(2)))
            .observation_space(Box::new(BoxSpace::new_with_universal_bounds(
                vec![4],
                -1.0,
                1.0,
                &device,
            )))
            .device_strategy(ReplayDeviceStrategy::OneDevice(device.clone()))
            .replay_capacity(16)
            .batch_size(2)
            .training_start(2)
            .training_horizon(4)
            .stabilization_configuration(SACStabilizationConfiguration::stable_discrete())
            .q_value_clip(0.5)
            .logger(&mut logger)
            .build()
            .unwrap();
        let deterministic = agent
            .act_deterministic(&Tensor::zeros((1, 4), DType::F32, &device).unwrap())
            .unwrap();
        assert_eq!(deterministic.dims(), &[1]);
        let mut env =
            VectorizedGymWrapper::from(vec![FixedEnv::new(device.clone()), FixedEnv::new(device)]);
        agent.learn(&mut env, 2).unwrap();
        agent.learn(&mut env, 2).unwrap();

        assert_eq!(actor_steps.load(Ordering::Relaxed), 3);
        assert_eq!(critic_steps_1.load(Ordering::Relaxed), 3);
        assert_eq!(critic_steps_2.load(Ordering::Relaxed), 3);
        assert_eq!(alpha_steps.load(Ordering::Relaxed), 3);
        assert_eq!(logger.updates, 3);
        assert_eq!(logger.collections, 2);
        assert_eq!(logger.entropy_change_updates, 3);
        assert_eq!(logger.entropy_change_losses[0], 0.0);
        assert_eq!(logger.target_entropies, vec![0.5, 0.0, 0.0]);
    }
}

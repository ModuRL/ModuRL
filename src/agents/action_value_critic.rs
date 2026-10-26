//! Action-value networks, optimization, target snapshots, and ensemble aggregation.

use crate::models::Forward;
use bon::bon;
use burn::{
    module::{Module, ModuleMapper, ModuleVisitor, Param},
    optim::{GradientsParams, ModuleOptimizer},
    tensor::{DType, Device, Float, Int, Tensor, TensorReadError, kind::Basic},
};

/// Estimates action values (Q values) for replay actions and policy candidates.
/// Ranks O and A include batch and event axes. Rank C adds one candidate axis to A.
/// Associated kinds specify native inputs; implementations define dtype and device requirements.
pub trait ActionValueCriticNetwork<const O: usize = 2, const A: usize = 2, const C: usize = 3>:
    Module
{
    type ObservationKind: Basic;
    type ActionKind: Basic;
    type Error;

    /// Evaluates replay `observations` shaped `[batch, ...observation_shape]` and `actions`
    /// shaped `[batch, ...action_shape]`, returning Q values shaped `[batch, 1]`.
    fn replay_values(
        &self,
        observations: &Tensor<O, Self::ObservationKind>,
        actions: &Tensor<A, Self::ActionKind>,
    ) -> Result<Tensor<2>, Self::Error>;

    /// Evaluates `observations` shaped `[batch, ...observation_shape]` and candidates
    /// shaped `[batch, candidates, ...action_shape]`, returning
    /// `[batch, candidates, 1]`.
    fn policy_values(
        &self,
        observations: &Tensor<O, Self::ObservationKind>,
        candidate_actions: &Tensor<C, Self::ActionKind>,
    ) -> Result<Tensor<3>, Self::Error>;

    /// Values used by the actor loss.
    ///
    /// Implementations should omit critic-parameter gradient paths while
    /// preserving gradients with respect to candidate actions when those
    /// actions are differentiable. `observations` is `[batch, ...observation_shape]`,
    /// `candidate_actions` is `[batch, candidates, ...action_shape]`, and the
    /// result is `[batch, candidates, 1]`.
    fn actor_values(
        &self,
        observations: &Tensor<O, Self::ObservationKind>,
        candidate_actions: &Tensor<C, Self::ActionKind>,
    ) -> Result<Tensor<3>, Self::Error>;
}

/// Failures from critic construction, aggregation, and optimization.
#[derive(Debug, thiserror::Error)]
pub enum ActionValueCriticError<E = std::convert::Infallible> {
    #[error("critic model failed: {0}")]
    ModelError(#[source] E),
    #[error("critic tensor operation failed: {0}")]
    TensorError(#[from] TensorReadError),
    #[error("invalid Polyak coefficient {tau}; expected a finite value in 0..=1")]
    InvalidPolyakCoefficient { tau: f64 },
    #[error("learning rate must be finite and positive")]
    InvalidLearningRate,
    #[error("critic inputs must have nonempty batch, candidate, and event axes")]
    EmptyInput,
    #[error("critic batches differ: {observations} observations and {actions} actions")]
    BatchMismatch { observations: usize, actions: usize },
    #[error("critic shapes differ: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("critic dtypes differ: expected {expected:?}, got {actual:?}")]
    DTypeMismatch { expected: DType, actual: DType },
    #[error("critic tensors must use the same device")]
    DeviceMismatch,
    #[error("discrete critic action indices must be in 0..{action_count}")]
    InvalidActionIndex { action_count: usize },
    #[error("the critic ensemble produced no values")]
    NoCriticValues,
}

/// Adapts a Float model that maps concatenated observations and actions to `[batch_size, 1]`.
/// Flattens event axes. Observations and actions must share Float dtype and device.
///
/// ```compile_fail
/// use burn::{nn::Relu, tensor::{Device, Tensor}};
/// use modurl::agents::action_value_critic::{ScalarStateActionCritic, ActionValueCriticNetwork};
/// let critic = ScalarStateActionCritic::new(Relu);
/// let observations = Tensor::<2>::ones([1, 2], &Device::flex());
/// let actions = Tensor::<2>::ones([1, 1], &Device::flex());
/// // Rank-2 replay actions require rank-3 candidates.
/// let _ = <_ as ActionValueCriticNetwork<2, 2, 2>>::policy_values(&critic, &observations, &actions);
/// ```
#[derive(Clone, Debug)]
pub struct ScalarStateActionCritic<M: Module> {
    module: M,
}

impl<M: Module> Module for ScalarStateActionCritic<M> {
    fn collect_devices(&self, devices: Vec<Device>) -> Vec<Device> {
        self.module.collect_devices(devices)
    }

    fn fork(self, device: &Device) -> Self {
        Self::new(self.module.fork(device))
    }

    fn to_device(self, device: &Device) -> Self {
        Self::new(self.module.to_device(device))
    }

    fn train(self) -> Self {
        Self::new(self.module.train())
    }

    fn valid(&self) -> Self {
        Self::new(self.module.valid())
    }

    fn visit<V: ModuleVisitor>(&self, visitor: &mut V) {
        self.module.visit(visitor);
    }

    fn map<T: ModuleMapper>(self, mapper: &mut T) -> Self {
        Self::new(self.module.map(mapper))
    }

    fn materialize(self) -> Self {
        Self::new(self.module.materialize())
    }
}

impl<M: Module> ScalarStateActionCritic<M> {
    /// Owns the model used to evaluate scalar Q values.
    pub fn new(module: M) -> Self {
        Self { module }
    }
}

/// Validates rank-O observations `[batch_size, ...observation_shape]` and rank-A actions `[batch_size, ...action_shape]`.
/// Candidates add an axis after batch. Requires nonempty axes, matching batches, and matching devices; preserves both native tensors.
fn validate_critic_inputs<E, const O: usize, const A: usize, K: Basic, L: Basic>(
    observations: &Tensor<O, K>,
    actions: &Tensor<A, L>,
) -> Result<(), ActionValueCriticError<E>> {
    const {
        assert!(
            O >= 2 && A >= 2,
            "critic inputs require batch and event axes"
        );
    }
    if observations.dims().contains(&0) || actions.dims().contains(&0) {
        return Err(ActionValueCriticError::EmptyInput);
    }
    if observations.dims()[0] != actions.dims()[0] {
        return Err(ActionValueCriticError::BatchMismatch {
            observations: observations.dims()[0],
            actions: actions.dims()[0],
        });
    }
    if observations.device() != actions.device() {
        return Err(ActionValueCriticError::DeviceMismatch);
    }
    Ok(())
}

impl<M, const O: usize, const A: usize, const C: usize> ActionValueCriticNetwork<O, A, C>
    for ScalarStateActionCritic<M>
where
    M: Module + Forward<2, 2, InputKind = Float, OutputKind = Float>,
{
    type ObservationKind = Float;
    type ActionKind = Float;
    type Error = ActionValueCriticError<M::Error>;

    /// Adds a size-one candidate axis to actions `[batch_size, ...action_shape]` and removes it from `[batch_size, 1, 1]` values.
    /// Observations `[batch_size, ...observation_shape]` retain their Float dtype and device. Returns `[batch_size, 1]` with critic gradients.
    fn replay_values(
        &self,
        observations: &Tensor<O>,
        actions: &Tensor<A>,
    ) -> Result<Tensor<2>, Self::Error> {
        const {
            assert!(C == A + 1, "candidate rank must be action rank + 1");
        }
        <Self as ActionValueCriticNetwork<O, A, C>>::policy_values(
            self,
            observations,
            &actions.clone().unsqueeze_dim::<C>(1),
        )
        .map(|values| values.squeeze_dim(1))
    }

    /// Broadcasts observations over candidates and flattens event axes into `[batch_size * candidate_count, observation_size + action_size]`.
    /// Returns `[batch_size, candidate_count, 1]`. Inputs must share Float dtype and device; preserves gradients.
    fn policy_values(
        &self,
        observations: &Tensor<O>,
        candidate_actions: &Tensor<C>,
    ) -> Result<Tensor<3>, Self::Error> {
        const {
            assert!(
                C == A + 1 && A >= 2,
                "candidates must add one axis to batched actions"
            );
        }

        validate_critic_inputs(observations, candidate_actions)?;
        if observations.dtype() != candidate_actions.dtype() {
            return Err(ActionValueCriticError::DTypeMismatch {
                expected: observations.dtype(),
                actual: candidate_actions.dtype(),
            });
        }

        let batch_size = observations.dims()[0];
        let candidate_count = candidate_actions.dims()[1];
        let observation_size = observations.dims()[1..].iter().product::<usize>();
        let action_size = candidate_actions.dims()[2..].iter().product::<usize>();

        let inputs = Tensor::cat(
            vec![
                observations
                    .clone()
                    .reshape([batch_size, observation_size])
                    .unsqueeze_dim::<3>(1)
                    .expand([batch_size, candidate_count, observation_size]),
                candidate_actions
                    .clone()
                    .reshape([batch_size, candidate_count, action_size]),
            ],
            2,
        )
        .reshape([batch_size * candidate_count, observation_size + action_size]);

        let values = self
            .module
            .forward(inputs)
            .map_err(ActionValueCriticError::ModelError)?;
        if values.dims() != [batch_size * candidate_count, 1] {
            return Err(ActionValueCriticError::ShapeMismatch {
                expected: vec![batch_size * candidate_count, 1],
                actual: values.dims().to_vec(),
            });
        }
        if values.device() != observations.device() {
            return Err(ActionValueCriticError::DeviceMismatch);
        }
        Ok(values.reshape([batch_size, candidate_count, 1]))
    }

    /// Evaluates observations and candidates with the `policy_values` layouts, returning `[batch_size, candidate_count, 1]`.
    /// Uses a snapshot with parameter gradients disabled. Preserves differentiable observation and action gradients without changing the online model.
    fn actor_values(
        &self,
        observations: &Tensor<O>,
        candidate_actions: &Tensor<C>,
    ) -> Result<Tensor<3>, Self::Error> {
        let frozen = Self::new(self.module.clone().set_require_grad(false));
        <Self as ActionValueCriticNetwork<O, A, C>>::policy_values(
            &frozen,
            observations,
            candidate_actions,
        )
    }
}

/// Adapts a model returning Float `[batch_size, action_count]` to scalar Int replay and candidate actions.
/// Observations preserve the model's native input kind and dtype. The model performs any required preprocessing.
#[derive(Clone, Debug)]
pub struct DiscreteVectorHeadCritic<M: Module> {
    module: M,
}

impl<M: Module> Module for DiscreteVectorHeadCritic<M> {
    fn collect_devices(&self, devices: Vec<Device>) -> Vec<Device> {
        self.module.collect_devices(devices)
    }

    fn fork(self, device: &Device) -> Self {
        Self::new(self.module.fork(device))
    }

    fn to_device(self, device: &Device) -> Self {
        Self::new(self.module.to_device(device))
    }

    fn train(self) -> Self {
        Self::new(self.module.train())
    }

    fn valid(&self) -> Self {
        Self::new(self.module.valid())
    }

    fn visit<V: ModuleVisitor>(&self, visitor: &mut V) {
        self.module.visit(visitor);
    }

    fn map<T: ModuleMapper>(self, mapper: &mut T) -> Self {
        Self::new(self.module.map(mapper))
    }

    fn materialize(self) -> Self {
        Self::new(self.module.materialize())
    }
}

impl<M: Module> DiscreteVectorHeadCritic<M> {
    /// Owns the model used to evaluate one value per discrete action.
    pub fn new(module: M) -> Self {
        Self { module }
    }
}

impl<M, const O: usize> ActionValueCriticNetwork<O, 2, 3> for DiscreteVectorHeadCritic<M>
where
    M: Module + Forward<O, 2, OutputKind = Float>,
{
    type ObservationKind = M::InputKind;
    type ActionKind = Int;
    type Error = ActionValueCriticError<M::Error>;

    /// Gathers Int actions `[batch_size, 1]` from Float model values `[batch_size, action_count]`, returning `[batch_size, 1]`.
    /// Observations `[batch_size, ...observation_shape]` retain their native kind and dtype. Inputs must share a device.
    fn replay_values(
        &self,
        observations: &Tensor<O, M::InputKind>,
        actions: &Tensor<2, Int>,
    ) -> Result<Tensor<2>, Self::Error> {
        validate_critic_inputs(observations, actions)?;
        if actions.dims()[1] != 1 {
            return Err(ActionValueCriticError::ShapeMismatch {
                expected: vec![actions.dims()[0], 1],
                actual: actions.dims().to_vec(),
            });
        }

        self.policy_values(observations, &actions.clone().unsqueeze_dim::<3>(1))
            .map(|values| values.squeeze_dim(1))
    }

    /// Gathers Int candidates `[batch_size, candidate_count, 1]` from Float model values `[batch_size, action_count]`.
    /// Returns `[batch_size, candidate_count, 1]`. Preserves native observations and model gradients; reads one bounds-check Boolean to the host.
    fn policy_values(
        &self,
        observations: &Tensor<O, M::InputKind>,
        candidate_actions: &Tensor<3, Int>,
    ) -> Result<Tensor<3>, Self::Error> {
        validate_critic_inputs(observations, candidate_actions)?;
        let [batch_size, candidate_count, scalar_size] = candidate_actions.dims();
        if scalar_size != 1 {
            return Err(ActionValueCriticError::ShapeMismatch {
                expected: vec![batch_size, candidate_count, 1],
                actual: candidate_actions.dims().to_vec(),
            });
        }

        let values = self
            .module
            .forward(observations.clone())
            .map_err(ActionValueCriticError::ModelError)?;
        if values.dims()[0] != batch_size || values.dims()[1] == 0 {
            return Err(ActionValueCriticError::ShapeMismatch {
                expected: vec![batch_size, values.dims()[1].max(1)],
                actual: values.dims().to_vec(),
            });
        }
        if values.device() != candidate_actions.device() {
            return Err(ActionValueCriticError::DeviceMismatch);
        }

        let action_count = values.dims()[1];
        let invalid = candidate_actions.clone().lower_elem(0).bool_or(
            candidate_actions
                .clone()
                .greater_equal_elem(action_count as i64),
        );
        if invalid.any().try_into_scalar::<bool>()? {
            return Err(ActionValueCriticError::InvalidActionIndex { action_count });
        }

        Ok(values
            .gather(1, candidate_actions.clone().squeeze_dim::<2>(2))
            .unsqueeze_dim(2))
    }

    /// Returns detached Float `[batch_size, candidate_count, 1]` for observations and Int candidates with the `policy_values` layouts.
    /// Discrete indices have no action gradients. Policy expectation weights retain their separate gradient paths.
    fn actor_values(
        &self,
        observations: &Tensor<O, M::InputKind>,
        candidate_actions: &Tensor<3, Int>,
    ) -> Result<Tensor<3>, Self::Error> {
        self.policy_values(observations, candidate_actions)
            .map(Tensor::detach)
    }
}

/// Owns one independently optimized online critic and a detached target initialized by cloning.
/// Online and target models share parameter IDs and initial values; optimizer updates leave the target snapshot unchanged.
///
/// ```
/// use burn::{optim::AdamWConfig, tensor::Device};
/// use modurl::{agents::action_value_critic::{ActionValueCritic, ScalarStateActionCritic}, models::MLP};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let device = Device::flex().autodiff();
/// let model = MLP::builder()
///     .input_size(5)
///     .output_size(1)
///     .hidden_layer_sizes(vec![16])
///     .options(&device)
///     .build()?;
/// let mut critic = ActionValueCritic::builder()
///     .online_network(ScalarStateActionCritic::new(model))
///     .optimizer(AdamWConfig::new().init())
///     .learning_rate(3e-4)
///     .build()?;
/// critic.polyak_update(0.005)?;
/// critic.hard_update()?;
/// # Ok(())
/// # }
/// ```
pub struct ActionValueCritic<N: Module> {
    online_network: N,
    target_network: N,
    optimizer: ModuleOptimizer,
    learning_rate: f64,
}

#[bon]
impl<N: Module> ActionValueCritic<N> {
    /// Returns the owned online model for parameter inspection and checkpoint snapshots.
    pub fn online_network(&self) -> &N {
        &self.online_network
    }

    /// Returns the learning rate used for online critic updates.
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Owns the model and optimizer, enables online training, and clones the target snapshot.
    /// Rejects a nonpositive or non-finite learning rate.
    #[builder]
    pub fn new(
        online_network: N,
        #[builder(into)] optimizer: ModuleOptimizer,
        learning_rate: f64,
    ) -> Result<Self, ActionValueCriticError> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(ActionValueCriticError::InvalidLearningRate);
        }

        let online_network = online_network.train();
        let target_network = online_network.clone().valid();

        Ok(Self {
            online_network,
            target_network,
            optimizer,
            learning_rate,
        })
    }

    /// Backpropagates Float loss `[1]` and updates online parameters, preserving their shapes, dtypes, devices, and IDs.
    /// Loss must come from this online model. Leaves the target snapshot unchanged; rejects a nonscalar loss layout.
    pub fn backward_step(&mut self, loss: &Tensor<1>) -> Result<(), ActionValueCriticError> {
        if loss.dims() != [1] {
            return Err(ActionValueCriticError::ShapeMismatch {
                expected: vec![1],
                actual: loss.dims().to_vec(),
            });
        }

        let gradients = GradientsParams::from_grads(loss.clone().backward(), &self.online_network);
        self.online_network =
            self.optimizer
                .step(self.learning_rate, self.online_network.clone(), gradients);
        Ok(())
    }

    /// Copies online parameters into a detached target snapshot, preserving parameter IDs and layouts.
    pub fn hard_update(&mut self) -> Result<(), ActionValueCriticError> {
        self.target_network = self.online_network.clone().valid();
        Ok(())
    }

    /// Updates rank-D target parameters to `tau * online + (1 - tau) * target`.
    /// Preserves parameter IDs, shapes, dtypes, and devices. Rejects a non-finite tau or a value outside 0..=1.
    pub fn polyak_update(&mut self, tau: f64) -> Result<(), ActionValueCriticError> {
        if !tau.is_finite() || !(0.0..=1.0).contains(&tau) {
            return Err(ActionValueCriticError::InvalidPolyakCoefficient { tau });
        }
        if tau == 0.0 {
            return Ok(());
        }
        if tau == 1.0 {
            return self.hard_update();
        }
        self.target_network =
            interpolate_target_model(&self.online_network, self.target_network.clone(), tau);
        Ok(())
    }

    /// Evaluates online observations `[batch_size, ...observation_shape]` and replay actions `[batch_size, ...action_shape]`.
    /// Returns Float `[batch_size, 1]`, retaining critic gradients and the network's input dtype and device requirements.
    pub fn online_replay_values<const O: usize, const A: usize, const C: usize>(
        &self,
        observations: &Tensor<O, N::ObservationKind>,
        actions: &Tensor<A, N::ActionKind>,
    ) -> Result<Tensor<2>, N::Error>
    where
        N: ActionValueCriticNetwork<O, A, C>,
    {
        self.online_network.replay_values(observations, actions)
    }

    /// Evaluates observations and candidates `[batch_size, candidate_count, ...action_shape]` with the network's native input requirements.
    /// Returns Float `[batch_size, candidate_count, 1]`, preserving action gradients and excluding critic-parameter gradients.
    pub fn online_actor_values<const O: usize, const A: usize, const C: usize>(
        &self,
        observations: &Tensor<O, N::ObservationKind>,
        actions: &Tensor<C, N::ActionKind>,
    ) -> Result<Tensor<3>, N::Error>
    where
        N: ActionValueCriticNetwork<O, A, C>,
    {
        self.online_network.actor_values(observations, actions)
    }

    /// Evaluates target observations and candidates `[batch_size, candidate_count, ...action_shape]` with the network's native input requirements.
    /// Returns detached Float `[batch_size, candidate_count, 1]`; observations retain their documented batch and event axes.
    pub fn target_policy_values<const O: usize, const A: usize, const C: usize>(
        &self,
        observations: &Tensor<O, N::ObservationKind>,
        actions: &Tensor<C, N::ActionKind>,
    ) -> Result<Tensor<3>, N::Error>
    where
        N: ActionValueCriticNetwork<O, A, C>,
    {
        self.target_network
            .policy_values(observations, actions)
            .map(Tensor::detach)
    }

    /// Evaluates target observations and replay actions `[batch_size, ...action_shape]` with the network's native input requirements.
    /// Returns detached Float `[batch_size, 1]`; observations retain their documented batch and event axes.
    pub fn target_replay_values<const O: usize, const A: usize, const C: usize>(
        &self,
        observations: &Tensor<O, N::ObservationKind>,
        actions: &Tensor<A, N::ActionKind>,
    ) -> Result<Tensor<2>, N::Error>
    where
        N: ActionValueCriticNetwork<O, A, C>,
    {
        self.target_network
            .replay_values(observations, actions)
            .map(Tensor::detach)
    }
}

/// Updates a cloned target model with tau * online + (1 - tau) * target and disables target gradients.
/// Parameter IDs, shapes, dtypes, and devices must match. Tau must be finite and in 0..=1.
pub(crate) fn interpolate_target_model<N: Module>(online: &N, target: N, tau: f64) -> N {
    if tau == 0.0 {
        return target;
    }
    if tau == 1.0 {
        return online.clone().valid();
    }
    let mut values = ActionValueParameterValues {
        values: GradientsParams::new(),
    };
    online.visit(&mut values);
    let mut mapper = ActionValueTargetUpdate {
        values: values.values,
        tau,
    };
    target.map(&mut mapper).valid()
}

struct ActionValueParameterValues {
    values: GradientsParams,
}

impl ModuleVisitor for ActionValueParameterValues {
    /// Stores detached rank-D online parameter values by ID, preserving shape, dtype, and device.
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        self.values.register(param.id, param.val().detach());
    }
}

struct ActionValueTargetUpdate {
    values: GradientsParams,
    tau: f64,
}

impl ModuleMapper for ActionValueTargetUpdate {
    /// Interpolates one rank-D target parameter, preserving ID, shape, dtype, and device.
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let Some(online) = self.values.get::<D>(param.id) else {
            unreachable!("cloned critic parameter IDs must match");
        };
        param.map(|target| online * self.tau + target.detach() * (1.0 - self.tau))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionValueCriticAggregationMode {
    /// Uses the arithmetic mean of all critic values.
    Mean,
    /// Uses the middle critic value, or the mean of the two middle values.
    Median,
    /// Uses the lowest critic value.
    Min,
    /// Uses the highest critic value.
    Max,
}

/// Aggregates Float rank-D tensors with identical shapes, dtypes, and devices, returning the same layout.
/// Rank S equals D + 1. Adds and removes a critic axis; typical layouts are `[batch_size, 1]` and `[batch_size, candidate_count, 1]`.
/// Preserves gradients. Min and Max split each pair's gradient equally at ties; Median follows native sort ordering at ties.
///
/// ```compile_fail
/// use burn::tensor::{Device, Tensor};
/// use modurl::agents::action_value_critic::{aggregate_critic_values, ActionValueCriticAggregationMode};
/// let values = [Tensor::<2>::ones([1, 1], &Device::flex())];
/// // Stacking rank-2 values requires rank 3.
/// let _ = aggregate_critic_values::<2, 2>(&values, ActionValueCriticAggregationMode::Mean);
/// ```
pub fn aggregate_critic_values<const D: usize, const S: usize>(
    values: &[Tensor<D>],
    mode: ActionValueCriticAggregationMode,
) -> Result<Tensor<D>, ActionValueCriticError> {
    const {
        assert!(S == D + 1, "stack rank must be value rank + 1");
    }
    let Some(first) = values.first() else {
        return Err(ActionValueCriticError::NoCriticValues);
    };

    for value in &values[1..] {
        if value.dims() != first.dims() {
            return Err(ActionValueCriticError::ShapeMismatch {
                expected: first.dims().to_vec(),
                actual: value.dims().to_vec(),
            });
        }
        if value.dtype() != first.dtype() {
            return Err(ActionValueCriticError::DTypeMismatch {
                expected: first.dtype(),
                actual: value.dtype(),
            });
        }
        if value.device() != first.device() {
            return Err(ActionValueCriticError::DeviceMismatch);
        }
    }

    Ok(match mode {
        ActionValueCriticAggregationMode::Mean => Tensor::stack::<S>(values.to_vec(), D)
            .mean_dim(D)
            .squeeze_dim(D),
        ActionValueCriticAggregationMode::Min | ActionValueCriticAggregationMode::Max => {
            values[1..].iter().fold(first.clone(), |current, value| {
                let ties = current.clone().equal(value.clone());
                let selected = match mode {
                    ActionValueCriticAggregationMode::Min => {
                        current.clone().min_pair(value.clone())
                    }
                    _ => current.clone().max_pair(value.clone()),
                };
                selected.mask_where(ties, current * 0.5 + value.clone() * 0.5)
            })
        }
        ActionValueCriticAggregationMode::Median => {
            let sorted = Tensor::stack::<S>(values.to_vec(), D).sort(D);
            let count = values.len();
            let upper = sorted.clone().narrow(D, count / 2, 1).squeeze_dim::<D>(D);
            if count % 2 == 1 {
                upper
            } else {
                (sorted.narrow(D, count / 2 - 1, 1).squeeze_dim::<D>(D) + upper) * 0.5
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::optim::SgdConfig;
    use std::convert::Infallible;

    #[derive(Module, Debug)]
    struct SumModel {
        weight: Param<Tensor<2>>,
    }

    impl Forward<2, 2> for SumModel {
        type InputKind = Float;
        type OutputKind = Float;
        type Error = Infallible;

        /// Maps Float `[batch_size, features]` through weights `[features, outputs]` to `[batch_size, outputs]`, preserving gradients.
        fn forward(&self, input: Tensor<2>) -> Result<Tensor<2>, Self::Error> {
            Ok(input.matmul(self.weight.val()))
        }
    }

    #[derive(Module, Debug)]
    struct IntegerModel;

    impl Forward<2, 2> for IntegerModel {
        type InputKind = Int;
        type OutputKind = Float;
        type Error = Infallible;

        /// Explicitly casts Int observations `[batch_size, action_count]` to Float, preserving shape and device.
        fn forward(&self, input: Tensor<2, Int>) -> Result<Tensor<2>, Self::Error> {
            assert_eq!(input.dtype(), DType::I64);
            Ok(input.float())
        }
    }

    #[derive(Debug, thiserror::Error)]
    #[error("custom critic failed")]
    struct CustomError;

    #[derive(Module, Debug)]
    struct CustomNetwork {
        weight: Param<Tensor<2>>,
        #[module(skip)]
        fail: bool,
    }

    impl ActionValueCriticNetwork for CustomNetwork {
        type ObservationKind = Int;
        type ActionKind = Float;
        type Error = CustomError;

        /// Evaluates observations and replay actions shaped `[batch, 1]`.
        fn replay_values(
            &self,
            observations: &Tensor<2, Int>,
            actions: &Tensor<2>,
        ) -> Result<Tensor<2>, Self::Error> {
            if self.fail {
                return Err(CustomError);
            }
            Ok(Tensor::cat(
                vec![
                    observations.clone().float().cast(self.weight.val().dtype()),
                    actions.clone(),
                ],
                1,
            )
            .matmul(self.weight.val()))
        }

        /// Evaluates observations `[batch, 1]` and candidates `[batch, candidates, 1]`.
        fn policy_values(
            &self,
            observations: &Tensor<2, Int>,
            candidate_actions: &Tensor<3>,
        ) -> Result<Tensor<3>, Self::Error> {
            if self.fail {
                return Err(CustomError);
            }
            let [batch_size, candidate_count, _] = candidate_actions.dims();
            let inputs = Tensor::cat(
                vec![
                    observations
                        .clone()
                        .float()
                        .cast(self.weight.val().dtype())
                        .unsqueeze_dim::<3>(1)
                        .expand([batch_size, candidate_count, 1]),
                    candidate_actions.clone(),
                ],
                2,
            )
            .reshape([batch_size * candidate_count, 2]);
            Ok(inputs
                .matmul(self.weight.val())
                .reshape([batch_size, candidate_count, 1]))
        }

        /// Evaluates candidates `[batch, candidates, 1]` without critic-parameter gradients.
        fn actor_values(
            &self,
            observations: &Tensor<2, Int>,
            candidate_actions: &Tensor<3>,
        ) -> Result<Tensor<3>, Self::Error> {
            self.clone()
                .set_require_grad(false)
                .policy_values(observations, candidate_actions)
        }
    }

    #[test]
    fn custom_action_value_critic_network_contract_is_supported() {
        let device = Device::flex().autodiff();
        let mut critic = ActionValueCritic::builder()
            .online_network(CustomNetwork {
                weight: Param::from_tensor(Tensor::ones([2, 1], (&device, DType::F64))),
                fail: false,
            })
            .optimizer(SgdConfig::new().init())
            .learning_rate(0.1)
            .build()
            .unwrap();
        let observations = Tensor::from_data([[2i64]], (&device, DType::I64));
        let actions = Tensor::from_data([[3.0f64]], (&device, DType::F64));
        let candidates =
            Tensor::from_data([[[3.0f64], [4.0]]], (&device, DType::F64)).require_grad();
        assert_eq!(
            numbers(
                critic
                    .online_replay_values::<2, 2, 3>(&observations, &actions)
                    .unwrap()
            ),
            [5.0]
        );
        assert_eq!(
            numbers(
                critic
                    .target_policy_values::<2, 2, 3>(&observations, &candidates)
                    .unwrap()
            ),
            [5.0, 6.0]
        );
        let gradients = critic
            .online_actor_values::<2, 2, 3>(&observations, &candidates)
            .unwrap()
            .sum()
            .backward();
        assert_eq!(numbers(candidates.grad(&gradients).unwrap()), [1.0, 1.0]);
        assert!(
            critic
                .online_network()
                .weight
                .val()
                .grad(&gradients)
                .is_none()
        );
        let loss = critic
            .online_replay_values::<2, 2, 3>(&observations, &actions)
            .unwrap()
            .sum();
        critic.backward_step(&loss).unwrap();
        assert_eq!(observations.dtype(), DType::I64);

        let failing = ActionValueCritic::builder()
            .online_network(CustomNetwork {
                weight: Param::from_tensor(Tensor::ones([2, 1], &device)),
                fail: true,
            })
            .optimizer(SgdConfig::new().init())
            .learning_rate(0.1)
            .build()
            .unwrap();
        assert!(matches!(
            failing.online_replay_values::<2, 2, 3>(&observations, &actions),
            Err(CustomError)
        ));
    }

    /// Creates an autodiff CPU model with weights `[features, 1]` in the requested Float dtype.
    fn sum_model(features: usize, dtype: DType) -> SumModel {
        let device = Device::flex().autodiff();
        SumModel {
            weight: Param::from_tensor(Tensor::ones([features, 1], (&device, dtype))),
        }
    }

    /// Reads rank-D Float values as flattened F64 host values; leaves the source shape and device unchanged.
    fn numbers<const D: usize>(value: Tensor<D>) -> Vec<f64> {
        value
            .into_data()
            .convert::<f64>()
            .try_to_vec::<f64>()
            .unwrap()
    }

    #[test]
    fn scalar_replay_and_candidates_preserve_layout_dtype_and_gradients() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        for dtype in [DType::F32, DType::F64] {
            let model = sum_model(3, dtype);
            let device = model.weight.val().device();
            let critic = ScalarStateActionCritic::new(model);
            let observations = Tensor::from_data([[1.0f64, 2.0], [3.0, 4.0]], (&device, dtype));
            let candidates =
                Tensor::from_data([[[10.0f64], [20.0]], [[30.0], [40.0]]], (&device, dtype));
            let values = <_ as ActionValueCriticNetwork<2, 2, 3>>::policy_values(
                &critic,
                &observations,
                &candidates,
            )
            .unwrap();
            assert_eq!(values.dims(), [2, 2, 1]);
            assert_eq!(values.dtype(), dtype);
            assert_eq!(values.device(), device);
            assert_eq!(numbers(values), [13.0, 23.0, 37.0, 47.0]);
            let actions = Tensor::from_data([[10.0f64], [30.0]], (&device, dtype));
            let values = <_ as ActionValueCriticNetwork<2, 2, 3>>::replay_values(
                &critic,
                &observations,
                &actions,
            )
            .unwrap();
            assert_eq!(values.dims(), [2, 1]);
            assert_eq!(numbers(values), [13.0, 37.0]);
        }
    }

    #[test]
    fn scalar_adapter_flattens_five_action_event_axes() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let model = sum_model(34, DType::F32);
        let device = model.weight.val().device();
        let critic = ScalarStateActionCritic::new(model);
        let observations = Tensor::from_data([[1.0f32, 2.0], [3.0, 4.0]], &device);
        let candidates = Tensor::ones([2, 2, 2, 2, 2, 2, 2], &device);
        assert_eq!(
            numbers(
                <_ as ActionValueCriticNetwork<2, 6, 7>>::policy_values(
                    &critic,
                    &observations,
                    &candidates
                )
                .unwrap()
            ),
            [35.0, 35.0, 39.0, 39.0]
        );
        let actions = Tensor::ones([2, 2, 2, 2, 2, 2], &device);
        assert_eq!(
            numbers(
                <_ as ActionValueCriticNetwork<2, 6, 7>>::replay_values(
                    &critic,
                    &observations,
                    &actions
                )
                .unwrap()
            ),
            [35.0, 39.0]
        );
    }

    #[test]
    fn scalar_actor_gradients_reach_actions_but_not_critic_parameters() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let model = sum_model(3, DType::F64);
        let weight = model.weight.val();
        let device = weight.device();
        let critic = ScalarStateActionCritic::new(model);
        let observations = Tensor::from_data([[1.0f64, 2.0]], (&device, DType::F64)).require_grad();
        let actions = Tensor::from_data([[[3.0f64], [4.0]]], (&device, DType::F64)).require_grad();
        let gradients = <_ as ActionValueCriticNetwork<2, 2, 3>>::actor_values(
            &critic,
            &observations,
            &actions,
        )
        .unwrap()
        .sum()
        .backward();
        assert_eq!(numbers(actions.grad(&gradients).unwrap()), [1.0, 1.0]);
        assert_eq!(numbers(observations.grad(&gradients).unwrap()), [2.0, 2.0]);
        assert!(weight.grad(&gradients).is_none());
        let replay_actions = Tensor::from_data([[3.0f64]], (&device, DType::F64));
        let gradients = <_ as ActionValueCriticNetwork<2, 2, 3>>::replay_values(
            &critic,
            &observations,
            &replay_actions,
        )
        .unwrap()
        .sum()
        .backward();
        assert_eq!(numbers(weight.grad(&gradients).unwrap()), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn discrete_adapter_gathers_indices_and_keeps_native_observations() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex();
        let critic = DiscreteVectorHeadCritic::new(IntegerModel);
        let observations =
            Tensor::<2, Int>::from_data([[1i64, 2, 3], [4, 5, 6]], (&device, DType::I64));
        let actions = Tensor::<2, Int>::from_data([[2i64], [0]], &device);
        assert_eq!(
            numbers(critic.replay_values(&observations, &actions).unwrap()),
            [3.0, 4.0]
        );
        let candidates =
            Tensor::<3, Int>::from_data([[[0i64], [1], [2]], [[0], [1], [2]]], &device);
        let values = critic.policy_values(&observations, &candidates).unwrap();
        assert_eq!(values.dims(), [2, 3, 1]);
        assert_eq!(numbers(values), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(observations.dtype(), DType::I64);
        for index in [-1i64, 3] {
            assert!(matches!(
                critic.replay_values(&observations, &Tensor::from_data([[index], [0]], &device)),
                Err(ActionValueCriticError::InvalidActionIndex { .. })
            ));
        }
        assert!(matches!(
            critic.replay_values(&observations, &Tensor::zeros([2, 2], &device)),
            Err(ActionValueCriticError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn discrete_actor_gradients_reach_weights_but_not_critic_parameters() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        let weight = Tensor::<2>::ones([2, 3], &device).require_grad();
        let critic = DiscreteVectorHeadCritic::new(SumModel {
            weight: Param::from_tensor(weight.clone()),
        });
        let observations = Tensor::from_data([[1.0f32, 2.0]], &device);
        let candidates = Tensor::<3, Int>::from_data([[[0i64], [1], [2]]], &device);
        let values = critic.actor_values(&observations, &candidates).unwrap();
        assert!(!values.is_require_grad());
        let weights = Tensor::from_data([[[0.2f32], [0.3], [0.5]]], &device).require_grad();
        let expected = numbers(values.clone());
        let gradients = (values * weights.clone()).sum().backward();
        assert_eq!(numbers(weights.grad(&gradients).unwrap()), expected);
        assert!(weight.grad(&gradients).is_none());
    }

    #[test]
    fn cloned_targets_are_independent_and_polyak_preserves_ids_and_dtypes() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        for dtype in [DType::F32, DType::F64] {
            let model = sum_model(3, dtype);
            let id = model.weight.id;
            let device = model.weight.val().device();
            let mut critic = ActionValueCritic::builder()
                .online_network(ScalarStateActionCritic::new(model))
                .optimizer(SgdConfig::new().init())
                .learning_rate(0.5)
                .build()
                .unwrap();
            assert_eq!(critic.target_network.module.weight.id, id);
            assert!(!critic.target_network.module.weight.val().is_require_grad());
            let observations = Tensor::ones([1, 2], (&device, dtype));
            let actions = Tensor::ones([1, 1], (&device, dtype));
            let loss = critic
                .online_replay_values::<2, 2, 3>(&observations, &actions)
                .unwrap()
                .sum();
            critic.backward_step(&loss).unwrap();
            assert_eq!(critic.online_network.module.weight.id, id);
            assert_eq!(numbers(critic.online_network.module.weight.val()), [0.5; 3]);
            assert_eq!(numbers(critic.target_network.module.weight.val()), [1.0; 3]);
            critic.polyak_update(0.25).unwrap();
            let target = &critic.target_network.module.weight;
            assert_eq!(target.id, id);
            assert_eq!(target.val().dims(), [3, 1]);
            assert_eq!(target.val().dtype(), dtype);
            assert_eq!(target.val().device(), device);
            assert_eq!(numbers(target.val()), [0.875; 3]);
            assert!(!target.val().is_require_grad());
            critic.polyak_update(0.0).unwrap();
            assert_eq!(
                numbers(critic.target_network.module.weight.val()),
                [0.875; 3]
            );
            critic.polyak_update(1.0).unwrap();
            assert_eq!(numbers(critic.target_network.module.weight.val()), [0.5; 3]);
            let candidates = Tensor::ones([1, 2, 1], (&device, dtype)).require_grad();
            let values = critic
                .target_policy_values::<2, 2, 3>(&observations, &candidates)
                .unwrap();
            assert!(!values.is_require_grad());
            assert_eq!(numbers(values), [1.5; 2]);
            let values = critic
                .target_replay_values::<2, 2, 3>(&observations, &actions)
                .unwrap();
            assert!(!values.is_require_grad());
            assert_eq!(numbers(values), [1.5]);
            let values = critic
                .online_actor_values::<2, 2, 3>(&observations, &candidates)
                .unwrap();
            assert_eq!(
                numbers(candidates.grad(&values.sum().backward()).unwrap()),
                [0.5; 2]
            );
            assert!(matches!(
                critic.backward_step(&Tensor::ones([2], &device)),
                Err(ActionValueCriticError::ShapeMismatch { .. })
            ));
            for tau in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
                assert!(matches!(
                    critic.polyak_update(tau),
                    Err(ActionValueCriticError::InvalidPolyakCoefficient { .. })
                ));
            }
        }
    }

    #[test]
    fn invalid_configuration_and_scalar_inputs_return_typed_errors() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        for rate in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                ActionValueCritic::builder()
                    .online_network(sum_model(3, DType::F32))
                    .optimizer(SgdConfig::new().init())
                    .learning_rate(rate)
                    .build(),
                Err(ActionValueCriticError::InvalidLearningRate)
            ));
        }
        let model = sum_model(3, DType::F32);
        let device = model.weight.val().device();
        let critic = ScalarStateActionCritic::new(model);
        let observations = Tensor::ones([2, 2], &device);
        assert!(matches!(
            <_ as ActionValueCriticNetwork<2, 2, 3>>::policy_values(
                &critic,
                &observations,
                &Tensor::ones([1, 2, 1], &device)
            ),
            Err(ActionValueCriticError::BatchMismatch { .. })
        ));
        assert!(matches!(
            <_ as ActionValueCriticNetwork<2, 2, 3>>::policy_values(
                &critic,
                &observations,
                &Tensor::zeros([2, 0, 1], &device)
            ),
            Err(ActionValueCriticError::EmptyInput)
        ));
        assert!(matches!(
            <_ as ActionValueCriticNetwork<2, 2, 3>>::policy_values(
                &critic,
                &observations,
                &Tensor::ones([2, 2, 1], (&device, DType::F64))
            ),
            Err(ActionValueCriticError::DTypeMismatch { .. })
        ));
        let wrong_output = ScalarStateActionCritic::new(burn::nn::Relu);
        assert!(matches!(
            <_ as ActionValueCriticNetwork<2, 2, 3>>::policy_values(
                &wrong_output,
                &observations,
                &Tensor::ones([2, 2, 1], &device)
            ),
            Err(ActionValueCriticError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn native_model_target_matches_before_and_after_online_initialization() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        use crate::models::MLP;

        let device = Device::flex().autodiff();
        let model = MLP::builder()
            .input_size(3)
            .output_size(1)
            .hidden_layer_sizes(vec![4])
            .options(&device)
            .build()
            .unwrap();
        let mut critic = ActionValueCritic::builder()
            .online_network(ScalarStateActionCritic::new(model))
            .optimizer(SgdConfig::new().init())
            .learning_rate(0.01)
            .build()
            .unwrap();
        let observations = Tensor::from_data([[1.0f32, 2.0]], &device);
        let actions = Tensor::from_data([[3.0f32]], &device);
        let initial_target = numbers(
            critic
                .target_replay_values::<2, 2, 3>(&observations, &actions)
                .unwrap(),
        );
        let online = critic
            .online_replay_values::<2, 2, 3>(&observations, &actions)
            .unwrap();
        assert_eq!(numbers(online.clone()), initial_target);
        critic.backward_step(&online.sum()).unwrap();
        assert_eq!(
            numbers(
                critic
                    .target_replay_values::<2, 2, 3>(&observations, &actions)
                    .unwrap()
            ),
            initial_target
        );
        critic.hard_update().unwrap();
        assert_eq!(
            numbers(
                critic
                    .target_replay_values::<2, 2, 3>(&observations, &actions)
                    .unwrap()
            ),
            numbers(
                critic
                    .online_replay_values::<2, 2, 3>(&observations, &actions)
                    .unwrap()
            )
        );
    }

    #[test]
    fn aggregation_preserves_all_modes_even_medians_and_gradients() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        for (mode, expected, gradient) in [
            (ActionValueCriticAggregationMode::Mean, 2.0, [1.0 / 3.0; 3]),
            (ActionValueCriticAggregationMode::Min, 1.0, [1.0, 0.0, 0.0]),
            (ActionValueCriticAggregationMode::Max, 3.0, [0.0, 0.0, 1.0]),
            (
                ActionValueCriticAggregationMode::Median,
                2.0,
                [0.0, 1.0, 0.0],
            ),
        ] {
            let values = [1.0f64, 2.0, 3.0]
                .map(|value| Tensor::from_data([[value]], (&device, DType::F64)).require_grad());
            let result = aggregate_critic_values::<2, 3>(&values, mode).unwrap();
            assert_eq!(numbers(result.clone()), [expected]);
            assert_eq!(result.dtype(), DType::F64);
            let gradients = result.sum().backward();
            for (value, expected) in values.iter().zip(gradient) {
                let actual = value.grad(&gradients).map(numbers).unwrap_or(vec![0.0])[0];
                assert!(
                    (actual - expected).abs() < 1e-12,
                    "{mode:?}: {actual} != {expected}"
                );
            }
        }
        let values = [1.0f64, 2.0, 10.0, 4.0]
            .map(|value| Tensor::from_data([[value]], (&device, DType::F64)).require_grad());
        let median =
            aggregate_critic_values::<2, 3>(&values, ActionValueCriticAggregationMode::Median)
                .unwrap();
        assert_eq!(numbers(median.clone()), [3.0]);
        let gradients = median.sum().backward();
        assert_eq!(numbers(values[1].grad(&gradients).unwrap()), [0.5]);
        assert_eq!(numbers(values[3].grad(&gradients).unwrap()), [0.5]);
        for mode in [
            ActionValueCriticAggregationMode::Min,
            ActionValueCriticAggregationMode::Max,
        ] {
            let tied =
                [1.0f32; 3].map(|value| Tensor::from_data([[value]], &device).require_grad());
            let gradients = aggregate_critic_values::<2, 3>(&tied, mode)
                .unwrap()
                .sum()
                .backward();
            for (value, expected) in tied.iter().zip([0.25, 0.25, 0.5]) {
                assert_eq!(numbers(value.grad(&gradients).unwrap()), [expected]);
            }
        }
    }

    #[test]
    fn aggregation_rejects_empty_mismatched_shapes_and_dtypes() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex();
        assert!(matches!(
            aggregate_critic_values::<2, 3>(&[], ActionValueCriticAggregationMode::Min),
            Err(ActionValueCriticError::NoCriticValues)
        ));
        for (shape, dtype) in [([2, 1], DType::F32), ([1, 1], DType::F64)] {
            assert!(matches!(
                aggregate_critic_values::<2, 3>(
                    &[
                        Tensor::ones([1, 1], &device),
                        Tensor::ones(shape, (&device, dtype))
                    ],
                    ActionValueCriticAggregationMode::Mean
                ),
                Err(ActionValueCriticError::ShapeMismatch { .. }
                    | ActionValueCriticError::DTypeMismatch { .. })
            ));
        }
    }
}

//! Q-value critics and ensemble aggregation for Soft Actor-Critic.

use bon::bon;
use burn::{
    module::{Module, ModuleMapper, ModuleVisitor},
    optim::ModuleOptimizer,
    tensor::{DType, Device, Float, Int, Tensor, TensorReadError, kind::Basic},
};

use crate::{agents::action_value_critic, models::Forward};

/// A Q network that can evaluate replay actions and policy candidates.
/// Ranks O and A include batch and event axes. Rank C adds one candidate axis to A.
/// Associated kinds specify native inputs; implementations define dtype and device requirements.
pub trait SACCriticNetwork<const O: usize = 2, const A: usize = 2, const C: usize = 3>:
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

/// Failures from SAC critic construction, aggregation, and optimization.
#[derive(Debug, thiserror::Error)]
pub enum SACCriticError<E = std::convert::Infallible> {
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

impl<E> From<action_value_critic::ActionValueCriticError<E>> for SACCriticError<E> {
    fn from(error: action_value_critic::ActionValueCriticError<E>) -> Self {
        match error {
            action_value_critic::ActionValueCriticError::ModelError(error) => {
                Self::ModelError(error)
            }
            action_value_critic::ActionValueCriticError::TensorError(error) => {
                Self::TensorError(error)
            }
            action_value_critic::ActionValueCriticError::InvalidPolyakCoefficient { tau } => {
                Self::InvalidPolyakCoefficient { tau }
            }
            action_value_critic::ActionValueCriticError::InvalidLearningRate => {
                Self::InvalidLearningRate
            }
            action_value_critic::ActionValueCriticError::EmptyInput => Self::EmptyInput,
            action_value_critic::ActionValueCriticError::BatchMismatch {
                observations,
                actions,
            } => Self::BatchMismatch {
                observations,
                actions,
            },
            action_value_critic::ActionValueCriticError::ShapeMismatch { expected, actual } => {
                Self::ShapeMismatch { expected, actual }
            }
            action_value_critic::ActionValueCriticError::DTypeMismatch { expected, actual } => {
                Self::DTypeMismatch { expected, actual }
            }
            action_value_critic::ActionValueCriticError::DeviceMismatch => Self::DeviceMismatch,
            action_value_critic::ActionValueCriticError::InvalidActionIndex { action_count } => {
                Self::InvalidActionIndex { action_count }
            }
            action_value_critic::ActionValueCriticError::NoCriticValues => Self::NoCriticValues,
        }
    }
}

/// Adapts a Float model that maps concatenated observations and actions to `[batch_size, 1]`.
/// Flattens event axes. Observations and actions must share Float dtype and device.
///
/// ```compile_fail
/// use burn::{nn::Relu, tensor::{Device, Tensor}};
/// use modurl::agents::sac::{ScalarStateActionCritic, SACCriticNetwork};
/// let critic = ScalarStateActionCritic::new(Relu);
/// let observations = Tensor::<2>::ones([1, 2], &Device::flex());
/// let actions = Tensor::<2>::ones([1, 1], &Device::flex());
/// // Rank-2 replay actions require rank-3 candidates.
/// let _ = <_ as SACCriticNetwork<2, 2, 2>>::policy_values(&critic, &observations, &actions);
/// ```
#[derive(Clone, Debug)]
pub struct ScalarStateActionCritic<M: Module> {
    network: action_value_critic::ScalarStateActionCritic<M>,
}

impl<M: Module> Module for ScalarStateActionCritic<M> {
    fn collect_devices(&self, devices: Vec<Device>) -> Vec<Device> {
        self.network.collect_devices(devices)
    }

    fn fork(self, device: &Device) -> Self {
        Self {
            network: self.network.fork(device),
        }
    }

    fn to_device(self, device: &Device) -> Self {
        Self {
            network: self.network.to_device(device),
        }
    }

    fn train(self) -> Self {
        Self {
            network: self.network.train(),
        }
    }

    fn valid(&self) -> Self {
        Self {
            network: self.network.valid(),
        }
    }

    fn visit<V: ModuleVisitor>(&self, visitor: &mut V) {
        self.network.visit(visitor);
    }

    fn map<T: ModuleMapper>(self, mapper: &mut T) -> Self {
        Self {
            network: self.network.map(mapper),
        }
    }

    fn materialize(self) -> Self {
        Self {
            network: self.network.materialize(),
        }
    }
}

impl<M: Module> ScalarStateActionCritic<M> {
    /// Owns the model used to evaluate scalar Q values.
    pub fn new(module: M) -> Self {
        Self {
            network: action_value_critic::ScalarStateActionCritic::new(module),
        }
    }
}

impl<M, const O: usize, const A: usize, const C: usize> SACCriticNetwork<O, A, C>
    for ScalarStateActionCritic<M>
where
    M: Module + Forward<2, 2, InputKind = Float, OutputKind = Float>,
{
    type ObservationKind = Float;
    type ActionKind = Float;
    type Error = SACCriticError<M::Error>;

    /// Adds a size-one candidate axis to actions `[batch_size, ...action_shape]` and removes it from `[batch_size, 1, 1]` values.
    /// Observations `[batch_size, ...observation_shape]` retain their Float dtype and device. Returns `[batch_size, 1]` with critic gradients.
    fn replay_values(
        &self,
        observations: &Tensor<O>,
        actions: &Tensor<A>,
    ) -> Result<Tensor<2>, Self::Error> {
        <action_value_critic::ScalarStateActionCritic<M> as action_value_critic::ActionValueCriticNetwork<O, A, C>>::replay_values(
            &self.network,
            observations,
            actions,
        )
        .map_err(Into::into)
    }

    /// Broadcasts observations over candidates and flattens event axes into `[batch_size * candidate_count, observation_size + action_size]`.
    /// Returns `[batch_size, candidate_count, 1]`. Inputs must share Float dtype and device; preserves gradients.
    fn policy_values(
        &self,
        observations: &Tensor<O>,
        candidate_actions: &Tensor<C>,
    ) -> Result<Tensor<3>, Self::Error> {
        <action_value_critic::ScalarStateActionCritic<M> as action_value_critic::ActionValueCriticNetwork<O, A, C>>::policy_values(
            &self.network,
            observations,
            candidate_actions,
        )
        .map_err(Into::into)
    }

    /// Evaluates observations and candidates with the `policy_values` layouts, returning `[batch_size, candidate_count, 1]`.
    /// Uses a snapshot with parameter gradients disabled. Preserves differentiable observation and action gradients without changing the online model.
    fn actor_values(
        &self,
        observations: &Tensor<O>,
        candidate_actions: &Tensor<C>,
    ) -> Result<Tensor<3>, Self::Error> {
        <action_value_critic::ScalarStateActionCritic<M> as action_value_critic::ActionValueCriticNetwork<O, A, C>>::actor_values(
            &self.network,
            observations,
            candidate_actions,
        )
        .map_err(Into::into)
    }
}

/// Adapts a model returning Float `[batch_size, action_count]` to scalar Int replay and candidate actions.
/// Observations preserve the model's native input kind and dtype. The model performs any required preprocessing.
#[derive(Clone, Debug)]
pub struct DiscreteVectorHeadCritic<M: Module> {
    network: action_value_critic::DiscreteVectorHeadCritic<M>,
}

impl<M: Module> Module for DiscreteVectorHeadCritic<M> {
    fn collect_devices(&self, devices: Vec<Device>) -> Vec<Device> {
        self.network.collect_devices(devices)
    }

    fn fork(self, device: &Device) -> Self {
        Self {
            network: self.network.fork(device),
        }
    }

    fn to_device(self, device: &Device) -> Self {
        Self {
            network: self.network.to_device(device),
        }
    }

    fn train(self) -> Self {
        Self {
            network: self.network.train(),
        }
    }

    fn valid(&self) -> Self {
        Self {
            network: self.network.valid(),
        }
    }

    fn visit<V: ModuleVisitor>(&self, visitor: &mut V) {
        self.network.visit(visitor);
    }

    fn map<T: ModuleMapper>(self, mapper: &mut T) -> Self {
        Self {
            network: self.network.map(mapper),
        }
    }

    fn materialize(self) -> Self {
        Self {
            network: self.network.materialize(),
        }
    }
}

impl<M: Module> DiscreteVectorHeadCritic<M> {
    /// Owns the model used to evaluate one value per discrete action.
    pub fn new(module: M) -> Self {
        Self {
            network: action_value_critic::DiscreteVectorHeadCritic::new(module),
        }
    }
}

impl<M, const O: usize> SACCriticNetwork<O, 2, 3> for DiscreteVectorHeadCritic<M>
where
    M: Module + Forward<O, 2, OutputKind = Float>,
{
    type ObservationKind = M::InputKind;
    type ActionKind = Int;
    type Error = SACCriticError<M::Error>;

    /// Gathers Int actions `[batch_size, 1]` from Float model values `[batch_size, action_count]`, returning `[batch_size, 1]`.
    /// Observations `[batch_size, ...observation_shape]` retain their native kind and dtype. Inputs must share a device.
    fn replay_values(
        &self,
        observations: &Tensor<O, M::InputKind>,
        actions: &Tensor<2, Int>,
    ) -> Result<Tensor<2>, Self::Error> {
        <action_value_critic::DiscreteVectorHeadCritic<M> as action_value_critic::ActionValueCriticNetwork<O, 2, 3>>::replay_values(
            &self.network,
            observations,
            actions,
        )
        .map_err(Into::into)
    }

    /// Gathers Int candidates `[batch_size, candidate_count, 1]` from Float model values `[batch_size, action_count]`.
    /// Returns `[batch_size, candidate_count, 1]`. Preserves native observations and model gradients; reads one bounds-check Boolean to the host.
    fn policy_values(
        &self,
        observations: &Tensor<O, M::InputKind>,
        candidate_actions: &Tensor<3, Int>,
    ) -> Result<Tensor<3>, Self::Error> {
        <action_value_critic::DiscreteVectorHeadCritic<M> as action_value_critic::ActionValueCriticNetwork<O, 2, 3>>::policy_values(
            &self.network,
            observations,
            candidate_actions,
        )
        .map_err(Into::into)
    }

    /// Returns detached Float `[batch_size, candidate_count, 1]` for observations and Int candidates with the `policy_values` layouts.
    /// Discrete indices have no action gradients. Policy expectation weights retain their separate gradient paths.
    fn actor_values(
        &self,
        observations: &Tensor<O, M::InputKind>,
        candidate_actions: &Tensor<3, Int>,
    ) -> Result<Tensor<3>, Self::Error> {
        <action_value_critic::DiscreteVectorHeadCritic<M> as action_value_critic::ActionValueCriticNetwork<O, 2, 3>>::actor_values(
            &self.network,
            observations,
            candidate_actions,
        )
        .map_err(Into::into)
    }
}

/// Owns one independently optimized online critic and a detached target initialized by cloning.
/// Online and target models share parameter IDs and initial values; optimizer updates leave the target snapshot unchanged.
///
/// ```
/// use burn::{optim::AdamWConfig, tensor::Device};
/// use modurl::prelude::{MLP, SACCritic, ScalarStateActionCritic};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let device = Device::flex().autodiff();
/// let model = MLP::builder()
///     .input_size(5)
///     .output_size(1)
///     .hidden_layer_sizes(vec![16])
///     .options(&device)
///     .build()?;
/// let mut critic = SACCritic::builder()
///     .online_network(ScalarStateActionCritic::new(model))
///     .optimizer(AdamWConfig::new().init())
///     .learning_rate(3e-4)
///     .build()?;
/// critic.polyak_update(0.005)?;
/// critic.hard_update()?;
/// # Ok(())
/// # }
/// ```
pub struct SACCritic<N: Module> {
    critic: action_value_critic::ActionValueCritic<N>,
}

#[bon]
impl<N: Module> SACCritic<N> {
    /// Returns the owned online model for parameter inspection and checkpoint snapshots.
    pub fn online_network(&self) -> &N {
        self.critic.online_network()
    }

    /// Owns the model and optimizer, enables online training, and clones the target snapshot.
    /// Rejects a nonpositive or non-finite learning rate.
    #[builder]
    pub fn new(
        online_network: N,
        #[builder(into)] optimizer: ModuleOptimizer,
        learning_rate: f64,
    ) -> Result<Self, SACCriticError> {
        Ok(Self {
            critic: action_value_critic::ActionValueCritic::builder()
                .online_network(online_network)
                .optimizer(optimizer)
                .learning_rate(learning_rate)
                .build()?,
        })
    }

    /// Backpropagates Float loss `[1]` and updates online parameters, preserving their shapes, dtypes, devices, and IDs.
    /// Loss must come from this online model. Leaves the target snapshot unchanged; rejects a nonscalar loss layout.
    pub fn backward_step(&mut self, loss: &Tensor<1>) -> Result<(), SACCriticError> {
        self.critic.backward_step(loss).map_err(Into::into)
    }

    /// Copies online parameters into a detached target snapshot, preserving parameter IDs and layouts.
    pub fn hard_update(&mut self) -> Result<(), SACCriticError> {
        self.critic.hard_update().map_err(Into::into)
    }

    /// Updates rank-D target parameters to `tau * online + (1 - tau) * target`.
    /// Preserves parameter IDs, shapes, dtypes, and devices. Rejects a non-finite tau or a value outside 0..=1.
    pub fn polyak_update(&mut self, tau: f64) -> Result<(), SACCriticError> {
        self.critic.polyak_update(tau).map_err(Into::into)
    }

    /// Evaluates online observations `[batch_size, ...observation_shape]` and replay actions `[batch_size, ...action_shape]`.
    /// Returns Float `[batch_size, 1]`, retaining critic gradients and the network's input dtype and device requirements.
    pub fn online_replay_values<const O: usize, const A: usize, const C: usize>(
        &self,
        observations: &Tensor<O, N::ObservationKind>,
        actions: &Tensor<A, N::ActionKind>,
    ) -> Result<Tensor<2>, N::Error>
    where
        N: SACCriticNetwork<O, A, C>,
    {
        self.critic
            .online_network()
            .replay_values(observations, actions)
    }

    /// Evaluates observations and candidates `[batch_size, candidate_count, ...action_shape]` with the network's native input requirements.
    /// Returns Float `[batch_size, candidate_count, 1]`, preserving action gradients and excluding critic-parameter gradients.
    pub fn online_actor_values<const O: usize, const A: usize, const C: usize>(
        &self,
        observations: &Tensor<O, N::ObservationKind>,
        actions: &Tensor<C, N::ActionKind>,
    ) -> Result<Tensor<3>, N::Error>
    where
        N: SACCriticNetwork<O, A, C>,
    {
        self.critic
            .online_network()
            .actor_values(observations, actions)
    }

    /// Evaluates target observations and candidates `[batch_size, candidate_count, ...action_shape]` with the network's native input requirements.
    /// Returns detached Float `[batch_size, candidate_count, 1]`; observations retain their documented batch and event axes.
    pub fn target_policy_values<const O: usize, const A: usize, const C: usize>(
        &self,
        observations: &Tensor<O, N::ObservationKind>,
        actions: &Tensor<C, N::ActionKind>,
    ) -> Result<Tensor<3>, N::Error>
    where
        N: SACCriticNetwork<O, A, C>,
    {
        self.critic
            .target_network()
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
        N: SACCriticNetwork<O, A, C>,
    {
        self.critic
            .target_network()
            .replay_values(observations, actions)
            .map(Tensor::detach)
    }
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

impl From<SACCriticAggregationMode> for action_value_critic::ActionValueCriticAggregationMode {
    fn from(mode: SACCriticAggregationMode) -> Self {
        match mode {
            SACCriticAggregationMode::Mean => Self::Mean,
            SACCriticAggregationMode::Median => Self::Median,
            SACCriticAggregationMode::Min => Self::Min,
            SACCriticAggregationMode::Max => Self::Max,
        }
    }
}

/// Aggregates Float rank-D tensors with identical shapes, dtypes, and devices, returning the same layout.
/// Rank S equals D + 1. Adds and removes a critic axis; typical layouts are `[batch_size, 1]` and `[batch_size, candidate_count, 1]`.
/// Preserves gradients. Min and Max split each pair's gradient equally at ties; Median follows native sort ordering at ties.
///
/// ```compile_fail
/// use burn::tensor::{Device, Tensor};
/// use modurl::agents::sac::{aggregate_critic_values, SACCriticAggregationMode};
/// let values = [Tensor::<2>::ones([1, 1], &Device::flex())];
/// // Stacking rank-2 values requires rank 3.
/// let _ = aggregate_critic_values::<2, 2>(&values, SACCriticAggregationMode::Mean);
/// ```
pub fn aggregate_critic_values<const D: usize, const S: usize>(
    values: &[Tensor<D>],
    mode: SACCriticAggregationMode,
) -> Result<Tensor<D>, SACCriticError> {
    action_value_critic::aggregate_critic_values::<D, S>(values, mode.into()).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{module::Param, optim::SgdConfig, tensor::Device};
    use std::convert::Infallible;

    #[derive(Debug, thiserror::Error)]
    #[error("custom critic failed")]
    struct CustomError;

    #[derive(Module, Debug)]
    struct CustomNetwork {
        weight: Param<Tensor<2>>,
        #[module(skip)]
        fail: bool,
    }

    impl SACCriticNetwork for CustomNetwork {
        type ObservationKind = Int;
        type ActionKind = Float;
        type Error = CustomError;

        /// Combines Int observations [batch_size, 1] and Float actions [batch_size, 1] into Float Q values [batch_size, 1].
        /// Casts observations to the parameter dtype; keeps online parameter gradients.
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

        /// Broadcasts Int observations [batch_size, 1] over Float candidates [batch_size, candidate_count, 1].
        /// Returns Float Q values [batch_size, candidate_count, 1], preserving model gradients.
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

        /// Returns Float [batch_size, candidate_count, 1] from Int observations and Float candidates with the policy layouts.
        /// Excludes parameter gradients while preserving candidate-action gradients.
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

    #[derive(Module, Debug)]
    struct SumModel {
        weight: Param<Tensor<2>>,
    }

    impl Forward<2, 2> for SumModel {
        type InputKind = Float;
        type OutputKind = Float;
        type Error = Infallible;

        /// Maps Float inputs [batch_size, features] through parameters [features, 1] to Q values [batch_size, 1].
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

        /// Converts Int observations [batch_size, action_count] to Float action values with unchanged axes.
        fn forward(&self, input: Tensor<2, Int>) -> Result<Tensor<2>, Self::Error> {
            assert_eq!(input.dtype(), DType::I64);
            Ok(input.float())
        }
    }

    /// Reads a rank-D Float tensor with arbitrary shape into flattened F64 host values; leaves its axes and device unchanged.
    fn numbers<const D: usize>(values: Tensor<D>) -> Vec<f64> {
        values
            .into_data()
            .convert::<f64>()
            .try_to_vec::<f64>()
            .unwrap()
    }

    #[test]
    fn custom_sac_network_contract_remains_supported() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        let model = CustomNetwork {
            weight: Param::from_tensor(Tensor::ones([2, 1], (&device, DType::F64))),
            fail: false,
        };
        let mut critic = SACCritic::builder()
            .online_network(model)
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
        let values = critic
            .online_actor_values::<2, 2, 3>(&observations, &candidates)
            .unwrap();
        let gradients = values.sum().backward();
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
        assert_eq!(
            numbers(
                critic
                    .target_replay_values::<2, 2, 3>(&observations, &actions)
                    .unwrap()
            ),
            [5.0]
        );
        assert_eq!(observations.dtype(), DType::I64);
        let failing = SACCritic::builder()
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
        let error = SACCriticError::from(action_value_critic::ActionValueCriticError::ModelError(
            CustomError,
        ));
        assert!(matches!(error, SACCriticError::ModelError(CustomError)));
        assert_eq!(
            std::error::Error::source(&error).unwrap().to_string(),
            "custom critic failed"
        );
    }

    #[test]
    fn scalar_sac_adapter_keeps_error_types_and_gradient_boundaries() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        let model = SumModel {
            weight: Param::from_tensor(Tensor::ones([3, 1], (&device, DType::F64))),
        };
        let parameters = model.weight.val();
        let network = ScalarStateActionCritic::new(model);
        let observations = Tensor::from_data([[1.0f64, 2.0]], (&device, DType::F64));
        let actions = Tensor::from_data([[[3.0f64], [4.0]]], (&device, DType::F64)).require_grad();
        let values =
            <_ as SACCriticNetwork<2, 2, 3>>::policy_values(&network, &observations, &actions)
                .unwrap();
        assert_eq!(numbers(values.clone()), [6.0, 7.0]);
        assert!(parameters.grad(&values.sum().backward()).is_some());
        let actor =
            <_ as SACCriticNetwork<2, 2, 3>>::actor_values(&network, &observations, &actions)
                .unwrap();
        let gradients = actor.sum().backward();
        assert_eq!(numbers(actions.grad(&gradients).unwrap()), [1.0, 1.0]);
        assert!(parameters.grad(&gradients).is_none());
        assert!(matches!(
            <_ as SACCriticNetwork<2, 2, 3>>::policy_values(
                &network,
                &observations,
                &Tensor::ones([1, 2, 1], &device)
            ),
            Err(SACCriticError::DTypeMismatch {
                expected: DType::F64,
                actual: DType::F32
            })
        ));
        assert!(matches!(
            <_ as SACCriticNetwork<2, 2, 3>>::policy_values(
                &network,
                &observations,
                &Tensor::ones([2, 2, 1], (&device, DType::F64))
            ),
            Err(SACCriticError::BatchMismatch {
                observations: 1,
                actions: 2
            })
        ));
    }

    #[test]
    fn discrete_sac_adapter_preserves_native_inputs_and_index_errors() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex();
        let network = DiscreteVectorHeadCritic::new(IntegerModel);
        let observations = Tensor::from_data([[1i64, 2, 3], [4, 5, 6]], (&device, DType::I64));
        let actions = Tensor::from_data([[2i64], [0]], &device);
        assert_eq!(
            numbers(network.replay_values(&observations, &actions).unwrap()),
            [3.0, 4.0]
        );
        let candidates = Tensor::from_data([[[0i64], [2]], [[1], [0]]], &device);
        assert_eq!(
            numbers(network.policy_values(&observations, &candidates).unwrap()),
            [1.0, 3.0, 5.0, 4.0]
        );
        for index in [-1i64, 3] {
            assert!(matches!(
                network.replay_values(&observations, &Tensor::from_data([[index], [0]], &device)),
                Err(SACCriticError::InvalidActionIndex { action_count: 3 })
            ));
        }
        assert!(matches!(
            network.replay_values(&observations, &Tensor::zeros([2, 2], &device)),
            Err(SACCriticError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn sac_wrapper_keeps_target_updates_and_optimizer_validation() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        for dtype in [DType::F32, DType::F64] {
            let device = Device::flex().autodiff();
            let mut critic = SACCritic::builder()
                .online_network(ScalarStateActionCritic::new(SumModel {
                    weight: Param::from_tensor(Tensor::ones([3, 1], (&device, dtype))),
                }))
                .optimizer(SgdConfig::new().init())
                .learning_rate(0.5)
                .build()
                .unwrap();
            let observations = Tensor::ones([1, 2], (&device, dtype));
            let actions = Tensor::ones([1, 1], (&device, dtype));
            let loss = critic
                .online_replay_values::<2, 2, 3>(&observations, &actions)
                .unwrap()
                .sum();
            critic.backward_step(&loss).unwrap();
            assert_eq!(
                numbers(
                    critic
                        .online_replay_values::<2, 2, 3>(&observations, &actions)
                        .unwrap()
                ),
                [1.5]
            );
            assert_eq!(
                numbers(
                    critic
                        .target_replay_values::<2, 2, 3>(&observations, &actions)
                        .unwrap()
                ),
                [3.0]
            );
            critic.polyak_update(0.25).unwrap();
            let target = critic
                .target_replay_values::<2, 2, 3>(&observations, &actions)
                .unwrap();
            assert_eq!(target.dtype(), dtype);
            assert!(!target.is_require_grad());
            assert_eq!(numbers(target), [2.625]);
            critic.polyak_update(0.0).unwrap();
            assert_eq!(
                numbers(
                    critic
                        .target_replay_values::<2, 2, 3>(&observations, &actions)
                        .unwrap()
                ),
                [2.625]
            );
            critic.polyak_update(1.0).unwrap();
            assert_eq!(
                numbers(
                    critic
                        .target_replay_values::<2, 2, 3>(&observations, &actions)
                        .unwrap()
                ),
                [1.5]
            );
            critic.hard_update().unwrap();
            assert!(matches!(
                critic.backward_step(&Tensor::ones([2], &device)),
                Err(SACCriticError::ShapeMismatch { .. })
            ));
            for tau in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
                assert!(matches!(
                    critic.polyak_update(tau),
                    Err(SACCriticError::InvalidPolyakCoefficient { .. })
                ));
            }
        }
        for rate in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                SACCritic::builder()
                    .online_network(burn::nn::Relu)
                    .optimizer(SgdConfig::new().init())
                    .learning_rate(rate)
                    .build(),
                Err(SACCriticError::InvalidLearningRate)
            ));
        }
    }

    #[test]
    fn sac_aggregation_preserves_modes_gradients_and_error_variants() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        for (mode, expected) in [
            (SACCriticAggregationMode::Mean, 4.0),
            (SACCriticAggregationMode::Median, 3.0),
            (SACCriticAggregationMode::Min, 1.0),
            (SACCriticAggregationMode::Max, 9.0),
        ] {
            let values = [1.0f64, 2.0, 9.0, 4.0]
                .map(|value| Tensor::from_data([[value]], (&device, DType::F64)));
            assert_eq!(
                numbers(aggregate_critic_values::<2, 3>(&values, mode).unwrap()),
                [expected]
            );
        }
        let values = [1.0f64; 2]
            .map(|value| Tensor::from_data([[value]], (&device, DType::F64)).require_grad());
        let gradients = aggregate_critic_values::<2, 3>(&values, SACCriticAggregationMode::Min)
            .unwrap()
            .sum()
            .backward();
        for value in &values {
            assert_eq!(numbers(value.grad(&gradients).unwrap()), [0.5]);
        }
        assert!(matches!(
            aggregate_critic_values::<2, 3>(&[], SACCriticAggregationMode::Mean),
            Err(SACCriticError::NoCriticValues)
        ));
        assert!(matches!(
            aggregate_critic_values::<2, 3>(
                &[Tensor::ones([1, 1], &device), Tensor::ones([2, 1], &device)],
                SACCriticAggregationMode::Min
            ),
            Err(SACCriticError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            aggregate_critic_values::<2, 3>(
                &[
                    Tensor::ones([1, 1], &device),
                    Tensor::ones([1, 1], (&device, DType::F64))
                ],
                SACCriticAggregationMode::Max
            ),
            Err(SACCriticError::DTypeMismatch { .. })
        ));
    }
}

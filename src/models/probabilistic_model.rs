use std::{fmt, num::NonZeroUsize};

use burn::{
    module::{Module, ModuleMapper, ModuleVisitor},
    tensor::{Device, Tensor, kind::Basic},
};

use crate::{
    distributions::{DifferentiableExpectation, Distribution, ExpectationTerms},
    models::Forward,
};

/// Samples and evaluates policy actions from batched observations.
/// Rank `O` counts observation axes, including the batch axis. Rank `A` counts sampled action axes.
/// Implementations define the observation kind and the dtype and device requirements of their model.
/// Actions use the `Float` kind; an action map converts policy actions to environment actions when needed.
pub trait ProbabilisticPolicy<const O: usize = 2, const A: usize = 2> {
    type Error;
    type ObservationKind: Basic;

    /// Samples `[batch_size, ...action_shape]` from observations `[batch_size, ...observation_shape]`.
    /// Keeps the batch axis. The model defines observation conversions; the distribution defines the action dtype and device.
    fn sample(
        &self,
        observations: Tensor<O, Self::ObservationKind>,
    ) -> Result<Tensor<A>, Self::Error>;

    /// Returns the distribution's modal actions `[batch_size, ...action_shape]` for observations `[batch_size, ...observation_shape]`.
    /// Keeps the batch axis. The model defines observation conversions; the distribution defines the action dtype and device.
    fn mode(
        &self,
        observations: Tensor<O, Self::ObservationKind>,
    ) -> Result<Tensor<A>, Self::Error>;

    /// Evaluates actions `[batch_size, ...action_shape]` for observations `[batch_size, ...observation_shape]`.
    /// Both returned float tensors have shape `[batch_size, 1]`, with one log probability and entropy statistic per item.
    /// Action batch size, dtype, and device must match the model's distribution parameters. The calculation retains gradient paths.
    fn log_prob_and_entropy(
        &self,
        observations: Tensor<O, Self::ObservationKind>,
        actions: Tensor<A>,
    ) -> Result<(Tensor<2>, Tensor<2>), Self::Error>;
}

/// Supplies candidate actions for an exact or sampled policy expectation, such as the objective used by SAC.
/// Rank `C` includes batch and candidate axes. Candidate kind is independent of observation and sampled action kinds.
pub trait ExpectationPolicy<const O: usize = 2, const A: usize = 2, const C: usize = 3>:
    ProbabilisticPolicy<O, A>
{
    type CandidateKind: Basic;

    /// Builds rank-`C` candidates `[batch_size, candidate_count, ...action_shape]` from rank-`O` batched observations.
    /// Log probabilities and weights have shape `[batch_size, candidate_count, 1]`.
    /// The distribution determines candidate count and kind. The calculation preserves available gradient paths.
    fn expectation(
        &self,
        observations: Tensor<O, Self::ObservationKind>,
        samples: NonZeroUsize,
    ) -> Result<ExpectationTerms<C, Self::CandidateKind>, Self::Error>;

    /// Returns scalar target entropy from observations `[batch_size, ...observation_shape]`.
    /// Detaches model outputs before computing the target. The target has no batch or action axes.
    fn default_target_entropy(
        &self,
        observations: Tensor<O, Self::ObservationKind>,
    ) -> Result<f64, Self::Error>;
}

/// Groups component and native tensor types without supplying execution behavior.
/// Observation and parameter types describe `[batch_size, ...observation_shape]` and `[batch_size, ...parameter_shape]`.
/// Action types describe `[batch_size, ...action_shape]`. Policy implementations constrain these types to native Burn tensors.
pub trait PolicyTypes: fmt::Debug {
    type Model: fmt::Debug;
    type Distribution: fmt::Debug;
    type Observation;
    type Parameters;
    type Action;
}

impl<M: fmt::Debug, D: fmt::Debug, const O: usize, const P: usize, const A: usize, K: Basic>
    PolicyTypes for (M, D, (Tensor<O, K>, Tensor<P>, Tensor<A>))
{
    type Model = M;
    type Distribution = D;
    type Observation = Tensor<O, K>;
    type Parameters = Tensor<P>;
    type Action = Tensor<A>;
}

/// Owns a native Burn model and a fixed distribution.
/// `T` groups their types and the native observation, parameter, and action tensor types.
/// Stores both components directly. The tensor signature describes types without storing tensor values.
/// Every model implements both Burn's `Module` and the tensor execution contract `Forward`.
/// Burn parameter traversal, optimizer updates, and records visit only the owned model.
/// Callers and components must follow the documented tensor contracts; the policy does not validate their tensors.
#[derive(Debug)]
pub struct ProbabilisticPolicyModel<T: PolicyTypes> {
    module: T::Model,
    distribution: T::Distribution,
}

impl<T: PolicyTypes> Clone for ProbabilisticPolicyModel<T>
where
    T::Model: Clone,
    T::Distribution: Clone,
{
    fn clone(&self) -> Self {
        Self {
            module: self.module.clone(),
            distribution: self.distribution.clone(),
        }
    }
}

impl<T: PolicyTypes> ProbabilisticPolicyModel<T> {
    /// Returns the native model whose parameters are executed, optimized, and saved.
    pub fn module(&self) -> &T::Model {
        &self.module
    }

    /// Returns the fixed distribution configuration used to interpret model outputs.
    pub fn distribution(&self) -> &T::Distribution {
        &self.distribution
    }
}

impl<M, D, const O: usize, const P: usize, const A: usize, K: Basic>
    ProbabilisticPolicyModel<(M, D, (Tensor<O, K>, Tensor<P>, Tensor<A>))>
where
    M: Module + Forward<O, P, K>,
    D: Distribution<P, A> + fmt::Debug,
{
    /// Creates a policy with a default distribution and an owned native model.
    /// Maps rank-`O` observations to rank-`P` float parameters and samples rank-`A` actions.
    /// Every rank includes a batch axis. Construction does not initialize or detach model parameters.
    pub fn new(module: M) -> Self
    where
        D: Default,
    {
        Self::with_distribution(module, D::default())
    }

    /// Connects an owned native model to a distribution using `Forward<O, P, K>`.
    /// Maps `[batch_size, ...observation_shape]` to float `[batch_size, ...parameter_shape]`, preserving batch size.
    /// The model defines observation conversions and owns all parameters visited by Burn.
    pub fn with_distribution(module: M, distribution: D) -> Self {
        const {
            assert!(O >= 1, "policy observations require a batch axis");
            assert!(P >= 1, "policy parameters require a batch axis");
            assert!(A >= 1, "policy actions require a batch axis");
        }
        Self {
            module,
            distribution,
        }
    }
}

impl<T, D, E, const O: usize, const P: usize, const A: usize, K: Basic> ProbabilisticPolicyModel<T>
where
    T: PolicyTypes<
            Distribution = D,
            Observation = Tensor<O, K>,
            Parameters = Tensor<P>,
            Action = Tensor<A>,
        >,
    T::Model: Module + Forward<O, P, K, Error = E>,
    D: Distribution<P, A>,
{
    /// Maps rank-`O` observations `[batch_size, ...observation_shape]` to rank-`P` float parameters.
    /// The model must preserve batch size; dtype, device, and gradient behavior remain under its control.
    fn outputs(
        &self,
        observations: Tensor<O, K>,
    ) -> Result<Tensor<P>, ProbabilisticPolicyModelError<E, D::Error>> {
        const {
            assert!(O >= 1, "policy observations require a batch axis");
            assert!(P >= 1, "policy parameters require a batch axis");
            assert!(A >= 1, "policy actions require a batch axis");
        }
        self.module
            .forward(observations)
            .map_err(ProbabilisticPolicyModelError::ModuleError)
    }
}

impl<T: PolicyTypes> Module for ProbabilisticPolicyModel<T>
where
    T::Model: Module,
    T::Distribution: Clone + fmt::Debug + Send,
{
    fn collect_devices(&self, devices: Vec<Device>) -> Vec<Device> {
        self.module.collect_devices(devices)
    }

    fn fork(self, device: &Device) -> Self {
        Self {
            module: self.module.fork(device),
            ..self
        }
    }

    fn to_device(self, device: &Device) -> Self {
        Self {
            module: self.module.to_device(device),
            ..self
        }
    }

    fn train(self) -> Self {
        Self {
            module: self.module.train(),
            ..self
        }
    }

    fn valid(&self) -> Self {
        Self {
            module: self.module.valid(),
            distribution: self.distribution.clone(),
        }
    }

    fn visit<V: ModuleVisitor>(&self, visitor: &mut V) {
        self.module.visit(visitor);
    }

    fn map<M: ModuleMapper>(self, mapper: &mut M) -> Self {
        Self {
            module: self.module.map(mapper),
            ..self
        }
    }

    fn materialize(self) -> Self {
        Self {
            module: self.module.materialize(),
            ..self
        }
    }
}

/// Preserves the model and distribution error causes.
#[derive(Debug, thiserror::Error)]
pub enum ProbabilisticPolicyModelError<ME, DE> {
    #[error("policy model execution failed: {0}")]
    ModuleError(#[source] ME),
    #[error("policy distribution failed: {0}")]
    DistError(#[source] DE),
}

impl<T, D, E, const O: usize, const P: usize, const A: usize, K: Basic> ProbabilisticPolicy<O, A>
    for ProbabilisticPolicyModel<T>
where
    T: PolicyTypes<
            Distribution = D,
            Observation = Tensor<O, K>,
            Parameters = Tensor<P>,
            Action = Tensor<A>,
        >,
    T::Model: Module + Forward<O, P, K, Error = E>,
    D: Distribution<P, A>,
{
    type Error = ProbabilisticPolicyModelError<E, D::Error>;
    type ObservationKind = K;

    /// Samples rank-`A` actions `[batch_size, ...action_shape]` from rank-`O` observations `[batch_size, ...observation_shape]`.
    /// The model must preserve batch size. Returned actions share the parameter batch size, floating-point dtype, and device.
    /// The calculation retains gradient paths supplied by the model and distribution.
    fn sample(&self, observations: Tensor<O, K>) -> Result<Tensor<A>, Self::Error> {
        let outputs = self.outputs(observations)?;
        self.distribution
            .sample(outputs)
            .map_err(ProbabilisticPolicyModelError::DistError)
    }

    /// Returns rank-`A` transformed modes `[batch_size, ...action_shape]` from rank-`O` observations `[batch_size, ...observation_shape]`.
    /// The result preserves batch size and uses the distribution parameters' floating-point dtype and device.
    fn mode(&self, observations: Tensor<O, K>) -> Result<Tensor<A>, Self::Error> {
        let outputs = self.outputs(observations)?;
        self.distribution
            .mode(outputs)
            .map_err(ProbabilisticPolicyModelError::DistError)
    }

    /// Evaluates rank-`A` actions `[batch_size, ...action_shape]` under parameters produced from rank-`O` batched observations.
    /// Actions must share parameter batch size, dtype, and device. Returns two `[batch_size, 1]` float tensors with those properties.
    /// Preserves model and distribution gradient paths; the distribution determines whether entropy is analytic or estimated.
    fn log_prob_and_entropy(
        &self,
        observations: Tensor<O, K>,
        actions: Tensor<A>,
    ) -> Result<(Tensor<2>, Tensor<2>), Self::Error> {
        let outputs = self.outputs(observations)?;
        let evaluation = self
            .distribution
            .dist_eval(outputs, actions)
            .map_err(ProbabilisticPolicyModelError::DistError)?;
        Ok((evaluation.log_prob().clone(), evaluation.entropy().clone()))
    }
}

impl<T, D, E, const O: usize, const P: usize, const A: usize, const C: usize, K: Basic>
    ExpectationPolicy<O, A, C> for ProbabilisticPolicyModel<T>
where
    T: PolicyTypes<
            Distribution = D,
            Observation = Tensor<O, K>,
            Parameters = Tensor<P>,
            Action = Tensor<A>,
        >,
    T::Model: Module + Forward<O, P, K, Error = E>,
    D: DifferentiableExpectation<P, A, C>,
{
    type CandidateKind = D::CandidateKind;

    /// Produces rank-`C` candidates `[batch_size, candidate_count, ...action_shape]` from rank-`O` batched observations.
    /// Log probabilities and weights are `[batch_size, candidate_count, 1]` with the parameters' floating-point dtype and device.
    /// Candidate kind comes from the distribution. Candidate device and batch size match the parameters; gradient paths remain available.
    fn expectation(
        &self,
        observations: Tensor<O, K>,
        samples: NonZeroUsize,
    ) -> Result<ExpectationTerms<C, Self::CandidateKind>, Self::Error> {
        const {
            assert!(C >= 2, "policy candidates require batch and candidate axes");
        }
        let outputs = self.outputs(observations)?;
        self.distribution
            .expectation(outputs, samples)
            .map_err(ProbabilisticPolicyModelError::DistError)
    }

    /// Returns scalar target entropy for rank-`O` observations `[batch_size, ...observation_shape]`.
    /// Detaches rank-`P` distribution parameters before calculating the fixed target; no tensor axes remain in the result.
    fn default_target_entropy(&self, observations: Tensor<O, K>) -> Result<f64, Self::Error> {
        let outputs = self.outputs(observations)?.detach();
        self.distribution
            .default_target_entropy(&outputs)
            .map_err(ProbabilisticPolicyModelError::DistError)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        distributions::{
            CategoricalDistribution, GaussianDistribution, GaussianDistributionError,
            TanhTransform, TransformedDistribution,
        },
        models::{MLP, ModelError},
    };
    use burn::{
        nn::{Initializer, Linear},
        optim::{GradientsParams, SgdConfig},
        tensor::{DType, Float, Int},
    };
    use std::{
        convert::Infallible,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
    };

    /// Builds a zero-output MLP mapping `[batch_size, input_size]` to `[batch_size, output_size]` on the requested device and dtype.
    fn zero_model(input_size: usize, output_size: usize, device: &Device, dtype: DType) -> MLP {
        MLP::builder()
            .input_size(input_size)
            .output_size(output_size)
            .options((device, dtype))
            .hidden_layer_sizes(vec![])
            .output_initializer(Initializer::Zeros)
            .bias_initializer(Initializer::Zeros)
            .build()
            .unwrap()
    }

    #[derive(Module, Debug)]
    struct ObservationModel {
        layer: Linear,
    }

    impl Forward<4, 2> for ObservationModel {
        type Error = ModelError;

        /// Maps image observations `[batch_size, 1, 2, 2]` to float parameters `[batch_size, 2]`.
        /// Combines image axes into four features, preserving dtype, device, and gradient paths.
        fn forward(&self, observations: Tensor<4>) -> Result<Tensor<2>, ModelError> {
            if observations.dims()[1..] != [1, 2, 2] {
                return Err(ModelError::InputFeatures {
                    expected: 4,
                    actual: observations.dims()[1..].iter().product(),
                });
            }
            Forward::forward(&self.layer, observations.flatten(1, 3))
        }
    }

    impl Forward<2, 2, Int> for ObservationModel {
        type Error = ModelError;

        /// Casts integer observations `[batch_size, 2]` to the layer's float dtype and returns `[batch_size, 2]`.
        fn forward(&self, observations: Tensor<2, Int>) -> Result<Tensor<2>, ModelError> {
            let dtype = self.layer.weight.val().dtype();
            Forward::forward(&self.layer, observations.float().cast(dtype))
        }
    }

    impl Forward<2, 2, burn::tensor::Bool> for ObservationModel {
        type Error = ModelError;

        /// Casts boolean observations `[batch_size, 2]` to the layer's float dtype and returns `[batch_size, 2]`.
        fn forward(
            &self,
            observations: Tensor<2, burn::tensor::Bool>,
        ) -> Result<Tensor<2>, ModelError> {
            let dtype = self.layer.weight.val().dtype();
            Forward::forward(&self.layer, observations.float().cast(dtype))
        }
    }

    #[derive(Module, Debug)]
    struct OffsetModel {
        layer: Linear,
        offset: f64,
    }

    impl Forward<2, 2> for OffsetModel {
        type Error = ModelError;

        /// Maps `[batch_size, 2]` observations to `[batch_size, 2]` parameters and adds a fixed offset.
        /// Preserves dtype, device, and gradients through the owned layer.
        fn forward(&self, observations: Tensor<2>) -> Result<Tensor<2>, ModelError> {
            Ok(Forward::forward(&self.layer, observations)? + self.offset)
        }
    }

    fn observation_model(features: usize, device: &Device, dtype: DType) -> ObservationModel {
        ObservationModel {
            layer: Linear {
                weight: Initializer::Zeros
                    .init([features, 2], device)
                    .init_mapper(move |tensor| tensor.cast(dtype)),
                bias: Some(
                    Initializer::Zeros
                        .init([2], device)
                        .init_mapper(move |tensor| tensor.cast(dtype)),
                ),
            },
        }
    }

    #[test]
    fn automatic_constructors_preserve_shapes_dtypes_and_gaussian_statistics() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex();
        for dtype in [DType::F32, DType::F64] {
            for action_count in [1, 2, 4] {
                let policy = ProbabilisticPolicyModel::<(_, GaussianDistribution, _)>::new(
                    zero_model(4, action_count * 2, &device, dtype),
                );
                for batch_size in [1, 3, 10] {
                    let observations = Tensor::zeros([batch_size, 4], (&device, dtype));
                    let actions = policy.sample(observations.clone()).unwrap();
                    assert_eq!(actions.dims(), [batch_size, action_count]);
                    assert_eq!(actions.dtype(), dtype);
                    let modes = policy.mode(observations.clone()).unwrap();
                    assert_eq!(modes.abs().max().into_scalar::<f64>(), 0.0);
                    let (log_prob, entropy) = policy
                        .log_prob_and_entropy(
                            observations,
                            Tensor::zeros([batch_size, action_count], (&device, dtype)),
                        )
                        .unwrap();
                    assert_eq!(log_prob.dims(), [batch_size, 1]);
                    assert_eq!(entropy.dims(), [batch_size, 1]);
                    assert_eq!(log_prob.dtype(), dtype);
                    assert_eq!(entropy.dtype(), dtype);
                    let expected_log_prob =
                        -0.5 * action_count as f64 * (2.0 * std::f64::consts::PI).ln();
                    let expected_entropy =
                        action_count as f64 * (0.5 * (2.0 * std::f64::consts::PI).ln() + 0.5);
                    for value in log_prob
                        .into_data()
                        .convert::<f64>()
                        .try_to_vec::<f64>()
                        .unwrap()
                    {
                        assert!((value - expected_log_prob).abs() < 1e-5);
                    }
                    for value in entropy
                        .into_data()
                        .convert::<f64>()
                        .try_to_vec::<f64>()
                        .unwrap()
                    {
                        assert!((value - expected_entropy).abs() < 1e-5);
                    }
                }
            }
        }
    }

    struct PolicyOwner<T: PolicyTypes> {
        policy: ProbabilisticPolicyModel<T>,
    }

    #[test]
    fn type_group_is_inferred_for_an_owning_consumer() {
        let device = Device::flex();
        let owner = PolicyOwner {
            policy: ProbabilisticPolicyModel::with_distribution(
                zero_model(2, 2, &device, DType::F64),
                GaussianDistribution::default(),
            ),
        };
        let modes = owner
            .policy
            .mode(Tensor::ones([3, 2], (&device, DType::F64)))
            .unwrap();
        assert_eq!(modes.dims(), [3, 1]);
        assert_eq!(modes.dtype(), DType::F64);
        assert_eq!(
            owner.policy.module().output_layer.weight.val().dims(),
            [2, 2]
        );
        assert_eq!(owner.policy.clone().num_params(), owner.policy.num_params());
    }

    #[test]
    fn configured_distribution_preserves_multidimensional_actions_and_candidates() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        let policy = ProbabilisticPolicyModel::with_distribution(
            zero_model(4, 12, &device, DType::F64),
            GaussianDistribution::<3>::new([2, 3]).unwrap(),
        );
        let observations = Tensor::ones([3, 4], (&device, DType::F64)).require_grad();
        let actions = policy.sample(observations.clone()).unwrap();
        assert_eq!(actions.dims(), [3, 2, 3]);
        let (log_prob, entropy) = policy
            .log_prob_and_entropy(observations.clone(), actions.detach())
            .unwrap();
        assert_eq!(log_prob.dims(), [3, 1]);
        assert_eq!(entropy.dims(), [3, 1]);
        let terms = policy
            .expectation(observations.clone(), NonZeroUsize::new(4).unwrap())
            .unwrap();
        assert_eq!(terms.actions().dims(), [3, 4, 2, 3]);
        assert_eq!(terms.log_probabilities().dims(), [3, 4, 1]);
        assert_eq!(terms.weights().dims(), [3, 4, 1]);
        assert_eq!(terms.actions().dtype(), DType::F64);
        let gradients = terms.actions().clone().sum().backward();
        assert!(observations.grad(&gradients).is_some());
        assert_eq!(policy.default_target_entropy(observations).unwrap(), -6.0);
    }

    #[test]
    fn categorical_candidates_have_integer_kind_and_independent_rank() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex();
        let policy = ProbabilisticPolicyModel::with_distribution(
            zero_model(2, 4, &device, DType::F64),
            CategoricalDistribution,
        );
        let observations = Tensor::zeros([3, 2], (&device, DType::F64));
        assert_eq!(policy.sample(observations.clone()).unwrap().dims(), [3, 4]);
        let terms: ExpectationTerms<3, Int> = policy
            .expectation(observations.clone(), NonZeroUsize::MIN)
            .unwrap();
        assert_eq!(terms.actions().dims(), [3, 4, 1]);
        assert_eq!(
            terms
                .actions()
                .clone()
                .into_data()
                .convert::<i64>()
                .try_to_vec::<i64>()
                .unwrap(),
            vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
        );
        assert_eq!(
            terms
                .weights()
                .clone()
                .into_data()
                .try_to_vec::<f64>()
                .unwrap(),
            vec![0.25; 12]
        );
        assert!(
            (policy.default_target_entropy(observations).unwrap() - 0.98 * 4.0f64.ln()).abs()
                < 1e-12
        );
    }

    #[test]
    fn transformed_policy_preserves_bounds_and_parameter_gradients() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        let policy = ProbabilisticPolicyModel::with_distribution(
            zero_model(2, 4, &device, DType::F64),
            TransformedDistribution::new(GaussianDistribution::default(), TanhTransform),
        );
        let observations = Tensor::ones([3, 2], (&device, DType::F64));
        let terms = policy
            .expectation(observations, NonZeroUsize::new(4).unwrap())
            .unwrap();
        assert!(terms.actions().clone().abs().max().into_scalar::<f64>() <= 1.0);
        let gradients = terms.log_probabilities().clone().sum().backward();
        let gradients = GradientsParams::from_grads(gradients, &policy);
        let mut optimizer = SgdConfig::new().init();
        let policy = optimizer.step(0.01, policy, gradients);
        let gradients = policy
            .mode(Tensor::ones([3, 2], (&device, DType::F64)))
            .unwrap()
            .sum()
            .backward();
        assert!(
            policy
                .module()
                .output_layer
                .weight
                .val()
                .grad(&gradients)
                .is_some()
        );
    }

    #[test]
    fn custom_models_accept_image_and_integer_observations() {
        let device = Device::flex();
        let image_policy = ProbabilisticPolicyModel::with_distribution(
            observation_model(4, &device, DType::F64),
            GaussianDistribution::default(),
        );
        assert_eq!(
            image_policy
                .mode(Tensor::<4>::zeros([3, 1, 2, 2], (&device, DType::F64)))
                .unwrap()
                .dims(),
            [3, 1]
        );
        assert_eq!(image_policy.num_params(), 10);
        let integer_policy = ProbabilisticPolicyModel::with_distribution(
            observation_model(2, &device, DType::F64),
            GaussianDistribution::default(),
        );
        let actions = integer_policy
            .mode(Tensor::<2, Int>::zeros([3, 2], &device))
            .unwrap();
        assert_eq!(actions.dims(), [3, 1]);
        assert_eq!(actions.dtype(), DType::F64);
    }

    #[test]
    fn boolean_observation_model_converts_explicitly() {
        let device = Device::flex();
        let policy = ProbabilisticPolicyModel::with_distribution(
            observation_model(2, &device, DType::F32),
            GaussianDistribution::default(),
        );
        assert_eq!(
            policy
                .mode(Tensor::<2, burn::tensor::Bool>::from_data(
                    [[true, false]],
                    &device
                ))
                .unwrap()
                .dims(),
            [1, 1]
        );
    }

    #[test]
    fn custom_model_uses_updated_owned_parameters_and_native_records() {
        let device = Device::flex().autodiff();
        let offset = 0.75;
        let policy = ProbabilisticPolicyModel::with_distribution(
            OffsetModel {
                layer: observation_model(2, &device, DType::F64).layer,
                offset,
            },
            GaussianDistribution::default(),
        );
        assert!(format!("{policy:?}").contains("ProbabilisticPolicyModel"));
        let observations = Tensor::ones([1, 2], (&device, DType::F64));
        let before = policy.mode(observations.clone()).unwrap();
        assert!((before.clone().into_scalar::<f64>() - offset).abs() < 1e-12);
        let gradients = GradientsParams::from_grads(before.sum().backward(), &policy);
        let mut optimizer = SgdConfig::new().init();
        let policy = optimizer.step(0.1, policy, gradients);
        let expected = offset - 0.3;
        assert!(
            (policy
                .mode(observations.clone())
                .unwrap()
                .into_scalar::<f64>()
                - expected)
                .abs()
                < 1e-12
        );
        let record = policy.clone().into_record();
        let restored = ProbabilisticPolicyModel::with_distribution(
            OffsetModel {
                layer: observation_model(2, &device, DType::F64).layer,
                offset,
            },
            GaussianDistribution::default(),
        )
        .try_load_record(record)
        .unwrap();
        assert!(
            (restored
                .mode(observations.clone())
                .unwrap()
                .into_scalar::<f64>()
                - expected)
                .abs()
                < 1e-12
        );
        let inference = policy.valid().materialize();
        assert!(
            !inference
                .mode(observations.clone().without_autodiff())
                .unwrap()
                .is_autodiff()
        );
        let trained = inference.train();
        let gradients = trained.mode(observations).unwrap().sum().backward();
        assert!(
            trained
                .module()
                .layer
                .weight
                .val()
                .grad(&gradients)
                .is_some()
        );
    }

    #[test]
    fn errors_preserve_model_and_distribution_causes() {
        let device = Device::flex();
        let policy = ProbabilisticPolicyModel::with_distribution(
            zero_model(2, 2, &device, DType::F64),
            GaussianDistribution::default(),
        );
        assert!(matches!(
            policy.mode(Tensor::zeros([3, 1], (&device, DType::F64))),
            Err(ProbabilisticPolicyModelError::ModuleError(
                ModelError::InputFeatures { .. }
            ))
        ));
        let bad_distribution = ProbabilisticPolicyModel::with_distribution(
            zero_model(2, 3, &device, DType::F32),
            GaussianDistribution::default(),
        );
        assert!(matches!(
            bad_distribution.mode(Tensor::zeros([3, 2], &device)),
            Err(ProbabilisticPolicyModelError::DistError(
                GaussianDistributionError::InvalidOutputWidth { output_width: 3 }
            ))
        ));
    }

    #[derive(Clone, Debug)]
    struct EntropyProbe {
        detached: Arc<AtomicBool>,
    }

    impl Distribution for EntropyProbe {
        type Error = Infallible;

        /// Returns rank-2 parameters `[batch_size, features]` unchanged as the action fixture, preserving dtype and device.
        fn sample(&self, outputs: Tensor<2>) -> Result<Tensor<2>, Self::Error> {
            Ok(outputs)
        }

        /// Returns rank-2 parameters `[batch_size, features]` unchanged as the mode fixture, preserving dtype and device.
        fn mode(&self, outputs: Tensor<2>) -> Result<Tensor<2>, Self::Error> {
            Ok(outputs)
        }

        /// Evaluates fixture actions `[batch_size, features]` and returns sums `[batch_size, 1]` with the parameters' dtype and device.
        fn dist_eval(
            &self,
            outputs: Tensor<2>,
            _actions: Tensor<2>,
        ) -> Result<crate::distributions::DistEval, Self::Error> {
            let sums = outputs.sum_dim(1);
            Ok(crate::distributions::DistEval::new(sums.clone(), sums).unwrap())
        }
    }

    impl DifferentiableExpectation for EntropyProbe {
        type CandidateKind = Float;

        /// Adds a size-one candidate axis to `[batch_size, features]`, returning `[batch_size, 1, features]` and statistics `[batch_size, 1]`.
        fn expectation(
            &self,
            outputs: Tensor<2>,
            _samples: NonZeroUsize,
        ) -> Result<ExpectationTerms, Self::Error> {
            let batch_size = outputs.dims()[0];
            let options = (&outputs.device(), outputs.dtype());
            Ok(ExpectationTerms::new(
                outputs.clone().unsqueeze_dim(1),
                Tensor::zeros([batch_size, 1, 1], options),
                Tensor::ones([batch_size, 1, 1], options),
            )
            .unwrap())
        }

        /// Records whether `[batch_size, features]` outputs are detached, then returns a scalar target.
        fn default_target_entropy(&self, outputs: &Tensor<2>) -> Result<f64, Self::Error> {
            self.detached
                .store(!outputs.is_tracked(), Ordering::Relaxed);
            Ok(-1.0)
        }
    }

    #[test]
    fn target_entropy_detaches_outputs_without_freezing_model_parameters() {
        let device = Device::flex().autodiff();
        let detached = Arc::new(AtomicBool::new(false));
        let model = zero_model(2, 2, &device, DType::F32);
        let parameter = model.output_layer.weight.val();
        let policy = ProbabilisticPolicyModel::with_distribution(
            model,
            EntropyProbe {
                detached: detached.clone(),
            },
        );
        assert_eq!(
            policy
                .default_target_entropy(Tensor::ones([1, 2], &device))
                .unwrap(),
            -1.0
        );
        assert!(detached.load(Ordering::Relaxed));
        let gradients = policy
            .mode(Tensor::ones([1, 2], &device))
            .unwrap()
            .sum()
            .backward();
        assert!(parameter.grad(&gradients).is_some());
    }
}

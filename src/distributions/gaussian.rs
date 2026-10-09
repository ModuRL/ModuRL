use std::num::NonZeroUsize;

use burn::tensor::{Distribution as RandomDistribution, Float, Tensor};

use crate::distributions::{
    DifferentiableExpectation, DistEval, Distribution, DistributionTensorError, ExpectationTerms,
};
use crate::tensor_rank::NextRank;

/// Samples independent Gaussian values from parameters `[batch, 2 * event_size]`,
/// where `event_size` is the number of values in one action.
/// Means occupy the first half of the parameter axis, followed by log standard
/// deviations. Action batches have rank `A`: `[batch, ...action_shape]`.
/// Drawing multiple samples adds an axis: `[batch, samples, ...action_shape]`,
/// with rank `A + 1`, inferred through [`NextRank`]. Vector actions use ranks 2 and 3, respectively.
///
/// The default uses vector actions `[batch, features]`. Other action shapes use
/// `[batch, ...action_shape]`; statistics always reduce to `[batch]`.
/// Multiple-sample expectations support action ranks 1 through 1024, as defined by `NextRank`.
/// Samples are not squashed or clipped. A separate action map prepares them
/// for the environment; probability calculations use the original samples.
#[derive(Clone, Debug)]
pub struct GaussianDistribution<const A: usize = 2> {
    action_shape: Option<[usize; A]>,
}

impl Default for GaussianDistribution<2> {
    fn default() -> Self {
        Self { action_shape: None }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GaussianDistributionError {
    #[error("Gaussian tensor validation failed: {0}")]
    TensorError(#[from] DistributionTensorError),
    #[error("the Gaussian action shape is too large")]
    ActionShapeTooLarge,
    #[error("a Gaussian distribution requires a nonzero action dimension")]
    ZeroActionDimension,
    #[error("invalid Gaussian output width {output_width}")]
    InvalidOutputWidth { output_width: usize },
}

impl<const A: usize> GaussianDistribution<A> {
    /// Sets the shape of one action. Parameters remain `[batch, 2 * event_size]`.
    /// `action_shape` has `E` axes. Adding batch requires `A = E + 1`.
    /// Drawing multiple samples adds one axis, inferred through `NextRank`.
    pub fn new<const E: usize>(
        action_shape: [usize; E],
    ) -> Result<Self, GaussianDistributionError> {
        const {
            assert!(A == E + 1, "Gaussian action rank must be event rank + 1");
        }
        let event_size = action_shape.iter().try_fold(1usize, |size, dimension| {
            size.checked_mul(*dimension)
                .ok_or(GaussianDistributionError::ActionShapeTooLarge)
        })?;
        if event_size == 0 {
            return Err(GaussianDistributionError::ZeroActionDimension);
        }
        if event_size > usize::MAX / 2 {
            return Err(GaussianDistributionError::ActionShapeTooLarge);
        }
        // Reserve a batch axis of length one before the action dimensions.
        let mut shape = [1; A];
        shape[1..].copy_from_slice(&action_shape);
        Ok(Self {
            action_shape: Some(shape),
        })
    }

    /// Returns the configured action shape without the batch axis.
    pub fn action_shape(&self) -> Option<&[usize]> {
        self.action_shape.as_ref().map(|shape| &shape[1..])
    }

    /// Splits `[batch, 2 * event_size]` into mean and log-standard-deviation
    /// tensors `[batch, ...action_shape]`, both rank `A`, preserving dtype,
    /// device, and gradients.
    fn parameters(
        &self,
        outputs: Tensor<2>,
    ) -> Result<(Tensor<A>, Tensor<A>), GaussianDistributionError> {
        const {
            assert!(A >= 1, "Gaussian actions require a batch axis");
        }
        let [batch_size, output_width] = outputs.dims();
        if output_width == 0 || output_width % 2 != 0 {
            return Err(GaussianDistributionError::InvalidOutputWidth { output_width });
        }
        let half = output_width / 2;
        let mut shape = self.action_shape.unwrap_or([1; A]);
        if self.action_shape.is_some() {
            let event_size = shape[1..].iter().product::<usize>();
            if event_size != half {
                return Err(DistributionTensorError::ShapeMismatch {
                    field: "Gaussian parameters",
                    expected: vec![batch_size, event_size * 2],
                    actual: outputs.dims().to_vec(),
                }
                .into());
            }
        } else {
            shape[A - 1] = half;
        }
        shape[0] = batch_size;
        let mean = outputs
            .clone()
            .slice([0..batch_size, 0..half])
            .reshape(shape);
        let log_std = outputs
            .slice([0..batch_size, half..output_width])
            .reshape(shape);
        Ok((mean, log_std))
    }
}

/// Sums the action dimensions of rank-`A` values `[batch, ...action_shape]`,
/// returning `[batch]` in the same batch order. Scalar actions have no trailing axes.
fn sum_event_dimensions<const A: usize>(values: Tensor<A>) -> Tensor<1> {
    const {
        assert!(A >= 1, "event reduction requires a batch axis");
    }
    let batch_size = values.dims()[0];
    let event_size = values.dims()[1..].iter().product::<usize>();
    values
        .reshape([batch_size, event_size])
        .sum_dim(1)
        .squeeze_dim(1)
}

impl<const A: usize> Distribution<2, A> for GaussianDistribution<A> {
    type Error = GaussianDistributionError;

    /// Draws actions from parameters `[batch, 2 * event_size]`, returning
    /// `[batch, ...action_shape]` of rank `A`, retaining gradients through mean
    /// and standard deviation. Noise shares parameter dtype and device.
    fn sample(&self, outputs: Tensor<2>) -> Result<Tensor<A>, Self::Error> {
        let (mean, log_std) = self.parameters(outputs)?;
        let noise = mean.random_like(RandomDistribution::Normal(0.0, 1.0));
        Ok(mean + noise * log_std.exp())
    }

    /// Extracts means `[batch, ...action_shape]` from `[batch, 2 * event_size]` parameters.
    fn mode(&self, outputs: Tensor<2>) -> Result<Tensor<A>, Self::Error> {
        Ok(self.parameters(outputs)?.0)
    }

    /// Evaluates rank-`A` actions `[batch, ...action_shape]` under rank-2
    /// parameters `[batch, 2 * event_size]`, summing action dimensions to `[batch]`
    /// log probability and entropy. Actions share parameter dtype and device.
    fn dist_eval(&self, outputs: Tensor<2>, actions: Tensor<A>) -> Result<DistEval, Self::Error> {
        let (mean, log_std) = self.parameters(outputs)?;
        let normalized_diff = (actions - mean).square() / log_std.clone().exp().square();
        let normalization = (2.0 * std::f64::consts::PI).ln();
        let log_prob =
            sum_event_dimensions((normalized_diff + log_std.clone() * 2.0 + normalization) * -0.5);
        let entropy = sum_event_dimensions(
            log_std + 0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E).ln(),
        );
        Ok(DistEval::new(log_prob, entropy)?)
    }
}

impl<const A: usize, const C: usize> DifferentiableExpectation<2, A, C> for GaussianDistribution<A>
where
    Tensor<A>: NextRank<Next = Tensor<C>>,
{
    type CandidateKind = Float;

    /// Draws multiple actions `[batch, samples, ...action_shape]` of rank `C` from
    /// `[batch, 2 * event_size]` parameters. Log probabilities and uniform
    /// weights are `[batch, samples]`; `C = A + 1` inserts the sample axis.
    fn expectation(
        &self,
        outputs: Tensor<2>,
        samples: NonZeroUsize,
    ) -> Result<ExpectationTerms<C>, Self::Error> {
        const {
            assert!(C == A + 1, "Gaussian samples axis requires C == A + 1");
        }
        let sample_count = samples.get();
        let mut actions = Vec::with_capacity(sample_count);
        let mut log_probabilities = Vec::with_capacity(sample_count);
        for _ in 0..sample_count {
            let sample = self.sample(outputs.clone())?;
            log_probabilities.push(
                self.dist_eval(outputs.clone(), sample.clone())?
                    .log_prob()
                    .clone(),
            );
            actions.push(sample);
        }
        let actions = Tensor::<A>::stack::<C>(actions, 1);
        let log_probabilities = Tensor::<1>::stack::<2>(log_probabilities, 1);
        let weights = Tensor::<2>::full(
            log_probabilities.dims(),
            1.0 / sample_count as f64,
            (&log_probabilities.device(), log_probabilities.dtype()),
        );
        Ok(ExpectationTerms::new(actions, log_probabilities, weights)?)
    }

    /// Returns scalar target entropy `-event_size` from `[batch, 2 * event_size]`
    /// parameters. The batch axis is excluded, including for empty batches.
    fn default_target_entropy(&self, outputs: &Tensor<2>) -> Result<f64, Self::Error> {
        let mean = self.parameters(outputs.clone())?.0;
        Ok(-(mean.dims()[1..].iter().product::<usize>() as f64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{DType, Device};

    #[test]
    fn evaluation_matches_closed_form_in_both_precisions() {
        let device = Device::flex();
        let distribution = GaussianDistribution::default();
        for dtype in [DType::F32, DType::F64] {
            let log_std = 0.5f64.ln();
            let outputs = Tensor::from_data(
                [[0.0f64, 0.0, 0.0, log_std, log_std, log_std]],
                (&device, dtype),
            );
            let evaluation = distribution
                .dist_eval(outputs, Tensor::zeros([1, 3], (&device, dtype)))
                .unwrap();
            let log_probability = evaluation.log_prob().clone().into_scalar::<f64>();
            let entropy = evaluation.entropy().clone().into_scalar::<f64>();
            let expected_log_probability =
                3.0 * (-log_std - 0.5 * (2.0 * std::f64::consts::PI).ln());
            let expected_entropy =
                3.0 * (log_std + 0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E).ln());
            assert!((log_probability - expected_log_probability).abs() < 1e-6);
            assert!((entropy - expected_entropy).abs() < 1e-6);
            assert_eq!(evaluation.entropy().dtype(), dtype);
        }
    }

    #[test]
    fn mode_and_scalar_events_preserve_configured_shapes() {
        let device = Device::flex();
        let distribution = GaussianDistribution::default();
        let outputs = Tensor::from_floats([[0.25, -0.5, 0.75, -1.0, 0.0, 1.0]], &device);
        assert_eq!(
            distribution
                .mode(outputs)
                .unwrap()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            vec![0.25, -0.5, 0.75]
        );
        let scalar = GaussianDistribution::<1>::new([]).unwrap();
        assert_eq!(
            scalar
                .mode(Tensor::from_floats([[2.0, 0.0], [3.0, 0.0]], &device))
                .unwrap()
                .dims(),
            [2]
        );
    }

    #[test]
    fn scalar_expectation_infers_native_candidate_rank() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex();
        let distribution = GaussianDistribution::<1>::new([]).unwrap();
        let terms: ExpectationTerms<2> = distribution
            .expectation(
                Tensor::zeros([2, 2], &device),
                NonZeroUsize::new(3).unwrap(),
            )
            .unwrap();
        let actions: Tensor<2> = terms.actions().clone();
        assert_eq!(actions.dims(), [2, 3]);
        assert_eq!(terms.log_probabilities().dims(), [2, 3]);
        assert_eq!(terms.weights().dims(), [2, 3]);
    }

    #[test]
    fn reparameterization_retains_mean_and_log_std_gradients() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        let mean = Tensor::<2>::from_floats([[0.0, 0.0]], &device).require_grad();
        let log_std = Tensor::<2>::zeros([1, 2], &device).require_grad();
        let outputs = Tensor::cat(vec![mean.clone(), log_std.clone()], 1);
        let distribution = GaussianDistribution::default();
        let terms = distribution
            .expectation(outputs.clone(), NonZeroUsize::MIN)
            .unwrap();
        assert_eq!(terms.actions().dims(), [1, 1, 2]);
        assert_eq!(
            terms
                .weights()
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            vec![1.0]
        );
        assert_eq!(distribution.default_target_entropy(&outputs).unwrap(), -2.0);
        let sample = terms
            .actions()
            .clone()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let gradients = terms.actions().clone().sum().backward();
        assert_eq!(
            mean.grad(&gradients)
                .unwrap()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            vec![1.0, 1.0]
        );
        let std_gradient = log_std
            .grad(&gradients)
            .unwrap()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        for (actual, expected) in std_gradient.iter().zip(sample) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn high_rank_events_and_uniform_candidate_weights_are_preserved() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex();
        let distribution = GaussianDistribution::<6>::new([2, 1, 2, 1, 2]).unwrap();
        assert_eq!(
            distribution.action_shape(),
            Some([2, 1, 2, 1, 2].as_slice())
        );
        let outputs = Tensor::<2>::zeros([3, 16], &device);
        let sample = distribution.sample(outputs.clone()).unwrap();
        assert_eq!(sample.dims(), [3, 2, 1, 2, 1, 2]);
        assert_eq!(
            distribution
                .dist_eval(outputs.clone(), sample)
                .unwrap()
                .entropy()
                .dims(),
            [3]
        );
        let terms: ExpectationTerms<7> = distribution
            .expectation(outputs.clone(), NonZeroUsize::new(4).unwrap())
            .unwrap();
        assert_eq!(terms.actions().dims(), [3, 4, 2, 1, 2, 1, 2]);
        assert_eq!(terms.log_probabilities().dims(), [3, 4]);
        assert_eq!(
            terms
                .weights()
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            vec![0.25; 12]
        );
        assert_eq!(distribution.default_target_entropy(&outputs).unwrap(), -8.0);
    }

    #[test]
    fn native_sampling_preserves_dtype_and_seeded_randomness() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex();
        let distribution = GaussianDistribution::default();
        for dtype in [DType::F32, DType::F64] {
            let outputs = Tensor::<2>::zeros([8, 4], (&device, dtype));
            device.seed(123);
            let first = distribution.sample(outputs.clone()).unwrap();
            device.seed(123);
            let second = distribution.sample(outputs).unwrap();
            assert_eq!(first.dims(), [8, 2]);
            assert_eq!(first.dtype(), dtype);
            assert_eq!(first.into_data(), second.into_data());
        }
    }

    #[test]
    fn invalid_shapes_and_empty_batch_entropy_are_handled() {
        let device = Device::flex();
        let distribution = GaussianDistribution::<3>::new([2, 3]).unwrap();
        assert!(matches!(
            distribution.mode(Tensor::zeros([1, 10], &device)),
            Err(GaussianDistributionError::TensorError(
                DistributionTensorError::ShapeMismatch { .. }
            ))
        ));
        assert!(matches!(
            GaussianDistribution::<4>::new([2, 0, 3]),
            Err(GaussianDistributionError::ZeroActionDimension)
        ));
        assert!(matches!(
            GaussianDistribution::<3>::new([usize::MAX, 2]),
            Err(GaussianDistributionError::ActionShapeTooLarge)
        ));
        let flat = GaussianDistribution::default();
        assert!(matches!(
            flat.sample(Tensor::zeros([1, 3], &device)),
            Err(GaussianDistributionError::InvalidOutputWidth { output_width: 3 })
        ));
        assert_eq!(
            flat.default_target_entropy(&Tensor::zeros([0, 4], &device))
                .unwrap(),
            -2.0
        );
    }
}

use std::num::NonZeroUsize;

use burn::tensor::{
    DType, Distribution as RandomDistribution, Int, Tensor,
    activation::{log_softmax, softmax},
};

use crate::distributions::{
    DifferentiableExpectation, DistEval, Distribution, DistributionTensorError, ExpectationTerms,
};

/// Categorical operations over scores (logits) `[batch, categories]`.
/// Samples and modes keep that shape; evaluation returns `[batch]` statistics.
/// Computing an average over all categories uses integer indices
/// `[batch, categories]`, rather than the scores used for action selection.
#[derive(Clone, Copy, Debug, Default)]
pub struct CategoricalDistribution;

#[derive(Debug, thiserror::Error)]
pub enum CategoricalDistributionError {
    #[error("categorical tensor validation failed: {0}")]
    TensorError(#[from] DistributionTensorError),
    #[error("a categorical distribution requires at least one category")]
    NoCategories,
    #[error("categorical sampling requires a floating-point dtype, got {0:?}")]
    UnsupportedLogitDType(DType),
    #[error("category count {0} exceeds the integer index range")]
    TooManyCategories(usize),
}

impl CategoricalDistribution {
    /// Validates logits `[batch, categories]` and returns the two axis lengths.
    fn validate(outputs: &Tensor<2>) -> Result<[usize; 2], CategoricalDistributionError> {
        let shape = outputs.dims();
        if shape[1] == 0 {
            return Err(CategoricalDistributionError::NoCategories);
        }
        Ok(shape)
    }

    /// Generates Gumbel noise `[batch, categories]` matching the logits' shape,
    /// dtype, and device. Uniform endpoints are clamped to keep logarithms finite.
    fn gumbel_noise(outputs: &Tensor<2>) -> Result<Tensor<2>, CategoricalDistributionError> {
        let info =
            outputs
                .dtype()
                .finfo()
                .ok_or(CategoricalDistributionError::UnsupportedLogitDType(
                    outputs.dtype(),
                ))?;
        let uniform = outputs
            .random_like(RandomDistribution::Uniform(0.0, 1.0))
            .clamp(info.min_positive, 1.0 - info.epsilon);
        Ok(uniform.log().neg().log().neg())
    }
}

impl Distribution<2, 2> for CategoricalDistribution {
    type Error = CategoricalDistributionError;

    /// Adds Gumbel noise to logits `[batch, categories]`, preserving both axes.
    fn sample(&self, outputs: Tensor<2>) -> Result<Tensor<2>, Self::Error> {
        Self::validate(&outputs)?;
        let noise = Self::gumbel_noise(&outputs)?;
        Ok(outputs + noise)
    }

    /// Returns scores `[batch, categories]` unchanged so an action map can
    /// select the category with the highest score.
    fn mode(&self, outputs: Tensor<2>) -> Result<Tensor<2>, Self::Error> {
        Self::validate(&outputs)?;
        Ok(outputs)
    }

    /// Evaluates sampled scores and model logits, both `[batch, categories]`, reducing
    /// log probability and entropy to `[batch]`. Inputs share dtype and device.
    fn dist_eval(&self, outputs: Tensor<2>, actions: Tensor<2>) -> Result<DistEval, Self::Error> {
        Self::validate(&outputs)?;
        let log_probs = log_softmax(outputs.clone(), 1);
        let indices = actions.argmax(1);
        let log_prob = log_probs.clone().gather(1, indices).squeeze_dim(1);
        let entropy = (softmax(outputs, 1) * log_probs)
            .sum_dim(1)
            .neg()
            .squeeze_dim(1);
        Ok(DistEval::new(log_prob, entropy)?)
    }
}

impl DifferentiableExpectation<2, 2, 2> for CategoricalDistribution {
    type CandidateKind = Int;

    /// Lists all category indices `[batch, categories]` from logits with the
    /// same shape. Log probabilities and normalized weights preserve both axes.
    fn expectation(
        &self,
        outputs: Tensor<2>,
        _samples: NonZeroUsize,
    ) -> Result<ExpectationTerms<2, Int>, Self::Error> {
        let [batch_size, categories] = Self::validate(&outputs)?;
        let end = i64::try_from(categories)
            .map_err(|_| CategoricalDistributionError::TooManyCategories(categories))?;
        let actions = Tensor::<1, Int>::arange(0..end, &outputs.device())
            .unsqueeze::<2>()
            .expand([batch_size, categories]);
        let log_probabilities = log_softmax(outputs.clone(), 1);
        let weights = softmax(outputs, 1);
        Ok(ExpectationTerms::new(actions, log_probabilities, weights)?)
    }

    /// Computes scalar target entropy from logits `[batch, categories]` without
    /// altering the tensor. The target is 98% of uniform categorical entropy.
    fn default_target_entropy(&self, outputs: &Tensor<2>) -> Result<f64, Self::Error> {
        Ok(0.98 * (Self::validate(outputs)?[1] as f64).ln())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;

    #[test]
    fn log_softmax_matches_reference_for_large_logits() {
        let device = Device::flex();
        for dtype in [DType::F32, DType::F64] {
            let logits = Tensor::<2>::from_data(
                [[1.0f64, 2.0, 3.0], [1001.0, 1002.0, 1003.0]],
                (&device, dtype),
            );
            let result = log_softmax(logits, 1)
                .cast(DType::F64)
                .into_data()
                .try_to_vec::<f64>()
                .unwrap();
            let expected = [-2.40760596444438, -1.40760596444438, -0.40760596444438];
            for (index, value) in result.iter().enumerate() {
                assert!((value - expected[index % 3]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn expectation_enumerates_every_category_per_batch_row() {
        let device = Device::flex();
        let logits = Tensor::from_floats([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]], &device);
        let terms = CategoricalDistribution
            .expectation(logits.clone(), NonZeroUsize::MIN)
            .unwrap();
        assert_eq!(terms.actions().dims(), [2, 3]);
        assert_eq!(
            terms
                .actions()
                .clone()
                .into_data()
                .try_to_vec::<i32>()
                .unwrap(),
            vec![0, 1, 2, 0, 1, 2]
        );
        let sums = terms
            .weights()
            .clone()
            .sum_dim(1)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert!(sums.iter().all(|sum| (sum - 1.0).abs() < 1e-6));
        assert!(
            (CategoricalDistribution
                .default_target_entropy(&logits)
                .unwrap()
                - 0.98 * 3.0f64.ln())
            .abs()
                < 1e-12
        );
    }

    #[test]
    fn evaluation_matches_uniform_entropy_and_preserves_gradients() {
        let device = Device::flex().autodiff();
        let logits = Tensor::<2>::from_floats([[0.0, 0.0]], &device).require_grad();
        let actions = Tensor::from_floats([[2.0, 1.0]], &device);
        let evaluation = CategoricalDistribution
            .dist_eval(logits.clone(), actions)
            .unwrap();
        assert!((evaluation.log_prob().clone().into_scalar::<f32>() + 2.0f32.ln()).abs() < 1e-6);
        assert!((evaluation.entropy().clone().into_scalar::<f32>() - 2.0f32.ln()).abs() < 1e-6);
        let gradients = evaluation.log_prob().clone().sum().backward();
        assert_eq!(
            logits
                .grad(&gradients)
                .unwrap()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            vec![0.5, -0.5]
        );
    }

    #[test]
    fn exact_expectation_weights_have_the_policy_gradient() {
        let device = Device::flex().autodiff();
        let logits = Tensor::<2>::from_floats([[0.0, 0.0]], &device).require_grad();
        let terms = CategoricalDistribution
            .expectation(logits.clone(), NonZeroUsize::MIN)
            .unwrap();
        let payoff = terms.actions().clone().float();
        let gradients = (terms.weights().clone() * payoff).sum().backward();
        assert_eq!(
            logits
                .grad(&gradients)
                .unwrap()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            vec![-0.25, 0.25]
        );
    }

    #[test]
    fn samples_preserve_dtype_shape_and_seeded_randomness() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex();
        for dtype in [DType::F32, DType::F64] {
            let logits = Tensor::<2>::zeros([8, 3], (&device, dtype));
            device.seed(42);
            let first = CategoricalDistribution.sample(logits.clone()).unwrap();
            device.seed(42);
            let second = CategoricalDistribution.sample(logits.clone()).unwrap();
            assert_eq!(first.dims(), [8, 3]);
            assert_eq!(first.dtype(), dtype);
            assert_eq!(first.clone().into_data(), second.into_data());
            assert!(
                first
                    .cast(DType::F64)
                    .into_data()
                    .try_to_vec::<f64>()
                    .unwrap()
                    .iter()
                    .all(|value| value.is_finite())
            );
            assert_eq!(
                CategoricalDistribution
                    .mode(logits.clone())
                    .unwrap()
                    .into_data(),
                logits.into_data()
            );
        }
    }

    #[test]
    fn empty_categories_return_errors_and_single_category_entropy_is_zero() {
        let device = Device::flex();
        assert!(matches!(
            CategoricalDistribution.sample(Tensor::zeros([2, 0], &device)),
            Err(CategoricalDistributionError::NoCategories)
        ));
        let single = Tensor::<2>::zeros([2, 1], &device);
        let evaluation = CategoricalDistribution
            .dist_eval(single.clone(), single.clone())
            .unwrap();
        assert_eq!(
            evaluation
                .entropy()
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            vec![0.0, 0.0]
        );
        assert_eq!(
            CategoricalDistribution
                .default_target_entropy(&single)
                .unwrap(),
            0.0
        );
    }
}

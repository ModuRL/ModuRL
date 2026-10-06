use std::num::NonZeroUsize;

use burn::tensor::{DType, Float, Tensor, kind::Basic};
mod categorical;
mod gaussian;
mod transformed;
pub use categorical::{CategoricalDistribution, CategoricalDistributionError};
pub use gaussian::{GaussianDistribution, GaussianDistributionError};
pub use transformed::{
    AffineTransform, AffineTransformError, DistributionTransform, TanhTransform,
    TransformedDistribution, TransformedDistributionError,
};

/// Invalid dimensions or incompatible distribution statistics.
#[derive(Debug, thiserror::Error)]
pub enum DistributionTensorError {
    #[error("a distribution expectation requires at least one candidate")]
    NoCandidates,
    #[error("{field} has shape {actual:?}, expected {expected:?}")]
    ShapeMismatch {
        field: &'static str,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("{field} is on a different device")]
    DeviceMismatch { field: &'static str },
    #[error("{field} has dtype {actual:?}, expected {expected:?}")]
    DTypeMismatch {
        field: &'static str,
        expected: DType,
        actual: DType,
    },
}

/// Validates that paired distribution statistics have matching dimensions,
/// dtype, and device. Used for entropy versus log probability and expectation
/// weights versus candidate log probabilities; `field` identifies the invalid
/// statistic in the returned error.
/// Both inputs are rank-`D` tensors with identical `[...]` shapes: `[batch]`
/// for evaluation statistics, or `[batch, candidates]` for expectation
/// statistics. Validation preserves all axes and returns no tensor.
fn validate_statistics<const D: usize>(
    field: &'static str,
    expected: &Tensor<D>,
    actual: &Tensor<D>,
) -> Result<(), DistributionTensorError> {
    if actual.dims() != expected.dims() {
        return Err(DistributionTensorError::ShapeMismatch {
            field,
            expected: expected.dims().to_vec(),
            actual: actual.dims().to_vec(),
        });
    }
    if actual.dtype() != expected.dtype() {
        return Err(DistributionTensorError::DTypeMismatch {
            field,
            expected: expected.dtype(),
            actual: actual.dtype(),
        });
    }
    if actual.device() != expected.device() {
        return Err(DistributionTensorError::DeviceMismatch { field });
    }
    Ok(())
}

pub struct DistEval {
    log_prob: Tensor<1>,
    entropy: Tensor<1>,
}

impl DistEval {
    /// Creates an evaluation with `log_prob` and `entropy` both shaped
    /// `[batch_size]`.
    /// Both statistics must share batch size, dtype, and device.
    pub fn new(log_prob: Tensor<1>, entropy: Tensor<1>) -> Result<Self, DistributionTensorError> {
        validate_statistics("entropy", &log_prob, &entropy)?;
        Ok(Self { log_prob, entropy })
    }

    pub fn log_prob(&self) -> &Tensor<1> {
        &self.log_prob
    }

    pub fn entropy(&self) -> &Tensor<1> {
        &self.entropy
    }
}

/// Interprets model output tensors as a probability distribution family.
///
/// Each implementation defines its own output layout. For example,
/// [`GaussianDistribution`] expects a rank-2 `[mean, log_std]` tensor, while
/// [`CategoricalDistribution`] interprets its output as category logits.
/// `P` is the parameter rank and `A` the latent sample rank. These samples are
/// floats; decoding them into environment actions is a separate operation.
pub trait Distribution<const P: usize = 2, const A: usize = 2> {
    type Error;
    /// Samples from `outputs` shaped `[batch_size, ...parameter_shape]` and
    /// returns values shaped `[batch_size, ...event_shape]`.
    fn sample(&self, outputs: Tensor<P>) -> Result<Tensor<A>, Self::Error>;
    /// Returns modal values shaped `[batch_size, ...event_shape]` for `outputs`
    /// shaped `[batch_size, ...parameter_shape]`.
    fn mode(&self, outputs: Tensor<P>) -> Result<Tensor<A>, Self::Error>;
    /// Evaluates `actions` shaped `[batch_size, ...event_shape]` under `outputs`
    /// shaped `[batch_size, ...parameter_shape]`.
    ///
    /// Both returned statistics are shaped `[batch_size]`.
    fn dist_eval(&self, outputs: Tensor<P>, actions: Tensor<A>) -> Result<DistEval, Self::Error>;
}

/// The differentiable candidates used to evaluate a policy expectation.
#[derive(Clone, Debug)]
pub struct ExpectationTerms<const D: usize = 3, K: Basic = Float> {
    /// Candidate actions shaped `[batch, candidates, ...event_shape]`.
    actions: Tensor<D, K>,
    /// Candidate log probabilities shaped `[batch, candidates]`.
    log_probabilities: Tensor<2>,
    /// Normalized expectation weights shaped `[batch, candidates]`.
    weights: Tensor<2>,
}

impl<const D: usize, K: Basic> ExpectationTerms<D, K> {
    /// Creates expectation terms using the common
    /// `[batch, candidates, ...event_shape]` action layout.
    ///
    /// `log_probabilities` and `weights` must both be shaped
    /// `[batch, candidates]`, sharing dtype and device. Candidate actions must
    /// share the batch/candidate dimensions and device. Callers supply
    /// normalized weights; this constructor does not normalize or detach inputs.
    pub fn new(
        actions: Tensor<D, K>,
        log_probabilities: Tensor<2>,
        weights: Tensor<2>,
    ) -> Result<Self, DistributionTensorError> {
        const {
            assert!(
                D >= 2,
                "expectation actions require batch and candidate axes"
            );
        }
        let [batch_size, candidate_count] = log_probabilities.dims();
        if candidate_count == 0 {
            return Err(DistributionTensorError::NoCandidates);
        }
        validate_statistics("expectation weights", &log_probabilities, &weights)?;
        if actions.dims()[0] != batch_size || actions.dims()[1] != candidate_count {
            let mut expected = actions.dims().to_vec();
            expected[0] = batch_size;
            expected[1] = candidate_count;
            return Err(DistributionTensorError::ShapeMismatch {
                field: "expectation actions",
                expected,
                actual: actions.dims().to_vec(),
            });
        }
        if actions.device() != log_probabilities.device() {
            return Err(DistributionTensorError::DeviceMismatch {
                field: "expectation actions",
            });
        }
        Ok(Self {
            actions,
            log_probabilities,
            weights,
        })
    }

    pub fn actions(&self) -> &Tensor<D, K> {
        &self.actions
    }

    pub fn log_probabilities(&self) -> &Tensor<2> {
        &self.log_probabilities
    }

    pub fn weights(&self) -> &Tensor<2> {
        &self.weights
    }

    pub fn into_parts(self) -> (Tensor<D, K>, Tensor<2>, Tensor<2>) {
        (self.actions, self.log_probabilities, self.weights)
    }
}

/// A distribution whose expectation can be optimized by backpropagation.
/// Candidate rank `C` and kind are independent of latent sample rank `A`:
/// categorical candidates are integer indices, while Gaussian candidates are
/// continuous values with an additional candidate axis.
pub trait DifferentiableExpectation<const P: usize = 2, const A: usize = 2, const C: usize = 3>:
    Distribution<P, A>
{
    type CandidateKind: Basic;
    /// Returns differentiable action candidates and their expectation weights.
    ///
    /// `outputs` is shaped `[batch_size, ...parameter_shape]`. Returned actions
    /// are `[batch_size, candidates, ...event_shape]`; returned log
    /// probabilities and weights are `[batch_size, candidates]`.
    ///
    /// `samples` is the requested Monte Carlo sample count. Distributions with
    /// tractable finite support may instead enumerate that support exactly.
    fn expectation(
        &self,
        outputs: Tensor<P>,
        samples: NonZeroUsize,
    ) -> Result<ExpectationTerms<C, Self::CandidateKind>, Self::Error>;

    /// Returns the conventional SAC target entropy for `outputs` shaped
    /// `[batch_size, ...parameter_shape]`.
    fn default_target_entropy(&self, outputs: &Tensor<P>) -> Result<f64, Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Int};

    #[test]
    fn expectation_terms_enforce_common_batch_and_candidate_dimensions() {
        let device = Device::flex();
        let actions = Tensor::<4>::zeros([2, 3, 4, 5], &device);
        let probabilities = Tensor::<2>::zeros([2, 3], &device);
        assert!(
            ExpectationTerms::new(actions, probabilities.clone(), probabilities.clone()).is_ok()
        );

        let wrong_candidates = Tensor::<4>::zeros([2, 4, 4, 5], &device);
        assert!(matches!(
            ExpectationTerms::new(
                wrong_candidates,
                probabilities.clone(),
                probabilities.clone()
            ),
            Err(DistributionTensorError::ShapeMismatch {
                field: "expectation actions",
                ..
            })
        ));
        let wrong_weights = Tensor::<2>::zeros([2, 1], &device);
        assert!(matches!(
            ExpectationTerms::new(
                Tensor::<2, Int>::zeros([2, 3], &device),
                probabilities,
                wrong_weights
            ),
            Err(DistributionTensorError::ShapeMismatch {
                field: "expectation weights",
                ..
            })
        ));
    }

    #[test]
    fn expectation_terms_validate_candidate_count_and_statistic_dtype() {
        let device = Device::flex();
        let result = ExpectationTerms::new(
            Tensor::<2, Int>::zeros([2, 0], &device),
            Tensor::zeros([2, 0], &device),
            Tensor::zeros([2, 0], &device),
        );
        assert!(matches!(result, Err(DistributionTensorError::NoCandidates)));
        let result = ExpectationTerms::new(
            Tensor::<3>::zeros([2, 3, 4], &device),
            Tensor::zeros([2, 3], (&device, DType::F32)),
            Tensor::zeros([2, 3], (&device, DType::F64)),
        );
        assert!(matches!(
            result,
            Err(DistributionTensorError::DTypeMismatch {
                field: "expectation weights",
                ..
            })
        ));
    }

    #[test]
    fn integer_candidates_preserve_float_statistic_gradients() {
        let device = Device::flex().autodiff();
        let actions = Tensor::<2, Int>::from_data([[0i32, 1]], &device);
        let log_probabilities = Tensor::<2>::from_floats([[-1.0, -2.0]], &device).require_grad();
        let weights = Tensor::<2>::from_floats([[0.25, 0.75]], &device).require_grad();
        let terms =
            ExpectationTerms::new(actions, log_probabilities.clone(), weights.clone()).unwrap();
        assert_eq!(terms.actions().dims(), [1, 2]);
        assert_eq!(
            terms
                .actions()
                .clone()
                .into_data()
                .try_to_vec::<i32>()
                .unwrap(),
            vec![0, 1]
        );
        let (_, logs, probabilities) = terms.into_parts();
        let gradients = (logs.sum() + probabilities.sum()).backward();
        for statistic in [log_probabilities, weights] {
            assert_eq!(
                statistic
                    .grad(&gradients)
                    .unwrap()
                    .into_data()
                    .try_to_vec::<f32>()
                    .unwrap(),
                vec![1.0, 1.0]
            );
        }
    }

    #[test]
    fn continuous_candidates_retain_their_gradient_path() {
        let device = Device::flex().autodiff();
        let actions = Tensor::<3>::from_floats([[[1.0, 2.0], [3.0, 4.0]]], &device).require_grad();
        let terms = ExpectationTerms::new(
            actions.clone(),
            Tensor::zeros([1, 2], &device),
            Tensor::from_floats([[0.5, 0.5]], &device),
        )
        .unwrap();
        let gradients = terms.actions().clone().sum().backward();
        assert_eq!(
            actions
                .grad(&gradients)
                .unwrap()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            vec![1.0; 4]
        );
    }

    #[test]
    fn evaluation_statistics_share_batch_shape_and_precision() {
        let device = Device::flex();
        let evaluation = DistEval::new(
            Tensor::from_data([1.0f64, 2.0], (&device, DType::F64)),
            Tensor::from_data([3.0f64, 4.0], (&device, DType::F64)),
        )
        .unwrap();
        assert_eq!(evaluation.log_prob().dtype(), DType::F64);
        assert_eq!(evaluation.entropy().dims(), [2]);
        assert!(matches!(
            DistEval::new(Tensor::zeros([2], &device), Tensor::zeros([3], &device)),
            Err(DistributionTensorError::ShapeMismatch {
                field: "entropy",
                ..
            })
        ));
        assert!(matches!(
            DistEval::new(
                Tensor::zeros([2], (&device, DType::F32)),
                Tensor::zeros([2], (&device, DType::F64))
            ),
            Err(DistributionTensorError::DTypeMismatch {
                field: "entropy",
                ..
            })
        ));
    }
}

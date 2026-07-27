use std::num::NonZeroUsize;

use candle_core::Tensor;
mod categorical;
mod gaussian;
mod transformed;
pub use categorical::{CategoricalDistribution, CategoricalDistributionError};
pub use gaussian::{GaussianDistribution, GaussianDistributionError};
pub use transformed::{
    AffineTransform, AffineTransformError, DistributionTransform, TanhTransform,
    TransformedDistribution, TransformedDistributionError,
};

pub struct DistEval {
    log_prob: Tensor,
    entropy: Tensor,
}

impl DistEval {
    /// Creates an evaluation with `log_prob` and `entropy` both shaped
    /// `[batch_size]`.
    pub fn new(log_prob: Tensor, entropy: Tensor) -> Self {
        Self { log_prob, entropy }
    }

    pub fn log_prob(&self) -> &Tensor {
        &self.log_prob
    }

    pub fn entropy(&self) -> &Tensor {
        &self.entropy
    }
}

/// Interprets model output tensors as a probability distribution family.
///
/// Each implementation defines its own output layout. For example,
/// [`GaussianDistribution`] expects a rank-2 `[mean, log_std]` tensor, while
/// [`CategoricalDistribution`] interprets its output as category logits.
pub trait Distribution {
    type Error;
    /// Samples from `outputs` shaped `[batch_size, ...parameter_shape]` and
    /// returns values shaped `[batch_size, ...event_shape]`.
    fn sample(&self, outputs: &Tensor) -> Result<Tensor, Self::Error>;
    /// Returns modal values shaped `[batch_size, ...event_shape]` for `outputs`
    /// shaped `[batch_size, ...parameter_shape]`.
    fn mode(&self, outputs: &Tensor) -> Result<Tensor, Self::Error>;
    /// Evaluates `actions` shaped `[batch_size, ...event_shape]` under `outputs`
    /// shaped `[batch_size, ...parameter_shape]`.
    ///
    /// Both returned statistics are shaped `[batch_size]`.
    fn dist_eval(&self, outputs: &Tensor, actions: &Tensor) -> Result<DistEval, Self::Error>;
}

/// The differentiable candidates used to evaluate a policy expectation.
#[derive(Clone, Debug)]
pub struct ExpectationTerms {
    /// Candidate actions shaped `[batch, candidates, ...event_shape]`.
    actions: Tensor,
    /// Candidate log probabilities shaped `[batch, candidates]`.
    log_probabilities: Tensor,
    /// Normalized expectation weights shaped `[batch, candidates]`.
    weights: Tensor,
}

impl ExpectationTerms {
    /// Creates expectation terms using the common
    /// `[batch, candidates, ...event_shape]` action layout.
    ///
    /// `log_probabilities` and `weights` must both be shaped
    /// `[batch, candidates]`.
    pub fn new(
        actions: Tensor,
        log_probabilities: Tensor,
        weights: Tensor,
    ) -> candle_core::Result<Self> {
        let (batch_size, candidate_count) = log_probabilities.dims2()?;
        if candidate_count == 0 {
            return Err(candle_core::Error::EmptyTensor {
                op: "distribution expectation candidate axis",
            });
        }
        if weights.dims() != log_probabilities.dims() {
            return Err(candle_core::Error::UnexpectedShape {
                msg: "expectation weights must match log probabilities".into(),
                expected: log_probabilities.shape().clone(),
                got: weights.shape().clone(),
            });
        }
        if actions.rank() < 2 {
            return Err(candle_core::Error::DimOutOfRange {
                shape: actions.shape().clone(),
                dim: 1,
                op: "distribution expectation actions",
            });
        }
        if actions.dim(0)? != batch_size || actions.dim(1)? != candidate_count {
            let mut expected = actions.dims().to_vec();
            expected[0] = batch_size;
            expected[1] = candidate_count;
            return Err(candle_core::Error::UnexpectedShape {
                msg: "expectation actions must match the batch and candidate dimensions".into(),
                expected: expected.into(),
                got: actions.shape().clone(),
            });
        }
        Ok(Self {
            actions,
            log_probabilities,
            weights,
        })
    }

    pub fn actions(&self) -> &Tensor {
        &self.actions
    }

    pub fn log_probabilities(&self) -> &Tensor {
        &self.log_probabilities
    }

    pub fn weights(&self) -> &Tensor {
        &self.weights
    }

    pub fn into_parts(self) -> (Tensor, Tensor, Tensor) {
        (self.actions, self.log_probabilities, self.weights)
    }
}

/// A distribution whose expectation can be optimized by backpropagation.
pub trait DifferentiableExpectation: Distribution {
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
        outputs: &Tensor,
        samples: NonZeroUsize,
    ) -> Result<ExpectationTerms, Self::Error>;

    /// Returns the conventional SAC target entropy for `outputs` shaped
    /// `[batch_size, ...parameter_shape]`.
    fn default_target_entropy(&self, outputs: &Tensor) -> Result<f64, Self::Error>;

    /// A descriptive alias for [`DifferentiableExpectation::expectation`],
    /// accepting `outputs` shaped `[batch_size, ...parameter_shape]`.
    fn differentiable_expectation(
        &self,
        outputs: &Tensor,
        samples: NonZeroUsize,
    ) -> Result<ExpectationTerms, Self::Error> {
        self.expectation(outputs, samples)
    }

    /// Returns expectation terms for `outputs` shaped
    /// `[batch_size, ...parameter_shape]`.
    fn expectation_terms(
        &self,
        outputs: &Tensor,
        samples: NonZeroUsize,
    ) -> Result<ExpectationTerms, Self::Error> {
        self.expectation(outputs, samples)
    }
}

#[cfg(test)]
mod tests {
    use super::ExpectationTerms;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn expectation_terms_enforce_common_batch_and_candidate_dimensions() {
        let actions = Tensor::zeros(&[2, 3, 4, 5], DType::F32, &Device::Cpu).unwrap();
        let probabilities = Tensor::zeros((2, 3), DType::F32, &Device::Cpu).unwrap();
        assert!(
            ExpectationTerms::new(actions, probabilities.clone(), probabilities.clone()).is_ok()
        );

        let wrong_candidates = Tensor::zeros(&[2, 4, 4, 5], DType::F32, &Device::Cpu).unwrap();
        assert!(matches!(
            ExpectationTerms::new(
                wrong_candidates,
                probabilities.clone(),
                probabilities.clone()
            ),
            Err(candle_core::Error::UnexpectedShape { .. })
        ));
        let wrong_weights = Tensor::zeros((2, 1), DType::F32, &Device::Cpu).unwrap();
        assert!(matches!(
            ExpectationTerms::new(
                Tensor::zeros((2, 3), DType::U32, &Device::Cpu).unwrap(),
                probabilities,
                wrong_weights
            ),
            Err(candle_core::Error::UnexpectedShape { .. })
        ));
    }
}

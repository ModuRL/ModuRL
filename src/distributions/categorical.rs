use std::num::NonZeroUsize;

use candle_core::{D, Device, Tensor};
use candle_nn::ops::softmax;

use crate::distributions::{DifferentiableExpectation, DistEval, Distribution, ExpectationTerms};

/// Stateless categorical distribution operations over unnormalized logits
/// shaped `[batch_size, category_count]`.
///
/// Samples and modes preserve that latent-logit shape. Distribution evaluation
/// accepts latent actions with the same shape and reduces statistics to
/// `[batch_size]`. Exact expectation candidates are scalar category indices
/// shaped `[batch_size, category_count]`.
#[derive(Clone, Copy, Debug, Default)]
pub struct CategoricalDistribution;

#[derive(Debug)]
pub enum CategoricalDistributionError {
    TensorError(candle_core::Error),
    NoCategories,
}

impl From<candle_core::Error> for CategoricalDistributionError {
    fn from(error: candle_core::Error) -> Self {
        Self::TensorError(error)
    }
}

/// Applies log-softmax to `tensor` of arbitrary shape `[...]`, preserving that
/// shape.
fn log_softmax(tensor: &Tensor, dim: D) -> Result<Tensor, candle_core::Error> {
    let max = tensor.max_keepdim(dim)?.expand(tensor.dims())?;
    let exps = (tensor - &max)?.exp()?;
    let sum_exps = exps.sum_keepdim(dim)?.expand(tensor.dims())?;
    let log_sum_exps = sum_exps.log()?;
    tensor - max - log_sum_exps
}

impl CategoricalDistribution {
    /// Validates categorical `outputs` shaped `[batch_size, category_count]`.
    fn validate(outputs: &Tensor) -> Result<(usize, usize), CategoricalDistributionError> {
        let shape = outputs.dims2()?;
        if shape.1 == 0 {
            return Err(CategoricalDistributionError::NoCategories);
        }
        Ok(shape)
    }

    /// Converts logits shaped `[batch_size, category_count]` to probabilities
    /// with the same shape.
    fn probs(&self, outputs: &Tensor) -> Result<Tensor, CategoricalDistributionError> {
        Self::validate(outputs)?;
        Ok(softmax(outputs, D::Minus1)?)
    }

    /// Converts logits shaped `[batch_size, category_count]` to log
    /// probabilities with the same shape.
    fn log_probs(&self, outputs: &Tensor) -> Result<Tensor, CategoricalDistributionError> {
        Self::validate(outputs)?;
        Ok(log_softmax(outputs, D::Minus1)?)
    }

    fn gumbel_noise(shape: &[usize], device: &Device) -> Result<Tensor, candle_core::Error> {
        // sample uniform(0,1), then transform: -log(-log(U))
        let u = Tensor::rand(0f32, 1f32, shape, device)?;
        let gumbel = (-1.0 * (&u.log()?))?.log()?;
        gumbel.neg() // -log(-log(u))
    }
}

impl Distribution for CategoricalDistribution {
    type Error = CategoricalDistributionError;

    /// Samples perturbed logits `[batch_size, category_count]` from logits with
    /// that same shape.
    fn sample(&self, outputs: &Tensor) -> Result<Tensor, Self::Error> {
        Self::validate(outputs)?;
        let shape = outputs.dims();
        let device = outputs.device();

        // add Gumbel noise to logits
        let noise = Self::gumbel_noise(shape, device)?;

        Ok((outputs + noise)?)
    }

    /// Returns modal logits `[batch_size, category_count]` from logits with that
    /// same shape.
    fn mode(&self, outputs: &Tensor) -> Result<Tensor, Self::Error> {
        Self::validate(outputs)?;
        Ok(outputs.clone())
    }

    /// Evaluates latent actions `[batch_size, category_count]` under logits of
    /// the same shape, returning statistics `[batch_size]`.
    fn dist_eval(&self, outputs: &Tensor, actions: &Tensor) -> Result<DistEval, Self::Error> {
        let log_probs = self.log_probs(outputs)?; // [batch, num_classes]

        let actions_argmax = actions.argmax(D::Minus1)?; // [batch]
        let log_prob = log_probs
            .gather(&actions_argmax.unsqueeze(1)?, D::Minus1)?
            .squeeze(1)?;

        // entropy = -∑ p * log p over classes
        let probs = self.probs(outputs)?;
        let entropy = (probs.clone() * log_probs)?.sum(D::Minus1)?.neg()?;

        Ok(DistEval::new(log_prob, entropy))
    }
}

impl DifferentiableExpectation for CategoricalDistribution {
    /// Enumerates scalar candidates `[batch_size, category_count]` for logits
    /// `[batch_size, category_count]`; log probabilities and weights have the
    /// same two-dimensional shape.
    fn expectation(
        &self,
        outputs: &Tensor,
        _samples: NonZeroUsize,
    ) -> Result<ExpectationTerms, Self::Error> {
        let (batch_size, categories) = Self::validate(outputs)?;
        let actions = Tensor::arange(0u32, categories as u32, outputs.device())?
            .broadcast_left(batch_size)?;
        let log_probabilities = self.log_probs(outputs)?;
        let weights = self.probs(outputs)?;
        Ok(ExpectationTerms::new(actions, log_probabilities, weights)?)
    }

    /// Computes target entropy from logits shaped
    /// `[batch_size, category_count]`.
    fn default_target_entropy(&self, outputs: &Tensor) -> Result<f64, Self::Error> {
        // A near-uniform policy is a useful default without making exact
        // uniformity the entropy optimizer's only fixed point.
        Ok(0.98 * (Self::validate(outputs)?.1 as f64).ln())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_softmax() {
        // Test against PyTorch's F.log_softmax
        // x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        // F.log_softmax(x, dim=-1) =
        // tensor([[-2.4076, -1.4076, -0.4076],
        //         [-2.4076, -1.4076, -0.4076]])

        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device).unwrap();

        let result = log_softmax(&x, D::Minus1).unwrap();
        let result_vec = result.to_vec2::<f32>().unwrap();

        // Expected values from PyTorch
        let expected = [
            vec![-2.407606f32, -1.4076059, -0.40760595],
            vec![-2.407606f32, -1.4076059, -0.40760595],
        ];

        for i in 0..2 {
            for j in 0..3 {
                let diff = (result_vec[i][j] - expected[i][j]).abs();
                assert!(
                    diff < 1e-5,
                    "Mismatch at [{}, {}]: got {}, expected {}, diff {}",
                    i,
                    j,
                    result_vec[i][j],
                    expected[i][j],
                    diff
                );
            }
        }
    }

    #[test]
    fn expectation_enumerates_actions_with_normalized_weights() {
        let logits = Tensor::from_vec(vec![0.0f32, 1.0, 2.0], (1, 3), &Device::Cpu).unwrap();
        let distribution = CategoricalDistribution;
        let terms = distribution
            .expectation(&logits, NonZeroUsize::MIN)
            .unwrap();

        assert_eq!(terms.actions().dims(), &[1, 3]);
        assert_eq!(
            terms.actions().to_vec2::<u32>().unwrap(),
            vec![vec![0, 1, 2]]
        );
        let weight_sum = terms
            .weights()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!((weight_sum - 1.0).abs() < 1e-6);
        assert!(
            (distribution.default_target_entropy(&logits).unwrap() - 0.98 * 3.0f64.ln()).abs()
                < 1e-12
        );
    }

    #[test]
    fn categorical_distribution_requires_a_category() {
        let outputs = Tensor::zeros((2, 0), candle_core::DType::F32, &Device::Cpu).unwrap();
        assert!(matches!(
            CategoricalDistribution.sample(&outputs),
            Err(CategoricalDistributionError::NoCategories)
        ));
    }
}

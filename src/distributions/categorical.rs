use candle_core::{Device, Tensor, D};
use candle_nn::ops::softmax;

use crate::distributions::{DistEval, Distribution};

/// Categorical distribution parameterized by unnormalized logits.
pub struct CategoricalDistribution {
    logits: Tensor, // shape: [batch, num_classes]
}

impl CategoricalDistribution {
    fn probs(&self) -> Result<Tensor, candle_core::Error> {
        softmax(&self.logits, D::Minus1)
    }

    fn log_probs(&self) -> Result<Tensor, candle_core::Error> {
        let probs = self.probs()?;
        probs.log()
    }

    fn gumbel_noise(shape: &[usize], device: &Device) -> Result<Tensor, candle_core::Error> {
        // sample uniform(0,1), then transform: -log(-log(U))
        let u = Tensor::rand(0f32, 1f32, shape, device)?;
        let gumbel = (-1.0 * (&u.log()?))?.log()?;
        Ok(gumbel.neg()?) // -log(-log(u))
    }
}

impl Distribution for CategoricalDistribution {
    type Error = candle_core::Error;

    fn sample(&self) -> Tensor {
        let shape = self.logits.dims();
        let device = self.logits.device();

        // add Gumbel noise to logits
        let noise = Self::gumbel_noise(&shape, device).unwrap();
        let noisy_logits = (&self.logits + noise).unwrap();

        noisy_logits
    }

    fn dist_eval(&self, actions: &Tensor) -> Result<DistEval, Self::Error> {
        let log_probs = self.log_probs()?; // [batch, num_classes]

        let actions_argmax = actions.argmax(D::Minus1)?; // [batch]
        let log_prob = log_probs
            .gather(&actions_argmax.unsqueeze(1)?, D::Minus1)?
            .squeeze(1)?;

        // entropy = -∑ p * log p over classes
        let probs = self.probs()?;
        let entropy = (probs.clone() * log_probs)?.sum(D::Minus1)?.neg()?;

        Ok(DistEval::new(log_prob, entropy))
    }

    fn from_outputs(outputs: &Tensor) -> Self {
        Self {
            logits: outputs.clone(),
        }
    }
}

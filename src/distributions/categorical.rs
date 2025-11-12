use candle_core::{D, Device, Tensor};
use candle_nn::ops::softmax;

use crate::distributions::{DistEval, Distribution};

/// Categorical distribution parameterized by unnormalized logits.
pub struct CategoricalDistribution {
    logits: Tensor, // shape: [batch, num_classes]
}

fn log_softmax(tensor: &Tensor, dim: D) -> Result<Tensor, candle_core::Error> {
    let max = tensor.max_keepdim(dim.clone())?.expand(tensor.dims())?;
    let exps = (tensor - &max)?.exp()?;
    let sum_exps = exps.sum_keepdim(dim.clone())?.expand(tensor.dims())?;
    let log_sum_exps = sum_exps.log()?;
    (tensor - max - log_sum_exps).map_err(Into::into)
}

impl CategoricalDistribution {
    fn probs(&self) -> Result<Tensor, candle_core::Error> {
        softmax(&self.logits, D::Minus1)
    }

    fn log_probs(&self) -> Result<Tensor, candle_core::Error> {
        log_softmax(&self.logits, D::Minus1)
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
            logits: outputs.clamp(-20.0, 20.0).unwrap(),
        }
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
        let expected = vec![
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
}

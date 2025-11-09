use candle_core::Tensor;

use crate::distributions::{DistEval, Distribution};

pub struct GuassianDistribution {
    action_mean: Tensor,
    action_log_std: Tensor,
}

impl Distribution for GuassianDistribution {
    type Error = candle_core::Error;

    fn sample(&self) -> Tensor {
        let mut unit_normal =
            Tensor::randn(0.0, 1.0, self.action_mean.dims(), self.action_mean.device()).unwrap();
        let dtype = self.action_mean.dtype();
        unit_normal = unit_normal.to_dtype(dtype).unwrap();
        let action_std = self.action_log_std.exp().unwrap();
        
        (&self.action_mean + unit_normal * action_std).unwrap()
    }

    fn dist_eval(&self, actions: &Tensor) -> Result<DistEval, Self::Error> {
        let action_log_std = self.action_log_std.clamp(-2.0, 2.0)?;
        let action_std = action_log_std.exp()?;
        let action = actions;
        let action_mean = &self.action_mean;
        // Break down each component
        let action_diff = (action.clone() - action_mean.clone())?;
        let normalized_diff = (action_diff.powf(2.0)? / action_std.powf(2.0))?;
        let log_det_term = (2.0 * &action_log_std)?;
        let normalization = action.dims()[1] as f64 * (2.0 * std::f64::consts::PI).ln();

        let total_inside = ((&normalized_diff + &log_det_term)? + normalization)?;
        let log_prob = -0.5 * total_inside;

        let log_prob = log_prob?.sum(1)?;
        // calculate the entropy
        let entropy = ((2.0 * &action_log_std)?
            + 0.5 * (std::f64::consts::PI * 2.0 * std::f64::consts::E).ln())?
        .sum(1)?;

        Ok(DistEval::new(log_prob, entropy))
    }

    fn from_outputs(outputs: &Tensor) -> Self {
        // Split the outputs into mean and log_std
        let half = outputs.dims()[1] / 2;
        let action_mean = outputs.narrow(1, 0, half).unwrap();
        let action_log_std = outputs.narrow(1, half, half).unwrap();
        Self {
            action_mean,
            action_log_std,
        }
    }
}

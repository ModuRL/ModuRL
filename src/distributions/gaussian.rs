use candle_core::Tensor;

use crate::distributions::{DistEval, Distribution};

/// A batch of independent Gaussian action distributions.
///
/// `GaussianDistribution` expects a rank-2 parameter tensor with shape
/// `[batch_size, 2 * action_size]`. The first half of dimension 1 contains the
/// action means. The second half contains their log standard deviations.
/// Sampling returns `[batch_size, action_size]`; log probability and entropy
/// each reduce the independent action components to `[batch_size]`.
///
/// The distribution does not squash or clamp samples. A bounded [`BoxSpace`]
/// clamps the separate action tensor sent to an environment while PPO retains
/// the original sample for probability calculations.
///
/// [`BoxSpace`]: crate::spaces::BoxSpace
pub struct GaussianDistribution {
    action_mean: Tensor,
    action_log_std: Tensor,
}

impl Distribution for GaussianDistribution {
    type Error = candle_core::Error;

    fn sample(&self) -> Tensor {
        let mut unit_normal =
            Tensor::randn(0.0, 1.0, self.action_mean.dims(), self.action_mean.device()).unwrap();
        let dtype = self.action_mean.dtype();
        unit_normal = unit_normal.to_dtype(dtype).unwrap();
        let action_std = self.action_log_std.exp().unwrap();

        (&self.action_mean + unit_normal * action_std).unwrap()
    }

    fn mode(&self) -> Tensor {
        self.action_mean.clone()
    }

    fn dist_eval(&self, actions: &Tensor) -> Result<DistEval, Self::Error> {
        let action_log_std = &self.action_log_std;
        let action_std = action_log_std.exp()?;
        let action = actions;
        let action_mean = &self.action_mean;
        // Break down each component
        let action_diff = (action.clone() - action_mean.clone())?;
        let normalized_diff = (action_diff.powf(2.0)? / action_std.powf(2.0))?;
        let log_det_term = (2.0 * action_log_std)?;
        // This term is applied per action component before the reduction.
        let normalization = (2.0 * std::f64::consts::PI).ln();

        let total_inside = ((&normalized_diff + &log_det_term)? + normalization)?;
        let log_prob = -0.5 * total_inside;

        let log_prob = log_prob?.sum(1)?;
        // H[N(mu, sigma^2)] = log(sigma) + 0.5 * log(2*pi*e)
        // for each independent action component.
        let entropy = (action_log_std
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn diagonal_gaussian_entropy_and_log_prob_match_closed_form() {
        const ACTION_DIMS: usize = 3;
        let log_std = 0.5_f32.ln();
        let outputs = Tensor::from_vec(
            [vec![0.0_f32; ACTION_DIMS], vec![log_std; ACTION_DIMS]].concat(),
            (1, ACTION_DIMS * 2),
            &Device::Cpu,
        )
        .unwrap();
        let actions =
            Tensor::zeros((1, ACTION_DIMS), candle_core::DType::F32, &Device::Cpu).unwrap();

        let distribution = GaussianDistribution::from_outputs(&outputs);
        let evaluation = distribution.dist_eval(&actions).unwrap();
        let actual_entropy = evaluation.entropy().to_vec1::<f32>().unwrap()[0];
        let actual_log_prob = evaluation.log_prob().to_vec1::<f32>().unwrap()[0];

        let component_entropy =
            log_std + 0.5 * (2.0 * std::f32::consts::PI * std::f32::consts::E).ln();
        let expected_entropy = ACTION_DIMS as f32 * component_entropy;
        let component_log_prob = -log_std - 0.5 * (2.0 * std::f32::consts::PI).ln();
        let expected_log_prob = ACTION_DIMS as f32 * component_log_prob;

        assert!((actual_entropy - expected_entropy).abs() < 1e-6);
        assert!((actual_log_prob - expected_log_prob).abs() < 1e-6);
    }

    #[test]
    fn mode_is_the_configured_mean() {
        let outputs = Tensor::from_vec(
            vec![0.25_f32, -0.5, 0.75, -1.0, 0.0, 1.0],
            (1, 6),
            &Device::Cpu,
        )
        .unwrap();
        let distribution = GaussianDistribution::from_outputs(&outputs);

        assert_eq!(
            distribution.mode().to_vec2::<f32>().unwrap(),
            vec![vec![0.25, -0.5, 0.75]]
        );
    }
}

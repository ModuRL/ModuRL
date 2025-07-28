use candle_core::{Error, Tensor};
use candle_nn::Module;

use crate::actors::ppo::print_tensor_stats;

use super::MLP;

pub trait ProbabilisticActor {
    type Error;
    fn sample(&self, state: &Tensor) -> Result<Tensor, Self::Error>;
    fn log_prob_and_entropy(
        &self,
        state: &Tensor,
        action: &Tensor,
    ) -> Result<(Tensor, Tensor), Self::Error>;
}

pub struct MLPProbabilisticActor {
    mlp: MLP,
}

impl MLPProbabilisticActor {
    pub fn new(mlp: MLP) -> Self {
        Self { mlp }
    }
}

impl ProbabilisticActor for MLPProbabilisticActor {
    type Error = Error;
    fn sample(&self, state: &Tensor) -> Result<Tensor, Self::Error> {
        let output = self.mlp.forward(state)?;
        // First half of the output is the mean, second half is the std
        let action_mean = output.narrow(1, 0, output.dims()[1] / 2)?;
        let action_std = output.narrow(1, output.dims()[1] / 2, output.dims()[1] / 2)?;
        let mut unit_normal = Tensor::randn(0.0, 1.0, action_mean.dims(), action_mean.device())?;
        let dtype = action_mean.dtype();
        unit_normal = unit_normal.to_dtype(dtype)?;
        let action = action_mean + action_std.exp() * unit_normal;
        action
    }

    fn log_prob_and_entropy(
        &self,
        state: &Tensor,
        action: &Tensor,
    ) -> Result<(Tensor, Tensor), Self::Error> {
        let state = state.squeeze(1)?;
        let action = action.squeeze(1)?;
        let output = self.mlp.forward(&state)?;

        let action_mean = output.narrow(1, 0, output.dims()[1] / 2)?;
        let mut log_std = output.narrow(1, output.dims()[1] / 2, output.dims()[1] / 2)?;
        log_std = log_std.clamp(-5.0, 2.0)?;
        let action_std = (&log_std).exp()?;
        // Break down each component
        let action_diff = (action.clone() - action_mean.clone())?;
        let normalized_diff = (action_diff.powf(2.0)? / action_std.powf(2.0))?;
        let log_det_term = (2.0 * &log_std)?;
        let normalization = action.dims()[1] as f64 * (2.0 * std::f64::consts::PI).ln();

        let total_inside = ((&normalized_diff + &log_det_term)? + normalization)?;
        let log_prob = -0.5 * total_inside;

        let log_prob = log_prob?.sum(1)?;
        // calculate the entropy
        let entropy = ((2.0 * &log_std)?
            + 0.5 * (std::f64::consts::PI * 2.0 * std::f64::consts::E).ln())?
        .sum(1)?;

        if action_mean.dims()[0] > 2 {
            print_tensor_stats("Raw Output", &output);

            println!("=== LOG PROB BREAKDOWN ===");
            print_tensor_stats("Action Mean", &action_mean);
            print_tensor_stats("Action Std", &action_std);
            print_tensor_stats("Log Std", &log_std);
            print_tensor_stats("Action Diff", &action_diff);
            print_tensor_stats("Normalized Diff", &normalized_diff);
            print_tensor_stats("Log Det Term", &log_det_term);
            println!("Normalization constant: {:.3}", normalization);

            print_tensor_stats("Log Prob", &log_prob);
            print_tensor_stats("Entropy", &entropy);
            print_tensor_stats("Action", &action);
        }

        Ok((log_prob, entropy))
    }
}

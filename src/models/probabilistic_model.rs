use candle_core::{Error, Tensor};
use candle_nn::Module;

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
        let mut action_std = output.narrow(1, output.dims()[1] / 2, output.dims()[1] / 2)?;
        action_std = action_std.exp()?;
        let log_std = (action_std.clone() + f32::EPSILON as f64)?.log()?;
        let log_prob = -0.5
            * ((((action.clone() - action_mean.clone())?.powf(2.0)
                / ((&action_std).powf(2.0))?)?
                + 2.0 * &log_std)?
                + action.dims()[1] as f64 * (2.0 * std::f64::consts::PI).ln())?;
        let log_prob = log_prob?;
        // calculate the entropy
        let entropy = log_std + 0.5 * (std::f64::consts::PI * 2.0 * std::f64::consts::E).ln();

        Ok((log_prob, entropy?))
    }
}

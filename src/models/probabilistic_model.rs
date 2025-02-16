use candle_core::{Error, Tensor};
use candle_nn::{init::NormalOrUniform, Module};
use rand::{distr::Distribution, Rng};

use super::MLP;

pub trait ProbabilisticActor {
    type Error;
    fn sample(&self, state: &Tensor) -> Result<Tensor, Self::Error>;
    fn log_prob_and_entropy(
        &self,
        state: &Tensor,
        action: &Tensor,
    ) -> Result<(Tensor, Tensor), Self::Error> {
        let _ = state;
        let _ = action;
        unimplemented!()
    }
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
        let unit_normal = Tensor::randn(0.0, 1.0, action_mean.dims(), action_mean.device())?;
        let action = action_mean + action_std * unit_normal;
        action
    }

    // I hate this... I need to refactor this
    fn log_prob_and_entropy(
        &self,
        state: &Tensor,
        action: &Tensor,
    ) -> Result<(Tensor, Tensor), Self::Error> {
        let output = self.mlp.forward(state)?;
        let action_mean = output.narrow(1, 0, output.dims()[1] / 2)?;
        let action_std = output.narrow(1, output.dims()[1] / 2, output.dims()[1] / 2)?;
        // calculate the probability of the action
        let log_std = action_std.log()?;
        let log_prob = (-1.0
            * ((action - action_mean.clone())? * (action - action_mean.clone())?
                / (action_std.clone() * action_std)?)?
            - log_std.clone())?
            - 0.5 * (2.0 * std::f64::consts::PI).ln();
        let log_prob = log_prob?;
        // calculate the entropy
        let entropy = log_std + 0.5 * (std::f64::consts::PI * 2.0 * std::f64::consts::E).ln();

        Ok((log_prob, entropy?))
    }
}

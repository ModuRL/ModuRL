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
        let output = self.mlp.forward(state).unwrap();
        // First half of the output is the mean, second half is the std
        let action_mean = output.narrow(1, 0, output.dims()[1] / 2).unwrap();
        let action_std = output
            .narrow(1, output.dims()[1] / 2, output.dims()[1] / 2)
            .unwrap();
        let mut unit_normal =
            Tensor::randn(0.0, 1.0, action_mean.dims(), action_mean.device()).unwrap();
        let dtype = action_mean.dtype();
        unit_normal = unit_normal.to_dtype(dtype).unwrap();
        let action = action_mean + action_std * unit_normal;
        action
    }

    fn log_prob_and_entropy(
        &self,
        state: &Tensor,
        action: &Tensor,
    ) -> Result<(Tensor, Tensor), Self::Error> {
        let state = state.squeeze(1).unwrap();
        let action = action.squeeze(1).unwrap();
        let output = self.mlp.forward(&state).unwrap();
        let action_mean = output.narrow(1, 0, output.dims()[1] / 2).unwrap();
        let action_std = output
            .narrow(1, output.dims()[1] / 2, output.dims()[1] / 2)
            .unwrap();
        let action_std = action_std.abs().unwrap(); // make sure std is positive
                                                    // calculate the probability of the action
        let log_std = (action_std.log().unwrap() + f32::EPSILON as f64).unwrap();
        let log_prob = -0.5
            * ((((action.clone() - action_mean.clone()).unwrap()
                * (action - action_mean.clone()).unwrap()
                / (action_std.clone() * action_std).unwrap())
            .unwrap()
                + log_std.clone())
            .unwrap()
                + (2.0 * std::f64::consts::PI).ln())
            .unwrap();
        let log_prob = log_prob.unwrap();
        // calculate the entropy
        let entropy = log_std + 0.5 * (std::f64::consts::PI * 2.0 * std::f64::consts::E).ln();

        Ok((log_prob, entropy.unwrap()))
    }
}

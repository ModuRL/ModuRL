use crate::gym::Gym;
use candle_core::{Tensor, Var};
mod dqn;
pub use dqn::*;

pub trait Actor {
    type Error;
    /// Observation must be a tensor of shape [batch_size, observation_size].
    fn act(&mut self, observation: &Tensor) -> Result<Tensor, Self::Error>;
    fn learn(
        &mut self,
        env: &mut dyn Gym<Error = Self::Error>,
        num_episodes: usize,
    ) -> Result<(), Self::Error>;
    fn save(&self, vars: Vec<Var>, path: &str) -> Result<(), Self::Error>;
}

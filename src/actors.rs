use crate::gym::VectorizedGym;
use candle_core::Tensor;
pub mod dqn;
pub mod ppo;

pub trait Actor {
    type Error;
    type GymError;
    /// Observation must be a tensor of shape [batch_size, observation_size].
    fn act(&mut self, observation: &Tensor) -> Result<Tensor, Self::Error>;
    fn learn(
        &mut self,
        env: &mut dyn VectorizedGym<Error = Self::GymError>,
        num_timesteps: usize,
    ) -> Result<(), Self::Error>;
}

use crate::gym::VectorizedGym;
use candle_core::Tensor;
pub mod ppo;
pub mod q_learning;

pub trait Agent {
    type Error;
    type GymError;
    type SpaceError;

    /// Observation must be a tensor of shape [batch_size, observation_size].
    fn act(&mut self, observation: &Tensor) -> Result<Tensor, Self::Error>;
    fn learn(
        &mut self,
        env: &mut dyn VectorizedGym<Error = Self::GymError, SpaceError = Self::SpaceError>,
        num_timesteps: usize,
    ) -> Result<(), Self::Error>;
}

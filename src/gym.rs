use candle_core::Tensor;
mod common_gyms;
use crate::Space;
pub use common_gyms::*;

pub trait Gym {
    fn get_name(&self) -> &str;
    /// Returns the next state, reward, and done flag.
    fn step(&mut self, action: Tensor) -> (Tensor, f32, bool);
    /// Resets the environment to its initial state. Returns the initial state.
    fn reset(&mut self) -> Tensor;
    /// Returns the observation space.
    fn observation_space(&self) -> Box<dyn Space>;
    /// Returns the action space.
    fn action_space(&self) -> Box<dyn Space>;
}

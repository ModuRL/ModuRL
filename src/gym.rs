use crate::spaces::Space;
use candle_core::Tensor;
pub mod common_gyms;

pub trait Gym {
    type Error;
    fn get_name(&self) -> &str;
    /// Returns the next state, reward, and done flag.
    fn step(&mut self, action: Tensor) -> Result<StepInfo, Self::Error>;
    /// Resets the environment to its initial state. Returns the initial state.
    fn reset(&mut self) -> Result<Tensor, Self::Error>;
    /// Returns the observation space.
    fn observation_space(&self) -> Box<dyn Space>;
    /// Returns the action space.
    fn action_space(&self) -> Box<dyn Space>;
}

pub struct StepInfo {
    pub state: Tensor,
    pub reward: f32,
    pub done: bool,
    pub truncated: bool,
}

use ndarray::ArrayD;
mod common_gyms;
pub use common_gyms::*;

pub trait Gym<I, O> {
    fn get_name(&self) -> &str;
    /// Returns the next state, reward, and done flag.
    fn step(&mut self, action: ArrayD<I>) -> (ArrayD<O>, f32, bool);
    /// Resets the environment to its initial state. Returns the initial state.
    fn reset(&mut self) -> ArrayD<O>;
}

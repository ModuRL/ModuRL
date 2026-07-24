//! Episode time-limit wrapper.

use candle_core::Tensor;

use crate::gym::{Gym, ResetInfo, StepInfo};

/// Truncates an episode after a fixed number of environment steps.
pub struct TimeLimitGym<G> {
    gym: G,
    max_episode_steps: u32,
    elapsed_steps: u32,
}

impl<G> TimeLimitGym<G> {
    pub fn new(gym: G, max_episode_steps: u32) -> Self {
        assert!(
            max_episode_steps > 0,
            "max_episode_steps must be at least 1"
        );
        Self {
            gym,
            max_episode_steps,
            elapsed_steps: 0,
        }
    }
}

impl<G, I> Gym<I> for TimeLimitGym<G>
where
    G: Gym<I>,
{
    type Error = <G as Gym<I>>::Error;
    type SpaceError = <G as Gym<I>>::SpaceError;

    fn reset(&mut self) -> Result<ResetInfo<I>, Self::Error> {
        self.elapsed_steps = 0;
        self.gym.reset()
    }

    /// Forwards one unbatched environment action shaped `action_shape`.
    fn step(&mut self, action: Tensor) -> Result<StepInfo<I>, Self::Error> {
        let mut info = self.gym.step(action)?;
        self.elapsed_steps += 1;
        if self.elapsed_steps >= self.max_episode_steps && !info.done {
            info.truncated = true;
        }
        Ok(info)
    }

    fn action_space(&self) -> Box<dyn crate::spaces::Space<Error = Self::SpaceError>> {
        self.gym.action_space()
    }

    fn observation_space(&self) -> Box<dyn crate::spaces::Space<Error = Self::SpaceError>> {
        self.gym.observation_space()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        gym::Gym,
        wrappers::test_support::{TestGym, action},
    };

    use super::TimeLimitGym;

    #[test]
    fn truncates_at_limit_and_resets_elapsed_steps() {
        let gym = TestGym::new([
            TestGym::step(1.0, 0.0, false, false, 1),
            TestGym::step(2.0, 0.0, false, false, 2),
            TestGym::step(3.0, 0.0, false, false, 3),
        ]);
        let mut wrapper = TimeLimitGym::new(gym, 2);

        wrapper.reset().unwrap();
        let first = wrapper.step(action()).unwrap();
        let second = wrapper.step(action()).unwrap();
        wrapper.reset().unwrap();
        let after_reset = wrapper.step(action()).unwrap();

        assert!(!first.truncated);
        assert!(second.truncated);
        assert!(!after_reset.truncated);
    }
}

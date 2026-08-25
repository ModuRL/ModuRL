//! Wrappers that transform rewards.

use candle_core::Tensor;

use crate::gym::{Gym, ResetInfo, StepInfo};

#[derive(Debug, thiserror::Error)]
pub enum ClipRewardGymError<E> {
    #[error("wrapped gym error: {0}")]
    GymError(#[source] E),
}

/// Maps each nonzero reward to its sign.
pub struct ClipRewardGym<G> {
    gym: G,
}

impl<G> ClipRewardGym<G> {
    pub fn new(gym: G) -> Self {
        Self { gym }
    }
}

impl<G, I> Gym<I> for ClipRewardGym<G>
where
    G: Gym<I>,
{
    type Error = ClipRewardGymError<<G as Gym<I>>::Error>;
    type SpaceError = <G as Gym<I>>::SpaceError;

    fn reset(&mut self) -> Result<ResetInfo<I>, Self::Error> {
        self.gym.reset().map_err(ClipRewardGymError::GymError)
    }

    /// Forwards one unbatched environment action shaped `action_shape`.
    fn step(&mut self, action: Tensor) -> Result<StepInfo<I>, Self::Error> {
        let mut info = self
            .gym
            .step(action)
            .map_err(ClipRewardGymError::GymError)?;
        // `f32::signum()` maps positive zero to `1.0` and negative zero to
        // `-1.0`. Atari rewards are usually zero, so preserve zero explicitly.
        if info.reward != 0.0 {
            info.reward = info.reward.signum();
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

    use super::ClipRewardGym;

    #[test]
    fn maps_nonzero_rewards_to_their_sign_and_preserves_zero() {
        let gym = TestGym::new([
            TestGym::step(1.0, 2.5, false, false, 1),
            TestGym::step(1.0, -2.5, false, false, 2),
            TestGym::step(1.0, 0.0, false, false, 3),
            TestGym::step(1.0, -0.0, false, false, 4),
        ]);
        let mut wrapper = ClipRewardGym::new(gym);

        let positive = wrapper.step(action()).unwrap();
        let negative = wrapper.step(action()).unwrap();
        let positive_zero = wrapper.step(action()).unwrap();
        let negative_zero = wrapper.step(action()).unwrap();

        assert_eq!(positive.reward, 1.0);
        assert_eq!(negative.reward, -1.0);
        assert_eq!(positive_zero.reward, 0.0);
        assert_eq!(negative_zero.reward, 0.0);
        assert_eq!(negative_zero.info.sequence, 4);
    }
}

//! Wrappers that transform rewards.

use candle_core::Tensor;

use crate::gym::{Gym, ResetInfo, StepInfo};

#[derive(Debug)]
pub enum ClipRewardGymError<E> {
    GymError(E),
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
        info.reward = info.reward.signum();
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
    fn maps_reward_to_its_sign_and_preserves_metadata() {
        let gym = TestGym::new([TestGym::step(1.0, 2.5, false, false, 7)]);
        let mut wrapper = ClipRewardGym::new(gym);

        let step = wrapper.step(action()).unwrap();

        assert_eq!(step.reward, 1.0);
        assert_eq!(step.info.sequence, 7);
    }
}

//! Wrappers that transform rewards.

use burn::tensor::Tensor;

use crate::{
    gym::{Gym, ResetInfo, StepInfo},
    spaces::{ActionSpace, ObservationSpace},
};

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

impl<G, I, const O: usize, const A: usize, const BO: usize, const BA: usize> Gym<I, O, A, BO, BA>
    for ClipRewardGym<G>
where
    G: Gym<I, O, A, BO, BA>,
{
    type Error = ClipRewardGymError<G::Error>;
    type ObservationSpace = G::ObservationSpace;
    type ActionSpace = G::ActionSpace;

    /// Returns the inner gym's unbatched rank-`O` observation, preserving its kind, dtype, device, and item axes.
    fn reset(
        &mut self,
    ) -> Result<ResetInfo<I, O, <Self::ObservationSpace as ObservationSpace<BO>>::Kind>, Self::Error>
    {
        self.gym.reset().map_err(ClipRewardGymError::GymError)
    }

    /// Forwards an unbatched rank-`A` action and returns an unbatched rank-`O` observation; scalars use `[1]`.
    /// Preserves tensor shapes, kinds, dtypes, and devices, subject to the inner gym's input requirements.
    fn step(
        &mut self,
        action: Tensor<A, <Self::ActionSpace as ActionSpace<BA>>::Kind>,
    ) -> Result<StepInfo<I, O, <Self::ObservationSpace as ObservationSpace<BO>>::Kind>, Self::Error>
    {
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

    fn action_space(&self) -> Self::ActionSpace {
        self.gym.action_space()
    }

    fn observation_space(&self) -> Self::ObservationSpace {
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

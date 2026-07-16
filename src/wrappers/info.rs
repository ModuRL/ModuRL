//! Wrappers that add environment metadata.

use candle_core::Tensor;

use crate::gym::{Gym, ResetInfo, StepInfo};

/// Environment metadata augmented with the reward before outer wrappers transform it.
#[derive(Debug, Clone)]
pub struct RawRewardInfo<I> {
    pub inner: I,
    pub raw_reward: Option<f32>,
}

/// Records each reward in [`RawRewardInfo`] for outer wrappers and loggers.
pub struct RecordRawRewardGym<G> {
    gym: G,
}

impl<G> RecordRawRewardGym<G> {
    pub fn new(gym: G) -> Self {
        Self { gym }
    }
}

impl<G, I> Gym<RawRewardInfo<I>> for RecordRawRewardGym<G>
where
    G: Gym<I>,
{
    type Error = <G as Gym<I>>::Error;
    type SpaceError = <G as Gym<I>>::SpaceError;

    fn reset(&mut self) -> Result<ResetInfo<RawRewardInfo<I>>, Self::Error> {
        let reset = self.gym.reset()?;
        Ok(ResetInfo {
            state: reset.state,
            info: RawRewardInfo {
                inner: reset.info,
                raw_reward: None,
            },
        })
    }

    fn step(&mut self, action: Tensor) -> Result<StepInfo<RawRewardInfo<I>>, Self::Error> {
        let step = self.gym.step(action)?;
        Ok(StepInfo {
            state: step.state,
            reward: step.reward,
            done: step.done,
            truncated: step.truncated,
            info: RawRewardInfo {
                inner: step.info,
                raw_reward: Some(step.reward),
            },
        })
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

    use super::RecordRawRewardGym;

    #[test]
    fn adds_raw_reward_without_changing_the_transition() {
        let gym = TestGym::new([TestGym::step(1.0, 2.5, false, false, 7)]);
        let mut wrapper = RecordRawRewardGym::new(gym);

        let reset = wrapper.reset().unwrap();
        assert_eq!(reset.info.raw_reward, None);

        let step = wrapper.step(action()).unwrap();
        assert_eq!(step.reward, 2.5);
        assert_eq!(step.info.raw_reward, Some(2.5));
        assert_eq!(step.info.inner.sequence, 7);
    }
}

//! Wrappers that add environment metadata.

use burn::tensor::Tensor;

use crate::{
    gym::{Gym, ResetInfo, StepInfo},
    spaces::{ActionSpace, ObservationSpace},
};

/// Return and length of an episode completed by the wrapped environment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EpisodeStatistics {
    /// Sum of rewards from the episode.
    pub episode_return: f32,
    /// Number of calls to the wrapped environment's `step` method.
    pub episode_length: usize,
}

/// Environment metadata augmented with statistics for a completed episode.
#[derive(Clone, Debug)]
pub struct EpisodeStatisticsInfo<I> {
    /// Metadata produced by the wrapped environment.
    pub inner: I,
    /// Statistics emitted on the transition that ends an episode.
    pub completed_episode: Option<EpisodeStatistics>,
}

/// Records episode return and length using the wrapped environment's original
/// termination and truncation signals.
pub struct RecordEpisodeStatisticsGym<G> {
    gym: G,
    episode_return: f32,
    episode_length: usize,
}

impl<G> RecordEpisodeStatisticsGym<G> {
    pub fn new(gym: G) -> Self {
        Self {
            gym,
            episode_return: 0.0,
            episode_length: 0,
        }
    }
}

impl<G, I, const O: usize, const A: usize, const BO: usize, const BA: usize>
    Gym<EpisodeStatisticsInfo<I>, O, A, BO, BA> for RecordEpisodeStatisticsGym<G>
where
    G: Gym<I, O, A, BO, BA>,
{
    type Error = G::Error;
    type ObservationSpace = G::ObservationSpace;
    type ActionSpace = G::ActionSpace;

    /// Returns the inner gym's unbatched rank-`O` observation, preserving its kind, dtype, device, and item axes.
    fn reset(
        &mut self,
    ) -> Result<
        ResetInfo<
            EpisodeStatisticsInfo<I>,
            O,
            <Self::ObservationSpace as ObservationSpace<BO>>::Kind,
        >,
        Self::Error,
    > {
        self.episode_return = 0.0;
        self.episode_length = 0;
        let reset = self.gym.reset()?;
        Ok(ResetInfo {
            observation: reset.observation,
            info: EpisodeStatisticsInfo {
                inner: reset.info,
                completed_episode: None,
            },
        })
    }

    /// Forwards an unbatched rank-`A` action and returns an unbatched rank-`O` observation; scalars use `[1]`.
    /// Preserves tensor shapes, kinds, dtypes, and devices, subject to the inner gym's input requirements.
    fn step(
        &mut self,
        action: Tensor<A, <Self::ActionSpace as ActionSpace<BA>>::Kind>,
    ) -> Result<
        StepInfo<
            EpisodeStatisticsInfo<I>,
            O,
            <Self::ObservationSpace as ObservationSpace<BO>>::Kind,
        >,
        Self::Error,
    > {
        let step = self.gym.step(action)?;
        self.episode_return += step.reward;
        self.episode_length += 1;

        let completed_episode = if step.done || step.truncated {
            let statistics = EpisodeStatistics {
                episode_return: self.episode_return,
                episode_length: self.episode_length,
            };
            self.episode_return = 0.0;
            self.episode_length = 0;
            Some(statistics)
        } else {
            None
        };

        Ok(StepInfo {
            observation: step.observation,
            reward: step.reward,
            done: step.done,
            truncated: step.truncated,
            info: EpisodeStatisticsInfo {
                inner: step.info,
                completed_episode,
            },
        })
    }

    fn action_space(&self) -> Self::ActionSpace {
        self.gym.action_space()
    }

    fn observation_space(&self) -> Self::ObservationSpace {
        self.gym.observation_space()
    }
}

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

impl<G, I, const O: usize, const A: usize, const BO: usize, const BA: usize>
    Gym<RawRewardInfo<I>, O, A, BO, BA> for RecordRawRewardGym<G>
where
    G: Gym<I, O, A, BO, BA>,
{
    type Error = G::Error;
    type ObservationSpace = G::ObservationSpace;
    type ActionSpace = G::ActionSpace;

    /// Returns the inner gym's unbatched rank-`O` observation, preserving its kind, dtype, device, and item axes.
    fn reset(
        &mut self,
    ) -> Result<
        ResetInfo<RawRewardInfo<I>, O, <Self::ObservationSpace as ObservationSpace<BO>>::Kind>,
        Self::Error,
    > {
        let reset = self.gym.reset()?;
        Ok(ResetInfo {
            observation: reset.observation,
            info: RawRewardInfo {
                inner: reset.info,
                raw_reward: None,
            },
        })
    }

    /// Forwards an unbatched rank-`A` action and returns an unbatched rank-`O` observation; scalars use `[1]`.
    /// Preserves tensor shapes, kinds, dtypes, and devices, subject to the inner gym's input requirements.
    fn step(
        &mut self,
        action: Tensor<A, <Self::ActionSpace as ActionSpace<BA>>::Kind>,
    ) -> Result<
        StepInfo<RawRewardInfo<I>, O, <Self::ObservationSpace as ObservationSpace<BO>>::Kind>,
        Self::Error,
    > {
        let step = self.gym.step(action)?;
        Ok(StepInfo {
            observation: step.observation,
            reward: step.reward,
            done: step.done,
            truncated: step.truncated,
            info: RawRewardInfo {
                inner: step.info,
                raw_reward: Some(step.reward),
            },
        })
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

    use super::{EpisodeStatistics, RecordEpisodeStatisticsGym, RecordRawRewardGym};

    #[test]
    fn records_statistics_at_the_wrapped_episode_boundary() {
        let gym = TestGym::new([
            TestGym::step(1.0, 2.5, false, false, 1),
            TestGym::step(2.0, -0.5, true, false, 2),
        ]);
        let mut wrapper = RecordEpisodeStatisticsGym::new(gym);

        assert!(wrapper.reset().unwrap().info.completed_episode.is_none());
        assert!(
            wrapper
                .step(action())
                .unwrap()
                .info
                .completed_episode
                .is_none()
        );
        assert_eq!(
            wrapper.step(action()).unwrap().info.completed_episode,
            Some(EpisodeStatistics {
                episode_return: 2.0,
                episode_length: 2,
            })
        );
    }

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

//! Episode time-limit wrapper.

use burn::tensor::Tensor;

use crate::{
    gym::{Gym, ResetInfo, StepInfo},
    spaces::{ActionSpace, ObservationSpace},
};

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

impl<G, I, const O: usize, const A: usize, const BO: usize, const BA: usize> Gym<I, O, A, BO, BA>
    for TimeLimitGym<G>
where
    G: Gym<I, O, A, BO, BA>,
{
    type Error = G::Error;
    type ObservationSpace = G::ObservationSpace;
    type ActionSpace = G::ActionSpace;

    /// Resets the step count and returns the inner gym's unbatched rank-`O` observation and metadata.
    /// Preserves the observation kind, dtype, device, and item axes; scalar observations use `[1]`.
    fn reset(
        &mut self,
    ) -> Result<ResetInfo<I, O, <Self::ObservationSpace as ObservationSpace<BO>>::Kind>, Self::Error>
    {
        self.elapsed_steps = 0;
        self.gym.reset()
    }

    /// Forwards an unbatched rank-`A` action and returns an unbatched rank-`O` observation; scalars use `[1]`.
    /// Preserves tensor shapes, kinds, dtypes, and devices. The inner gym's input requirements still apply.
    /// Marks truncation at the step limit unless the inner gym reports termination.
    fn step(
        &mut self,
        action: Tensor<A, <Self::ActionSpace as ActionSpace<BA>>::Kind>,
    ) -> Result<StepInfo<I, O, <Self::ObservationSpace as ObservationSpace<BO>>::Kind>, Self::Error>
    {
        let mut info = self.gym.step(action)?;
        self.elapsed_steps += 1;
        if self.elapsed_steps >= self.max_episode_steps && !info.done {
            info.truncated = true;
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
        gym::{Gym, ResetInfo, StepInfo},
        spaces::{BoxSpace, Discrete},
    };
    use burn::tensor::{DType, Device, Int, Tensor};
    use std::convert::Infallible;

    use super::TimeLimitGym;

    struct TestGym {
        done: bool,
        truncated: bool,
    }

    impl Gym<u32> for TestGym {
        type Error = Infallible;
        type ObservationSpace = BoxSpace<2>;
        type ActionSpace = Discrete;

        /// Returns unbatched F64 observations `[2]` on the fixture device and reset metadata.
        fn reset(&mut self) -> Result<ResetInfo<u32>, Self::Error> {
            Ok(ResetInfo {
                observation: Tensor::from_data([1.0f64, 2.0], (&Device::flex(), DType::F64)),
                info: 7,
            })
        }

        /// Accepts a scalar Int action `[1]` and returns F64 observations `[2]` with the action in metadata.
        fn step(&mut self, action: Tensor<1, Int>) -> Result<StepInfo<u32>, Self::Error> {
            Ok(StepInfo {
                observation: self.reset()?.observation,
                reward: 3.0,
                done: self.done,
                truncated: self.truncated,
                info: action.into_scalar::<i32>() as u32,
            })
        }

        fn action_space(&self) -> Self::ActionSpace {
            Discrete::new(4)
        }

        fn observation_space(&self) -> Self::ObservationSpace {
            BoxSpace::new_unbounded([1, 2], &Device::flex())
        }
    }

    /// Creates one scalar Int action `[1]` on the fixture device.
    fn action() -> Tensor<1, Int> {
        Tensor::from_data([2i32], &Device::flex())
    }

    #[test]
    fn truncates_at_limit_and_resets_elapsed_steps() {
        let gym = TestGym {
            done: false,
            truncated: false,
        };
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

    #[test]
    fn preserves_termination_and_existing_truncation() {
        for (done, truncated) in [(true, false), (false, true), (true, true)] {
            let mut wrapper = TimeLimitGym::new(TestGym { done, truncated }, 1);
            let step = wrapper.step(action()).unwrap();
            assert_eq!((step.done, step.truncated), (done, truncated));
        }
    }

    #[test]
    fn preserves_observations_rewards_metadata_and_spaces() {
        let mut wrapper = TimeLimitGym::new(
            TestGym {
                done: false,
                truncated: false,
            },
            1,
        );
        let reset = wrapper.reset().unwrap();
        assert_eq!(reset.info, 7);
        assert_eq!(reset.observation.dims(), [2]);
        assert_eq!(reset.observation.dtype(), DType::F64);
        let step = wrapper.step(action()).unwrap();
        assert_eq!(step.observation.dims(), [2]);
        assert_eq!(step.observation.dtype(), DType::F64);
        assert_eq!(step.observation.device(), reset.observation.device());
        assert_eq!(
            step.observation.into_data().try_to_vec::<f64>().unwrap(),
            vec![1.0, 2.0]
        );
        assert_eq!(step.reward, 3.0);
        assert_eq!(step.info, 2);
        assert_eq!(wrapper.observation_space().shape(), vec![2]);
        assert!(wrapper.action_space().shape().is_empty());
    }
}

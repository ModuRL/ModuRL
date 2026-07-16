//! Environment-independent wrappers for [`crate::gym::Gym`].
//!
//! The modules follow Gymnasium's conventional categories: observation
//! transformations, reward transformations, and episode-control wrappers.

pub mod observation;
pub mod reward;
pub mod time_limit;

pub use observation::{FrameStackGym, FrameStackGymError, MaxAndSkipGym, MaxAndSkipGymError};
pub use reward::{ClipRewardGym, ClipRewardGymError};
pub use time_limit::TimeLimitGym;

#[cfg(test)]
mod test_support {
    use std::collections::VecDeque;

    use candle_core::{Device, Tensor};

    use crate::{
        gym::{Gym, ResetInfo, StepInfo},
        spaces::{BoxSpace, Discrete, Space},
    };

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) struct TestInfo {
        pub(super) sequence: u32,
    }

    pub(super) struct TestGym {
        steps: VecDeque<StepInfo<TestInfo>>,
        device: Device,
        reset_count: u32,
    }

    impl TestGym {
        pub(super) fn new(steps: impl IntoIterator<Item = StepInfo<TestInfo>>) -> Self {
            Self {
                steps: steps.into_iter().collect(),
                device: Device::Cpu,
                reset_count: 0,
            }
        }

        pub(super) fn step(
            state: f32,
            reward: f32,
            done: bool,
            truncated: bool,
            sequence: u32,
        ) -> StepInfo<TestInfo> {
            StepInfo {
                state: Tensor::new(state, &Device::Cpu).unwrap(),
                reward,
                done,
                truncated,
                info: TestInfo { sequence },
            }
        }
    }

    impl Gym<TestInfo> for TestGym {
        type Error = candle_core::Error;
        type SpaceError = candle_core::Error;

        fn step(&mut self, _action: Tensor) -> Result<StepInfo<TestInfo>, Self::Error> {
            Ok(self.steps.pop_front().expect("test step script exhausted"))
        }

        fn reset(&mut self) -> Result<ResetInfo<TestInfo>, Self::Error> {
            self.reset_count += 1;
            Ok(ResetInfo {
                state: Tensor::new(100.0 * self.reset_count as f32, &self.device)?,
                info: TestInfo { sequence: 0 },
            })
        }

        fn observation_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
            Box::new(BoxSpace::new(
                Tensor::new(-10_000.0f32, &self.device).unwrap(),
                Tensor::new(10_000.0f32, &self.device).unwrap(),
            ))
        }

        fn action_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
            Box::new(Discrete::new(4))
        }
    }

    pub(super) fn action() -> Tensor {
        Tensor::new(0u32, &Device::Cpu).unwrap()
    }

    pub(super) fn scalar(tensor: &Tensor) -> f32 {
        tensor.to_scalar::<f32>().unwrap()
    }
}

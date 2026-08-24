//! Environment-independent wrappers for [`crate::gym::Gym`].
//!
//! The modules follow Gymnasium's conventional categories: observation
//! transformations, reward transformations, and episode-control wrappers.

use candle_core::Tensor;

use crate::{
    gym::{MultiGym, MultiGymStepInfo},
    spaces::Space,
};

/// An error raised while mapping tensors around a multi-environment gym.
#[derive(Debug, thiserror::Error)]
pub enum TensorMapMultiGymError<E> {
    /// An error returned by the wrapped environment.
    #[error("wrapped gym error: {0}")]
    Gym(#[source] E),
    /// An error returned by either tensor-mapping function.
    #[error("tensor mapping failed: {0}")]
    Candle(#[source] candle_core::Error),
}

/// Maps every tensor crossing a [`MultiGym`] boundary.
///
/// `map_input` transforms batched actions before the inner environment sees
/// them. `map_output` transforms reset observations and every tensor returned
/// by a step: states, rewards, and terminal states.
pub struct TensorMapMultiGymWrapper<G, FInput, FOutput> {
    gym: G,
    map_input: FInput,
    map_output: FOutput,
}

impl<G, FInput, FOutput> TensorMapMultiGymWrapper<G, FInput, FOutput> {
    /// Creates a wrapper with input and output tensor transformations.
    pub fn new(gym: G, map_input: FInput, map_output: FOutput) -> Self {
        Self {
            gym,
            map_input,
            map_output,
        }
    }

    /// Returns a shared reference to the wrapped environment.
    pub fn inner(&self) -> &G {
        &self.gym
    }

    /// Returns a mutable reference to the wrapped environment.
    pub fn inner_mut(&mut self) -> &mut G {
        &mut self.gym
    }

    /// Unwraps and returns the inner environment.
    pub fn into_inner(self) -> G {
        self.gym
    }
}

impl<G, FInput, FOutput, I> MultiGym<I> for TensorMapMultiGymWrapper<G, FInput, FOutput>
where
    G: MultiGym<I>,
    FInput: FnMut(Tensor) -> Result<Tensor, candle_core::Error>,
    FOutput: FnMut(Tensor) -> Result<Tensor, candle_core::Error>,
{
    type Error = TensorMapMultiGymError<G::Error>;
    type SpaceError = G::SpaceError;

    /// Maps batched `action` shaped `[num_envs, ...action_shape]` before
    /// stepping the inner environment, then maps every tensor in the returned
    /// transition.
    fn step(&mut self, action: Tensor) -> Result<MultiGymStepInfo<I>, Self::Error> {
        let action = (self.map_input)(action).map_err(TensorMapMultiGymError::Candle)?;
        let mut step = self.gym.step(action).map_err(TensorMapMultiGymError::Gym)?;
        step.states = (self.map_output)(step.states).map_err(TensorMapMultiGymError::Candle)?;
        step.rewards = (self.map_output)(step.rewards).map_err(TensorMapMultiGymError::Candle)?;
        for state in step.terminal_states.iter_mut().flatten() {
            *state = (self.map_output)(state.clone()).map_err(TensorMapMultiGymError::Candle)?;
        }
        Ok(step)
    }

    fn observation_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
        self.gym.observation_space()
    }

    fn action_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
        self.gym.action_space()
    }

    fn num_envs(&self) -> usize {
        self.gym.num_envs()
    }

    fn reset(&mut self) -> Result<Tensor, Self::Error> {
        let observation = self.gym.reset().map_err(TensorMapMultiGymError::Gym)?;
        (self.map_output)(observation).map_err(TensorMapMultiGymError::Candle)
    }
}

pub mod info;
pub mod normalize;
pub mod observation;
pub mod reward;
pub mod time_limit;

pub use info::{
    EpisodeStatistics, EpisodeStatisticsInfo, RawRewardInfo, RecordEpisodeStatisticsGym,
    RecordRawRewardGym,
};
pub use normalize::{NormalizeObservationGym, NormalizeObservationGymError, NormalizeRewardGym};
pub use observation::{FrameStackGym, FrameStackGymError, MaxAndSkipGym, MaxAndSkipGymError};
pub use reward::{ClipRewardGym, ClipRewardGymError};
pub use time_limit::TimeLimitGym;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spaces::BoxSpace;
    use candle_core::Device;

    struct TensorMapTestGym;

    impl MultiGym for TensorMapTestGym {
        type Error = candle_core::Error;
        type SpaceError = candle_core::Error;

        /// Steps both test environments with an action tensor of shape `[2]`.
        fn step(&mut self, action: Tensor) -> Result<MultiGymStepInfo, Self::Error> {
            let actions = action.flatten_all()?.to_vec1::<f32>()?;
            let states = Tensor::from_vec(
                vec![2.0, 0.0, actions[0], 2.0, 1.0, actions[1]],
                (2, 3),
                &Device::Cpu,
            )?;
            Ok(MultiGymStepInfo {
                rewards: Tensor::from_vec(actions, 2, &Device::Cpu)?,
                terminal_states: vec![None, Some(states.get(1)?)],
                states,
                infos: vec![(), ()],
                dones: vec![false, true],
                truncateds: vec![false, false],
            })
        }

        fn observation_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
            Box::new(BoxSpace::new_unbounded(vec![3], &Device::Cpu))
        }

        fn action_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
            Box::new(BoxSpace::new_unbounded(vec![1], &Device::Cpu))
        }

        fn num_envs(&self) -> usize {
            2
        }

        fn reset(&mut self) -> Result<Tensor, Self::Error> {
            Tensor::from_vec(
                vec![2.0f32, 0.0, -1.0, 2.0, 1.0, -1.0],
                (2, 3),
                &Device::Cpu,
            )
        }
    }

    #[test]
    fn tensor_map_multi_gym_maps_every_boundary_tensor() {
        let mut gym = TensorMapMultiGymWrapper::new(
            TensorMapTestGym,
            |tensor: Tensor| tensor * 2.0,
            |tensor: Tensor| tensor + 10.0,
        );

        assert_eq!(
            gym.reset().unwrap().to_vec2::<f32>().unwrap(),
            vec![vec![12.0, 10.0, 9.0], vec![12.0, 11.0, 9.0]]
        );

        let actions = Tensor::from_vec(vec![1.0f32, 2.0], (2, 1), &Device::Cpu).unwrap();
        let step = gym.step(actions).unwrap();
        assert_eq!(
            step.states.to_vec2::<f32>().unwrap(),
            vec![vec![12.0, 10.0, 12.0], vec![12.0, 11.0, 14.0]]
        );
        assert_eq!(step.rewards.to_vec1::<f32>().unwrap(), vec![12.0, 14.0]);
        assert_eq!(
            step.terminal_states[1]
                .as_ref()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![12.0, 11.0, 14.0]
        );
    }
}

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

        /// Steps with one scalar discrete action shaped `[]`.
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

    /// Reads one scalar tensor shaped `[]`.
    pub(super) fn scalar(tensor: &Tensor) -> f32 {
        tensor.to_scalar::<f32>().unwrap()
    }
}

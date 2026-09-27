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

/// Maps batched actions before stepping the wrapped environment.
/// Reset and all outputs pass through unchanged. Space descriptions are forwarded.
pub struct InputMapMultiGymWrapper<G, F> {
    gym: G,
    map_input: F,
}

impl<G, F> InputMapMultiGymWrapper<G, F> {
    /// Creates a wrapper that transforms batched actions before dispatch.
    pub fn new<I>(gym: G, map_input: F) -> Self
    where
        G: MultiGym<I>,
        F: FnMut(Tensor) -> candle_core::Result<Tensor>,
    {
        Self { gym, map_input }
    }
    pub fn inner(&self) -> &G {
        &self.gym
    }
    pub fn inner_mut(&mut self) -> &mut G {
        &mut self.gym
    }
    pub fn into_inner(self) -> G {
        self.gym
    }
}

impl<G, F, I> MultiGym<I> for InputMapMultiGymWrapper<G, F>
where
    G: MultiGym<I>,
    F: FnMut(Tensor) -> candle_core::Result<Tensor>,
{
    type Error = TensorMapMultiGymError<G::Error>;
    type SpaceError = G::SpaceError;

    /// Maps `action` shaped `[num_envs, ...action_shape]` before dispatch.
    fn step(&mut self, action: Tensor) -> Result<MultiGymStepInfo<I>, Self::Error> {
        let action = (self.map_input)(action).map_err(TensorMapMultiGymError::Candle)?;
        self.gym.step(action).map_err(TensorMapMultiGymError::Gym)
    }
    fn reset(&mut self) -> Result<Tensor, Self::Error> {
        self.gym.reset().map_err(TensorMapMultiGymError::Gym)
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
}

/// Maps each reset or step output with one callback invocation.
///
/// Separate callbacks map reset observations and the complete step result.
/// Actions and space descriptions are forwarded unchanged.
/// Observation space must be kept consistent with the inner gym.
pub struct OutputMapMultiGymWrapper<G, FReset, FStep> {
    gym: G,
    map_reset: FReset,
    map_step: FStep,
}

impl<G, FReset, FStep> OutputMapMultiGymWrapper<G, FReset, FStep> {
    /// Creates a wrapper that transforms reset observations and step results.
    pub fn new<I>(gym: G, map_reset: FReset, map_step: FStep) -> Self
    where
        G: MultiGym<I>,
        FReset: FnMut(Tensor) -> candle_core::Result<Tensor>,
        FStep: FnMut(MultiGymStepInfo<I>) -> candle_core::Result<MultiGymStepInfo<I>>,
    {
        Self {
            gym,
            map_reset,
            map_step,
        }
    }
    pub fn inner(&self) -> &G {
        &self.gym
    }
    pub fn inner_mut(&mut self) -> &mut G {
        &mut self.gym
    }
    pub fn into_inner(self) -> G {
        self.gym
    }
}

impl<G, FReset, FStep, I> MultiGym<I> for OutputMapMultiGymWrapper<G, FReset, FStep>
where
    G: MultiGym<I>,
    FReset: FnMut(Tensor) -> candle_core::Result<Tensor>,
    FStep: FnMut(MultiGymStepInfo<I>) -> candle_core::Result<MultiGymStepInfo<I>>,
{
    type Error = TensorMapMultiGymError<G::Error>;
    type SpaceError = G::SpaceError;

    /// Forwards `action` shaped `[num_envs, ...action_shape]` and maps the full output.
    fn step(&mut self, action: Tensor) -> Result<MultiGymStepInfo<I>, Self::Error> {
        let step = self.gym.step(action).map_err(TensorMapMultiGymError::Gym)?;
        (self.map_step)(step).map_err(TensorMapMultiGymError::Candle)
    }
    fn reset(&mut self) -> Result<Tensor, Self::Error> {
        let states = self.gym.reset().map_err(TensorMapMultiGymError::Gym)?;
        (self.map_reset)(states).map_err(TensorMapMultiGymError::Candle)
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
}

/// Transfers current and terminal observations together, then restores their shapes.
fn transfer_batched_observations<I>(
    step: &mut MultiGymStepInfo<I>,
    device: &candle_core::Device,
) -> candle_core::Result<()> {
    let count = step.states.dim(0)?;
    let mut parts = vec![step.states.clone()];
    for state in step.terminal_states.iter().flatten() {
        parts.push(state.unsqueeze(0)?);
    }
    let terminal_count = parts.len() - 1;
    let packed = if terminal_count == 0 {
        parts.remove(0)
    } else {
        Tensor::cat(&parts, 0)?
    };
    let packed = packed.to_device(device)?;
    step.states = packed.narrow(0, 0, count)?;
    for (offset, state) in step.terminal_states.iter_mut().flatten().enumerate() {
        *state = packed.narrow(0, count + offset, 1)?.squeeze(0)?;
    }
    Ok(())
}

/// Transfers actions and observations at the batch boundary. CPU simulators
/// should produce CPU observations; their RNG may remain on a separate device.
pub struct DeviceMultiGymWrapper<G> {
    gym: G,
    environment_device: candle_core::Device,
    agent_device: candle_core::Device,
}

impl<G> DeviceMultiGymWrapper<G> {
    pub fn new(
        gym: G,
        environment_device: candle_core::Device,
        agent_device: candle_core::Device,
    ) -> Self {
        Self {
            gym,
            environment_device,
            agent_device,
        }
    }

    pub fn inner(&self) -> &G {
        &self.gym
    }
    pub fn inner_mut(&mut self) -> &mut G {
        &mut self.gym
    }
    pub fn into_inner(self) -> G {
        self.gym
    }
}

impl<G: MultiGym<I>, I> MultiGym<I> for DeviceMultiGymWrapper<G> {
    type Error = TensorMapMultiGymError<G::Error>;
    type SpaceError = G::SpaceError;

    /// Transfers actions `[num_envs, ...action_shape]` once before dispatch.
    fn step(&mut self, action: Tensor) -> Result<MultiGymStepInfo<I>, Self::Error> {
        let action = action
            .to_device(&self.environment_device)
            .map_err(TensorMapMultiGymError::Candle)?;
        let mut step = self.gym.step(action).map_err(TensorMapMultiGymError::Gym)?;
        let transfer = |step: &mut MultiGymStepInfo<I>| -> candle_core::Result<()> {
            // Include terminal observations in the same upload as reset states.
            // Skip packing entirely when no transfer is necessary.
            if !step.states.device().same_device(&self.agent_device) {
                transfer_batched_observations(step, &self.agent_device)?;
            }
            step.rewards = step.rewards.to_device(&self.agent_device)?;
            Ok(())
        };
        transfer(&mut step).map_err(TensorMapMultiGymError::Candle)?;
        Ok(step)
    }

    fn reset(&mut self) -> Result<Tensor, Self::Error> {
        self.gym
            .reset()
            .map_err(TensorMapMultiGymError::Gym)?
            .to_device(&self.agent_device)
            .map_err(TensorMapMultiGymError::Candle)
    }
    fn num_envs(&self) -> usize {
        self.gym.num_envs()
    }
    fn observation_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
        self.gym.observation_space()
    }
    fn action_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
        self.gym.action_space()
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
    fn output_mapping_receives_one_complete_output_per_operation() {
        let calls = std::cell::RefCell::new(Vec::new());
        let input = InputMapMultiGymWrapper::new(TensorMapTestGym, |action: Tensor| action * 2.0);
        let mut gym = OutputMapMultiGymWrapper::new(
            input,
            |states: Tensor| {
                calls.borrow_mut().push("reset");
                states + 10.0
            },
            |mut step: MultiGymStepInfo| {
                calls.borrow_mut().push("step");
                step.states = (step.states + 10.0)?;
                for state in step.terminal_states.iter_mut().flatten() {
                    *state = (state.clone() + 10.0)?;
                }
                Ok(step)
            },
        );
        assert_eq!(
            gym.reset().unwrap().to_vec2::<f32>().unwrap(),
            vec![vec![12.0, 10.0, 9.0], vec![12.0, 11.0, 9.0]]
        );
        let step = gym
            .step(Tensor::new(&[1.0f32, 2.0], &Device::Cpu).unwrap())
            .unwrap();
        assert_eq!(
            step.states.to_vec2::<f32>().unwrap(),
            vec![vec![12.0, 10.0, 12.0], vec![12.0, 11.0, 14.0]]
        );
        assert_eq!(step.rewards.to_vec1::<f32>().unwrap(), vec![2.0, 4.0]);
        assert_eq!(step.dones, vec![false, true]);
        assert_eq!(step.truncateds, vec![false, false]);
        assert_eq!(step.infos, vec![(), ()]);
        assert!(step.terminal_states[0].is_none());
        assert_eq!(
            step.terminal_states[1]
                .as_ref()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![12.0, 11.0, 14.0]
        );
        assert_eq!(*calls.borrow(), vec!["reset", "step"]);
    }

    #[test]
    fn output_mapping_propagates_callback_errors() {
        let mut gym = OutputMapMultiGymWrapper::new(
            TensorMapTestGym,
            |_| Err(candle_core::Error::Msg("reset mapping failed".into())),
            |_| Err(candle_core::Error::Msg("step mapping failed".into())),
        );
        assert!(
            matches!(gym.reset(), Err(TensorMapMultiGymError::Candle(error))
            if error.to_string() == "reset mapping failed")
        );
        assert!(
            matches!(gym.step(Tensor::new(&[1.0f32, 2.0], &Device::Cpu).unwrap()),
            Err(TensorMapMultiGymError::Candle(error)) if error.to_string() == "step mapping failed")
        );
    }

    #[test]
    fn device_wrapper_transfers_reset_step_and_terminal_observations() {
        let device = Device::cuda_if_available(0).unwrap();
        let mut gym =
            super::DeviceMultiGymWrapper::new(TensorMapTestGym, Device::Cpu, device.clone());
        assert!(gym.reset().unwrap().device().same_device(&device));
        let step = gym
            .step(Tensor::new(&[1.0f32, 2.0], &device).unwrap())
            .unwrap();
        assert!(step.states.device().same_device(&device));
        assert!(step.rewards.device().same_device(&device));
        assert!(
            step.terminal_states[1]
                .as_ref()
                .unwrap()
                .device()
                .same_device(&device)
        );
        assert_eq!(
            step.transition_next_states()
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            vec![vec![2.0, 0.0, 1.0], vec![2.0, 1.0, 2.0]]
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

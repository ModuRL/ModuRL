//! Environment-independent wrappers for [`crate::gym::Gym`].
//!
//! The modules group observation transformations, reward transformations, and episode controls.

use crate::tensor_rank::{NextRank, PrevRank};
use burn::tensor::{Device, Slice, Tensor, kind::Basic};
use std::convert::Infallible;

use crate::{
    gym::{MultiGym, MultiGymStepInfo},
    spaces::{ActionSpace, ObservationSpace},
};

/// Reports an environment failure or a caller-defined mapping failure.
#[derive(Debug, thiserror::Error)]
pub enum TensorMapMultiGymError<E, M = Infallible> {
    #[error("wrapped gym error: {0}")]
    Gym(#[source] E),
    #[error("tensor mapping failed: {0}")]
    Mapping(#[source] M),
}

/// Maps batched actions before dispatch. Reset, outputs, and space descriptions pass through unchanged.
pub struct InputMapMultiGymWrapper<G, F> {
    gym: G,
    map_input: F,
}

impl<G, F> InputMapMultiGymWrapper<G, F> {
    /// Creates a callback for rank-`A` actions `[num_envs, ...action_shape]` in the inner action kind.
    /// The callback must preserve the layout and meet the inner gym's dtype and device requirements.
    pub fn new<I, E, const O: usize, const A: usize>(gym: G, map_input: F) -> Self
    where
        G: MultiGym<I, O, A>,
        Tensor<O, <G::ObservationSpace as ObservationSpace<O>>::Kind>: PrevRank,
        F: FnMut(
            Tensor<A, <G::ActionSpace as ActionSpace<A>>::Kind>,
        ) -> Result<Tensor<A, <G::ActionSpace as ActionSpace<A>>::Kind>, E>,
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

impl<G, F, I, E, const O: usize, const A: usize, const U: usize, K: Basic, AK: Basic>
    MultiGym<I, O, A> for InputMapMultiGymWrapper<G, F>
where
    G: MultiGym<I, O, A>,
    G::ObservationSpace: ObservationSpace<O, Kind = K>,
    G::ActionSpace: ActionSpace<A, Kind = AK>,
    Tensor<O, K>: PrevRank<Prev = Tensor<U, K>>,
    Tensor<U, K>: NextRank<Next = Tensor<O, K>>,
    F: FnMut(Tensor<A, AK>) -> Result<Tensor<A, AK>, E>,
{
    type Error = TensorMapMultiGymError<G::Error, E>;
    type ObservationSpace = G::ObservationSpace;
    type ActionSpace = G::ActionSpace;

    /// Maps rank-`A` actions `[num_envs, ...action_shape]` and returns rank-`O` observations `[num_envs, ...observation_shape]`.
    /// Terminal observations keep unbatched rank `U`; rewards keep `[num_envs]`. Output kinds, dtypes, and devices remain unchanged.
    fn step(
        &mut self,
        action: Tensor<A, AK>,
    ) -> Result<MultiGymStepInfo<I, Tensor<U, K>>, Self::Error> {
        let action = (self.map_input)(action).map_err(TensorMapMultiGymError::Mapping)?;
        self.gym.step(action).map_err(TensorMapMultiGymError::Gym)
    }

    /// Returns unchanged rank-`O` observations `[num_envs, ...observation_shape]` in the inner kind, dtype, and device.
    fn reset(&mut self) -> Result<Tensor<O, K>, Self::Error> {
        self.gym.reset().map_err(TensorMapMultiGymError::Gym)
    }

    fn observation_space(&self) -> Self::ObservationSpace {
        self.gym.observation_space()
    }

    fn action_space(&self) -> Self::ActionSpace {
        self.gym.action_space()
    }

    fn num_envs(&self) -> usize {
        self.gym.num_envs()
    }
}

/// Maps each reset or complete step output with one callback invocation.
/// Actions and space descriptions pass through unchanged. Callbacks must keep observations consistent with the declared space.
pub struct OutputMapMultiGymWrapper<G, FReset, FStep> {
    gym: G,
    map_reset: FReset,
    map_step: FStep,
}

impl<G, FReset, FStep> OutputMapMultiGymWrapper<G, FReset, FStep> {
    /// Creates callbacks for rank-`O` observations `[num_envs, ...observation_shape]` and complete step results.
    /// Step callbacks also receive rewards `[num_envs]` and unbatched rank-`U` terminal observations.
    /// Callbacks must preserve layouts and native kinds; they control output dtypes and devices.
    pub fn new<I, E, const O: usize, const A: usize, const U: usize>(
        gym: G,
        map_reset: FReset,
        map_step: FStep,
    ) -> Self
    where
        G: MultiGym<I, O, A>,
        Tensor<O, <G::ObservationSpace as ObservationSpace<O>>::Kind>:
            PrevRank<Prev = Tensor<U, <G::ObservationSpace as ObservationSpace<O>>::Kind>>,
        Tensor<U, <G::ObservationSpace as ObservationSpace<O>>::Kind>:
            NextRank<Next = Tensor<O, <G::ObservationSpace as ObservationSpace<O>>::Kind>>,
        FReset: FnMut(
            Tensor<O, <G::ObservationSpace as ObservationSpace<O>>::Kind>,
        )
            -> Result<Tensor<O, <G::ObservationSpace as ObservationSpace<O>>::Kind>, E>,
        FStep: FnMut(
            MultiGymStepInfo<I, Tensor<U, <G::ObservationSpace as ObservationSpace<O>>::Kind>>,
        ) -> Result<
            MultiGymStepInfo<I, Tensor<U, <G::ObservationSpace as ObservationSpace<O>>::Kind>>,
            E,
        >,
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

impl<G, FReset, FStep, I, E, const O: usize, const A: usize, const U: usize, K: Basic, AK: Basic>
    MultiGym<I, O, A> for OutputMapMultiGymWrapper<G, FReset, FStep>
where
    G: MultiGym<I, O, A>,
    G::ObservationSpace: ObservationSpace<O, Kind = K>,
    G::ActionSpace: ActionSpace<A, Kind = AK>,
    Tensor<O, K>: PrevRank<Prev = Tensor<U, K>>,
    Tensor<U, K>: NextRank<Next = Tensor<O, K>>,
    FReset: FnMut(Tensor<O, K>) -> Result<Tensor<O, K>, E>,
    FStep: FnMut(MultiGymStepInfo<I, Tensor<U, K>>) -> Result<MultiGymStepInfo<I, Tensor<U, K>>, E>,
{
    type Error = TensorMapMultiGymError<G::Error, E>;
    type ObservationSpace = G::ObservationSpace;
    type ActionSpace = G::ActionSpace;

    /// Forwards rank-`A` actions `[num_envs, ...action_shape]` and maps the complete step result.
    /// Observations retain rank `O`, rewards retain rank 1, and unbatched terminal observations retain rank `U`.
    fn step(
        &mut self,
        action: Tensor<A, AK>,
    ) -> Result<MultiGymStepInfo<I, Tensor<U, K>>, Self::Error> {
        let step = self.gym.step(action).map_err(TensorMapMultiGymError::Gym)?;
        (self.map_step)(step).map_err(TensorMapMultiGymError::Mapping)
    }

    /// Maps rank-`O` reset observations `[num_envs, ...observation_shape]` without changing the layout or kind.
    fn reset(&mut self) -> Result<Tensor<O, K>, Self::Error> {
        let observations = self.gym.reset().map_err(TensorMapMultiGymError::Gym)?;
        (self.map_reset)(observations).map_err(TensorMapMultiGymError::Mapping)
    }

    fn observation_space(&self) -> Self::ObservationSpace {
        self.gym.observation_space()
    }

    fn action_space(&self) -> Self::ActionSpace {
        self.gym.action_space()
    }

    fn num_envs(&self) -> usize {
        self.gym.num_envs()
    }
}

/// Transfers observations `[num_envs, ...observation_shape]` and unbatched rank-`U` terminal observations in one packed tensor.
/// Adds and removes the terminal batch axis, preserving kind, dtype, item axes, and gradients; scalar values use `[1]`.
fn transfer_batched_observations<I, const O: usize, K: Basic, const U: usize>(
    step: &mut MultiGymStepInfo<I, Tensor<U, K>>,
    device: &Device,
) where
    Tensor<U, K>: NextRank<Next = Tensor<O, K>>,
{
    let count = step.observations.dims()[0];
    let terminal_count = step.terminal_observations.iter().flatten().count();
    let mut parts = Vec::with_capacity(terminal_count + 1);
    parts.push(step.observations.clone());
    for observation in step.terminal_observations.iter().flatten() {
        let mut shape = [1; O];
        shape[1..].copy_from_slice(&observation.dims());
        parts.push(observation.clone().reshape(shape));
    }
    // Pack terminal rows with current observations so one transfer supplies both next-observation layouts.
    let packed = if terminal_count == 0 {
        parts.remove(0)
    } else {
        Tensor::cat(parts, 0)
    }
    .to_device(device);
    step.observations = packed.clone().slice([Slice::from(0..count)]);
    for (offset, observation) in step.terminal_observations.iter_mut().flatten().enumerate() {
        *observation = packed
            .clone()
            .slice([Slice::from(count + offset..count + offset + 1)])
            .reshape(observation.dims());
    }
}

/// Transfers actions to the environment device and observations and rewards to the agent device.
/// Space descriptions stay on the inner gym's device. CPU simulators can keep their own RNG placement.
pub struct DeviceMultiGymWrapper<G> {
    gym: G,
    environment_device: Device,
    agent_device: Device,
}

impl<G> DeviceMultiGymWrapper<G> {
    pub fn new(gym: G, environment_device: Device, agent_device: Device) -> Self {
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

impl<G, I, const O: usize, const A: usize, const U: usize, K: Basic, AK: Basic> MultiGym<I, O, A>
    for DeviceMultiGymWrapper<G>
where
    G: MultiGym<I, O, A>,
    G::ObservationSpace: ObservationSpace<O, Kind = K>,
    G::ActionSpace: ActionSpace<A, Kind = AK>,
    Tensor<O, K>: PrevRank<Prev = Tensor<U, K>>,
    Tensor<U, K>: NextRank<Next = Tensor<O, K>>,
{
    type Error = TensorMapMultiGymError<G::Error>;
    type ObservationSpace = G::ObservationSpace;
    type ActionSpace = G::ActionSpace;

    /// Transfers rank-`A` actions `[num_envs, ...action_shape]` to the environment device.
    /// Returns rank-`O` observations, rewards `[num_envs]`, and unbatched rank-`U` terminal observations on the agent device.
    /// Preserves tensor kinds, dtypes, item axes, and gradients.
    fn step(
        &mut self,
        action: Tensor<A, AK>,
    ) -> Result<MultiGymStepInfo<I, Tensor<U, K>>, Self::Error> {
        let action = action.to_device(&self.environment_device);
        let mut step = self.gym.step(action).map_err(TensorMapMultiGymError::Gym)?;
        if step.observations.device() != self.agent_device {
            transfer_batched_observations(&mut step, &self.agent_device);
        }
        step.rewards = step.rewards.to_device(&self.agent_device);
        Ok(step)
    }

    /// Transfers rank-`O` reset observations `[num_envs, ...observation_shape]` to the agent device, preserving kind and dtype.
    fn reset(&mut self) -> Result<Tensor<O, K>, Self::Error> {
        Ok(self
            .gym
            .reset()
            .map_err(TensorMapMultiGymError::Gym)?
            .to_device(&self.agent_device))
    }

    fn num_envs(&self) -> usize {
        self.gym.num_envs()
    }

    fn observation_space(&self) -> Self::ObservationSpace {
        self.gym.observation_space()
    }

    fn action_space(&self) -> Self::ActionSpace {
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
    use burn::tensor::{Bool, DType, Int};

    struct TensorMapTestGym;

    impl MultiGym for TensorMapTestGym {
        type Error = Infallible;
        type ObservationSpace = BoxSpace<2>;
        type ActionSpace = BoxSpace<2>;

        /// Returns Float observations `[2, 3]`, rewards `[2]`, and one terminal observation `[3]` from scalar actions `[2, 1]`.
        fn step(&mut self, action: Tensor<2>) -> Result<MultiGymStepInfo, Self::Error> {
            let actions = action.into_data().try_to_vec::<f32>().unwrap();
            let observations = Tensor::from_data(
                [[2.0, 0.0, actions[0]], [2.0, 1.0, actions[1]]],
                &Device::flex(),
            );
            Ok(MultiGymStepInfo {
                rewards: Tensor::from_data(actions.as_slice(), &Device::flex()),
                terminal_observations: vec![
                    None,
                    Some(observations.clone().slice([Slice::from(1..2)]).reshape([3])),
                ],
                observations,
                infos: vec![(), ()],
                dones: vec![false, true],
                truncateds: vec![false, false],
            })
        }

        fn observation_space(&self) -> Self::ObservationSpace {
            BoxSpace::new_unbounded([1, 3], &Device::flex())
        }

        fn action_space(&self) -> Self::ActionSpace {
            BoxSpace::new_unbounded([1, 1], &Device::flex())
        }

        fn num_envs(&self) -> usize {
            2
        }

        /// Returns Float reset observations `[2, 3]` on the fixture device.
        fn reset(&mut self) -> Result<Tensor<2>, Self::Error> {
            Ok(Tensor::from_data(
                [[2.0f32, 0.0, -1.0], [2.0, 1.0, -1.0]],
                &Device::flex(),
            ))
        }
    }

    #[derive(Debug, PartialEq, thiserror::Error)]
    #[error("mapping failed: {0}")]
    struct MappingError(&'static str);

    #[test]
    fn output_mapping_receives_one_complete_output_per_operation() {
        let calls = std::cell::RefCell::new(Vec::new());
        let input = InputMapMultiGymWrapper::new(TensorMapTestGym, |action: Tensor<2>| {
            Ok::<_, Infallible>(action * 2.0)
        });
        let mut gym = OutputMapMultiGymWrapper::new(
            input,
            |observations: Tensor<2>| {
                calls.borrow_mut().push("reset");
                Ok::<_, Infallible>(observations + 10.0)
            },
            |mut step: MultiGymStepInfo| {
                calls.borrow_mut().push("step");
                step.observations = step.observations + 10.0;
                for observation in step.terminal_observations.iter_mut().flatten() {
                    *observation = observation.clone() + 10.0;
                }
                Ok(step)
            },
        );
        assert_eq!(
            gym.reset()
                .unwrap()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            vec![12.0, 10.0, 9.0, 12.0, 11.0, 9.0]
        );
        let step = gym
            .step(Tensor::from_data([[1.0f32], [2.0]], &Device::flex()))
            .unwrap();
        assert_eq!(
            step.observations.into_data().try_to_vec::<f32>().unwrap(),
            vec![12.0, 10.0, 12.0, 12.0, 11.0, 14.0]
        );
        assert_eq!(
            step.rewards.into_data().try_to_vec::<f32>().unwrap(),
            vec![2.0, 4.0]
        );
        assert_eq!(step.dones, vec![false, true]);
        assert_eq!(step.truncateds, vec![false, false]);
        assert_eq!(step.infos, vec![(), ()]);
        assert!(step.terminal_observations[0].is_none());
        assert_eq!(
            step.terminal_observations[1]
                .clone()
                .unwrap()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            vec![12.0, 11.0, 14.0]
        );
        assert_eq!(*calls.borrow(), vec!["reset", "step"]);
    }

    #[test]
    fn mappings_propagate_callback_errors() {
        let mut gym = OutputMapMultiGymWrapper::new(
            TensorMapTestGym,
            |_| Err(MappingError("reset")),
            |_| Err(MappingError("step")),
        );
        assert!(matches!(
            gym.reset(),
            Err(TensorMapMultiGymError::Mapping(MappingError("reset")))
        ));
        assert!(matches!(
            gym.step(Tensor::from_data([[1.0f32], [2.0]], &Device::flex())),
            Err(TensorMapMultiGymError::Mapping(MappingError("step")))
        ));
        let mut gym =
            InputMapMultiGymWrapper::new(TensorMapTestGym, |_| Err(MappingError("input")));
        assert!(matches!(
            gym.step(Tensor::from_data([[1.0f32], [2.0]], &Device::flex())),
            Err(TensorMapMultiGymError::Mapping(MappingError("input")))
        ));
    }

    #[test]
    fn device_wrapper_preserves_reset_step_and_terminal_observations() {
        let device = Device::flex();
        let mut gym = DeviceMultiGymWrapper::new(TensorMapTestGym, device.clone(), device.clone());
        assert_eq!(gym.reset().unwrap().device(), device);
        let step = gym
            .step(Tensor::from_data([[1.0f32], [2.0]], &device))
            .unwrap();
        assert_eq!(step.observations.device(), device);
        assert_eq!(step.rewards.device(), device);
        assert_eq!(
            step.terminal_observations[1].as_ref().unwrap().device(),
            device
        );
        assert_eq!(
            step.transition_next_observations()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            vec![2.0, 0.0, 1.0, 2.0, 1.0, 2.0]
        );
    }

    #[test]
    fn packed_transfer_preserves_scalar_integer_and_boolean_layouts() {
        let device = Device::flex();
        let mut integers = MultiGymStepInfo::<(), Tensor<1, Int>> {
            observations: Tensor::<2, Int>::from_data([[4i32], [5]], &device),
            rewards: Tensor::zeros([2], &device),
            infos: vec![(), ()],
            dones: vec![false, true],
            truncateds: vec![false, false],
            terminal_observations: vec![None, Some(Tensor::from_data([8i32], &device))],
        };
        transfer_batched_observations(&mut integers, &device);
        assert_eq!(integers.observations.dims(), [2, 1]);
        assert_eq!(
            integers.terminal_observations[1].as_ref().unwrap().dims(),
            [1]
        );
        assert_eq!(
            integers
                .transition_next_observations()
                .into_data()
                .try_to_vec::<i32>()
                .unwrap(),
            vec![4, 8]
        );
        let mut booleans = MultiGymStepInfo::<(), Tensor<1, Bool>> {
            observations: Tensor::<2, Bool>::from_data([[true, false], [false, true]], &device),
            rewards: Tensor::zeros([2], &device),
            infos: vec![(), ()],
            dones: vec![false, true],
            truncateds: vec![false, false],
            terminal_observations: vec![None, Some(Tensor::from_data([true, true], &device))],
        };
        transfer_batched_observations(&mut booleans, &device);
        assert_eq!(booleans.observations.dims(), [2, 2]);
        assert_eq!(
            booleans.terminal_observations[1].as_ref().unwrap().dims(),
            [2]
        );
        assert_eq!(
            booleans
                .transition_next_observations()
                .into_data()
                .try_to_vec::<bool>()
                .unwrap(),
            vec![true, false, true, true]
        );
    }

    #[test]
    fn packed_transfer_preserves_f64_and_handles_no_terminal_observations() {
        let device = Device::flex();
        let mut step = TensorMapTestGym
            .step(Tensor::zeros([2, 1], &device))
            .unwrap();
        step.observations = step.observations.cast(DType::F64);
        for observation in step.terminal_observations.iter_mut().flatten() {
            *observation = observation.clone().cast(DType::F64);
        }
        transfer_batched_observations(&mut step, &device);
        assert_eq!(step.observations.dtype(), DType::F64);
        assert_eq!(
            step.terminal_observations[1].as_ref().unwrap().dtype(),
            DType::F64
        );
        step.terminal_observations = vec![None, None];
        transfer_batched_observations(&mut step, &device);
        assert_eq!(step.observations.dims(), [2, 3]);
    }
}

#[cfg(test)]
mod test_support {
    use std::collections::VecDeque;

    use burn::tensor::{Device, Int, Tensor};
    use std::convert::Infallible;

    use crate::{
        gym::{Gym, ResetInfo, StepInfo},
        spaces::{BoxSpace, Discrete},
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
                device: Device::flex(),
                reset_count: 0,
            }
        }

        /// Creates a transition with one scalar Float observation `[1]`.
        pub(super) fn step(
            observation: f32,
            reward: f32,
            done: bool,
            truncated: bool,
            sequence: u32,
        ) -> StepInfo<TestInfo> {
            StepInfo {
                observation: Tensor::from_data([observation], &Device::flex()),
                reward,
                done,
                truncated,
                info: TestInfo { sequence },
            }
        }
    }

    impl Gym<TestInfo> for TestGym {
        type Error = Infallible;
        type ObservationSpace = BoxSpace<2>;
        type ActionSpace = Discrete;

        /// Accepts one scalar Int action `[1]` and returns a scalar Float observation `[1]` on the fixture device.
        fn step(&mut self, _action: Tensor<1, Int>) -> Result<StepInfo<TestInfo>, Self::Error> {
            Ok(self.steps.pop_front().expect("test step script exhausted"))
        }

        /// Returns one scalar Float observation `[1]` on the fixture device.
        fn reset(&mut self) -> Result<ResetInfo<TestInfo>, Self::Error> {
            self.reset_count += 1;
            Ok(ResetInfo {
                observation: Tensor::from_data([100.0 * self.reset_count as f32], &self.device),
                info: TestInfo { sequence: 0 },
            })
        }

        fn observation_space(&self) -> Self::ObservationSpace {
            BoxSpace::new(
                Tensor::from_data([[-10_000.0f32]], &self.device),
                Tensor::from_data([[10_000.0f32]], &self.device),
            )
        }

        fn action_space(&self) -> Self::ActionSpace {
            Discrete::new(4)
        }
    }

    /// Creates one scalar Int action `[1]` on the fixture device.
    pub(super) fn action() -> Tensor<1, Int> {
        Tensor::from_data([0i32], &Device::flex())
    }

    /// Reads one scalar Float tensor `[1]` to the host.
    pub(super) fn scalar(tensor: &Tensor<1>) -> f32 {
        tensor.clone().into_scalar::<f32>()
    }
}

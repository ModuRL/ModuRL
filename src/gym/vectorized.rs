use super::{Gym, MultiGym, MultiGymStepInfo};
use crate::spaces::{ActionSpace, ObservationSpace};
use crate::tensor_rank::{NextRank, PrevRank};
use burn::tensor::{DType, Slice, Tensor, TensorData, kind::Basic};
use std::marker::PhantomData;

#[derive(Debug, thiserror::Error)]
pub enum VectorizedGymError<E> {
    #[error("at least one environment is required")]
    Empty,
    #[error("single environment failed: {0}")]
    Single(#[source] E),
    #[cfg(feature = "multithreading")]
    #[error("environment worker {environment_index} did not return the expected response")]
    WorkerDisconnected { environment_index: usize },
}

/// Batches independent Gym values by adding an observation batch axis and removing the action batch axis for each environment.
/// Scalars use `[1]` individually and `[num_envs, 1]` when batched.
/// Ended environments reset immediately. Saved terminal observations remain available for transition targets.
/// Reset after an error because some environments may already have advanced.
pub struct VectorizedGymWrapper<G, I = (), const BO: usize = 2, const BA: usize = 2> {
    envs: Vec<G>,
    _info: PhantomData<fn() -> I>,
}

impl<G, I, const BO: usize, const BA: usize, K: Basic, AK: Basic> VectorizedGymWrapper<G, I, BO, BA>
where
    G: Gym<I, BO, BA>,
    G::ObservationSpace: ObservationSpace<BO, Kind = K>,
    G::ActionSpace: ActionSpace<BA, Kind = AK>,
    Tensor<BO, K>: PrevRank,
    Tensor<BA, AK>: PrevRank,
{
    pub fn new(envs: Vec<G>) -> Result<Self, VectorizedGymError<G::Error>> {
        if envs.is_empty() {
            return Err(VectorizedGymError::Empty);
        }
        Ok(Self {
            envs,
            _info: PhantomData,
        })
    }

    pub fn envs(&self) -> &[G] {
        &self.envs
    }

    pub fn envs_mut(&mut self) -> &mut [G] {
        &mut self.envs
    }
}

impl<G, I, const BO: usize, const BA: usize, K: Basic, AK: Basic> From<G>
    for VectorizedGymWrapper<G, I, BO, BA>
where
    G: Gym<I, BO, BA>,
    G::ObservationSpace: ObservationSpace<BO, Kind = K>,
    G::ActionSpace: ActionSpace<BA, Kind = AK>,
    Tensor<BO, K>: PrevRank,
    Tensor<BA, AK>: PrevRank,
{
    fn from(env: G) -> Self {
        Self {
            envs: vec![env],
            _info: PhantomData,
        }
    }
}

impl<G, I, const BO: usize, const BA: usize, K: Basic, AK: Basic> TryFrom<Vec<G>>
    for VectorizedGymWrapper<G, I, BO, BA>
where
    G: Gym<I, BO, BA>,
    G::ObservationSpace: ObservationSpace<BO, Kind = K>,
    G::ActionSpace: ActionSpace<BA, Kind = AK>,
    Tensor<BO, K>: PrevRank,
    Tensor<BA, AK>: PrevRank,
{
    type Error = VectorizedGymError<G::Error>;

    fn try_from(envs: Vec<G>) -> Result<Self, Self::Error> {
        Self::new(envs)
    }
}

impl<G, I, const O: usize, const A: usize, const BO: usize, const BA: usize, K: Basic, AK: Basic>
    MultiGym<I, BO, BA> for VectorizedGymWrapper<G, I, BO, BA>
where
    G: Gym<I, BO, BA>,
    G::ObservationSpace: ObservationSpace<BO, Kind = K>,
    G::ActionSpace: ActionSpace<BA, Kind = AK>,
    Tensor<BO, K>: PrevRank<Prev = Tensor<O, K>>,
    Tensor<O, K>: NextRank<Next = Tensor<BO, K>>,
    Tensor<BA, AK>: PrevRank<Prev = Tensor<A, AK>>,
    Tensor<A, AK>: NextRank<Next = Tensor<BA, AK>>,
{
    type Error = VectorizedGymError<G::Error>;
    type ObservationSpace = G::ObservationSpace;
    type ActionSpace = G::ActionSpace;

    /// Removes the batch axis from rank-`BA` actions `[num_envs, ...action_shape]` before each Gym step, producing rank `A`.
    /// Stacks unbatched rank-`O` observations into rank `BO` and produces F32 rewards `[num_envs, 1]` on the observation device.
    /// Requires `BO = O + 1` and `BA = A + 1`, including scalar values.
    /// Inputs must meet each environment's dtype and device contract. Flags and metadata stay on the host.
    fn step(
        &mut self,
        action: Tensor<BA, AK>,
    ) -> Result<MultiGymStepInfo<I, Tensor<O, K>>, Self::Error> {
        let count = self.envs.len();
        let mut observations = Vec::with_capacity(count);
        let mut rewards = Vec::with_capacity(count);
        let mut infos = Vec::with_capacity(count);
        let mut dones = Vec::with_capacity(count);
        let mut truncateds = Vec::with_capacity(count);
        let mut terminal_observations = Vec::with_capacity(count);
        for (index, env) in self.envs.iter_mut().enumerate() {
            let actions = action.clone().slice([Slice::from(index..index + 1)]);
            let actions = actions.squeeze_dim::<A>(0);
            let mut step = env.step(actions).map_err(VectorizedGymError::Single)?;
            let ended = step.done || step.truncated;
            terminal_observations.push(ended.then(|| step.observation.clone()));
            if ended {
                // Save the transition's terminal observation before replacing it with the reset observation.
                step.observation = env.reset().map_err(VectorizedGymError::Single)?.observation;
            }
            observations.push(step.observation);
            rewards.push(step.reward);
            infos.push(step.info);
            dones.push(step.done);
            truncateds.push(step.truncated);
        }
        let observations: Tensor<BO, K> = Tensor::stack(observations, 0);
        let rewards = Tensor::from_data(
            TensorData::new(rewards, [count, 1]),
            (&observations.device(), DType::F32),
        );
        Ok(MultiGymStepInfo {
            observations,
            rewards,
            infos,
            dones,
            truncateds,
            terminal_observations,
        })
    }

    fn observation_space(&self) -> Self::ObservationSpace {
        self.envs[0].observation_space()
    }

    fn action_space(&self) -> Self::ActionSpace {
        self.envs[0].action_space()
    }

    fn num_envs(&self) -> usize {
        self.envs.len()
    }

    /// Stacks unbatched rank-`O` reset observations into `[num_envs, ...observation_shape]` of rank `BO`.
    /// Requires `BO = O + 1`, including scalar observations `[num_envs, 1]`.
    /// Observation kind, dtype, device, and item dimensions must match across environments. Gradients are preserved.
    fn reset(&mut self) -> Result<Tensor<BO, K>, Self::Error> {
        let mut observations = Vec::with_capacity(self.envs.len());
        for env in &mut self.envs {
            observations.push(env.reset().map_err(VectorizedGymError::Single)?.observation);
        }
        Ok(Tensor::stack(observations, 0))
    }
}

#[cfg(test)]
mod tests {
    use super::super::{ResetInfo, StepInfo, test_support::*};
    use super::*;
    use crate::spaces::{BoxSpace, Discrete};
    use burn::tensor::{Device, Int};

    #[test]
    fn vectorization_restores_unbatched_observations_and_preserves_metadata() {
        let mut env =
            VectorizedGymWrapper::new(vec![CounterEnv::new(0), CounterEnv::new(1)]).unwrap();
        assert_eq!(env.num_envs(), 2);
        assert_eq!(env.envs().len(), 2);
        assert_eq!(
            env.reset()
                .unwrap()
                .into_data()
                .try_to_vec::<f64>()
                .unwrap(),
            vec![0.0, 0.0, 1.0, 0.0]
        );
        let actions = Tensor::<2, Int>::from_data([[3], [4]], &Device::flex());
        let first = env.step(actions.clone()).unwrap();
        assert_eq!(first.observations.dims(), [2, 2]);
        assert_eq!(first.observations.dtype(), DType::F64);
        assert_eq!(first.rewards.dtype(), DType::F32);
        assert_eq!(
            first.rewards.into_data().try_to_vec::<f32>().unwrap(),
            vec![3.0, 4.0]
        );
        assert_eq!(
            first.infos,
            vec![
                TestInfo {
                    id: 0,
                    step: 1,
                    action: 3
                },
                TestInfo {
                    id: 1,
                    step: 1,
                    action: 4
                }
            ]
        );
        assert!(first.terminal_observations.iter().all(Option::is_none));
        let ended = env.step(actions).unwrap();
        assert_eq!(ended.dones, vec![true, false]);
        assert_eq!(ended.truncateds, vec![false, true]);
        assert_eq!(
            ended
                .observations
                .clone()
                .into_data()
                .try_to_vec::<f64>()
                .unwrap(),
            vec![0.0, 0.0, 1.0, 0.0]
        );
        assert!(
            ended
                .terminal_observations
                .iter()
                .flatten()
                .all(|observation| observation.dims() == [2])
        );
        assert_eq!(
            ended
                .transition_next_observations()
                .into_data()
                .try_to_vec::<f64>()
                .unwrap(),
            vec![0.0, 2.0, 1.0, 2.0]
        );
        assert_eq!(
            ended.observations.into_data().try_to_vec::<f64>().unwrap(),
            vec![0.0, 0.0, 1.0, 0.0]
        );
        assert!(env.envs_mut().iter().all(|env| env.step == 0));
    }

    #[test]
    fn single_environment_preserves_reset_metadata_and_from_conversion() {
        let mut single = CounterEnv::new(7);
        assert_eq!(single.reset().unwrap().observation.dims(), [2]);
        assert_eq!(
            single
                .step(Tensor::from_data([3i32], &Device::flex()))
                .unwrap()
                .observation
                .dims(),
            [2]
        );
        assert_eq!(
            single.reset().unwrap().info,
            TestInfo {
                id: 7,
                step: 0,
                action: 0
            }
        );
        let mut env = VectorizedGymWrapper::from(single);
        assert_eq!(env.reset().unwrap().dims(), [1, 2]);
    }

    struct IntegerEnv;

    impl Gym<(), 2, 2> for IntegerEnv {
        type Error = TestError;
        type ObservationSpace = Discrete;
        type ActionSpace = Discrete;

        /// Returns U64 observations `[1]` from Int actions `[1]` on the fixture CPU device.
        fn step(
            &mut self,
            action: Tensor<1, Int>,
        ) -> Result<StepInfo<(), Tensor<1, Int>>, Self::Error> {
            Ok(StepInfo {
                observation: action.cast(DType::U64),
                reward: 1.0,
                done: false,
                truncated: false,
                info: (),
            })
        }

        /// Returns U64 reset observations `[1]`.
        fn reset(&mut self) -> Result<ResetInfo<(), Tensor<1, Int>>, Self::Error> {
            Ok(ResetInfo {
                observation: Tensor::zeros([1], (&Device::flex(), DType::U64)),
                info: (),
            })
        }

        fn observation_space(&self) -> Self::ObservationSpace {
            Discrete::new(10)
        }

        fn action_space(&self) -> Self::ActionSpace {
            Discrete::new(10)
        }
    }

    #[test]
    fn integer_observations_keep_their_kind_and_dtype() {
        let mut env = VectorizedGymWrapper::new(vec![IntegerEnv, IntegerEnv]).unwrap();
        assert_eq!(env.reset().unwrap().dims(), [2, 1]);
        let step = env
            .step(Tensor::from_data([[3i32], [4]], &Device::flex()))
            .unwrap();
        assert_eq!(step.observations.dtype(), DType::U64);
        assert_eq!(
            step.observations.into_data().try_to_vec::<u64>().unwrap(),
            vec![3, 4]
        );
        assert_eq!(step.rewards.dtype(), DType::F32);
    }

    struct ImageEnv;

    impl Gym<(), 4, 2> for ImageEnv {
        type Error = TestError;
        type ObservationSpace = BoxSpace<4>;
        type ActionSpace = Discrete;

        /// Accepts scalar Int actions `[1]` and returns unbatched F64 image observations `[1, 2, 2]`.
        fn step(
            &mut self,
            _action: Tensor<1, Int>,
        ) -> Result<StepInfo<(), Tensor<3>>, Self::Error> {
            Ok(StepInfo {
                observation: Tensor::ones([1, 2, 2], (&Device::flex(), DType::F64)),
                reward: 1.0,
                done: true,
                truncated: false,
                info: (),
            })
        }

        /// Returns unbatched F64 image observations `[1, 2, 2]`.
        fn reset(&mut self) -> Result<ResetInfo<(), Tensor<3>>, Self::Error> {
            Ok(ResetInfo {
                observation: Tensor::zeros([1, 2, 2], (&Device::flex(), DType::F64)),
                info: (),
            })
        }

        fn observation_space(&self) -> Self::ObservationSpace {
            BoxSpace::new_unbounded([1, 1, 2, 2], &Device::flex())
        }

        fn action_space(&self) -> Self::ActionSpace {
            Discrete::new(2)
        }
    }

    #[test]
    fn vectorization_adds_image_batch_axis_and_keeps_terminals_unbatched() {
        let mut env = VectorizedGymWrapper::new(vec![ImageEnv, ImageEnv]).unwrap();
        assert_eq!(env.reset().unwrap().dims(), [2, 1, 2, 2]);
        let step = env
            .step(Tensor::zeros([2, 1], (&Device::flex(), DType::I32)))
            .unwrap();
        assert_eq!(step.observations.dims(), [2, 1, 2, 2]);
        assert_eq!(step.observations.dtype(), DType::F64);
        assert!(
            step.terminal_observations
                .iter()
                .flatten()
                .all(|observation| observation.dims() == [1, 2, 2])
        );
        let targets = step.transition_next_observations();
        assert_eq!(targets.dims(), [2, 1, 2, 2]);
        assert_eq!(
            targets.into_data().try_to_vec::<f64>().unwrap(),
            vec![1.0; 8]
        );
    }

    #[test]
    fn vectorization_removes_action_batch_axis_and_stacks_vector_observations() {
        let device = Device::flex();
        let mut env = VectorizedGymWrapper::new(vec![ContinuousEnv, ContinuousEnv]).unwrap();
        assert_eq!(env.reset().unwrap().dims(), [2, 2]);
        let step = env
            .step(Tensor::from_data(
                [[1.0f64, 2.0], [3.0, 4.0]],
                (&device, DType::F64),
            ))
            .unwrap();
        assert_eq!(step.observations.dims(), [2, 2]);
        assert_eq!(step.observations.dtype(), DType::F64);
        assert!(
            step.terminal_observations
                .iter()
                .flatten()
                .all(|observation| observation.dims() == [2])
        );
        assert_eq!(
            step.transition_next_observations()
                .into_data()
                .try_to_vec::<f64>()
                .unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn constructor_rejects_empty_environment_list() {
        assert!(matches!(
            VectorizedGymWrapper::<CounterEnv, TestInfo>::new(vec![]),
            Err(VectorizedGymError::Empty)
        ));
    }

    #[test]
    fn scalar_batches_preserve_item_axes_terminals_and_gradients() {
        let device = Device::flex().autodiff();
        let actions =
            Tensor::<2>::from_data([[2.0f64], [3.0]], (&device, DType::F64)).require_grad();
        let mut env = VectorizedGymWrapper::new(vec![ScalarEnv, ScalarEnv]).unwrap();
        assert_eq!(env.reset().unwrap().dims(), [2, 1]);
        let step = env.step(actions.clone()).unwrap();
        assert_eq!(step.observations.dims(), [2, 1]);
        assert_eq!(step.rewards.dims(), [2, 1]);
        assert_eq!(step.dones, [true, true]);
        assert_eq!(step.truncateds, [false, false]);
        assert!(
            step.terminal_observations
                .iter()
                .flatten()
                .all(|observation| observation.dims() == [1])
        );
        let targets = step.transition_next_observations();
        assert_eq!(targets.dims(), [2, 1]);
        assert_eq!(targets.dtype(), DType::F64);
        assert_eq!(
            targets.clone().into_data().try_to_vec::<f64>().unwrap(),
            [2.0, 3.0]
        );
        let gradients = targets.sum().backward();
        assert_eq!(
            actions
                .grad(&gradients)
                .unwrap()
                .into_data()
                .try_to_vec::<f64>()
                .unwrap(),
            [1.0, 1.0]
        );
    }

    #[test]
    fn scalar_frame_stacks_keep_frame_and_item_axes_when_batched() {
        use crate::wrappers::FrameStackGym;

        let device = Device::flex();
        let make_env =
            || FrameStackGym::<_>::new(ScalarEnv, 3, BoxSpace::new_unbounded([1, 3, 1], &device));
        let mut env = VectorizedGymWrapper::new(vec![make_env(), make_env()]).unwrap();
        assert_eq!(env.reset().unwrap().dims(), [2, 3, 1]);
        let step = env
            .step(Tensor::from_data([[2.0f64], [3.0]], (&device, DType::F64)))
            .unwrap();
        assert_eq!(step.observations.dims(), [2, 3, 1]);
        assert!(
            step.terminal_observations
                .iter()
                .flatten()
                .all(|observation| observation.dims() == [3, 1])
        );
        assert_eq!(
            step.transition_next_observations()
                .into_data()
                .try_to_vec::<f64>()
                .unwrap(),
            [0.0, 0.0, 2.0, 0.0, 0.0, 3.0]
        );
    }
}

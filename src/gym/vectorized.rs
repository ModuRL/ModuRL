use super::{Gym, MultiGym, MultiGymStepInfo};
use crate::spaces::{ActionSpace, ObservationSpace};
use burn::tensor::{DType, Slice, Tensor, TensorData};
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
/// Scalars use `[1]` individually and `[num_envs]` when batched.
/// Ended environments reset immediately. Saved terminal observations remain available for transition targets.
/// Reset after an error because some environments may already have advanced.
pub struct VectorizedGymWrapper<
    G,
    I = (),
    const O: usize = 1,
    const A: usize = 1,
    const BO: usize = 2,
    const BA: usize = 1,
> where
    G: Gym<I, O, A, BO, BA>,
{
    envs: Vec<G>,
    _info: PhantomData<fn() -> I>,
}

impl<G, I, const O: usize, const A: usize, const BO: usize, const BA: usize>
    VectorizedGymWrapper<G, I, O, A, BO, BA>
where
    G: Gym<I, O, A, BO, BA>,
{
    pub fn new(envs: Vec<G>) -> Result<Self, VectorizedGymError<G::Error>> {
        const {
            assert!(
                BO == O + 1 || (BO == 1 && O == 1),
                "batch observation rank must be single observation rank + 1, except scalar observations"
            );
            assert!(
                BA == A + 1 || (BA == 1 && A == 1),
                "batch action rank must be single action rank + 1, except scalar actions"
            );
        }
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

impl<G, I, const O: usize, const A: usize, const BO: usize, const BA: usize> From<G>
    for VectorizedGymWrapper<G, I, O, A, BO, BA>
where
    G: Gym<I, O, A, BO, BA>,
{
    fn from(env: G) -> Self {
        const {
            assert!(
                BO == O + 1 || (BO == 1 && O == 1),
                "batch observation rank must be single observation rank + 1, except scalar observations"
            );
            assert!(
                BA == A + 1 || (BA == 1 && A == 1),
                "batch action rank must be single action rank + 1, except scalar actions"
            );
        }
        Self {
            envs: vec![env],
            _info: PhantomData,
        }
    }
}

impl<G, I, const O: usize, const A: usize, const BO: usize, const BA: usize> TryFrom<Vec<G>>
    for VectorizedGymWrapper<G, I, O, A, BO, BA>
where
    G: Gym<I, O, A, BO, BA>,
{
    type Error = VectorizedGymError<G::Error>;

    fn try_from(envs: Vec<G>) -> Result<Self, Self::Error> {
        Self::new(envs)
    }
}

impl<G, I, const O: usize, const A: usize, const BO: usize, const BA: usize> MultiGym<I, BO, BA, O>
    for VectorizedGymWrapper<G, I, O, A, BO, BA>
where
    G: Gym<I, O, A, BO, BA>,
{
    type Error = VectorizedGymError<G::Error>;
    type ObservationSpace = G::ObservationSpace;
    type ActionSpace = G::ActionSpace;

    /// Removes the batch axis from rank-`BA` actions `[num_envs, ...action_shape]` before each Gym step, producing rank `A`.
    /// Stacks unbatched rank-`O` observations into rank `BO` and produces F32 rewards `[num_envs]` on the observation device.
    /// Requires `BO = O + 1` and `BA = A + 1`; scalar values instead use single and batch ranks of one.
    /// Inputs must meet each environment's dtype and device contract. Flags and metadata stay on the host.
    fn step(
        &mut self,
        action: Tensor<BA, <G::ActionSpace as ActionSpace<BA>>::Kind>,
    ) -> Result<
        MultiGymStepInfo<I, BO, <G::ObservationSpace as ObservationSpace<BO>>::Kind, O>,
        Self::Error,
    > {
        let count = self.envs.len();
        let mut observations = Vec::with_capacity(count);
        let mut rewards = Vec::with_capacity(count);
        let mut infos = Vec::with_capacity(count);
        let mut dones = Vec::with_capacity(count);
        let mut truncateds = Vec::with_capacity(count);
        let mut terminal_observations = Vec::with_capacity(count);
        for (index, env) in self.envs.iter_mut().enumerate() {
            let actions = action.clone().slice([Slice::from(index..index + 1)]);
            let actions = if const { BA == A + 1 } {
                actions.squeeze_dim::<A>(0)
            } else {
                actions.reshape([1; A])
            };
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
        let observations: Tensor<BO, <G::ObservationSpace as ObservationSpace<BO>>::Kind> =
            if const { BO == O + 1 } {
                Tensor::stack(observations, 0)
            } else {
                Tensor::cat(observations, 0).reshape([count; BO])
            };
        let rewards = Tensor::from_data(
            TensorData::new(rewards, [count]),
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
    /// Requires `BO = O + 1`, except scalar observations use rank one and concatenate to `[num_envs]`.
    /// Observation kind, dtype, device, and item dimensions must match across environments. Gradients are preserved.
    fn reset(
        &mut self,
    ) -> Result<Tensor<BO, <G::ObservationSpace as ObservationSpace<BO>>::Kind>, Self::Error> {
        let mut observations = Vec::with_capacity(self.envs.len());
        for env in &mut self.envs {
            observations.push(env.reset().map_err(VectorizedGymError::Single)?.observation);
        }
        Ok(if const { BO == O + 1 } {
            Tensor::stack(observations, 0)
        } else {
            Tensor::cat(observations, 0).reshape([self.envs.len(); BO])
        })
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
        let actions = Tensor::<1, Int>::from_data([3, 4], &Device::flex());
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

    impl Gym<(), 1, 1, 1, 1> for IntegerEnv {
        type Error = TestError;
        type ObservationSpace = Discrete;
        type ActionSpace = Discrete;

        /// Returns U64 observations `[1]` from Int actions `[1]` on the fixture CPU device.
        fn step(&mut self, action: Tensor<1, Int>) -> Result<StepInfo<(), 1, Int>, Self::Error> {
            Ok(StepInfo {
                observation: action.cast(DType::U64),
                reward: 1.0,
                done: false,
                truncated: false,
                info: (),
            })
        }

        /// Returns U64 reset observations `[1]`.
        fn reset(&mut self) -> Result<ResetInfo<(), 1, Int>, Self::Error> {
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
        assert_eq!(env.reset().unwrap().dims(), [2]);
        let step = env
            .step(Tensor::from_data([3i32, 4], &Device::flex()))
            .unwrap();
        assert_eq!(step.observations.dtype(), DType::U64);
        assert_eq!(
            step.observations.into_data().try_to_vec::<u64>().unwrap(),
            vec![3, 4]
        );
        assert_eq!(step.rewards.dtype(), DType::F32);
    }

    struct ImageEnv;

    impl Gym<(), 3, 1, 4, 1> for ImageEnv {
        type Error = TestError;
        type ObservationSpace = BoxSpace<4>;
        type ActionSpace = Discrete;

        /// Accepts scalar Int actions `[1]` and returns unbatched F64 image observations `[1, 2, 2]`.
        fn step(&mut self, _action: Tensor<1, Int>) -> Result<StepInfo<(), 3>, Self::Error> {
            Ok(StepInfo {
                observation: Tensor::ones([1, 2, 2], (&Device::flex(), DType::F64)),
                reward: 1.0,
                done: true,
                truncated: false,
                info: (),
            })
        }

        /// Returns unbatched F64 image observations `[1, 2, 2]`.
        fn reset(&mut self) -> Result<ResetInfo<(), 3>, Self::Error> {
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
            .step(Tensor::zeros([2], (&Device::flex(), DType::I32)))
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
}

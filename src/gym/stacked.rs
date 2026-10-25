use super::{MultiGym, MultiGymStepInfo, batch_metadata, batch_offsets, combine_steps};
use crate::spaces::{ActionSpace, ObservationSpace};
use crate::tensor_rank::{NextRank, PrevRank};
use burn::tensor::{Slice, Tensor, kind::Basic};
use std::marker::PhantomData;

#[derive(Debug, thiserror::Error)]
pub enum StackedMultiGymError<E> {
    #[error("at least one inner gym is required")]
    Empty,
    #[error("inner gym {gym_index} reported no batch slots")]
    EmptyInner { gym_index: usize },
    #[error("inner gym {gym_index} has observation shape {actual:?}, expected {expected:?}")]
    IncompatibleObservationShape {
        gym_index: usize,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("inner gym {gym_index} has action shape {actual:?}, expected {expected:?}")]
    IncompatibleActionShape {
        gym_index: usize,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    /// An inner gym changed its number of batch slots after construction.
    #[error("inner gym {gym_index} changed batch size from {expected} to {actual}")]
    ChangedBatchSize {
        gym_index: usize,
        expected: usize,
        actual: usize,
    },
    /// The supplied action does not have the combined batch size in dimension zero.
    #[error("invalid action batch: expected {expected} rows, got {actual:?}")]
    InvalidActionBatch {
        expected: usize,
        actual: Option<usize>,
    },
    /// An inner gym returned a tensor or vector that violates the [`MultiGym`] contract.
    #[error(
        "inner gym {gym_index} returned invalid {field} shape {actual:?}, expected {expected:?}"
    )]
    InvalidOutputShape {
        gym_index: usize,
        field: &'static str,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("inner gym {gym_index} failed: {error}")]
    Inner {
        gym_index: usize,
        #[source]
        error: E,
    },
    #[cfg(feature = "multithreading")]
    #[error("gym worker {gym_index} did not return the expected response")]
    WorkerDisconnected { gym_index: usize },
}

/// Flattens homogeneous MultiGym batches while preserving the order within each group.
/// Actions `[total_size, ...action_shape]` split into contiguous groups; observation batches concatenate on the same axis.
/// Inner gyms retain their auto-reset behavior. Reset the stack after an error because earlier groups may have advanced.
pub struct StackedMultiGym<G, I = (), const O: usize = 2, const A: usize = 2> {
    gyms: Vec<G>,
    group_offsets: Vec<usize>,
    observation_shape: Vec<usize>,
    _info: PhantomData<fn() -> I>,
}

impl<G, I, const O: usize, const A: usize, K: Basic, AK: Basic> StackedMultiGym<G, I, O, A>
where
    G: MultiGym<I, O, A>,
    G::ObservationSpace: ObservationSpace<O, Kind = K>,
    G::ActionSpace: ActionSpace<A, Kind = AK>,
    Tensor<A, AK>: PrevRank<Prev = <G::ActionSpace as ActionSpace<A>>::Action>,
    Tensor<O, K>: PrevRank<Prev = <G::ObservationSpace as ObservationSpace<O>>::Observation>,
{
    /// Connects nonempty groups with matching observation and action item shapes.
    pub fn new(gyms: Vec<G>) -> Result<Self, StackedMultiGymError<G::Error>> {
        const {
            assert!(A >= 2, "batched actions require batch and item axes");
        }
        let Some(first) = gyms.first() else {
            return Err(StackedMultiGymError::Empty);
        };
        let groups = gyms.iter().map(batch_metadata).collect::<Vec<_>>();
        let observation_shape = first.observation_space().shape();
        let action_shape = first.action_space().shape();
        let group_offsets = batch_offsets(&groups, &observation_shape, &action_shape)?;
        Ok(Self {
            gyms,
            group_offsets,
            observation_shape,
            _info: PhantomData,
        })
    }

    pub fn group_offsets(&self) -> &[usize] {
        &self.group_offsets
    }

    /// Returns the inner gyms in flattened batch order.
    pub fn gyms(&self) -> &[G] {
        &self.gyms
    }

    pub fn num_groups(&self) -> usize {
        self.gyms.len()
    }

    fn validate_batch_sizes(&self) -> Result<(), StackedMultiGymError<G::Error>> {
        for (gym_index, gym) in self.gyms.iter().enumerate() {
            let expected = super::group_size(&self.group_offsets, gym_index);
            let actual = gym.num_envs();
            if actual != expected {
                return Err(StackedMultiGymError::ChangedBatchSize {
                    gym_index,
                    expected,
                    actual,
                });
            }
        }
        Ok(())
    }
}

impl<G, I, const O: usize, const A: usize, K: Basic, AK: Basic> TryFrom<Vec<G>>
    for StackedMultiGym<G, I, O, A>
where
    G: MultiGym<I, O, A>,
    G::ObservationSpace: ObservationSpace<O, Kind = K>,
    G::ActionSpace: ActionSpace<A, Kind = AK>,
    Tensor<A, AK>: PrevRank<Prev = <G::ActionSpace as ActionSpace<A>>::Action>,
    Tensor<O, K>: PrevRank<Prev = <G::ObservationSpace as ObservationSpace<O>>::Observation>,
{
    type Error = StackedMultiGymError<G::Error>;

    fn try_from(gyms: Vec<G>) -> Result<Self, Self::Error> {
        Self::new(gyms)
    }
}

impl<G, I, const O: usize, const A: usize, const U: usize, K: Basic, AK: Basic> MultiGym<I, O, A>
    for StackedMultiGym<G, I, O, A>
where
    G: MultiGym<I, O, A>,
    G::ObservationSpace: ObservationSpace<O, Kind = K, Observation = Tensor<U, K>>,
    G::ActionSpace: ActionSpace<A, Kind = AK>,
    Tensor<A, AK>: PrevRank<Prev = <G::ActionSpace as ActionSpace<A>>::Action>,
    Tensor<O, K>: PrevRank<Prev = Tensor<U, K>>,
    Tensor<U, K>: NextRank<Next = Tensor<O, K>>,
{
    type Error = StackedMultiGymError<G::Error>;
    type ObservationSpace = G::ObservationSpace;
    type ActionSpace = G::ActionSpace;

    /// Splits rank-`A` actions `[total_size, ...action_shape]` across groups without removing the batch axis.
    /// Concatenates rank-`O` observations `[group_size, ...observation_shape]` and rewards `[group_size, 1]` in group order.
    /// Validates the action batch, output shapes, and metadata lengths before concatenating. Dtype and device follow the inner gyms' contracts.
    fn step(
        &mut self,
        action: Tensor<A, AK>,
    ) -> Result<MultiGymStepInfo<I, Tensor<U, K>>, Self::Error> {
        self.validate_batch_sizes()?;
        let expected = self.num_envs();
        let actual = action.dims().first().copied();
        if actual != Some(expected) {
            return Err(StackedMultiGymError::InvalidActionBatch { expected, actual });
        }
        let mut steps = Vec::with_capacity(self.gyms.len());
        for (gym_index, gym) in self.gyms.iter_mut().enumerate() {
            let range = self.group_offsets[gym_index]..self.group_offsets[gym_index + 1];
            let actions = action.clone().slice([Slice::from(range)]);
            steps.push(
                gym.step(actions)
                    .map_err(|error| StackedMultiGymError::Inner { gym_index, error })?,
            );
        }
        combine_steps(steps, &self.group_offsets, &self.observation_shape)
    }

    fn observation_space(&self) -> Self::ObservationSpace {
        self.gyms[0].observation_space()
    }

    fn action_space(&self) -> Self::ActionSpace {
        self.gyms[0].action_space()
    }

    fn num_envs(&self) -> usize {
        self.group_offsets[self.group_offsets.len() - 1]
    }

    /// Validates each reset observation batch/item shape before concatenating to `[total_size, ...observation_shape]`.
    /// Observation kind, dtype, and device follow the inner gyms. Gradient paths are preserved.
    fn reset(&mut self) -> Result<Tensor<O, K>, Self::Error> {
        self.validate_batch_sizes()?;
        let mut observations = Vec::with_capacity(self.gyms.len());
        for (gym_index, gym) in self.gyms.iter_mut().enumerate() {
            let observation = gym
                .reset()
                .map_err(|error| StackedMultiGymError::Inner { gym_index, error })?;
            super::validate_group_observations(
                gym_index,
                "reset observations",
                &observation.dims(),
                super::group_size(&self.group_offsets, gym_index),
                &self.observation_shape,
            )?;
            observations.push(observation);
        }
        Ok(Tensor::cat(observations, 0))
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::*;
    use burn::tensor::Device;

    #[test]
    fn stacked_batches_route_actions_and_preserve_slot_order() {
        let mut env =
            StackedMultiGym::new(vec![GroupEnv::new(10, 2), GroupEnv::new(20, 3)]).unwrap();
        assert_eq!(env.group_offsets(), &[0, 2, 5]);
        assert_eq!(env.num_groups(), 2);
        assert_eq!(env.num_envs(), 5);
        assert_eq!(env.reset().unwrap().dims(), [5, 3]);
        let actions = Tensor::from_data([[0.0], [1.0], [2.0], [3.0], [4.0]], &Device::flex());
        let step = env.step(actions).unwrap();
        assert_eq!(
            step.observations
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            vec![
                10.0, 0.0, 0.0, 10.0, 1.0, 1.0, 20.0, 0.0, 2.0, 20.0, 1.0, 3.0, 20.0, 2.0, 4.0
            ]
        );
        assert_eq!(
            step.rewards
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            vec![0.0, 1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(step.dones, vec![false, true, false, false, true]);
        assert_eq!(
            step.infos[2],
            SlotInfo {
                group: 20,
                slot: 0,
                action: 2.0
            }
        );
        assert_eq!(
            step.transition_next_observations()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            vec![
                10.0, 0.0, 0.0, 10.0, 1.0, -1.0, 20.0, 0.0, 2.0, 20.0, 1.0, 3.0, 20.0, 2.0, -4.0
            ]
        );
    }

    #[test]
    fn constructors_report_empty_groups_and_incompatible_spaces() {
        assert!(matches!(
            StackedMultiGym::<GroupEnv, SlotInfo, 2, 2>::new(vec![]),
            Err(StackedMultiGymError::Empty)
        ));
        assert!(matches!(
            StackedMultiGym::new(vec![GroupEnv::new(0, 0)]),
            Err(StackedMultiGymError::EmptyInner { gym_index: 0 })
        ));
        let mut wrong_observation = GroupEnv::new(1, 1);
        wrong_observation.observation_width = 2;
        assert!(matches!(
            StackedMultiGym::new(vec![GroupEnv::new(0, 1), wrong_observation]),
            Err(StackedMultiGymError::IncompatibleObservationShape { gym_index: 1, .. })
        ));
        let mut wrong_action = GroupEnv::new(1, 1);
        wrong_action.action_width = 2;
        assert!(matches!(
            StackedMultiGym::new(vec![GroupEnv::new(0, 1), wrong_action]),
            Err(StackedMultiGymError::IncompatibleActionShape { gym_index: 1, .. })
        ));
    }

    #[test]
    fn stacked_multi_gym_rejects_invalid_action_batches_before_stepping() {
        let mut env = StackedMultiGym::new(vec![GroupEnv::new(0, 2)]).unwrap();
        assert!(matches!(
            env.step(Tensor::zeros([3, 1], &Device::flex())),
            Err(StackedMultiGymError::InvalidActionBatch {
                expected: 2,
                actual: Some(3),
            })
        ));
    }

    #[test]
    fn stacked_multi_gym_validates_step_shapes_and_metadata() {
        let mut wrong_observations = GroupEnv::new(0, 2);
        wrong_observations.malformed_output = Some(MalformedGroupOutput::StepObservations);
        let mut env = StackedMultiGym::new(vec![wrong_observations]).unwrap();
        assert!(matches!(
            env.step(Tensor::zeros([2, 1], &Device::flex())),
            Err(StackedMultiGymError::InvalidOutputShape {
                gym_index: 0,
                field: "observations",
                expected,
                actual,
            }) if expected == vec![2, 3] && actual == vec![1, 3]
        ));

        let mut wrong_infos = GroupEnv::new(0, 2);
        wrong_infos.malformed_output = Some(MalformedGroupOutput::StepInfos);
        let mut env = StackedMultiGym::new(vec![wrong_infos]).unwrap();
        assert!(matches!(
            env.step(Tensor::zeros([2, 1], &Device::flex())),
            Err(StackedMultiGymError::InvalidOutputShape {
                gym_index: 0,
                field: "infos",
                expected,
                actual,
            }) if expected == vec![2] && actual == vec![1]
        ));
    }

    #[test]
    fn stacked_multi_gym_validates_reset_shapes() {
        let mut malformed = GroupEnv::new(0, 2);
        malformed.malformed_output = Some(MalformedGroupOutput::ResetObservations);
        let mut env = StackedMultiGym::new(vec![malformed]).unwrap();
        assert!(matches!(
            env.reset(),
            Err(StackedMultiGymError::InvalidOutputShape {
                gym_index: 0,
                field: "reset observations",
                expected,
                actual,
            }) if expected == vec![2, 3] && actual == vec![1, 3]
        ));
    }

    #[test]
    fn stacked_multi_gym_detects_changed_inner_batch_size() {
        let mut env = StackedMultiGym::new(vec![GroupEnv::new(0, 2)]).unwrap();
        env.gyms[0].count = 1;
        for result in [
            env.step(Tensor::zeros([2, 1], &Device::flex())).map(|_| ()),
            env.reset().map(|_| ()),
        ] {
            assert!(matches!(
                result,
                Err(StackedMultiGymError::ChangedBatchSize {
                    gym_index: 0,
                    expected: 2,
                    actual: 1,
                })
            ));
        }
    }

    #[test]
    fn errors_identify_the_failing_environment_group() {
        let mut failed = GroupEnv::new(1, 1);
        failed.fail_next = true;
        let mut env = StackedMultiGym::new(vec![GroupEnv::new(0, 1), failed]).unwrap();
        assert!(matches!(
            env.step(Tensor::zeros([2, 1], &Device::flex())),
            Err(StackedMultiGymError::Inner {
                gym_index: 1,
                error: TestError::Forced
            })
        ));
    }
}

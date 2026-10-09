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
    _info: PhantomData<fn() -> I>,
}

impl<G, I, const O: usize, const A: usize, K: Basic, AK: Basic> StackedMultiGym<G, I, O, A>
where
    G: MultiGym<I, O, A>,
    G::ObservationSpace: ObservationSpace<O, Kind = K>,
    G::ActionSpace: ActionSpace<A, Kind = AK>,
    Tensor<O, K>: PrevRank,
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
        let group_offsets = batch_offsets(
            &groups,
            &first.observation_space().shape(),
            &first.action_space().shape(),
        )?;
        Ok(Self {
            gyms,
            group_offsets,
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
}

impl<G, I, const O: usize, const A: usize, K: Basic, AK: Basic> TryFrom<Vec<G>>
    for StackedMultiGym<G, I, O, A>
where
    G: MultiGym<I, O, A>,
    G::ObservationSpace: ObservationSpace<O, Kind = K>,
    G::ActionSpace: ActionSpace<A, Kind = AK>,
    Tensor<O, K>: PrevRank,
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
    G::ObservationSpace: ObservationSpace<O, Kind = K>,
    G::ActionSpace: ActionSpace<A, Kind = AK>,
    Tensor<O, K>: PrevRank<Prev = Tensor<U, K>>,
    Tensor<U, K>: NextRank<Next = Tensor<O, K>>,
{
    type Error = StackedMultiGymError<G::Error>;
    type ObservationSpace = G::ObservationSpace;
    type ActionSpace = G::ActionSpace;

    /// Splits rank-`A` actions `[total_size, ...action_shape]` across groups without removing the batch axis.
    /// Concatenates rank-`O` observations `[group_size, ...observation_shape]` and rewards `[group_size]` in group order.
    /// Input dimensions, dtype, and device must satisfy the inner gyms' contracts.
    fn step(
        &mut self,
        action: Tensor<A, AK>,
    ) -> Result<MultiGymStepInfo<I, Tensor<U, K>>, Self::Error> {
        let mut steps = Vec::with_capacity(self.gyms.len());
        for (gym_index, gym) in self.gyms.iter_mut().enumerate() {
            let range = self.group_offsets[gym_index]..self.group_offsets[gym_index + 1];
            let actions = action.clone().slice([Slice::from(range)]);
            steps.push(
                gym.step(actions)
                    .map_err(|error| StackedMultiGymError::Inner { gym_index, error })?,
            );
        }
        Ok(combine_steps(steps))
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

    /// Concatenates reset observations `[group_size, ...observation_shape]` to `[total_size, ...observation_shape]` of rank `O`.
    /// All groups must share observation kind, dtype, device, and item dimensions. Gradient paths are preserved.
    fn reset(&mut self) -> Result<Tensor<O, K>, Self::Error> {
        let mut observations = Vec::with_capacity(self.gyms.len());
        for (gym_index, gym) in self.gyms.iter_mut().enumerate() {
            observations.push(
                gym.reset()
                    .map_err(|error| StackedMultiGymError::Inner { gym_index, error })?,
            );
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

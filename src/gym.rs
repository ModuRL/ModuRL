use crate::spaces::{ActionSpace, ObservationSpace};
use crate::tensor_rank::{NextRank, PrevRank};
use burn::tensor::{
    DType, IndexingUpdateOp, Int, Slice, Tensor, TensorData,
    kind::{Basic, Kind},
};

mod stacked;
#[cfg(test)]
mod test_support;
#[cfg(feature = "multithreading")]
mod threaded;
mod vectorized;

pub use stacked::{StackedMultiGym, StackedMultiGymError};
#[cfg(feature = "multithreading")]
pub use threaded::{MultithreadedStackedMultiGym, MultithreadedVectorizedGymWrapper};
pub use vectorized::{VectorizedGymError, VectorizedGymWrapper};

/// A single reinforcement-learning environment with unbatched tensors.
/// Ranks `O` and `A` describe observation and action batches. `PrevRank` determines each single tensor type.
/// Scalars use `[1]` individually and `[batch_size, 1]` in batches. Kinds come from the associated spaces.
/// `I` is reset and transition metadata; environments without metadata use `()`.
pub trait Gym<I = (), const O: usize = 2, const A: usize = 2>
where
    Tensor<O, <Self::ObservationSpace as ObservationSpace<O>>::Kind>: PrevRank,
    Tensor<A, <Self::ActionSpace as ActionSpace<A>>::Kind>: PrevRank,
{
    type Error;
    type ObservationSpace: ObservationSpace<O>;
    type ActionSpace: ActionSpace<A>;

    /// Steps with unbatched `action_shape`, one rank below `A`; scalar actions use `[1]`.
    /// Returns unbatched `observation_shape`, one rank below `O`, a scalar reward, and host flags.
    /// The environment defines supported dtypes and devices; CPU environments require CPU actions.
    fn step(
        &mut self,
        action: <Tensor<A, <Self::ActionSpace as ActionSpace<A>>::Kind> as PrevRank>::Prev,
    ) -> Result<
        StepInfo<
            I,
            <Tensor<O, <Self::ObservationSpace as ObservationSpace<O>>::Kind> as PrevRank>::Prev,
        >,
        Self::Error,
    >;

    /// Resets to unbatched `observation_shape`, one rank below `O`; scalar observations use `[1]`.
    /// The environment defines observation kind, dtype, and device.
    fn reset(
        &mut self,
    ) -> Result<
        ResetInfo<
            I,
            <Tensor<O, <Self::ObservationSpace as ObservationSpace<O>>::Kind> as PrevRank>::Prev,
        >,
        Self::Error,
    >;

    fn observation_space(&self) -> Self::ObservationSpace;

    fn action_space(&self) -> Self::ActionSpace;
}

/// An ordered batch of environment slots with native tensors.
/// Slots may be independent environments or coupled players in one world.
/// Implementations keep batch order and size fixed. Auto-reset implementations retain terminal observations.
/// `O` and `A` are batch ranks; `PrevRank` determines the unbatched terminal observation type.
pub trait MultiGym<I = (), const O: usize = 2, const A: usize = 2>
where
    Tensor<O, <Self::ObservationSpace as ObservationSpace<O>>::Kind>: PrevRank,
{
    type Error;
    type ObservationSpace: ObservationSpace<O>;
    type ActionSpace: ActionSpace<A>;

    /// Steps with `[num_envs, ...action_shape]` of rank `A` in the action space's kind.
    /// Returns observations `[num_envs, ...observation_shape]` of rank `O` and rewards `[num_envs]`.
    /// Callers must follow the environment's dtype and device requirements.
    fn step(
        &mut self,
        action: Tensor<A, <Self::ActionSpace as ActionSpace<A>>::Kind>,
    ) -> Result<
        MultiGymStepInfo<
            I,
            <Tensor<O, <Self::ObservationSpace as ObservationSpace<O>>::Kind> as PrevRank>::Prev,
        >,
        Self::Error,
    >;

    fn observation_space(&self) -> Self::ObservationSpace;

    fn action_space(&self) -> Self::ActionSpace;

    fn num_envs(&self) -> usize;

    /// Resets all slots to `[num_envs, ...observation_shape]` of rank `O` in the observation space's kind.
    /// The environment defines observation dtype and device. Reset before the first step.
    fn reset(
        &mut self,
    ) -> Result<Tensor<O, <Self::ObservationSpace as ObservationSpace<O>>::Kind>, Self::Error>;
}

/// An unbatched observation tensor `T` and environment metadata; scalar observations use native `Tensor<1>` with shape `[1]`.
/// Rank, kind, dtype, and device follow the environment's observation space.
#[derive(Debug, Clone)]
pub struct ResetInfo<I = (), T: NextRank = Tensor<1>> {
    pub observation: T,
    pub info: I,
}

/// One transition with an unbatched observation tensor `T`; scalar observations have shape `[1]`.
/// Reward and termination/truncation flags are host values. Kind, dtype, and device follow the environment.
#[derive(Debug, Clone)]
pub struct StepInfo<I = (), T: NextRank = Tensor<1>> {
    pub observation: T,
    pub reward: f32,
    pub done: bool,
    pub truncated: bool,
    pub info: I,
}

/// Batched transition results in environment-slot order.
/// `T` is the native unbatched terminal observation type. `T::Next` stores `[num_envs, ...observation_shape]`.
/// Scalar terminals `[1]` produce observation batches `[num_envs, 1]`. Rewards are Float `[num_envs]`.
/// Terminal observations share the observation dtype and device. Host metadata and flags contain one entry per slot.
/// Done and truncated remain separate.
#[derive(Debug, Clone)]
pub struct MultiGymStepInfo<I = (), T: NextRank = Tensor<1>> {
    pub observations: T::Next,
    pub rewards: Tensor<1>,
    pub infos: Vec<I>,
    pub dones: Vec<bool>,
    pub truncateds: Vec<bool>,
    pub terminal_observations: Vec<Option<T>>,
}

impl<I, const O: usize, K: Basic, const U: usize> MultiGymStepInfo<I, Tensor<U, K>>
where
    Tensor<U, K>: NextRank<Next = Tensor<O, K>>,
{
    /// Returns `[num_envs, ...observation_shape]` of rank `O` for transition targets.
    /// Replaces reset observations with saved terminal observations, preserving kind, dtype, device, and gradient paths.
    /// Adds the batch axis to unbatched terminal observations; scalar terminals `[1]` become rows `[1, 1]`.
    pub fn transition_next_observations(&self) -> Tensor<O, K> {
        let mut observations = self.observations.clone();
        let mut indices = Vec::new();
        let mut terminals = Vec::new();
        for (index, observation) in self.terminal_observations.iter().enumerate() {
            if let Some(observation) = observation {
                indices.push(index as i64);
                terminals.push(observation.clone());
            }
        }
        if !indices.is_empty() {
            if const { matches!(K::KIND, Kind::Bool) } {
                // Boolean indexed updates support OR, so replace terminal rows through native slice assignment.
                for (index, terminal) in indices.into_iter().zip(terminals) {
                    let start = index as usize;
                    let mut shape = observations.dims();
                    shape[0] = 1;
                    observations = observations
                        .slice_assign([Slice::from(start..start + 1)], terminal.reshape(shape));
                }
            } else {
                let count = indices.len();
                let rows = Tensor::<1, Int>::from_data(
                    TensorData::new(indices, [count]),
                    (&observations.device(), DType::I64),
                );
                observations = observations.select_assign(
                    0,
                    rows,
                    Tensor::stack(terminals, 0),
                    IndexingUpdateOp::Assign,
                );
            }
        }
        observations
    }
}

struct GymBatchMetadata {
    num_envs: usize,
    observation_shape: Vec<usize>,
    action_shape: Vec<usize>,
}

/// Collects slot count and item shapes so stacked gyms can check group compatibility and assign batch rows.
fn batch_metadata<G, I, const O: usize, const A: usize>(gym: &G) -> GymBatchMetadata
where
    G: MultiGym<I, O, A>,
    Tensor<O, <G::ObservationSpace as ObservationSpace<O>>::Kind>: PrevRank,
{
    GymBatchMetadata {
        num_envs: gym.num_envs(),
        observation_shape: gym.observation_space().shape(),
        action_shape: gym.action_space().shape(),
    }
}

/// Checks nonempty groups and matching item shapes, then returns cumulative row boundaries for splitting actions and combining observations.
fn batch_offsets<E>(
    groups: &[GymBatchMetadata],
    observation_shape: &[usize],
    action_shape: &[usize],
) -> Result<Vec<usize>, StackedMultiGymError<E>> {
    if groups.is_empty() {
        return Err(StackedMultiGymError::Empty);
    }
    let mut offsets = Vec::with_capacity(groups.len() + 1);
    offsets.push(0);
    let mut total = 0;
    for (gym_index, group) in groups.iter().enumerate() {
        if group.num_envs == 0 {
            return Err(StackedMultiGymError::EmptyInner { gym_index });
        }
        if group.observation_shape != observation_shape {
            return Err(StackedMultiGymError::IncompatibleObservationShape {
                gym_index,
                expected: observation_shape.to_vec(),
                actual: group.observation_shape.clone(),
            });
        }
        if group.action_shape != action_shape {
            return Err(StackedMultiGymError::IncompatibleActionShape {
                gym_index,
                expected: action_shape.to_vec(),
                actual: group.action_shape.clone(),
            });
        }
        total += group.num_envs;
        offsets.push(total);
    }
    Ok(offsets)
}

/// Concatenates observations `[group_size, ...observation_shape]` and rewards `[group_size]` along the batch axis.
/// Returns `[total_size, ...observation_shape]` of rank `O` and rewards `[total_size]`, preserving each tensor's kind, dtype, device, and gradients.
/// Groups must share item dimensions, dtypes, and devices. Metadata, flags, and terminal observations retain group order.
fn combine_steps<I, const O: usize, K: Basic, const U: usize>(
    steps: Vec<MultiGymStepInfo<I, Tensor<U, K>>>,
) -> MultiGymStepInfo<I, Tensor<U, K>>
where
    Tensor<U, K>: NextRank<Next = Tensor<O, K>>,
{
    let count = steps.iter().map(|step| step.infos.len()).sum();
    let mut observations = Vec::with_capacity(steps.len());
    let mut rewards = Vec::with_capacity(steps.len());
    let mut infos = Vec::with_capacity(count);
    let mut dones = Vec::with_capacity(count);
    let mut truncateds = Vec::with_capacity(count);
    let mut terminal_observations = Vec::with_capacity(count);
    for step in steps {
        observations.push(step.observations);
        rewards.push(step.rewards);
        infos.extend(step.infos);
        dones.extend(step.dones);
        truncateds.extend(step.truncateds);
        terminal_observations.extend(step.terminal_observations);
    }
    MultiGymStepInfo {
        observations: Tensor::cat(observations, 0),
        rewards: Tensor::cat(rewards, 0),
        infos,
        dones,
        truncateds,
        terminal_observations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Bool, Device};

    #[test]
    fn scalar_terminal_observations_use_one_element_tensors() {
        let device = Device::flex();
        let step = MultiGymStepInfo::<(), Tensor<1, Int>> {
            observations: Tensor::from_data([[8u64], [9]], (&device, DType::U64)),
            rewards: Tensor::zeros([2], &device),
            infos: vec![(); 2],
            dones: vec![true, false],
            truncateds: vec![false; 2],
            terminal_observations: vec![
                Some(Tensor::from_data([3u64], (&device, DType::U64))),
                None,
            ],
        };
        let targets = step.transition_next_observations();
        assert_eq!(targets.dims(), [2, 1]);
        assert_eq!(targets.dtype(), DType::U64);
        assert_eq!(targets.into_data().try_to_vec::<u64>().unwrap(), vec![3, 9]);
    }

    #[test]
    fn boolean_scalar_terminals_replace_complete_batch_rows() {
        let device = Device::flex();
        let step = MultiGymStepInfo::<(), Tensor<1, Bool>> {
            observations: Tensor::from_data([[true], [true]], &device),
            rewards: Tensor::zeros([2], &device),
            infos: vec![(); 2],
            dones: vec![true, false],
            truncateds: vec![false; 2],
            terminal_observations: vec![Some(Tensor::from_data([false], &device)), None],
        };
        let observations = step.transition_next_observations();
        assert_eq!(observations.dims(), [2, 1]);
        assert_eq!(
            observations.into_data().try_to_vec::<bool>().unwrap(),
            [false, true]
        );
    }

    #[test]
    fn boolean_terminal_observations_replace_true_reset_values() {
        let step = MultiGymStepInfo::<(), Tensor<1, Bool>> {
            observations: Tensor::from_data([[true, true], [true, false]], &Device::flex()),
            rewards: Tensor::zeros([2], &Device::flex()),
            infos: vec![(); 2],
            dones: vec![true, false],
            truncateds: vec![false; 2],
            terminal_observations: vec![
                Some(Tensor::from_data([false, false], &Device::flex())),
                None,
            ],
        };
        assert_eq!(
            step.transition_next_observations()
                .into_data()
                .iter::<bool>()
                .collect::<Vec<_>>(),
            vec![false, false, true, false]
        );
    }
}

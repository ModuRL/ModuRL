//! Wrappers that transform observations.

use crate::gym::{Gym, LegalActionsGym, PerfectStateGym, ResetInfo, StepInfo};
use crate::spaces::{ActionSpace, BoxSpace, ObservationSpace};
use crate::tensor_rank::{NextRank, PrevRank, TensorBatchTypes};
use burn::tensor::{
    DType, Int, Tensor, TensorData,
    kind::{Basic, Ordered},
};
use std::collections::VecDeque;

#[derive(Debug, thiserror::Error)]
pub enum LegalActionMaskGymError<E> {
    #[error("wrapped gym error: {0}")]
    GymError(#[source] E),
    #[error("legal action {action} is outside 0..{action_count}")]
    InvalidLegalAction { action: usize, action_count: usize },
}

/// Splits observations `[batch_size, feature_count + action_count]` into features and the trailing mask.
/// Returns a recoverable error when fewer than `action_count` columns are present.
pub(crate) fn split_legal_action_mask<K: Basic>(
    observations: Tensor<2, K>,
    action_count: usize,
) -> Result<(Tensor<2, K>, Tensor<2, K>), LegalActionMaskLayoutError> {
    let [batch_size, columns] = observations.dims();
    if columns < action_count {
        return Err(LegalActionMaskLayoutError {
            columns,
            action_count,
        });
    }

    let feature_count = columns - action_count;
    Ok((
        observations
            .clone()
            .slice([0..batch_size, 0..feature_count]),
        observations.slice([0..batch_size, feature_count..columns]),
    ))
}

#[derive(Debug, thiserror::Error)]
#[error(
    "observation has {columns} columns, fewer than the {action_count} legal-action mask columns"
)]
pub(crate) struct LegalActionMaskLayoutError {
    pub(crate) columns: usize,
    pub(crate) action_count: usize,
}

/// Appends the current discrete legal-action mask to flat observations.
pub struct LegalActionMaskGym<G> {
    gym: G,
}

impl<G> LegalActionMaskGym<G> {
    /// Wraps a flat-observation gym with a legal-action mask appended to each observation.
    pub fn new(gym: G) -> Self {
        Self { gym }
    }
}

/// Appends a mask to observation shape `[feature_count]`, preserving dtype and device.
fn append_legal_action_mask<E>(
    observation: Tensor<1>,
    legal_actions: Vec<usize>,
    action_count: usize,
) -> Result<Tensor<1>, LegalActionMaskGymError<E>> {
    let mut mask = vec![0.0f32; action_count];
    for action in legal_actions {
        if action >= action_count {
            return Err(LegalActionMaskGymError::InvalidLegalAction {
                action,
                action_count,
            });
        }
        mask[action] = 1.0;
    }
    let mask = Tensor::<1>::from_data(
        TensorData::new(mask, [action_count]),
        (&observation.device(), DType::F32),
    )
    .cast(observation.dtype());
    Ok(Tensor::cat(vec![observation, mask], 0))
}

impl<G, I> Gym<I, 2, 2> for LegalActionMaskGym<G>
where
    G: LegalActionsGym<I, 2, ObservationSpace = BoxSpace<2>>,
{
    type Error = LegalActionMaskGymError<G::Error>;
    type ObservationSpace = BoxSpace<2>;
    type ActionSpace = G::ActionSpace;

    fn reset(&mut self) -> Result<ResetInfo<I, Tensor<1>>, Self::Error> {
        let reset = self
            .gym
            .reset()
            .map_err(LegalActionMaskGymError::GymError)?;
        let legal_actions = self
            .gym
            .legal_actions()
            .map_err(LegalActionMaskGymError::GymError)?;
        Ok(ResetInfo {
            observation: append_legal_action_mask(
                reset.observation,
                legal_actions,
                self.gym.action_space().get_possible_values(),
            )?,
            info: reset.info,
        })
    }

    /// Steps the inner gym with an action shaped `[1]` and appends the post-step legal-action mask.
    fn step(&mut self, action: Tensor<1, Int>) -> Result<StepInfo<I, Tensor<1>>, Self::Error> {
        let step = self
            .gym
            .step(action)
            .map_err(LegalActionMaskGymError::GymError)?;
        let legal_actions = self
            .gym
            .legal_actions()
            .map_err(LegalActionMaskGymError::GymError)?;
        Ok(StepInfo {
            observation: append_legal_action_mask(
                step.observation,
                legal_actions,
                self.gym.action_space().get_possible_values(),
            )?,
            reward: step.reward,
            done: step.done,
            truncated: step.truncated,
            info: step.info,
        })
    }

    fn action_space(&self) -> Self::ActionSpace {
        self.gym.action_space()
    }

    fn observation_space(&self) -> Self::ObservationSpace {
        let inner_space = self.gym.observation_space();
        let action_count = self.gym.action_space().get_possible_values();
        let low = inner_space.low();
        let high = inner_space.high();
        let device = low.device();
        let dtype = low.dtype();
        let mask_low = Tensor::<2>::zeros([1, action_count], (&device, dtype));
        let mask_high = Tensor::<2>::ones([1, action_count], (&device, dtype));
        BoxSpace::new(
            Tensor::cat(vec![low.clone(), mask_low], 1),
            Tensor::cat(vec![high.clone(), mask_high], 1),
        )
    }
}

impl<G, I> LegalActionsGym<I, 2> for LegalActionMaskGym<G>
where
    G: LegalActionsGym<I, 2, ObservationSpace = BoxSpace<2>>,
{
    fn legal_actions(&self) -> Result<Vec<usize>, Self::Error> {
        self.gym
            .legal_actions()
            .map_err(LegalActionMaskGymError::GymError)
    }
}

impl<G, I> PerfectStateGym<I, 2, 2> for LegalActionMaskGym<G>
where
    G: LegalActionsGym<I, 2, ObservationSpace = BoxSpace<2>> + PerfectStateGym<I, 2, 2>,
{
    type State = G::State;

    fn save_state(&self) -> Result<Self::State, Self::Error> {
        self.gym
            .save_state()
            .map_err(LegalActionMaskGymError::GymError)
    }

    fn set_state(&mut self, state: &Self::State) -> Result<(), Self::Error> {
        self.gym
            .set_state(state)
            .map_err(LegalActionMaskGymError::GymError)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum FrameStackGymError<E> {
    #[error("wrapped gym error: {0}")]
    GymError(#[source] E),
}

/// Stacks recent observations along a leading frame axis, preserving kind, dtype, and device.
/// `BF` is the stacked batch rank. Two `PrevRank` mappings determine the inner single rank `O`.
/// The intermediate rank `F` is shared by inner batches and stacked single observations.
/// All observations require `F = O + 1` and `BF = F + 1`.
/// Scalar observations use `O = 1`, `F = 2`, and `BF = 3`, producing `[stack_size, 1]`.
/// Supported stacked batch ranks are 3 through 1025, as defined by `PrevRank`.
pub struct FrameStackGym<G, const BF: usize = 3, S = BoxSpace<BF>>
where
    S: ObservationSpace<BF, Item: PrevRank>,
{
    gym: G,
    stack_size: usize,
    frames: VecDeque<<S::Item as PrevRank>::Prev>,
    observation_space: S,
}

impl<G, const O: usize, const F: usize, const BF: usize, S, K: Basic> FrameStackGym<G, BF, S>
where
    S: ObservationSpace<BF, Kind = K, Item = Tensor<F, K>>,
    Tensor<BF, K>: PrevRank<Prev = Tensor<F, K>>,
    Tensor<F, K>: PrevRank<Prev = Tensor<O, K>> + NextRank<Next = Tensor<BF, K>>,
    Tensor<O, K>: NextRank<Next = Tensor<F, K>>,
{
    /// Creates a frame stack with a declared space shaped `[stack_size, ...inner_observation_shape]` per item.
    /// Scalars use `[stack_size, 1]`. `BoxSpace` bounds include a separate size-one batch axis.
    pub fn new(gym: G, stack_size: usize, observation_space: S) -> Self {
        assert!(stack_size > 0, "stack_size must be at least 1");
        assert_eq!(
            observation_space.shape().first().copied(),
            Some(stack_size),
            "the pre-stacked observation space must start with stack_size"
        );
        Self {
            gym,
            stack_size,
            frames: VecDeque::with_capacity(stack_size),
            observation_space,
        }
    }

    /// Adds a frame axis to rank-`O` observations, returning rank `F` with `[frame_count, ...observation_shape]`.
    /// Scalar `[1]` frames stack into `[frame_count, 1]`. Preserves kind, dtype, device, and gradients.
    fn stacked_observation(&self) -> Tensor<F, K> {
        let frames = self.frames.iter().cloned().collect::<Vec<_>>();
        Tensor::stack(frames, 0)
    }
}

impl<G, I, const O: usize, const BA: usize, const F: usize, const BF: usize, S, K: Basic, AK: Basic>
    Gym<I, BF, BA> for FrameStackGym<G, BF, S>
where
    G: Gym<I, F, BA>,
    G::ObservationSpace: ObservationSpace<F, Kind = K, Item = Tensor<O, K>>,
    G::ActionSpace: ActionSpace<BA, Kind = AK>,
    S: ObservationSpace<BF, Kind = K, Item = Tensor<F, K>> + Clone,
    Tensor<BF, K>: PrevRank<Prev = Tensor<F, K>>,
    Tensor<F, K>: PrevRank<Prev = Tensor<O, K>> + NextRank<Next = Tensor<BF, K>>,
    Tensor<O, K>: NextRank<Next = Tensor<F, K>>,
{
    type Error = FrameStackGymError<G::Error>;
    type ObservationSpace = S;
    type ActionSpace = G::ActionSpace;

    /// Repeats the unbatched rank-`O` reset observation to fill `[stack_size, ...observation_shape]` of rank `F`.
    /// Scalar inputs `[1]` produce `[stack_size, 1]`; all outputs preserve kind, dtype, and device.
    fn reset(&mut self) -> Result<ResetInfo<I, Tensor<F, K>>, Self::Error> {
        let reset = self.gym.reset().map_err(FrameStackGymError::GymError)?;
        self.frames.clear();
        self.frames
            .extend(std::iter::repeat_with(|| reset.observation.clone()).take(self.stack_size));
        Ok(ResetInfo {
            observation: self.stacked_observation(),
            info: reset.info,
        })
    }

    /// Forwards an unbatched rank-`BA - 1` action and appends the rank-`O` observation after dropping the oldest full-stack frame.
    /// Returns rank `F` with `[frame_count, ...observation_shape]`, or `[frame_count, 1]` for scalars, preserving kind, dtype, and device.
    fn step(
        &mut self,
        action: <Self::ActionSpace as TensorBatchTypes>::Item,
    ) -> Result<StepInfo<I, Tensor<F, K>>, Self::Error> {
        let step = self
            .gym
            .step(action)
            .map_err(FrameStackGymError::GymError)?;
        if self.frames.len() >= self.stack_size {
            self.frames.pop_front();
        }
        self.frames.push_back(step.observation);
        Ok(StepInfo {
            observation: self.stacked_observation(),
            reward: step.reward,
            done: step.done,
            truncated: step.truncated,
            info: step.info,
        })
    }

    fn action_space(&self) -> Self::ActionSpace {
        self.gym.action_space()
    }

    fn observation_space(&self) -> Self::ObservationSpace {
        self.observation_space.clone()
    }
}

impl<G, I, const O: usize, const BA: usize, const F: usize, const BF: usize, S, K: Basic, AK: Basic>
    PerfectStateGym<I, BF, BA> for FrameStackGym<G, BF, S>
where
    G: PerfectStateGym<I, F, BA>,
    G::ObservationSpace: ObservationSpace<F, Kind = K, Item = Tensor<O, K>>,
    G::ActionSpace: ActionSpace<BA, Kind = AK>,
    S: ObservationSpace<BF, Kind = K, Item = Tensor<F, K>> + Clone,
    Tensor<BF, K>: PrevRank<Prev = Tensor<F, K>>,
    Tensor<F, K>: PrevRank<Prev = Tensor<O, K>> + NextRank<Next = Tensor<BF, K>>,
    Tensor<O, K>: NextRank<Next = Tensor<F, K>>,
{
    type State = (G::State, VecDeque<Tensor<O, K>>);

    fn save_state(&self) -> Result<Self::State, Self::Error> {
        Ok((
            self.gym
                .save_state()
                .map_err(FrameStackGymError::GymError)?,
            self.frames.clone(),
        ))
    }

    fn set_state(&mut self, state: &Self::State) -> Result<(), Self::Error> {
        self.gym
            .set_state(&state.0)
            .map_err(FrameStackGymError::GymError)?;
        self.frames.clone_from(&state.1);
        Ok(())
    }
}

impl<G, I, const O: usize, const F: usize, const BF: usize, S, K: Basic> LegalActionsGym<I, BF>
    for FrameStackGym<G, BF, S>
where
    G: LegalActionsGym<I, F>,
    G::ObservationSpace: ObservationSpace<F, Kind = K, Item = Tensor<O, K>>,
    G::ActionSpace: ActionSpace<2, Kind = Int>,
    S: ObservationSpace<BF, Kind = K, Item = Tensor<F, K>> + Clone,
    Tensor<BF, K>: PrevRank<Prev = Tensor<F, K>>,
    Tensor<F, K>: PrevRank<Prev = Tensor<O, K>> + NextRank<Next = Tensor<BF, K>>,
    Tensor<O, K>: NextRank<Next = Tensor<F, K>>,
{
    fn legal_actions(&self) -> Result<Vec<usize>, Self::Error> {
        self.gym
            .legal_actions()
            .map_err(FrameStackGymError::GymError)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MaxAndSkipGymError<E> {
    #[error("wrapped gym error: {0}")]
    GymError(#[source] E),
}

/// Repeats each action, sums its rewards, and max-pools the final two observations.
pub struct MaxAndSkipGym<G> {
    gym: G,
    skip: usize,
}

impl<G> MaxAndSkipGym<G> {
    pub fn new(gym: G, skip: usize) -> Self {
        assert!(skip > 0, "skip must be at least 1");
        Self { gym, skip }
    }
}

impl<G, I, const O: usize, const BO: usize, const BA: usize, K: Basic, AK: Basic> Gym<I, BO, BA>
    for MaxAndSkipGym<G>
where
    G: Gym<I, BO, BA>,
    G::ObservationSpace: ObservationSpace<BO, Kind = K, Item = Tensor<O, K>>,
    G::ActionSpace: ActionSpace<BA, Kind = AK>,
    Tensor<BO, K>: PrevRank<Prev = Tensor<O, K>>,
    Tensor<O, K>: NextRank<Next = Tensor<BO, K>>,
    K: Ordered,
{
    type Error = MaxAndSkipGymError<G::Error>;
    type ObservationSpace = G::ObservationSpace;
    type ActionSpace = G::ActionSpace;

    /// Returns the inner gym's unbatched rank-`O` observation, preserving its shape, kind, dtype, and device.
    fn reset(&mut self) -> Result<ResetInfo<I, Tensor<O, K>>, Self::Error> {
        self.gym.reset().map_err(MaxAndSkipGymError::GymError)
    }

    /// Repeats an unbatched rank-`BA - 1` action and max-pools the final two unbatched rank-`O` observations.
    /// Preserves item axes, numeric kind, dtype, and device; scalars use `[1]`. Stops on termination or truncation.
    fn step(
        &mut self,
        action: <Self::ActionSpace as TensorBatchTypes>::Item,
    ) -> Result<StepInfo<I, Tensor<O, K>>, Self::Error> {
        let mut step = self
            .gym
            .step(action.clone())
            .map_err(MaxAndSkipGymError::GymError)?;
        let mut total_reward = 0.0;
        total_reward += step.reward;
        let mut second_last_observation = None;
        for _ in 1..self.skip {
            if step.done || step.truncated {
                break;
            }
            second_last_observation = Some(step.observation);
            step = self
                .gym
                .step(action.clone())
                .map_err(MaxAndSkipGymError::GymError)?;
            total_reward += step.reward;
        }
        if let Some(second_last) = second_last_observation {
            step.observation = step.observation.max_pair(second_last);
        }
        step.reward = total_reward;
        Ok(step)
    }

    fn action_space(&self) -> Self::ActionSpace {
        self.gym.action_space()
    }

    fn observation_space(&self) -> Self::ObservationSpace {
        self.gym.observation_space()
    }
}

impl<G, I, const O: usize, const BO: usize, const BA: usize, K: Basic, AK: Basic>
    PerfectStateGym<I, BO, BA> for MaxAndSkipGym<G>
where
    G: PerfectStateGym<I, BO, BA>,
    G::ObservationSpace: ObservationSpace<BO, Kind = K, Item = Tensor<O, K>>,
    G::ActionSpace: ActionSpace<BA, Kind = AK>,
    Tensor<BO, K>: PrevRank<Prev = Tensor<O, K>>,
    Tensor<O, K>: NextRank<Next = Tensor<BO, K>>,
    K: Ordered,
{
    type State = G::State;

    fn save_state(&self) -> Result<Self::State, Self::Error> {
        self.gym.save_state().map_err(MaxAndSkipGymError::GymError)
    }

    fn set_state(&mut self, state: &Self::State) -> Result<(), Self::Error> {
        self.gym
            .set_state(state)
            .map_err(MaxAndSkipGymError::GymError)
    }
}

impl<G, I, const O: usize, const BO: usize, K: Basic + Ordered> LegalActionsGym<I, BO>
    for MaxAndSkipGym<G>
where
    G: LegalActionsGym<I, BO>,
    G::ObservationSpace: ObservationSpace<BO, Kind = K, Item = Tensor<O, K>>,
    G::ActionSpace: ActionSpace<2, Kind = Int>,
    Tensor<BO, K>: PrevRank<Prev = Tensor<O, K>>,
    Tensor<O, K>: NextRank<Next = Tensor<BO, K>>,
{
    fn legal_actions(&self) -> Result<Vec<usize>, Self::Error> {
        self.gym
            .legal_actions()
            .map_err(MaxAndSkipGymError::GymError)
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor_rank::TensorBatchTypes;
    use crate::{
        gym::{Gym, LegalActionsGym, ResetInfo, StepInfo},
        spaces::{BoxSpace, Discrete, ObservationSpace},
        tensor_rank::PrevRank,
        wrappers::{
            state_test_support::{StateTestGym, action as state_action, assert_state_round_trip},
            test_support::{TestGym, action, scalar},
        },
    };
    use burn::tensor::{Bool, DType, Device, Int, Tensor, kind::Basic};
    use std::marker::PhantomData;

    use super::{FrameStackGym, LegalActionMaskGym, LegalActionMaskGymError, MaxAndSkipGym};

    struct InvalidLegalActionsGym(StateTestGym);

    impl Gym for InvalidLegalActionsGym {
        type Error = std::convert::Infallible;
        type ObservationSpace = BoxSpace<2>;
        type ActionSpace = Discrete;

        /// Accepts an action with shape `[1]` and returns the inner observation.
        fn step(&mut self, action: Tensor<1, Int>) -> Result<StepInfo, Self::Error> {
            self.0.step(action)
        }

        fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
            self.0.reset()
        }

        fn observation_space(&self) -> Self::ObservationSpace {
            self.0.observation_space()
        }

        fn action_space(&self) -> Self::ActionSpace {
            self.0.action_space()
        }
    }

    impl LegalActionsGym for InvalidLegalActionsGym {
        fn legal_actions(&self) -> Result<Vec<usize>, Self::Error> {
            Ok(vec![2])
        }
    }

    #[test]
    fn legal_action_mask_is_appended_on_reset_and_step() {
        let device = Device::flex();
        let mut wrapper = LegalActionMaskGym::new(StateTestGym::new(41));
        let reset = wrapper.reset().unwrap();
        assert_eq!(reset.observation.dims(), [3]);
        assert_eq!(
            reset
                .observation
                .clone()
                .narrow(0, 1, 2)
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            [1.0, 1.0]
        );
        let step = wrapper.step(state_action()).unwrap();
        let step_values = step.observation.into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(&step_values[1..], &[0.0, 1.0]);

        let space = wrapper.observation_space();
        assert_eq!(space.shape(), [3]);
        assert_eq!(
            space
                .low()
                .clone()
                .slice([0..1, 1..3])
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            [0.0, 0.0]
        );
        assert_eq!(
            space
                .high()
                .clone()
                .slice([0..1, 1..3])
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            [1.0, 1.0]
        );
        assert_eq!(reset.observation.device(), device);
    }

    #[test]
    fn legal_action_mask_rejects_actions_outside_the_action_space() {
        let mut wrapper = LegalActionMaskGym::new(InvalidLegalActionsGym(StateTestGym::new(43)));
        assert!(matches!(
            wrapper.reset(),
            Err(LegalActionMaskGymError::InvalidLegalAction {
                action: 2,
                action_count: 2
            })
        ));
    }

    #[derive(Clone)]
    struct TypedSpace<const D: usize, K> {
        shape: [usize; D],
        kind: PhantomData<K>,
    }

    impl<const D: usize, K: Basic> TensorBatchTypes for TypedSpace<D, K>
    where
        Tensor<D, K>: PrevRank,
    {
        type Kind = K;
        type Item = <Tensor<D, K> as PrevRank>::Prev;
        type Batch = Tensor<D, K>;
    }

    impl<const D: usize, K: Basic> ObservationSpace<D> for TypedSpace<D, K>
    where
        Tensor<D, K>: PrevRank,
    {
        /// Checks the rank-`D` batch's item axes against the declared item shape without inspecting values.
        fn contains(&self, values: &Tensor<D, K>) -> bool {
            values.dims()[1..] == self.shape[1..]
        }

        fn shape(&self) -> Vec<usize> {
            self.shape[1..].to_vec()
        }
    }

    #[test]
    fn frame_stack_adds_an_axis_to_vector_observations() {
        let device = Device::flex();
        let gym = crate::agents::test_support::FixedEnv::new(device.clone());
        let space = BoxSpace::new_unbounded([1, 2, 4], &device);
        let mut wrapper: FrameStackGym<_> = FrameStackGym::new(gym, 2, space);
        let reset = wrapper.reset().unwrap();
        assert_eq!(reset.observation.dims(), [2, 4]);
        assert_eq!(wrapper.step(action()).unwrap().observation.dims(), [2, 4]);
        assert_eq!(wrapper.observation_space().shape(), vec![2, 4]);
    }

    #[test]
    fn frame_stack_preserves_integer_and_boolean_frames() {
        let device = Device::flex();
        let mut integers = FrameStackGym::<_, 4, _>::new(
            (),
            2,
            TypedSpace::<4, Int> {
                shape: [1, 2, 1, 2],
                kind: PhantomData,
            },
        );
        integers
            .frames
            .push_back(Tensor::from_data([[1i32, 2]], (&device, DType::I64)));
        integers
            .frames
            .push_back(Tensor::from_data([[3i32, 4]], (&device, DType::I64)));
        let stacked = integers.stacked_observation();
        assert_eq!(stacked.dims(), [2, 1, 2]);
        assert_eq!(stacked.dtype(), DType::I64);
        assert_eq!(stacked.device(), device);
        assert_eq!(
            stacked.into_data().try_to_vec::<i64>().unwrap(),
            vec![1, 2, 3, 4]
        );
        let mut booleans = FrameStackGym::<_, 3, _>::new(
            (),
            2,
            TypedSpace::<3, Bool> {
                shape: [1, 2, 1],
                kind: PhantomData,
            },
        );
        booleans
            .frames
            .push_back(Tensor::from_data([true], &device));
        booleans
            .frames
            .push_back(Tensor::from_data([false], &device));
        let stacked = booleans.stacked_observation();
        assert_eq!(stacked.dims(), [2, 1]);
        assert_eq!(
            stacked.into_data().try_to_vec::<bool>().unwrap(),
            vec![true, false]
        );
    }

    #[test]
    fn frame_stack_duplicates_reset_observation_and_rolls_on_step() {
        let gym = TestGym::new([TestGym::step(2.0, 0.0, false, false, 1)]);
        let observation_space = BoxSpace::new_with_universal_bounds(
            [1, 4, 1],
            -100.0,
            100.0,
            &burn::tensor::Device::flex(),
        );
        let mut wrapper = FrameStackGym::<_, 3>::new(gym, 4, observation_space);

        let reset = wrapper.reset().unwrap();
        let step = wrapper.step(action()).unwrap();

        let observation_space = wrapper.observation_space();
        assert_eq!(observation_space.shape(), vec![4, 1]);
        assert!(observation_space.contains(&reset.observation.clone().unsqueeze::<3>()));
        assert_eq!(
            reset.observation.into_data().try_to_vec::<f32>().unwrap(),
            vec![100.0; 4]
        );
        assert_eq!(
            step.observation.into_data().try_to_vec::<f32>().unwrap(),
            vec![100.0, 100.0, 100.0, 2.0]
        );
        assert_eq!(step.info.sequence, 1);
    }

    #[test]
    fn max_and_skip_pools_final_observations_and_preserves_final_metadata() {
        let gym = TestGym::new([
            TestGym::step(1.0, 0.25, false, false, 1),
            TestGym::step(5.0, 0.25, false, false, 2),
            TestGym::step(3.0, 0.25, false, false, 3),
            TestGym::step(2.0, 0.25, false, false, 4),
        ]);
        let mut wrapper = MaxAndSkipGym::new(gym, 4);

        let step = wrapper.step(action()).unwrap();

        assert_eq!(scalar(&step.observation), 3.0);
        assert_eq!(step.reward, 1.0);
        assert_eq!(step.info.sequence, 4);
    }

    #[test]
    fn max_and_skip_stops_on_truncation() {
        let gym = TestGym::new([
            TestGym::step(1.0, 0.25, false, false, 1),
            TestGym::step(2.0, 0.25, false, true, 2),
            TestGym::step(3.0, 0.25, false, false, 3),
        ]);
        let mut wrapper = MaxAndSkipGym::new(gym, 4);

        let step = wrapper.step(action()).unwrap();

        assert_eq!(scalar(&step.observation), 2.0);
        assert_eq!(step.reward, 0.5);
        assert!(step.truncated);
        assert_eq!(step.info.sequence, 2);
    }

    #[test]
    fn frame_stack_restores_frames_and_inner_state() {
        let device = Device::flex();
        let observation_space = BoxSpace::new_unbounded([1, 2, 1], &device);
        assert_state_round_trip(
            FrameStackGym::<_, 3>::new(StateTestGym::new(29), 2, observation_space),
            state_action(),
            1,
            1,
            |observation| observation.into_data(),
            |_: &()| (),
        );
    }

    #[test]
    fn max_and_skip_restores_inner_state() {
        assert_state_round_trip(
            MaxAndSkipGym::new(StateTestGym::new(31), 2),
            state_action(),
            1,
            2,
            |observation| observation.into_data(),
            |_: &()| (),
        );
    }
}

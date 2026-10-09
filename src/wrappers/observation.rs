//! Wrappers that transform observations.

use crate::gym::{Gym, ResetInfo, StepInfo};
use crate::spaces::{ActionSpace, BoxSpace, ObservationSpace};
use burn::tensor::{Tensor, kind::Ordered};
use std::collections::VecDeque;

#[derive(Debug, thiserror::Error)]
pub enum FrameStackGymError<E> {
    #[error("wrapped gym error: {0}")]
    GymError(#[source] E),
}

/// Stacks recent observations along a leading frame axis, preserving kind, dtype, and device.
/// `O` and `BO` are inner single and batch ranks; `F` and `BF` are stacked single and batch ranks.
/// Nonscalar observations require `BO = O + 1`, `F = O + 1`, and `BF = F + 1`.
/// Scalar observations use `O = BO = F = 1` and `BF = 2`, producing `[stack_size]`.
pub struct FrameStackGym<
    G,
    const O: usize = 1,
    const BO: usize = 2,
    const F: usize = 2,
    const BF: usize = 3,
    S = BoxSpace<BF>,
> where
    S: ObservationSpace<BF>,
{
    gym: G,
    stack_size: usize,
    frames: VecDeque<Tensor<O, S::Kind>>,
    observation_space: S,
}

impl<G, const O: usize, const BO: usize, const F: usize, const BF: usize, S>
    FrameStackGym<G, O, BO, F, BF, S>
where
    S: ObservationSpace<BF>,
{
    /// Creates a frame stack with a declared space shaped `[stack_size, ...inner_observation_shape]` per item.
    /// Scalars use `[stack_size]`. `BoxSpace` bounds include a separate size-one batch axis.
    pub fn new(gym: G, stack_size: usize, observation_space: S) -> Self {
        const {
            assert!(
                BO == O + 1 || (O == 1 && BO == 1),
                "inner batch rank must be single rank + 1, except scalar observations"
            );
            assert!(
                if O == 1 && BO == 1 {
                    F == 1
                } else {
                    F == O + 1
                },
                "frame rank must be observation rank + 1, except scalar observations"
            );
            assert!(BF == F + 1, "stacked batch rank must be frame rank + 1");
        }
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
    /// Scalar `[1]` frames concatenate to `[frame_count]`. Preserves kind, dtype, device, and gradients.
    fn stacked_observation(&self) -> Tensor<F, S::Kind> {
        let frames = self.frames.iter().cloned().collect::<Vec<_>>();
        if const { O == 1 && BO == 1 } {
            Tensor::cat(frames, 0).reshape([self.frames.len(); F])
        } else {
            Tensor::stack(frames, 0)
        }
    }
}

impl<
    G,
    I,
    const O: usize,
    const A: usize,
    const BO: usize,
    const BA: usize,
    const F: usize,
    const BF: usize,
    S,
> Gym<I, F, A, BF, BA> for FrameStackGym<G, O, BO, F, BF, S>
where
    G: Gym<I, O, A, BO, BA>,
    S: ObservationSpace<BF, Kind = <G::ObservationSpace as ObservationSpace<BO>>::Kind> + Clone,
{
    type Error = FrameStackGymError<G::Error>;
    type ObservationSpace = S;
    type ActionSpace = G::ActionSpace;

    /// Repeats the unbatched rank-`O` reset observation to fill `[stack_size, ...observation_shape]` of rank `F`.
    /// Scalar inputs `[1]` produce `[stack_size]`; all outputs preserve kind, dtype, and device.
    fn reset(&mut self) -> Result<ResetInfo<I, F, S::Kind>, Self::Error> {
        let reset = self.gym.reset().map_err(FrameStackGymError::GymError)?;
        self.frames.clear();
        self.frames
            .extend(std::iter::repeat_with(|| reset.observation.clone()).take(self.stack_size));
        Ok(ResetInfo {
            observation: self.stacked_observation(),
            info: reset.info,
        })
    }

    /// Forwards an unbatched rank-`A` action and appends the rank-`O` observation after dropping the oldest full-stack frame.
    /// Returns rank `F` with `[frame_count, ...observation_shape]`, or `[frame_count]` for scalars, preserving kind, dtype, and device.
    fn step(
        &mut self,
        action: Tensor<A, <Self::ActionSpace as ActionSpace<BA>>::Kind>,
    ) -> Result<StepInfo<I, F, S::Kind>, Self::Error> {
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

impl<G, I, const O: usize, const A: usize, const BO: usize, const BA: usize> Gym<I, O, A, BO, BA>
    for MaxAndSkipGym<G>
where
    G: Gym<I, O, A, BO, BA>,
    <G::ObservationSpace as ObservationSpace<BO>>::Kind: Ordered,
{
    type Error = MaxAndSkipGymError<G::Error>;
    type ObservationSpace = G::ObservationSpace;
    type ActionSpace = G::ActionSpace;

    /// Returns the inner gym's unbatched rank-`O` observation, preserving its shape, kind, dtype, and device.
    fn reset(
        &mut self,
    ) -> Result<ResetInfo<I, O, <Self::ObservationSpace as ObservationSpace<BO>>::Kind>, Self::Error>
    {
        self.gym.reset().map_err(MaxAndSkipGymError::GymError)
    }

    /// Repeats an unbatched rank-`A` action and max-pools the final two unbatched rank-`O` observations.
    /// Preserves item axes, numeric kind, dtype, and device; scalars use `[1]`. Stops on termination or truncation.
    fn step(
        &mut self,
        action: Tensor<A, <Self::ActionSpace as ActionSpace<BA>>::Kind>,
    ) -> Result<StepInfo<I, O, <Self::ObservationSpace as ObservationSpace<BO>>::Kind>, Self::Error>
    {
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

#[cfg(test)]
mod tests {
    use crate::{
        gym::Gym,
        spaces::{BoxSpace, ObservationSpace},
        wrappers::test_support::{TestGym, action, scalar},
    };
    use burn::tensor::{Bool, DType, Device, Int, Tensor, kind::Basic};
    use std::marker::PhantomData;

    use super::{FrameStackGym, MaxAndSkipGym};

    #[derive(Clone)]
    struct TypedSpace<const D: usize, K> {
        shape: [usize; D],
        kind: PhantomData<K>,
    }

    impl<const D: usize, K: Basic> ObservationSpace<D> for TypedSpace<D, K> {
        type Kind = K;

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
        let mut integers = FrameStackGym::<_, 2, 3, 3, 4, _>::new(
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
        let mut booleans = FrameStackGym::<_, 1, 1, 1, 2, _>::new(
            (),
            2,
            TypedSpace::<2, Bool> {
                shape: [1, 2],
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
        assert_eq!(stacked.dims(), [2]);
        assert_eq!(
            stacked.into_data().try_to_vec::<bool>().unwrap(),
            vec![true, false]
        );
    }

    #[test]
    fn frame_stack_duplicates_reset_observation_and_rolls_on_step() {
        let gym = TestGym::new([TestGym::step(2.0, 0.0, false, false, 1)]);
        let observation_space = BoxSpace::new_with_universal_bounds(
            [1, 4],
            -100.0,
            100.0,
            &burn::tensor::Device::flex(),
        );
        let mut wrapper = FrameStackGym::<_, 1, 1, 1, 2>::new(gym, 4, observation_space);

        let reset = wrapper.reset().unwrap();
        let step = wrapper.step(action()).unwrap();

        let observation_space = wrapper.observation_space();
        assert_eq!(observation_space.shape(), vec![4]);
        assert!(observation_space.contains(&reset.observation.clone().unsqueeze::<2>()));
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
}

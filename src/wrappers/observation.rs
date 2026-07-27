//! Wrappers that transform observations.

use std::collections::VecDeque;

use candle_core::Tensor;

use crate::gym::{Gym, ResetInfo, StepInfo};

#[derive(Debug)]
pub enum FrameStackGymError<E> {
    GymError(E),
    CandleError(candle_core::Error),
}

/// Stacks the most recent observations along a leading frame dimension.
pub struct FrameStackGym<G> {
    gym: G,
    stack_size: usize,
    frames: VecDeque<Tensor>,
}

impl<G> FrameStackGym<G> {
    pub fn new(gym: G, stack_size: usize) -> Self {
        assert!(stack_size > 0, "stack_size must be at least 1");
        Self {
            gym,
            stack_size,
            frames: VecDeque::new(),
        }
    }
}

impl<G, I> Gym<I> for FrameStackGym<G>
where
    G: Gym<I>,
{
    type Error = FrameStackGymError<<G as Gym<I>>::Error>;
    type SpaceError = <G as Gym<I>>::SpaceError;

    fn reset(&mut self) -> Result<ResetInfo<I>, Self::Error> {
        let mut reset = self.gym.reset().map_err(FrameStackGymError::GymError)?;
        self.frames = std::iter::repeat_with(|| reset.state.clone())
            .take(self.stack_size)
            .collect();
        let frames = self.frames.iter().cloned().collect::<Vec<_>>();
        reset.state = Tensor::stack(&frames, 0).map_err(FrameStackGymError::CandleError)?;
        Ok(reset)
    }

    /// Forwards one unbatched environment action shaped `action_shape`.
    fn step(&mut self, action: Tensor) -> Result<StepInfo<I>, Self::Error> {
        let mut info = self
            .gym
            .step(action)
            .map_err(FrameStackGymError::GymError)?;
        if self.frames.len() >= self.stack_size {
            self.frames.pop_front();
        }
        self.frames.push_back(info.state.clone());
        let frames = self.frames.iter().cloned().collect::<Vec<_>>();
        info.state = Tensor::stack(&frames, 0).map_err(FrameStackGymError::CandleError)?;
        Ok(info)
    }

    fn action_space(&self) -> Box<dyn crate::spaces::Space<Error = Self::SpaceError>> {
        self.gym.action_space()
    }

    fn observation_space(&self) -> Box<dyn crate::spaces::Space<Error = Self::SpaceError>> {
        self.gym.observation_space()
    }
}

#[derive(Debug)]
pub enum MaxAndSkipGymError<E> {
    GymError(E),
    CandleError(candle_core::Error),
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

impl<G, I> Gym<I> for MaxAndSkipGym<G>
where
    G: Gym<I>,
{
    type Error = MaxAndSkipGymError<<G as Gym<I>>::Error>;
    type SpaceError = <G as Gym<I>>::SpaceError;

    fn reset(&mut self) -> Result<ResetInfo<I>, Self::Error> {
        self.gym.reset().map_err(MaxAndSkipGymError::GymError)
    }

    /// Forwards one unbatched environment action shaped `action_shape`.
    fn step(&mut self, action: Tensor) -> Result<StepInfo<I>, Self::Error> {
        let mut total_reward = 0.0;
        let mut last_observation = None;
        let mut second_last_observation = None;
        let mut final_step = None;

        for _ in 0..self.skip {
            let step = self
                .gym
                .step(action.clone())
                .map_err(MaxAndSkipGymError::GymError)?;
            total_reward += step.reward;
            second_last_observation = last_observation;
            last_observation = Some(step.state.clone());
            let finished = step.done || step.truncated;
            final_step = Some(step);
            if finished {
                break;
            }
        }

        let mut step = final_step.expect("skip is validated to be at least one");
        step.state = match second_last_observation {
            Some(second_last) => Tensor::maximum(
                last_observation
                    .as_ref()
                    .expect("at least one step was taken"),
                &second_last,
            )
            .map_err(MaxAndSkipGymError::CandleError)?,
            None => last_observation.expect("at least one step was taken"),
        };
        step.reward = total_reward;
        Ok(step)
    }

    fn action_space(&self) -> Box<dyn crate::spaces::Space<Error = Self::SpaceError>> {
        self.gym.action_space()
    }

    fn observation_space(&self) -> Box<dyn crate::spaces::Space<Error = Self::SpaceError>> {
        self.gym.observation_space()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        gym::Gym,
        wrappers::test_support::{TestGym, action, scalar},
    };

    use super::{FrameStackGym, MaxAndSkipGym};

    #[test]
    fn frame_stack_duplicates_reset_observation_and_rolls_on_step() {
        let gym = TestGym::new([TestGym::step(2.0, 0.0, false, false, 1)]);
        let mut wrapper = FrameStackGym::new(gym, 4);

        let reset = wrapper.reset().unwrap();
        let step = wrapper.step(action()).unwrap();

        assert_eq!(reset.state.to_vec1::<f32>().unwrap(), vec![100.0; 4]);
        assert_eq!(
            step.state.to_vec1::<f32>().unwrap(),
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

        assert_eq!(scalar(&step.state), 3.0);
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

        assert_eq!(scalar(&step.state), 2.0);
        assert_eq!(step.reward, 0.5);
        assert!(step.truncated);
        assert_eq!(step.info.sequence, 2);
    }
}

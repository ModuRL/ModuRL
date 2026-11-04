use std::convert::Infallible;

use burn::tensor::{Device, Int, Tensor, TensorData};
use rand::{RngExt, SeedableRng, rngs::Xoshiro256PlusPlus};

use crate::{
    gym::{Gym, PerfectStateGym, ResetInfo, StepInfo},
    spaces::{BoxSpace, Discrete},
    tensor_rank::TensorBatchTypes,
};

pub(super) struct StateTestGym {
    rng: Xoshiro256PlusPlus,
    step_count: usize,
    device: Device,
}

impl StateTestGym {
    pub(super) fn new(seed: u64) -> Self {
        Self {
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
            step_count: 0,
            device: Device::flex(),
        }
    }
}

impl Gym for StateTestGym {
    type Error = Infallible;
    type ObservationSpace = BoxSpace<2>;
    type ActionSpace = Discrete;

    fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
        self.step_count = 0;
        Ok(ResetInfo {
            observation: Tensor::from_data([self.rng.random::<f32>()], &self.device),
            info: (),
        })
    }

    /// Accepts one scalar Int action `[1]` and returns a scalar Float observation `[1]`.
    fn step(&mut self, _action: Tensor<1, Int>) -> Result<StepInfo, Self::Error> {
        self.step_count += 1;
        Ok(StepInfo {
            observation: Tensor::from_data([self.rng.random::<f32>()], &self.device),
            reward: self.rng.random::<f32>(),
            done: self.step_count.is_multiple_of(3),
            truncated: false,
            info: (),
        })
    }

    fn observation_space(&self) -> Self::ObservationSpace {
        BoxSpace::new_unbounded([1, 1], &self.device)
    }

    fn action_space(&self) -> Self::ActionSpace {
        Discrete::new(2).unwrap()
    }
}

impl PerfectStateGym for StateTestGym {
    type State = (Xoshiro256PlusPlus, usize);

    fn save_state(&self) -> Result<Self::State, Self::Error> {
        Ok((self.rng.clone(), self.step_count))
    }

    fn set_state(&mut self, state: &Self::State) -> Result<(), Self::Error> {
        self.rng.clone_from(&state.0);
        self.step_count = state.1;
        Ok(())
    }
}

pub(super) fn action() -> Tensor<1, Int> {
    Tensor::from_data([0i32], &Device::flex())
}

pub(super) fn assert_state_round_trip<G, I, const O: usize, const A: usize, T>(
    mut gym: G,
    action: <G::ActionSpace as TensorBatchTypes>::Item,
    prefix_steps: usize,
    replay_steps: usize,
    observation_data: impl Fn(<G::ObservationSpace as TensorBatchTypes>::Item) -> TensorData,
    project_info: impl Fn(&I) -> T,
) where
    G: PerfectStateGym<I, O, A>,
    G::Error: std::fmt::Debug,
    <G::ActionSpace as TensorBatchTypes>::Item: Clone,
    T: std::fmt::Debug + PartialEq,
{
    gym.reset().unwrap();
    for _ in 0..prefix_steps {
        gym.step(action.clone()).unwrap();
    }
    let state = gym.save_state().unwrap();

    let record = |gym: &mut G| {
        (0..replay_steps)
            .map(|_| {
                let step = gym.step(action.clone()).unwrap();
                (
                    observation_data(step.observation),
                    step.reward,
                    step.done,
                    step.truncated,
                    project_info(&step.info),
                )
            })
            .collect::<Vec<_>>()
    };

    let expected = record(&mut gym);
    gym.set_state(&state).unwrap();
    let actual = record(&mut gym);

    assert_eq!(actual, expected);
}

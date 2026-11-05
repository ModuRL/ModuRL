use std::convert::Infallible;

use burn::tensor::{Device, Int, Tensor, TensorData};
use rand::{RngExt, SeedableRng, rngs::Xoshiro256PlusPlus};

use crate::{
    gym::{Gym, LegalActionsGym, PerfectStateGym, PerfectStateMultiGym, ResetInfo, StepInfo},
    spaces::{BoxSpace, Discrete},
    tensor_rank::TensorBatchTypes,
};

pub(crate) struct StateTestGym {
    rng: Xoshiro256PlusPlus,
    step_count: usize,
    device: Device,
}

impl StateTestGym {
    pub(crate) fn new(seed: u64) -> Self {
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

impl LegalActionsGym for StateTestGym {
    fn legal_actions(&self) -> Result<Vec<usize>, Self::Error> {
        Ok(if self.step_count.is_multiple_of(2) {
            vec![0, 1]
        } else {
            vec![1]
        })
    }
}

pub(crate) fn action() -> Tensor<1, Int> {
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

#[derive(Debug, PartialEq)]
struct SlotStepRecord {
    observations: TensorData,
    rewards: TensorData,
    dones: Vec<bool>,
    truncateds: Vec<bool>,
    observation_rows: Vec<TensorData>,
    reward_rows: Vec<TensorData>,
}

pub(crate) fn assert_slot_round_trip<G, I, const O: usize, const A: usize>(
    make: impl Fn() -> G,
    actions: <G::ActionSpace as TensorBatchTypes>::Batch,
    slot: usize,
    prefix_steps: usize,
    replay_steps: usize,
) where
    G: PerfectStateMultiGym<I, O, A>,
    G::Error: std::fmt::Debug,
{
    let mut a = make();
    let mut b = make();
    a.reset().unwrap();
    b.reset().unwrap();
    for _ in 0..prefix_steps {
        a.step(actions.clone()).unwrap();
        b.step(actions.clone()).unwrap();
    }
    let state = a.save_state(slot).unwrap();

    let record = |gym: &mut G, count: usize| {
        (0..count)
            .map(|_| {
                let step = gym.step(actions.clone()).unwrap();
                let observation_rows = (0..gym.num_envs())
                    .map(|row| {
                        step.observations
                            .clone()
                            .slice_dim(0, row..row + 1)
                            .into_data()
                    })
                    .collect();
                let reward_rows = (0..gym.num_envs())
                    .map(|row| step.rewards.clone().slice_dim(0, row..row + 1).into_data())
                    .collect();
                SlotStepRecord {
                    observations: step.observations.into_data(),
                    rewards: step.rewards.into_data(),
                    dones: step.dones,
                    truncateds: step.truncateds,
                    observation_rows,
                    reward_rows,
                }
            })
            .collect::<Vec<_>>()
    };

    let first_half = record(&mut a, replay_steps);
    a.set_state(slot, &state).unwrap();
    let second_half = record(&mut a, replay_steps);
    let uninterrupted = record(&mut b, replay_steps * 2);

    for (first, second) in first_half.iter().zip(&second_half) {
        assert_eq!(first.observation_rows[slot], second.observation_rows[slot]);
        assert_eq!(first.reward_rows[slot], second.reward_rows[slot]);
        assert_eq!(first.dones[slot], second.dones[slot]);
        assert_eq!(first.truncateds[slot], second.truncateds[slot]);
    }

    for (index, (actual, expected)) in first_half
        .iter()
        .chain(&second_half)
        .zip(&uninterrupted)
        .enumerate()
    {
        for other_slot in 0..actual.dones.len() {
            if other_slot != slot {
                assert_eq!(
                    actual.observation_rows[other_slot], expected.observation_rows[other_slot],
                    "observation row differs at step {index}, slot {other_slot}",
                );
                assert_eq!(
                    actual.reward_rows[other_slot], expected.reward_rows[other_slot],
                    "reward row differs at step {index}, slot {other_slot}",
                );
                assert_eq!(actual.dones[other_slot], expected.dones[other_slot]);
                assert_eq!(
                    actual.truncateds[other_slot],
                    expected.truncateds[other_slot]
                );
            }
        }
    }

    for (actual, expected) in first_half.iter().zip(&uninterrupted) {
        assert_eq!(actual.observations, expected.observations);
        assert_eq!(actual.rewards, expected.rewards);
        assert_eq!(actual.dones, expected.dones);
        assert_eq!(actual.truncateds, expected.truncateds);
    }
}

use super::{Gym, MultiGym, MultiGymStepInfo, ResetInfo, StepInfo};
use crate::spaces::{BoxSpace, Discrete};
use burn::tensor::{DType, Device, Int, Tensor, TensorData};

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub(super) enum TestError {
    #[error("forced environment failure")]
    Forced,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TestInfo {
    pub(super) id: usize,
    pub(super) step: usize,
    pub(super) action: i32,
}

pub(super) struct CounterEnv {
    pub(super) id: usize,
    pub(super) step: usize,
    pub(super) truncate: bool,
    pub(super) fail_next: bool,
    pub(super) fail_reset: bool,
}

impl CounterEnv {
    pub(super) fn new(id: usize) -> Self {
        Self {
            id,
            step: 0,
            truncate: id % 2 == 1,
            fail_next: false,
            fail_reset: false,
        }
    }
}

impl Gym<TestInfo> for CounterEnv {
    type Error = TestError;
    type ObservationSpace = BoxSpace<2>;
    type ActionSpace = Discrete;

    /// Accepts scalar Int actions `[1]` and returns unbatched F64 observations `[2]` on the CPU fixture device.
    fn step(&mut self, action: Tensor<1, Int>) -> Result<StepInfo<TestInfo>, Self::Error> {
        assert_eq!(action.dims(), [1]);
        if self.fail_next {
            self.fail_next = false;
            return Err(TestError::Forced);
        }
        let action = action.into_scalar::<i32>();
        self.step += 1;
        let ended = self.step == 2;
        Ok(StepInfo {
            observation: Tensor::from_data(
                [self.id as f64, self.step as f64],
                (&Device::flex(), DType::F64),
            ),
            reward: action as f32,
            done: ended && !self.truncate,
            truncated: ended && self.truncate,
            info: TestInfo {
                id: self.id,
                step: self.step,
                action,
            },
        })
    }

    /// Returns unbatched F64 reset observations `[2]` with reset metadata.
    fn reset(&mut self) -> Result<ResetInfo<TestInfo>, Self::Error> {
        if self.step == 2 && self.fail_reset {
            self.fail_reset = false;
            return Err(TestError::Forced);
        }
        self.step = 0;
        Ok(ResetInfo {
            observation: Tensor::from_data([self.id as f64, 0.0], (&Device::flex(), DType::F64)),
            info: TestInfo {
                id: self.id,
                step: 0,
                action: 0,
            },
        })
    }

    fn observation_space(&self) -> Self::ObservationSpace {
        BoxSpace::new_unbounded([1, 2], &Device::flex())
    }

    fn action_space(&self) -> Self::ActionSpace {
        Discrete::new(10)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct SlotInfo {
    pub(super) group: usize,
    pub(super) slot: usize,
    pub(super) action: f32,
}

pub(super) struct GroupEnv {
    pub(super) group: usize,
    pub(super) count: usize,
    pub(super) observation_width: usize,
    pub(super) action_width: usize,
    pub(super) fail_next: bool,
    #[cfg(feature = "multithreading")]
    pub(super) gate: Option<std::sync::Arc<(std::sync::Mutex<GateState>, std::sync::Condvar)>>,
}

impl GroupEnv {
    pub(super) fn new(group: usize, count: usize) -> Self {
        Self {
            group,
            count,
            observation_width: 3,
            action_width: 1,
            fail_next: false,
            #[cfg(feature = "multithreading")]
            gate: None,
        }
    }
}

impl MultiGym<SlotInfo, 2, 2> for GroupEnv {
    type Error = TestError;
    type ObservationSpace = BoxSpace<2>;
    type ActionSpace = BoxSpace<2>;

    /// Accepts F32 actions `[count, 1]` and returns F32 observations `[count, 3]` and rewards `[count]`.
    fn step(&mut self, action: Tensor<2>) -> Result<MultiGymStepInfo<SlotInfo>, Self::Error> {
        if self.fail_next {
            self.fail_next = false;
            return Err(TestError::Forced);
        }
        #[cfg(feature = "multithreading")]
        if let Some(gate) = &self.gate {
            let (lock, ready) = &**gate;
            let mut state = lock.lock().unwrap();
            state.entered += 1;
            ready.notify_all();
            while !state.released {
                state = ready.wait(state).unwrap();
            }
        }
        let actions = action.into_data().try_to_vec::<f32>().unwrap();
        let observations: Vec<_> = actions
            .iter()
            .enumerate()
            .flat_map(|(slot, &action)| [self.group as f32, slot as f32, action])
            .collect();
        let infos = actions
            .iter()
            .enumerate()
            .map(|(slot, &action)| SlotInfo {
                group: self.group,
                slot,
                action,
            })
            .collect();
        let terminal_observations = actions
            .iter()
            .enumerate()
            .map(|(slot, &value)| {
                (slot + 1 == self.count).then(|| {
                    Tensor::from_data([self.group as f32, slot as f32, -value], &Device::flex())
                })
            })
            .collect();
        Ok(MultiGymStepInfo {
            observations: Tensor::from_data(
                TensorData::new(observations, [self.count, 3]),
                &Device::flex(),
            ),
            rewards: Tensor::from_data(TensorData::new(actions, [self.count]), &Device::flex()),
            infos,
            dones: (0..self.count).map(|slot| slot + 1 == self.count).collect(),
            truncateds: vec![false; self.count],
            terminal_observations,
        })
    }

    fn observation_space(&self) -> Self::ObservationSpace {
        BoxSpace::new_unbounded([1, self.observation_width], &Device::flex())
    }

    fn action_space(&self) -> Self::ActionSpace {
        BoxSpace::new_unbounded([1, self.action_width], &Device::flex())
    }

    fn num_envs(&self) -> usize {
        self.count
    }

    /// Returns F32 reset observations `[count, 3]` in slot order.
    fn reset(&mut self) -> Result<Tensor<2>, Self::Error> {
        let values = (0..self.count)
            .flat_map(|slot| [self.group as f32, slot as f32, 0.0])
            .collect();
        Ok(Tensor::from_data(
            TensorData::new(values, [self.count, 3]),
            &Device::flex(),
        ))
    }
}

#[cfg(feature = "multithreading")]
#[derive(Default)]
pub(super) struct GateState {
    pub(super) entered: usize,
    pub(super) released: bool,
}

pub(super) struct ContinuousEnv;

impl Gym<(), 1, 1, 2, 2> for ContinuousEnv {
    type Error = TestError;
    type ObservationSpace = BoxSpace<2>;
    type ActionSpace = BoxSpace<2>;

    /// Returns each unbatched F64 action vector `[2]` as the terminal observation `[2]` on the fixture CPU device.
    fn step(&mut self, action: Tensor<1>) -> Result<StepInfo, Self::Error> {
        assert_eq!(action.dims(), [2]);
        Ok(StepInfo {
            observation: action,
            reward: 1.0,
            done: true,
            truncated: false,
            info: (),
        })
    }

    /// Returns an unbatched F64 observation `[2]` on the fixture CPU device.
    fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
        Ok(ResetInfo {
            observation: Tensor::zeros([2], (&Device::flex(), DType::F64)),
            info: (),
        })
    }

    fn observation_space(&self) -> Self::ObservationSpace {
        BoxSpace::new_unbounded([1, 2], &Device::flex())
    }

    fn action_space(&self) -> Self::ActionSpace {
        BoxSpace::new_with_universal_bounds([1, 2], -10.0, 10.0, &Device::flex())
    }
}

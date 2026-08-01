use crate::gym::MultiGym;
use candle_core::Tensor;
pub mod a2c;
pub mod deterministic_actor_critic;
pub mod ppo;
pub mod q_learning;
mod replay_device_strategy;
pub mod sac;

pub use replay_device_strategy::ReplayDeviceStrategy;

pub trait Agent<I = ()> {
    type Error;
    type GymError;
    type SpaceError;

    /// Selects actions for observations shaped
    /// `[batch_size, ...observation_shape]`.
    ///
    /// Returns environment actions shaped
    /// `[batch_size, ...environment_action_shape]`.
    fn act(&mut self, observation: &Tensor) -> Result<Tensor, Self::Error>;
    fn learn(
        &mut self,
        env: &mut dyn MultiGym<I, Error = Self::GymError, SpaceError = Self::SpaceError>,
        num_timesteps: usize,
    ) -> Result<(), Self::Error>;
}

#[cfg(test)]
pub(crate) mod test_support {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::Optimizer;

    use crate::{
        gym::{Gym, ResetInfo, StepInfo},
        spaces::{BoxSpace, Discrete, Space},
    };

    pub(crate) struct FixedEnv {
        device: Device,
    }

    impl FixedEnv {
        pub(crate) fn new(device: Device) -> Self {
            Self { device }
        }
    }

    impl Gym for FixedEnv {
        type Error = candle_core::Error;
        type SpaceError = candle_core::Error;

        /// Steps with one scalar discrete action shaped `[]`.
        fn step(&mut self, _action: Tensor) -> Result<StepInfo, Self::Error> {
            Ok(StepInfo {
                state: Tensor::zeros(&[4], DType::F32, &self.device)?,
                reward: 1.0,
                done: false,
                truncated: false,
                info: (),
            })
        }

        fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
            Ok(ResetInfo {
                state: Tensor::zeros(&[4], DType::F32, &self.device)?,
                info: (),
            })
        }

        fn observation_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
            Box::new(BoxSpace::new(
                Tensor::zeros(&[4], DType::F32, &self.device).unwrap(),
                Tensor::ones(&[4], DType::F32, &self.device).unwrap(),
            ))
        }

        fn action_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
            Box::new(Discrete::new(2))
        }
    }

    pub(crate) struct CountingOptimizer {
        pub(crate) steps: usize,
        learning_rate: f64,
    }

    impl CountingOptimizer {
        pub(crate) fn with_learning_rate(learning_rate: f64) -> Self {
            Self {
                steps: 0,
                learning_rate,
            }
        }
    }

    impl Optimizer for CountingOptimizer {
        type Config = f64;

        fn new(
            _vars: Vec<candle_core::Var>,
            learning_rate: Self::Config,
        ) -> candle_core::Result<Self> {
            Ok(Self::with_learning_rate(learning_rate))
        }

        fn step(&mut self, _grads: &candle_core::backprop::GradStore) -> candle_core::Result<()> {
            self.steps += 1;
            Ok(())
        }

        fn learning_rate(&self) -> f64 {
            self.learning_rate
        }

        fn set_learning_rate(&mut self, learning_rate: f64) {
            self.learning_rate = learning_rate;
        }
    }

    pub(crate) struct FixedContinuousEnv {
        device: Device,
    }

    impl FixedContinuousEnv {
        pub(crate) fn new(device: Device) -> Self {
            Self { device }
        }
    }

    impl Gym for FixedContinuousEnv {
        type Error = candle_core::Error;
        type SpaceError = candle_core::Error;

        /// Steps with one continuous action shaped `[1]`.
        fn step(&mut self, _action: Tensor) -> Result<StepInfo, Self::Error> {
            Ok(StepInfo {
                state: Tensor::zeros(&[4], DType::F32, &self.device)?,
                reward: 1.0,
                done: false,
                truncated: false,
                info: (),
            })
        }

        fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
            Ok(ResetInfo {
                state: Tensor::zeros(&[4], DType::F32, &self.device)?,
                info: (),
            })
        }

        fn observation_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
            Box::new(BoxSpace::new_with_universal_bounds(
                vec![4],
                -1.0,
                1.0,
                &self.device,
            ))
        }

        fn action_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
            Box::new(BoxSpace::new_with_universal_bounds(
                vec![1],
                -1.0,
                1.0,
                &self.device,
            ))
        }
    }
}

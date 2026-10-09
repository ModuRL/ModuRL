use crate::{
    gym::MultiGym,
    spaces::{ActionSpace, ObservationSpace},
};
use burn::tensor::Tensor;

pub mod a2c;
pub mod deterministic_actor_critic;
pub mod ppo;
pub mod q_learning;
mod replay_device_strategy;
pub mod sac;

pub use replay_device_strategy::{ReplayDeviceStrategy, ReplayStorageConfig};

/// Selects native environment actions and trains through batched environment interaction.
/// Ranks `O` and `A` include the batch axis; `U` is the rank of each unbatched terminal observation.
/// Requires `O = U + 1`, including scalar observations `[batch_size, 1]`.
/// Associated spaces determine the native observation and action kinds.
pub trait Agent<I = (), const O: usize = 2, const A: usize = 2, const U: usize = 1> {
    type Error;
    type GymError;
    type SpaceError;
    type ObservationSpace: ObservationSpace<O>;
    type ActionSpace: ActionSpace<A, Error = Self::SpaceError>;

    /// Selects rank-`A` actions `[batch_size, ...action_shape]` from rank-`O` observations `[batch_size, ...observation_shape]`.
    /// Preserves the batch axis. The agent defines supported observation dtypes/devices and returned action dtypes/devices.
    fn act(
        &mut self,
        observations: &Tensor<O, <Self::ObservationSpace as ObservationSpace<O>>::Kind>,
    ) -> Result<Tensor<A, <Self::ActionSpace as ActionSpace<A>>::Kind>, Self::Error>;

    /// Trains with rank-`O` observation batches and rank-`A` action batches, each with `num_envs` rows.
    /// Terminal observations have unbatched rank `U`; scalar terminal observations use `[1]`.
    /// The environment must use the associated space types and meet the agent's dtype and device requirements.
    fn learn(
        &mut self,
        env: &mut dyn MultiGym<
            I,
            O,
            A,
            U,
            Error = Self::GymError,
            ObservationSpace = Self::ObservationSpace,
            ActionSpace = Self::ActionSpace,
        >,
        num_timesteps: usize,
    ) -> Result<(), Self::Error>;
}

#[cfg(test)]
pub(crate) mod test_support {
    use burn::tensor::{DType, Device, Int, Tensor};
    use candle_nn::Optimizer;
    use std::convert::Infallible;

    use crate::{
        gym::{Gym, ResetInfo, StepInfo},
        spaces::{BoxSpace, Discrete},
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
        type Error = Infallible;
        type ObservationSpace = BoxSpace<2>;
        type ActionSpace = Discrete;

        /// Accepts one scalar Int action `[1]` and returns unbatched F32 observations `[4]` on the fixture device.
        fn step(&mut self, _action: Tensor<1, Int>) -> Result<StepInfo, Self::Error> {
            Ok(StepInfo {
                observation: Tensor::zeros([4], (&self.device, DType::F32)),
                reward: 1.0,
                done: false,
                truncated: false,
                info: (),
            })
        }

        /// Returns unbatched F32 observations `[4]` on the fixture device.
        fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
            Ok(ResetInfo {
                observation: Tensor::zeros([4], (&self.device, DType::F32)),
                info: (),
            })
        }

        fn observation_space(&self) -> Self::ObservationSpace {
            BoxSpace::new(
                Tensor::zeros([1, 4], (&self.device, DType::F32)),
                Tensor::ones([1, 4], (&self.device, DType::F32)),
            )
        }

        fn action_space(&self) -> Self::ActionSpace {
            Discrete::new(2)
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

    impl Gym<(), 1, 1, 2, 2> for FixedContinuousEnv {
        type Error = Infallible;
        type ObservationSpace = BoxSpace<2>;
        type ActionSpace = BoxSpace<2>;

        /// Accepts one Float action vector `[1]` and returns unbatched F32 observations `[4]` on the fixture device.
        fn step(&mut self, _action: Tensor<1>) -> Result<StepInfo, Self::Error> {
            Ok(StepInfo {
                observation: Tensor::zeros([4], (&self.device, DType::F32)),
                reward: 1.0,
                done: false,
                truncated: false,
                info: (),
            })
        }

        /// Returns unbatched F32 observations `[4]` on the fixture device.
        fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
            Ok(ResetInfo {
                observation: Tensor::zeros([4], (&self.device, DType::F32)),
                info: (),
            })
        }

        fn observation_space(&self) -> Self::ObservationSpace {
            BoxSpace::new_with_universal_bounds([1, 4], -1.0, 1.0, &self.device)
        }

        fn action_space(&self) -> Self::ActionSpace {
            BoxSpace::new_with_universal_bounds([1, 1], -1.0, 1.0, &self.device)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Agent;
    use super::test_support::{FixedContinuousEnv, FixedEnv};
    use crate::{
        gym::{Gym, MultiGym, VectorizedGymError, VectorizedGymWrapper},
        spaces::{BoxSpace, Discrete, SpaceError},
    };
    use burn::tensor::{DType, Device, Int, Tensor};
    use std::convert::Infallible;

    struct FixedAgent;

    impl Agent for FixedAgent {
        type Error = VectorizedGymError<Infallible>;
        type GymError = VectorizedGymError<Infallible>;
        type SpaceError = SpaceError;
        type ObservationSpace = BoxSpace<2>;
        type ActionSpace = Discrete;

        /// Returns zero Int actions `[batch_size, 1]` for Float observations `[batch_size, features]` on the observation device.
        fn act(&mut self, observations: &Tensor<2>) -> Result<Tensor<2, Int>, Self::Error> {
            Ok(Tensor::zeros(
                [observations.dims()[0], 1],
                &observations.device(),
            ))
        }

        /// Steps observation batches `[num_envs, 4]` with scalar action batches `[num_envs, 1]`.
        fn learn(
            &mut self,
            env: &mut dyn MultiGym<
                Error = Self::GymError,
                ObservationSpace = Self::ObservationSpace,
                ActionSpace = Self::ActionSpace,
            >,
            num_timesteps: usize,
        ) -> Result<(), Self::Error> {
            let mut observations = env.reset()?;
            for _ in (0..num_timesteps).step_by(env.num_envs()) {
                observations = env.step(self.act(&observations)?)?.observations;
            }
            Ok(())
        }
    }

    #[test]
    fn agent_trait_object_accepts_native_tensors_and_environment_batches() {
        let device = Device::flex();
        let mut agent = FixedAgent;
        let agent: &mut dyn Agent<
            Error = VectorizedGymError<Infallible>,
            GymError = VectorizedGymError<Infallible>,
            SpaceError = SpaceError,
            ObservationSpace = BoxSpace<2>,
            ActionSpace = Discrete,
        > = &mut agent;
        let mut single = VectorizedGymWrapper::from(FixedEnv::new(device.clone()));
        let mut multiple = VectorizedGymWrapper::new(vec![
            FixedEnv::new(device.clone()),
            FixedEnv::new(device.clone()),
        ])
        .unwrap();
        let observations = multiple.reset().unwrap();
        let actions = agent.act(&observations).unwrap();
        assert_eq!(actions.dims(), [2, 1]);
        assert_eq!(actions.device(), device);
        assert_eq!(actions.into_data().try_to_vec::<i32>().unwrap(), vec![0, 0]);
        agent.learn(&mut single, 1).unwrap();
        agent.learn(&mut multiple, 2).unwrap();
    }

    #[test]
    fn counting_optimizer_counts_whole_updates_with_multiple_parameters() {
        use super::test_support::CountingOptimizer;
        use candle_nn::Optimizer;

        let device = candle_core::Device::Cpu;
        let first = candle_core::Var::new(&[1.0f32], &device).unwrap();
        let second = candle_core::Var::new(&[2.0f32], &device).unwrap();
        let loss = (first.as_tensor() + second.as_tensor()).unwrap();
        let mut optimizer = CountingOptimizer::new(vec![first, second], 0.1).unwrap();
        optimizer.backward_step(&loss).unwrap();
        assert_eq!(optimizer.steps, 1);
        optimizer.set_learning_rate(0.2);
        assert_eq!(optimizer.learning_rate(), 0.2);
    }

    #[test]
    fn fixed_environments_keep_single_and_batched_layouts() {
        let device = Device::flex();
        let mut discrete = FixedEnv::new(device.clone());
        assert_eq!(discrete.reset().unwrap().observation.dims(), [4]);
        assert_eq!(discrete.observation_space().shape(), vec![4]);
        assert_eq!(discrete.action_space().shape(), vec![1]);
        let mut discrete = VectorizedGymWrapper::from(discrete);
        assert_eq!(discrete.reset().unwrap().dims(), [1, 4]);
        let step = discrete
            .step(Tensor::<2, Int>::from_data([[0i32]], &device))
            .unwrap();
        assert_eq!(step.observations.dims(), [1, 4]);
        assert_eq!(step.observations.dtype(), DType::F32);
        assert_eq!(step.rewards.into_scalar::<f32>(), 1.0);
        assert_eq!(step.dones, vec![false]);
        assert_eq!(step.truncateds, vec![false]);

        let mut continuous = FixedContinuousEnv::new(device.clone());
        assert_eq!(continuous.reset().unwrap().observation.dims(), [4]);
        assert_eq!(continuous.action_space().shape(), vec![1]);
        let mut continuous = VectorizedGymWrapper::from(continuous);
        assert_eq!(continuous.reset().unwrap().dims(), [1, 4]);
        let step = continuous
            .step(Tensor::from_data([[0.0f32]], &device))
            .unwrap();
        assert_eq!(step.observations.dims(), [1, 4]);
        assert_eq!(step.rewards.into_scalar::<f32>(), 1.0);
        assert_eq!(step.dones, vec![false]);
        assert_eq!(step.truncateds, vec![false]);
    }
}

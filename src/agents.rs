use crate::tensor_rank::TensorBatchTypes;
use crate::{
    gym::{MultiGym, MultiGymStepInfo},
    spaces::{ActionSpace, ObservationSpace},
};

pub mod a2c;
pub mod action_value_critic;
pub mod deterministic_actor_critic;

pub mod ppo;
pub mod q_learning;
mod replay_device_strategy;
pub mod sac;

pub(crate) use replay_device_strategy::ReplayStorageConfig;

/// Actions returned by [`Agent::act_raw`].
///
/// `raw` is used for training; `environment` is passed to the environment.
#[derive(Debug, Clone)]
pub struct AgentActions<R, A> {
    pub raw: R,
    pub environment: A,
}

/// Selects native environment actions and trains through batched environment interaction.
/// Ranks `O` and `A` include the batch axis; `PrevRank` determines each unbatched terminal observation type.
/// Scalar observations use `[batch_size, 1]` and terminal observations use `[1]`.
/// Associated spaces determine the native observation and action kinds.
pub trait Agent<I = (), const O: usize = 2, const A: usize = 2> {
    type Error;
    type GymError;

    type ObservationSpace: ObservationSpace<O>;
    type ActionSpace: ActionSpace<A>;

    /// Raw policy output used by `learn_from_timesteps`.
    /// PPO/A2C use rank-L policy samples before action mapping; other agents use environment action batches.
    type RawAction;

    /// Selects actions from observations shaped `[batch_size, ...observation_shape]`.
    #[allow(clippy::type_complexity)]
    fn act_raw(
        &mut self,
        observations: &<Self::ObservationSpace as TensorBatchTypes>::Batch,
    ) -> Result<
        AgentActions<Self::RawAction, <Self::ActionSpace as TensorBatchTypes>::Batch>,
        Self::Error,
    >;

    /// Selects environment actions from observations shaped `[batch_size, ...observation_shape]`.
    fn act(
        &mut self,
        observations: &<Self::ObservationSpace as TensorBatchTypes>::Batch,
    ) -> Result<<Self::ActionSpace as TensorBatchTypes>::Batch, Self::Error> {
        Ok(self.act_raw(observations)?.environment)
    }

    /// Trains with rank-`O` observation batches and rank-`A` action batches, each with `num_envs` rows.
    /// Terminal observations have rank `O - 1`; scalar terminal observations use `[1]`.
    /// The environment must use the associated space types and meet the agent's dtype and device requirements.
    fn learn(
        &mut self,
        env: &mut dyn MultiGym<
            I,
            O,
            A,
            Error = Self::GymError,
            ObservationSpace = Self::ObservationSpace,
            ActionSpace = Self::ActionSpace,
        >,
        num_timesteps: usize,
    ) -> Result<(), Self::Error>;

    /// Stores one caller-collected step from observations shaped `[num_envs, ...observation_shape]`.
    /// `actions` must come from `act_raw` on those observations; steps must be consecutive for the same environments.
    /// Schedules need the builder's `training_horizon` because this method has no total timestep count.
    fn learn_from_timesteps(
        &mut self,
        observations: &<Self::ObservationSpace as TensorBatchTypes>::Batch,
        actions: AgentActions<Self::RawAction, <Self::ActionSpace as TensorBatchTypes>::Batch>,
        step: MultiGymStepInfo<I, <Self::ObservationSpace as TensorBatchTypes>::Item>,
    ) -> Result<(), Self::Error>;
}

#[cfg(test)]
pub(crate) mod test_support {
    use burn::tensor::{DType, Device, Int, Tensor};
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
            Discrete::new(2).unwrap()
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

    impl Gym<(), 2, 2> for FixedContinuousEnv {
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
    use super::test_support::{FixedContinuousEnv, FixedEnv};
    use super::{Agent, AgentActions};
    use crate::{
        gym::{Gym, MultiGym, MultiGymStepInfo, VectorizedGymError, VectorizedGymWrapper},
        spaces::{BoxSpace, Discrete},
    };
    use burn::tensor::{DType, Device, Int, Tensor};
    use std::convert::Infallible;

    struct FixedAgent;

    impl Agent for FixedAgent {
        type Error = VectorizedGymError<Infallible>;
        type GymError = VectorizedGymError<Infallible>;
        type ObservationSpace = BoxSpace<2>;
        type ActionSpace = Discrete;
        type RawAction = Tensor<2, Int>;

        /// Returns zero Int actions `[batch_size, 1]` for Float observations `[batch_size, features]` on the observation device.
        fn act_raw(
            &mut self,
            observations: &Tensor<2>,
        ) -> Result<AgentActions<Tensor<2, Int>, Tensor<2, Int>>, Self::Error> {
            let actions = Tensor::zeros([observations.dims()[0], 1], &observations.device());
            Ok(AgentActions {
                raw: actions.clone(),
                environment: actions,
            })
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

        /// Accepts observations and step data with matching environment batch shapes.
        fn learn_from_timesteps(
            &mut self,
            _observations: &Tensor<2>,
            _actions: AgentActions<Tensor<2, Int>, Tensor<2, Int>>,
            _step: MultiGymStepInfo<(), Tensor<1>>,
        ) -> Result<(), Self::Error> {
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
            ObservationSpace = BoxSpace<2>,
            ActionSpace = Discrete,
            RawAction = Tensor<2, Int>,
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

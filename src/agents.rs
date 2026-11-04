use crate::tensor_rank::TensorBatchTypes;
use crate::{
    gym::{MultiGym, MultiGymStepInfo},
    spaces::{ActionSpace, ObservationSpace},
};
use burn::tensor::{Tensor, kind::Basic};

pub mod a2c;
pub mod action_value_critic;
pub mod deterministic_actor_critic;

pub mod ppo;
pub mod q_learning;
mod replay_device_strategy;
pub mod sac;

pub(crate) use replay_device_strategy::ReplayStorageConfig;

/// Training actions selected by [`Agent::act_for_training`] and converted for the environment on demand.
pub trait ToEnvironmentActions {
    type Actions;
    type Error;

    /// Returns the action batch passed to `MultiGym::step`, shape `[num_envs, ...action_shape]`.
    fn to_environment_actions(&self) -> Result<Self::Actions, Self::Error>;
}

impl<const D: usize, K: Basic> ToEnvironmentActions for Tensor<D, K> {
    type Actions = Self;
    type Error = std::convert::Infallible;

    fn to_environment_actions(&self) -> Result<Self, Self::Error> {
        Ok(self.clone())
    }
}

/// Selects native environment actions and trains through batched environment interaction.
/// Ranks `O` and `A` include the batch axis; `PrevRank` determines each unbatched terminal observation type.
/// Scalar observations use `[batch_size, 1]` and terminal observations use `[1]`.
/// Associated spaces determine the native observation and action kinds.
pub trait Agent<I = (), const O: usize = 2, const A: usize = 2> {
    type Error: From<<Self::TrainingActions as ToEnvironmentActions>::Error>;
    type GymError;

    type ObservationSpace: ObservationSpace<O>;
    type ActionSpace: ActionSpace<A>;

    /// Training actions used by `learn_from_timesteps`.
    type TrainingActions: ToEnvironmentActions<
        Actions = <Self::ActionSpace as TensorBatchTypes>::Batch,
    >;

    /// Selects training actions from observations shaped `[batch_size, ...observation_shape]`.
    fn act_for_training(
        &mut self,
        observations: &<Self::ObservationSpace as TensorBatchTypes>::Batch,
    ) -> Result<Self::TrainingActions, Self::Error>;

    /// Selects environment actions from observations shaped `[batch_size, ...observation_shape]`.
    fn act(
        &mut self,
        observations: &<Self::ObservationSpace as TensorBatchTypes>::Batch,
    ) -> Result<<Self::ActionSpace as TensorBatchTypes>::Batch, Self::Error> {
        Ok(self
            .act_for_training(observations)?
            .to_environment_actions()?)
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
    /// `actions` must come from `act_for_training` on those observations; steps must be consecutive for the same environments.
    /// Schedules need the builder's `training_horizon` because this method has no total timestep count.
    fn learn_from_timesteps(
        &mut self,
        observations: &<Self::ObservationSpace as TensorBatchTypes>::Batch,
        training_actions: Self::TrainingActions,
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
    use super::{Agent, ToEnvironmentActions};
    use crate::{
        gym::{Gym, MultiGym, MultiGymStepInfo, VectorizedGymError, VectorizedGymWrapper},
        spaces::{BoxSpace, Discrete},
    };
    use burn::tensor::{DType, Device, Int, Tensor};
    use std::convert::Infallible;

    struct FixedAgent;

    impl Agent for FixedAgent {
        type Error = Box<dyn std::error::Error>;
        type GymError = VectorizedGymError<Infallible>;
        type ObservationSpace = BoxSpace<2>;
        type ActionSpace = Discrete;
        type TrainingActions = Tensor<2, Int>;

        /// Returns zero Int actions `[batch_size, 1]` for Float observations `[batch_size, features]` on the observation device.
        fn act_for_training(
            &mut self,
            observations: &Tensor<2>,
        ) -> Result<Tensor<2, Int>, Self::Error> {
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

        /// Accepts observations and step data with matching environment batch shapes.
        fn learn_from_timesteps(
            &mut self,
            _observations: &Tensor<2>,
            _training_actions: Tensor<2, Int>,
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
            Error = Box<dyn std::error::Error>,
            GymError = VectorizedGymError<Infallible>,
            ObservationSpace = BoxSpace<2>,
            ActionSpace = Discrete,
            TrainingActions = Tensor<2, Int>,
        > = &mut agent;
        let mut single = VectorizedGymWrapper::from(FixedEnv::new(device.clone()));
        let mut multiple = VectorizedGymWrapper::new(vec![
            FixedEnv::new(device.clone()),
            FixedEnv::new(device.clone()),
        ])
        .unwrap();
        let observations = multiple.reset().unwrap();
        let training_actions = (|| {
            let training_actions = agent.act_for_training(&observations)?;
            Ok::<_, Box<dyn std::error::Error>>(training_actions.to_environment_actions()?)
        })()
        .unwrap();
        let actions = agent.act(&observations).unwrap();
        assert_eq!(
            training_actions.into_data().try_to_vec::<i32>().unwrap(),
            vec![0, 0]
        );
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

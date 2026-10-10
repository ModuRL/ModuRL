use bon::bon;
use burn::{
    module::Module,
    optim::ModuleOptimizer,
    tensor::{Bool, DType, Int, Tensor, kind::Autodiff},
};

use super::{
    QAgentError, QCollectionLogEntry, QLearningAgent, QLearningLogger, QLearningTarget, QLogEntry,
    selected_action_q_values,
};
use crate::{
    agents::{Agent, ReplayStorageConfig},
    gym::MultiGym,
    models::Forward,
    objectives::bellman_targets,
    parameter_schedule::{LinearSchedule, ParameterSchedule},
    spaces::{Discrete, ObservationSpace, SpaceError},
    tensor_rank::{NextRank, PrevRank},
};

pub trait DDQNLogger<I = ()> {
    fn log(&mut self, info: &QLogEntry);

    fn log_collection(&mut self, _info: &QCollectionLogEntry<I>) {}
}

struct DDQNLoggingInfo<'a, I> {
    logger: &'a mut dyn DDQNLogger<I>,
}

impl<I> QLearningLogger<I> for Option<DDQNLoggingInfo<'_, I>> {
    fn log_update(&mut self, entry: &QLogEntry) {
        if let Some(info) = self {
            info.logger.log(entry);
        }
    }

    fn log_collection(&mut self, entry: &QCollectionLogEntry<I>) {
        if let Some(info) = self {
            info.logger.log_collection(entry);
        }
    }
}

struct DDQNTarget;

impl QLearningTarget for DDQNTarget {
    fn requires_online_next_q_values() -> bool {
        true
    }

    /// Computes Float targets [batch_size, 1] from rewards and Bool termination flags [batch_size, 1].
    /// Q tensors use [batch_size, action_count], the reward dtype, and the reward device.
    /// The online network selects next actions; the target network supplies their values.
    /// The shared training loop detaches the targets before computing the loss.
    fn target_q_values(
        rewards: &Tensor<2>,
        next_dones: &Tensor<2, Bool>,
        online_next_q_values: Option<&Tensor<2>>,
        target_next_q_values: &Tensor<2>,
        gamma: f32,
    ) -> Tensor<2> {
        let online_next_q_values = online_next_q_values
            .expect("DDQN target calculation requires online next-observation Q values");
        let next_actions = online_next_q_values.clone().argmax(1);
        let next_q_values = selected_action_q_values(target_next_q_values, &next_actions);
        bellman_targets(
            rewards.clone(),
            next_dones.clone(),
            next_q_values,
            f64::from(gamma),
        )
    }
}

/// Double Deep Q-Network agent with an owned Burn model and optimizer.
/// Rank R includes the observation batch axis. Native observation kind and dtype are preserved.
/// The model returns Float Q values [batch_size, action_count]; selected Int actions use [batch_size, 1].
pub struct DDQNAgent<'a, M, S, GE, I = (), const R: usize = 2>
where
    S: ObservationSpace<R>,
    S::Kind: Autodiff,
{
    inner: QLearningAgent<M, S, GE, DDQNTarget, R>,
    logging_info: Option<DDQNLoggingInfo<'a, I>>,
}

#[bon]
impl<'a, M, S, GE, I, const R: usize, ME: std::fmt::Debug> DDQNAgent<'a, M, S, GE, I, R>
where
    S: ObservationSpace<R>,
    S::Kind: Autodiff,
    M: Module + Forward<R, 2, S::Kind, Error = ME>,
    GE: std::fmt::Debug,
{
    /// Creates an agent for native observations [batch_size, ...observation_shape] and Int actions [batch_size, 1].
    /// The target model starts as a detached copy of the online model and receives periodic hard updates.
    /// Model parameters and Q outputs must use the optimization device and compute dtype.
    /// Replay stores detached observations in their native kind and first-batch dtype.
    /// Construction returns an error for invalid sizes, schedules, learning rate, dtype, or observation rank.
    #[builder]
    pub fn builder(
        action_space: Discrete,
        observation_space: S,
        online_q_network: M,
        #[builder(into)] optimizer: ModuleOptimizer,
        learning_rate: f64,
        #[builder(default = 1000)] target_update_interval: usize,
        #[builder(
            default = Box::new(LinearSchedule::new(1.0, 0.1)),
            with = |schedule: impl ParameterSchedule + 'static| Box::new(schedule)
        )]
        epsilon_schedule: Box<dyn ParameterSchedule>,
        #[builder(default = 10000)] replay_capacity: usize,
        #[builder(default = 32)] batch_size: usize,
        #[builder(default = 0.99)] gamma: f32,
        #[builder(default = 4)] update_frequency: usize,
        #[builder(default = 1000)] training_start: usize,
        training_horizon: usize,
        logger: Option<&'a mut dyn DDQNLogger<I>>,
        replay_storage_config: ReplayStorageConfig,
        #[builder(default = DType::F32)] dtype: DType,
    ) -> Result<Self, QAgentError<GE, ME>> {
        let inner = QLearningAgent::<M, S, GE, DDQNTarget, R>::builder()
            .action_space(action_space)
            .observation_space(observation_space)
            .online_q_network(online_q_network)
            .optimizer(optimizer)
            .learning_rate(learning_rate)
            .target_update_interval(target_update_interval)
            .epsilon_schedule(move |progress| epsilon_schedule.value(progress))
            .replay_capacity(replay_capacity)
            .batch_size(batch_size)
            .gamma(gamma)
            .update_frequency(update_frequency)
            .training_start(training_start)
            .training_horizon(training_horizon)
            .replay_storage_config(replay_storage_config)
            .dtype(dtype)
            .build()?;
        Ok(Self {
            inner,
            logging_info: logger.map(|logger| DDQNLoggingInfo { logger }),
        })
    }

    pub fn get_action_space(&self) -> &Discrete {
        self.inner.get_action_space()
    }

    pub fn get_observation_space(&self) -> &S {
        self.inner.get_observation_space()
    }
}

impl<M, S, GE, I, const R: usize, const U: usize> Agent<I, R, 2> for DDQNAgent<'_, M, S, GE, I, R>
where
    S: ObservationSpace<R>,
    S::Kind: Autodiff,
    M: Module + Forward<R, 2, S::Kind>,
    M::Error: std::fmt::Debug,
    GE: std::fmt::Debug,
    Tensor<R, S::Kind>: PrevRank<Prev = Tensor<U, S::Kind>>,
    Tensor<U, S::Kind>: NextRank<Next = Tensor<R, S::Kind>>,
{
    type Error = QAgentError<GE, M::Error>;
    type GymError = GE;
    type SpaceError = SpaceError;
    type ObservationSpace = S;
    type ActionSpace = Discrete;

    /// Selects U32 Int actions [batch_size, 1] for native rank-R observations [batch_size, ...observation_shape].
    /// Transfers observations to the optimization device and preserves their kind and dtype.
    fn act(&mut self, observations: &Tensor<R, S::Kind>) -> Result<Tensor<2, Int>, Self::Error> {
        self.inner.act(observations)
    }

    /// Trains with native observations [num_envs, ...observation_shape] and Int actions [num_envs, 1].
    /// Uses terminal observations for targets and keeps termination distinct from truncation.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            name = "ddqn.learn",
            target = "modurl::performance",
            skip_all,
            fields(num_timesteps)
        )
    )]
    fn learn(
        &mut self,
        env: &mut dyn MultiGym<I, R, 2, Error = GE, ObservationSpace = S, ActionSpace = Discrete>,
        num_timesteps: usize,
    ) -> Result<(), Self::Error> {
        self.inner.learn(env, num_timesteps, &mut self.logging_info)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        agents::q_learning::tests::{ObservationFixture, network},
        agents::{ReplayDeviceStrategy, test_support::FixedEnv},
        gym::{VectorizedGymError, VectorizedGymWrapper},
        parameter_schedule::ConstantSchedule,
    };
    use burn::{optim::SgdConfig, tensor::Device};
    use std::{convert::Infallible, marker::PhantomData};

    #[derive(Default)]
    struct RecordingLogger {
        update_timesteps: Vec<usize>,
        collection_timesteps: Vec<usize>,
    }

    impl DDQNLogger for RecordingLogger {
        fn log(&mut self, entry: &QLogEntry) {
            assert_eq!(entry.q_values.dims(), [1, 1]);
            assert_eq!(entry.replay_rewards.dims(), [1, 1]);
            self.update_timesteps.push(entry.collection_timestep);
        }

        fn log_collection(&mut self, entry: &QCollectionLogEntry) {
            assert_eq!(entry.collection_rewards.dims(), [2, 1]);
            self.collection_timesteps.push(entry.collection_timestep);
        }
    }

    #[test]
    fn public_agent_runs_actions_collection_training_and_logging() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        for dtype in [DType::F32, DType::F64] {
            let device = Device::flex().autodiff();
            let mut env = VectorizedGymWrapper::new(vec![
                FixedEnv::new(device.clone().without_autodiff()),
                FixedEnv::new(device.clone().without_autodiff()),
            ])
            .unwrap();
            let mut logger = RecordingLogger::default();
            let mut agent = DDQNAgent::builder()
                .action_space(Discrete::new(2))
                .observation_space(env.observation_space())
                .online_q_network(network(&device, dtype))
                .optimizer(SgdConfig::new().init())
                .learning_rate(0.01)
                .dtype(dtype)
                .epsilon_schedule(ConstantSchedule::new(0.0))
                .replay_capacity(4)
                .batch_size(1)
                .training_start(1)
                .update_frequency(1)
                .target_update_interval(2)
                .training_horizon(4)
                .logger(&mut logger)
                .replay_storage_config(ReplayStorageConfig::new(ReplayDeviceStrategy::OneDevice(
                    device.clone(),
                )))
                .build()
                .unwrap();
            assert_eq!(agent.get_action_space().get_possible_values(), 2);
            assert_eq!(agent.get_observation_space().shape(), [4]);
            let actions = agent.act(&Tensor::zeros([2, 4], &device)).unwrap();
            assert_eq!(actions.dims(), [2, 1]);
            assert_eq!(actions.into_data().try_to_vec::<u32>().unwrap(), [1, 1]);
            let initial_parameters = agent.inner.online_q_network.values.val().into_data();
            agent.learn(&mut env, 2).unwrap();
            assert_eq!(agent.inner.optimization_steps, 2);
            let updated_parameters = agent.inner.online_q_network.values.val().into_data();
            assert_ne!(initial_parameters, updated_parameters);
            assert_eq!(agent.inner.dtype, dtype);
            drop(agent);
            assert_eq!(logger.collection_timesteps, [2]);
            assert_eq!(logger.update_timesteps, [1, 2]);
        }
    }

    #[test]
    fn public_agent_accepts_native_boolean_observations_with_extra_item_axes() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        let mut agent = DDQNAgent::<_, _, VectorizedGymError<Infallible>, (), 3>::builder()
            .action_space(Discrete::new(2))
            .observation_space(ObservationFixture::<3, Bool> { _kind: PhantomData })
            .online_q_network(network(&device, DType::F64))
            .optimizer(SgdConfig::new().init())
            .learning_rate(0.01)
            .dtype(DType::F64)
            .epsilon_schedule(ConstantSchedule::new(0.0))
            .replay_capacity(4)
            .batch_size(1)
            .training_horizon(4)
            .replay_storage_config(ReplayStorageConfig::new(ReplayDeviceStrategy::OneDevice(
                device.clone(),
            )))
            .build()
            .unwrap();
        let observations = Tensor::<3, Bool>::from_data([[[true]], [[false]]], &device);
        let actions = agent.act(&observations).unwrap();
        assert_eq!(actions.dims(), [2, 1]);
        assert_eq!(actions.into_data().try_to_vec::<u32>().unwrap(), [1, 1]);
    }

    #[test]
    fn targets_select_online_actions_and_evaluate_target_values() {
        for dtype in [DType::F32, DType::F64] {
            let device = Device::flex().autodiff();
            let rewards = Tensor::<2>::from_data([[1.0f64], [2.0]], (&device, dtype));
            let terminated = Tensor::<2, Bool>::from_data([[false], [true]], &device);
            let target =
                Tensor::<2>::from_data([[5.0f64, 7.0, 9.0], [4.0, 8.0, 6.0]], (&device, dtype))
                    .require_grad();
            let online =
                Tensor::<2>::from_data([[3.0f64, 100.0, 0.0], [10.0, 2.0, 1.0]], (&device, dtype));
            let values =
                DDQNTarget::target_q_values(&rewards, &terminated, Some(&online), &target, 0.9);
            assert_eq!(values.dims(), [2, 1]);
            assert_eq!(values.dtype(), dtype);
            let data = values
                .clone()
                .into_data()
                .convert::<f64>()
                .try_to_vec::<f64>()
                .unwrap();
            assert!((data[0] - 7.3).abs() < 1e-6);
            assert_eq!(data[1], 2.0);
            // Detachment belongs to the shared training loop, not the target formula.
            let gradients = values.sum().backward();
            let target_gradient = target
                .grad(&gradients)
                .unwrap()
                .into_data()
                .convert::<f64>()
                .try_to_vec::<f64>()
                .unwrap();
            assert!((target_gradient[1] - 0.9).abs() < 1e-6);
            assert_eq!(&target_gradient[3..], &[0.0; 3]);
        }
    }
}

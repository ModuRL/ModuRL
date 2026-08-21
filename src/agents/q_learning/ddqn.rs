use bon::bon;
use candle_core::{Error, Tensor};
use candle_nn::{Optimizer, VarMap};

use super::{
    QAgentError, QCollectionLogEntry, QLearningAgent, QLearningLogger, QLearningTarget, QLogEntry,
    selected_action_q_values,
};
use crate::{
    agents::{Agent, ReplayDeviceStrategy},
    gym::MultiGym,
    objectives::bellman_targets,
    parameter_schedule::{LinearSchedule, ParameterSchedule},
    spaces::{Discrete, Space},
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

    /// Computes `[batch]` targets from reward/done vectors `[batch]` and online
    /// and target Q values `[batch, action_count]`.
    fn target_q_values(
        rewards: &Tensor,
        next_dones: &Tensor,
        online_next_q_values: Option<&Tensor>,
        target_next_q_values: &Tensor,
        gamma: f32,
    ) -> Result<Tensor, Error> {
        let next_actions = online_next_q_values
            .expect("DDQN target calculation requires online next-state Q-values")
            .argmax(1)?;
        let next_q_values = selected_action_q_values(target_next_q_values, &next_actions)?
            .squeeze(1)?
            .detach();
        bellman_targets(rewards, next_dones, &next_q_values, f64::from(gamma))
    }
}

/// Double Deep Q-Network agent.
pub struct DDQNAgent<'a, O, GE, SE, I = ()>
where
    O: Optimizer,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    inner: QLearningAgent<'a, O, GE, SE, DDQNTarget>,
    logging_info: Option<DDQNLoggingInfo<'a, I>>,
}

#[bon]
impl<'a, O, GE, SE, I> DDQNAgent<'a, O, GE, SE, I>
where
    O: Optimizer,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    #[builder]
    pub fn new(
        #[builder(with = |network: impl candle_core::Module + 'static| Box::new(network))]
        target_q_network: Box<dyn candle_core::Module>,
        #[builder(with = |network: impl candle_core::Module + 'static| Box::new(network))]
        online_q_network: Box<dyn candle_core::Module>,
        target_vars: &'a mut VarMap,
        online_vars: &'a VarMap,
        action_space: Discrete,
        observation_space: Box<dyn Space<Error = SE>>,
        optimizer: O,
        #[builder(
            default = Box::new(LinearSchedule::new(1.0, 0.1)),
            with = |schedule: impl ParameterSchedule + 'static| Box::new(schedule)
        )]
        epsilon_schedule: Box<dyn ParameterSchedule>,
        #[builder(default = 10000)] replay_capacity: usize,
        /// Number of environments whose transitions are inserted together.
        environment_count: usize,
        #[builder(default = 32)] batch_size: usize,
        #[builder(default = 0.99)] gamma: f32,
        #[builder(default = 4)] update_frequency: usize,
        #[builder(default = 1000)] training_start: usize,
        #[builder(default = 1000)] target_update_interval: usize,
        training_horizon: usize,
        logger: Option<&'a mut dyn DDQNLogger<I>>,
        device_strategy: ReplayDeviceStrategy,
        #[builder(default = candle_core::DType::F32)] dtype: candle_core::DType,
    ) -> Result<Self, QAgentError<GE, SE>> {
        let inner = QLearningAgent::<'a, O, GE, SE, DDQNTarget>::builder()
            .action_space(action_space)
            .observation_space(observation_space)
            .target_q_network(target_q_network)
            .online_q_network(online_q_network)
            .target_vars(target_vars)
            .online_vars(online_vars)
            .optimizer(optimizer)
            .target_update_interval(target_update_interval)
            .epsilon_schedule(epsilon_schedule)
            .replay_capacity(replay_capacity)
            .environment_count(environment_count)
            .batch_size(batch_size)
            .gamma(gamma)
            .update_frequency(update_frequency)
            .training_start(training_start)
            .training_horizon(training_horizon)
            .device_strategy(device_strategy)
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

    pub fn get_observation_space(&self) -> &dyn Space<Error = SE> {
        self.inner.get_observation_space()
    }
}

impl<'a, O, GE, SE, I> Agent<I> for DDQNAgent<'a, O, GE, SE, I>
where
    O: Optimizer,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    type Error = QAgentError<GE, SE>;
    type GymError = GE;
    type SpaceError = SE;

    /// Selects scalar discrete actions `[batch]` for observations
    /// `[batch, ...observation_shape]`.
    fn act(&mut self, observation: &Tensor) -> Result<Tensor, Self::Error> {
        self.inner.act(observation)
    }

    fn learn(
        &mut self,
        env: &mut dyn MultiGym<I, Error = Self::GymError, SpaceError = Self::SpaceError>,
        num_timesteps: usize,
    ) -> Result<(), Self::Error> {
        self.inner.learn(env, num_timesteps, &mut self.logging_info)
    }
}

#[cfg(test)]
mod tests {
    use super::{DDQNAgent, DDQNLogger, DDQNTarget};
    use crate::{
        agents::{
            Agent, ReplayDeviceStrategy,
            q_learning::QLearningTarget,
            test_support::{CountingOptimizer, FixedEnv},
        },
        gym::{MultiGym, VectorizedGymWrapper},
        models::MLP,
        parameter_schedule::ConstantSchedule,
        spaces::Discrete,
    };
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    fn q_network(var_map: &VarMap, device: &Device) -> MLP {
        MLP::builder()
            .input_size(4)
            .output_size(2)
            .vb(VarBuilder::from_varmap(var_map, DType::F32, device))
            .hidden_layer_sizes(vec![2])
            .build()
            .unwrap()
    }

    #[derive(Default)]
    struct RecordingLogger {
        update_timesteps: Vec<usize>,
        collection_timesteps: Vec<usize>,
    }

    impl DDQNLogger for RecordingLogger {
        fn log(&mut self, entry: &super::QLogEntry) {
            self.update_timesteps.push(entry.collection_timestep);
        }

        fn log_collection(&mut self, entry: &super::QCollectionLogEntry) {
            self.collection_timesteps.push(entry.collection_timestep);
        }
    }

    #[test]
    fn public_agent_runs_actions_collection_training_and_logging() {
        let device = Device::Cpu;
        let mut env: VectorizedGymWrapper<FixedEnv> =
            vec![FixedEnv::new(device.clone()), FixedEnv::new(device.clone())].into();
        let online_vars = VarMap::new();
        let mut target_vars = VarMap::new();
        let mut logger = RecordingLogger::default();
        let mut agent = DDQNAgent::builder()
            .action_space(Discrete::new(2))
            .observation_space(env.observation_space())
            .online_q_network(q_network(&online_vars, &device))
            .target_q_network(q_network(&target_vars, &device))
            .online_vars(&online_vars)
            .target_vars(&mut target_vars)
            .optimizer(CountingOptimizer::with_learning_rate(1e-3))
            .epsilon_schedule(ConstantSchedule::new(0.0))
            .replay_capacity(4)
            .environment_count(2)
            .batch_size(1)
            .training_start(1)
            .update_frequency(1)
            .target_update_interval(2)
            .training_horizon(4)
            .logger(&mut logger)
            .device_strategy(ReplayDeviceStrategy::OneDevice(device.clone()))
            .build()
            .unwrap();

        assert_eq!(agent.get_action_space().get_possible_values(), 2);
        assert_eq!(agent.get_observation_space().shape(), vec![4]);
        let actions = agent
            .act(&Tensor::zeros((2, 4), DType::F32, &device).unwrap())
            .unwrap();
        assert_eq!(actions.dims(), &[2]);

        agent.learn(&mut env, 2).unwrap();
        assert_eq!(agent.inner.optimizer.steps, 2);

        assert_eq!(logger.collection_timesteps, vec![2]);
        assert_eq!(logger.update_timesteps, vec![1, 2]);
    }

    #[test]
    fn targets_select_online_actions_and_evaluate_with_target_values() {
        let device = Device::Cpu;
        let rewards = Tensor::from_vec(vec![1.0f32, 2.0], 2, &device).unwrap();
        let dones = Tensor::from_vec(vec![0.0f32, 1.0], 2, &device).unwrap();
        let online =
            Tensor::from_vec(vec![3.0f32, 100.0, 0.0, 10.0, 2.0, 1.0], (2, 3), &device).unwrap();
        let target =
            Tensor::from_vec(vec![5.0f32, 7.0, 9.0, 4.0, 8.0, 6.0], (2, 3), &device).unwrap();
        let values = DDQNTarget::target_q_values(&rewards, &dones, Some(&online), &target, 0.9)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!((values[0] - 7.3).abs() < 1e-6);
        assert_eq!(values[1], 2.0);
    }
}

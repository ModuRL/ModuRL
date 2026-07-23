use bon::bon;
use candle_core::{Error, Tensor};
use candle_nn::{Optimizer, VarMap};

use super::{
    QAgentError, QCollectionLogEntry, QLearningAgent, QLearningDeviceStrategy, QLearningLogger,
    QLearningTarget, QLogEntry, bellman_targets, selected_action_q_values,
};
use crate::{
    agents::Agent,
    gym::VectorizedGym,
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
        bellman_targets(rewards, next_dones, &next_q_values, gamma)
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
        target_q_network: Box<dyn candle_core::Module>,
        online_q_network: Box<dyn candle_core::Module>,
        target_vars: &'a mut VarMap,
        online_vars: &'a VarMap,
        action_space: Discrete,
        observation_space: Box<dyn Space<Error = SE>>,
        optimizer: O,
        #[builder(default = Box::new(LinearSchedule::new(1.0, 0.1)))] epsilon_schedule: Box<
            dyn ParameterSchedule,
        >,
        #[builder(default = 10000)] replay_capacity: usize,
        #[builder(default = 32)] batch_size: usize,
        #[builder(default = 0.99)] gamma: f32,
        #[builder(default = 4)] update_frequency: usize,
        #[builder(default = 1000)] training_start: usize,
        #[builder(default = 1000)] target_update_interval: usize,
        training_horizon: usize,
        logger: Option<&'a mut dyn DDQNLogger<I>>,
        device_strategy: QLearningDeviceStrategy,
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
            .batch_size(batch_size)
            .gamma(gamma)
            .update_frequency(update_frequency)
            .training_start(training_start)
            .training_horizon(training_horizon)
            .device_strategy(device_strategy)
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
        env: &mut dyn VectorizedGym<I, Error = Self::GymError, SpaceError = Self::SpaceError>,
        num_timesteps: usize,
    ) -> Result<(), Self::Error> {
        self.inner.learn(env, num_timesteps, &mut self.logging_info)
    }
}

#[cfg(test)]
mod tests {
    use super::DDQNTarget;
    use crate::agents::q_learning::QLearningTarget;
    use candle_core::{Device, Tensor};

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

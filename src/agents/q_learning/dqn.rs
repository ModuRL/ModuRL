use bon::bon;
use candle_core::{Error, Tensor};
use candle_nn::{Optimizer, VarMap};

use super::{
    QAgentError, QCollectionLogEntry, QLearningAgent, QLearningDeviceStrategy, QLearningLogger,
    QLearningTarget, QLogEntry, bellman_targets,
};
use crate::{
    agents::Agent,
    gym::VectorizedGym,
    parameter_schedule::{LinearSchedule, ParameterSchedule},
    spaces::{Discrete, Space},
};

pub trait DQNLogger<I = ()> {
    fn log(&mut self, info: &QLogEntry);

    fn log_collection(&mut self, _info: &QCollectionLogEntry<I>) {}
}

struct DQNLoggingInfo<'a, I> {
    logger: &'a mut dyn DQNLogger<I>,
}

impl<I> QLearningLogger<I> for Option<DQNLoggingInfo<'_, I>> {
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

struct DQNTarget;

impl QLearningTarget for DQNTarget {
    fn requires_online_next_q_values() -> bool {
        false
    }

    fn target_q_values(
        rewards: &Tensor,
        next_dones: &Tensor,
        _online_next_q_values: Option<&Tensor>,
        target_next_q_values: &Tensor,
        gamma: f32,
    ) -> Result<Tensor, Error> {
        bellman_targets(
            rewards,
            next_dones,
            &target_next_q_values.max(1)?.detach(),
            gamma,
        )
    }
}

/// Deep Q-Network agent.
pub struct DQNAgent<'a, O, GE, SE, I = ()>
where
    O: Optimizer,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    inner: QLearningAgent<'a, O, GE, SE, DQNTarget>,
    logging_info: Option<DQNLoggingInfo<'a, I>>,
}

#[bon]
impl<'a, O, GE, SE, I> DQNAgent<'a, O, GE, SE, I>
where
    O: Optimizer,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    #[builder]
    pub fn new(
        action_space: Discrete,
        observation_space: Box<dyn Space<Error = SE>>,
        target_q_network: Box<dyn candle_core::Module>,
        online_q_network: Box<dyn candle_core::Module>,
        target_vars: &'a mut VarMap,
        online_vars: &'a VarMap,
        optimizer: O,
        #[builder(default = 1000)] target_update_interval: usize,
        #[builder(default = Box::new(LinearSchedule::new(1.0, 0.1)))] epsilon_schedule: Box<
            dyn ParameterSchedule,
        >,
        #[builder(default = 10000)] replay_capacity: usize,
        #[builder(default = 32)] batch_size: usize,
        #[builder(default = 0.99)] gamma: f32,
        #[builder(default = 4)] update_frequency: usize,
        #[builder(default = 1000)] training_start: usize,
        training_horizon: usize,
        logger: Option<&'a mut dyn DQNLogger<I>>,
        device_strategy: QLearningDeviceStrategy,
    ) -> Result<Self, QAgentError<GE, SE>> {
        let inner = QLearningAgent::<'a, O, GE, SE, DQNTarget>::builder()
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
            logging_info: logger.map(|logger| DQNLoggingInfo { logger }),
        })
    }

    pub fn get_action_space(&self) -> &Discrete {
        self.inner.get_action_space()
    }

    pub fn get_observation_space(&self) -> &dyn Space<Error = SE> {
        self.inner.get_observation_space()
    }
}

impl<'a, O, GE, SE, I> Agent<I> for DQNAgent<'a, O, GE, SE, I>
where
    O: Optimizer,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    type Error = QAgentError<GE, SE>;
    type GymError = GE;
    type SpaceError = SE;

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
    use super::DQNTarget;
    use crate::agents::q_learning::{QLearningTarget, selected_action_q_values};
    use candle_core::{Device, Tensor};

    #[test]
    fn targets_use_target_max_and_mask_terminal_transitions() {
        let device = Device::Cpu;
        let rewards = Tensor::from_vec(vec![1.0f32, 2.0], 2, &device).unwrap();
        let dones = Tensor::from_vec(vec![0.0f32, 1.0], 2, &device).unwrap();
        let target =
            Tensor::from_vec(vec![10.0f32, 1.0, 3.0, 2.0, 4.0, 0.0], (2, 3), &device).unwrap();
        let values = DQNTarget::target_q_values(&rewards, &dones, None, &target, 0.9)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(values, vec![10.0, 2.0]);
    }

    #[test]
    fn mse_matches_selected_values_and_bellman_targets() {
        let device = Device::Cpu;
        let q_values =
            Tensor::from_vec(vec![1.0f32, 5.0, 2.0, 7.0, 3.0, 4.0], (2, 3), &device).unwrap();
        let actions = Tensor::from_vec(vec![1u32, 2], 2, &device).unwrap();
        let rewards = Tensor::from_vec(vec![1.0f32, 2.0], 2, &device).unwrap();
        let dones = Tensor::from_vec(vec![0.0f32, 1.0], 2, &device).unwrap();
        let target =
            Tensor::from_vec(vec![10.0f32, 1.0, 3.0, 2.0, 4.0, 0.0], (2, 3), &device).unwrap();

        let selected = selected_action_q_values(&q_values, &actions).unwrap();
        let targets = DQNTarget::target_q_values(&rewards, &dones, None, &target, 0.9)
            .unwrap()
            .unsqueeze(1)
            .unwrap();
        let loss = candle_nn::loss::mse(&selected, &targets).unwrap();
        assert!((loss.to_vec0::<f32>().unwrap() - 14.5).abs() < 1e-6);
    }
}

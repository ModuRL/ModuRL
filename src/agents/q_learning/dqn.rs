use bon::bon;
use candle_core::{Error, IndexOp, Tensor};
use candle_nn::{Optimizer, VarMap};

use super::super::Agent;
use super::{
    bellman_targets, epsilon_greedy_actions, selected_action_q_values, validate_configuration,
    validate_epsilon, QLearningConfigurationError, QLearningDeviceStrategy,
};
use crate::{
    buffers::{
        experience,
        experience_replay::{ExperienceReplay, ExperienceReplayError},
    },
    gym::{VectorizedGym, VectorizedStepInfo},
    parameter_schedule::{LinearSchedule, ParameterSchedule, ScheduleProgress},
    spaces::{Discrete, Space},
    tensor_operations::tensor_has_nan,
};
use std::ops::Deref;

struct DQNLoggingInfo<'a> {
    logger: &'a mut dyn DQNLogger,
    epoch: usize,
    timestep: usize,
}

impl<'a> DQNLoggingInfo<'a> {
    fn new(logger: &'a mut dyn DQNLogger) -> Self {
        Self {
            logger,
            epoch: 0,
            timestep: 0,
        }
    }
}

pub struct DQNLogEntry {
    pub loss: Tensor,
    pub epsilon: f64,
    pub learning_rate: f32,
    pub q_values: Tensor,
    pub rewards: Tensor,
    pub epoch: usize,
    pub timestep: usize,
}

pub trait DQNLogger {
    fn log(&mut self, info: &DQNLogEntry);
}

#[derive(Debug)]
pub enum DQNAgentError<GE, SE>
where
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    TensorError(candle_core::Error),
    ConfigurationError(QLearningConfigurationError),
    GymError(GE),
    SpaceError(SE),
}

impl<GE, SE> From<candle_core::Error> for DQNAgentError<GE, SE>
where
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    fn from(err: candle_core::Error) -> Self {
        DQNAgentError::TensorError(err)
    }
}

impl<GE, SE> From<QLearningConfigurationError> for DQNAgentError<GE, SE>
where
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    fn from(err: QLearningConfigurationError) -> Self {
        DQNAgentError::ConfigurationError(err)
    }
}

#[derive(Clone)]
struct DQNAgentExperience {
    state: Tensor,
    next_state: Tensor,
    action: Tensor,
    reward: f32,
    /// Whether the next state is terminal (1.0 if terminal, 0.0 otherwise)
    next_done: f32,
}

impl experience::Experience for DQNAgentExperience {
    type Error = candle_core::Error;
    fn get_elements(&self) -> Result<Vec<Tensor>, Self::Error> {
        Ok(vec![
            self.state.clone(),
            self.next_state.clone(),
            self.action.clone(),
            Tensor::from_vec(vec![self.reward], [].to_vec(), self.state.device())?,
            Tensor::from_vec(vec![self.next_done], [].to_vec(), self.state.device())?,
        ])
    }
}

impl DQNAgentExperience {
    pub fn new(
        state: Tensor,
        next_state: Tensor,
        action: Tensor,
        reward: f32,
        next_done: f32,
    ) -> Self {
        Self {
            state,
            next_state,
            action,
            reward,
            next_done,
        }
    }
}

/// Deep Q-Network agent.
/// The DQN agent uses a neural network to approximate the Q-values of the environment for each action.
/// The agent then picks the action with the highest Q-value. (Greedy)
/// The agent also sometimes picks a random action with probability epsilon. (Exploration)
pub struct DQNAgent<'a, O, GE, SE>
where
    O: Optimizer,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    online_q_network: Box<dyn candle_core::Module>,
    target_q_network: Box<dyn candle_core::Module>,
    target_vars: &'a mut VarMap,
    online_vars: &'a VarMap,
    target_update_interval: usize,
    optimizer: O,
    current_epsilon: f64,
    epsilon_schedule: Box<dyn ParameterSchedule>,
    schedule_progress: ScheduleProgress,
    action_space: Discrete,
    observation_space: Box<dyn Space<Error = SE>>,
    experience_replay: ExperienceReplay<DQNAgentExperience>,
    gamma: f32,
    update_frequency: usize,
    training_start: usize,
    logging_info: Option<DQNLoggingInfo<'a>>,
    device_strategy: QLearningDeviceStrategy,
    _phantom: std::marker::PhantomData<GE>,
}

#[bon]
impl<'a, O, GE, SE> DQNAgent<'a, O, GE, SE>
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
        logger: Option<&'a mut dyn DQNLogger>,
        device_strategy: QLearningDeviceStrategy,
    ) -> Result<Self, DQNAgentError<GE, SE>> {
        let experience_replay = ExperienceReplay::new(
            replay_capacity,
            batch_size,
            device_strategy.storage_device(),
        );
        let initial_epsilon = epsilon_schedule.value(0.0);
        let final_epsilon = epsilon_schedule.value(1.0);
        validate_configuration(
            replay_capacity,
            batch_size,
            gamma,
            initial_epsilon,
            final_epsilon,
            update_frequency,
            target_update_interval,
            training_horizon,
        )?;

        let mut agent = Self {
            online_q_network,
            target_q_network,
            training_start,
            target_vars,
            online_vars,
            target_update_interval,
            optimizer,
            current_epsilon: initial_epsilon,
            epsilon_schedule,
            schedule_progress: ScheduleProgress::new(training_horizon),
            action_space,
            observation_space,
            experience_replay,
            gamma,
            update_frequency,
            logging_info: logger.map(DQNLoggingInfo::new),
            _phantom: std::marker::PhantomData,
            device_strategy,
        };

        agent.update_target_network();
        Ok(agent)
    }
}

impl<'a, O, GE, SE> DQNAgent<'a, O, GE, SE>
where
    O: Optimizer,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    pub fn get_action_space(&self) -> &Discrete {
        &self.action_space
    }

    pub fn get_observation_space(&self) -> &dyn Space<Error = SE> {
        &*self.observation_space
    }

    fn target_q_values(
        rewards: &Tensor,
        next_dones: &Tensor,
        target_next_q_values: &Tensor,
        gamma: f32,
    ) -> Result<Tensor, Error> {
        let next_q_values = target_next_q_values.max(1)?.detach();
        bellman_targets(rewards, next_dones, &next_q_values, gamma)
    }

    fn optimize(&mut self) -> Result<(), Error> {
        if self.experience_replay.len() < self.experience_replay.get_batch_size() {
            return Ok(()); // Not enough samples to train.
        }

        let training_batch = self.experience_replay.sample();
        let training_batch = match training_batch {
            Ok(b) => b,
            Err(e) => match e {
                ExperienceReplayError::ExperienceError(ee) => return Err(ee),
                ExperienceReplayError::TensorError(te) => return Err(te),
            },
        };
        let mut elements = training_batch.get_elements();
        for elements in &mut elements {
            *elements = elements.to_device(&self.device_strategy.optimization_device())?;
        }
        let states = elements[0].clone();
        let next_states = elements[1].clone();
        let actions = elements[2].clone();
        let rewards = elements[3].clone();
        let next_dones = elements[4].clone();

        let mut state_action_q_values = self.online_q_network.forward(&states)?;
        state_action_q_values = state_action_q_values.squeeze(1)?;
        state_action_q_values = selected_action_q_values(&state_action_q_values, &actions)?;

        let target_next_q_values = self.target_q_network.forward(&next_states)?;
        let mut target_q_values =
            Self::target_q_values(&rewards, &next_dones, &target_next_q_values, self.gamma)?;

        target_q_values = target_q_values
            .reshape(&[target_q_values.shape().dims()[0], 1])?
            .detach();
        let loss = candle_nn::loss::mse(&state_action_q_values, &target_q_values)?;

        if let Some(logging_info) = &mut self.logging_info {
            let log_entry = DQNLogEntry {
                loss: loss.clone(),
                epsilon: self.current_epsilon,
                learning_rate: self.optimizer.learning_rate() as f32,
                q_values: state_action_q_values.clone(),
                rewards,
                epoch: logging_info.epoch,
                timestep: logging_info.timestep,
            };
            logging_info.logger.log(&log_entry);
            logging_info.epoch += 1;
            logging_info.timestep += 1;
        }

        if !tensor_has_nan(&loss)? {
            self.optimizer.backward_step(&loss)?;
        }

        Ok(())
    }

    // I don't LOVE the way we do this but sadly with the structure of candle I don't see a better way
    fn update_target_network(&mut self) {
        let online_vars = self.online_vars;
        let online_data = online_vars.data().lock().unwrap();

        for (name, online_var) in online_data.deref().iter() {
            self.target_vars
                .set_one(name, online_var.as_tensor())
                .expect("failed to match var names in target and online varmaps, make sure they are the same");
        }
    }
}

impl<'a, O, GE, SE> Agent for DQNAgent<'a, O, GE, SE>
where
    O: Optimizer,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    type Error = DQNAgentError<GE, SE>;
    type GymError = GE;
    type SpaceError = SE;

    fn act(&mut self, observation: &Tensor) -> Result<Tensor, Self::Error> {
        Ok(epsilon_greedy_actions(
            observation,
            self.current_epsilon,
            &self.action_space,
            |observations| self.online_q_network.forward(observations),
        )?)
    }

    fn learn(
        &mut self,
        env: &mut dyn VectorizedGym<Error = Self::GymError, SpaceError = Self::SpaceError>,
        num_timesteps: usize,
    ) -> Result<(), Self::Error> {
        let mut elapsed_timesteps = 0;
        let mut observations = env.reset().map_err(DQNAgentError::GymError)?;
        while elapsed_timesteps < num_timesteps {
            self.current_epsilon = validate_epsilon(
                self.schedule_progress
                    .parameter(self.epsilon_schedule.as_ref()),
            )?;

            let action = self.act(&observations)?;
            let step_info = env.step(action.clone()).map_err(DQNAgentError::GymError)?;
            let training_next_observations = step_info.transition_next_states()?;
            let VectorizedStepInfo {
                states: next_observations,
                rewards,
                dones,
                truncateds: _,
                terminal_states: _,
            } = step_info;

            let rewards = rewards.chunk(env.num_envs(), 0)?;
            let action = action.chunk(env.num_envs(), 0)?;
            let this_observations = observations.chunk(env.num_envs(), 0)?;
            observations = next_observations.clone();
            let next_observations = training_next_observations.chunk(env.num_envs(), 0)?;

            for i in 0..env.num_envs() {
                let reward = rewards[i].i(0)?.to_scalar::<f32>()?;
                let next_done = dones[i];
                let mut next_observation = next_observations[i].clone().squeeze(0)?.detach();
                let mut observation = this_observations[i].clone().squeeze(0)?.detach();
                let mut action = action[i].clone().squeeze(0)?.detach();

                for element in [&mut observation, &mut next_observation, &mut action] {
                    *element = element.to_device(&self.device_strategy.storage_device())?;
                }

                // Add the experience to the replay buffer.
                self.experience_replay.add(DQNAgentExperience::new(
                    observation.clone(),
                    next_observation,
                    action,
                    reward,
                    if next_done { 1.0 } else { 0.0 },
                ));
            }

            let first_training_timestep = self.schedule_progress.elapsed_steps();
            for timestep_offset in 1..=env.num_envs() {
                elapsed_timesteps += 1;
                let training_timestep = first_training_timestep.saturating_add(timestep_offset);
                if training_timestep % self.update_frequency == 0
                    && training_timestep >= self.training_start
                {
                    self.optimize()?;
                }
                if training_timestep % self.target_update_interval == 0 {
                    self.update_target_network();
                }
            }
            self.schedule_progress.advance_steps(env.num_envs());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{AdamW, VarBuilder, VarMap};

    use super::super::{selected_action_q_values, QLearningDeviceStrategy};
    use super::DQNAgent;
    use crate::{
        agents::{
            test_support::{CountingOptimizer, FixedEnv},
            Agent,
        },
        gym::{VectorizedGym, VectorizedGymWrapper},
        models::MLP,
        parameter_schedule::LinearSchedule,
        spaces::Discrete,
        tensor_operations::tanh,
    };

    type TestDQNAgent = DQNAgent<'static, AdamW, (), ()>;

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn targets_use_target_max_and_mask_terminal_transitions() {
        let device = Device::Cpu;
        let rewards = Tensor::from_vec(vec![1.0f32, 2.0], 2, &device).unwrap();
        let next_dones = Tensor::from_vec(vec![0.0f32, 1.0], 2, &device).unwrap();
        let target_next_q_values =
            Tensor::from_vec(vec![10.0f32, 1.0, 3.0, 2.0, 4.0, 0.0], (2, 3), &device).unwrap();

        let targets =
            TestDQNAgent::target_q_values(&rewards, &next_dones, &target_next_q_values, 0.9)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
        assert_close(targets[0], 10.0);
        assert_close(targets[1], 2.0);
    }

    #[test]
    fn mse_matches_selected_values_and_bellman_targets() {
        let device = Device::Cpu;
        let q_values =
            Tensor::from_vec(vec![1.0f32, 5.0, 2.0, 7.0, 3.0, 4.0], (2, 3), &device).unwrap();
        let actions = Tensor::from_vec(vec![1u32, 2], 2, &device).unwrap();
        let rewards = Tensor::from_vec(vec![1.0f32, 2.0], 2, &device).unwrap();
        let next_dones = Tensor::from_vec(vec![0.0f32, 1.0], 2, &device).unwrap();
        let target_next_q_values =
            Tensor::from_vec(vec![10.0f32, 1.0, 3.0, 2.0, 4.0, 0.0], (2, 3), &device).unwrap();

        let selected = selected_action_q_values(&q_values, &actions).unwrap();
        let targets =
            TestDQNAgent::target_q_values(&rewards, &next_dones, &target_next_q_values, 0.9)
                .unwrap()
                .unsqueeze(1)
                .unwrap();
        let loss = candle_nn::loss::mse(&selected, &targets).unwrap();

        assert_close(loss.to_vec0::<f32>().unwrap(), 14.5);
    }

    fn q_network(var_map: &VarMap, device: &Device) -> MLP {
        MLP::builder()
            .input_size(4)
            .output_size(2)
            .vb(VarBuilder::from_varmap(var_map, DType::F32, device))
            .activation(Box::new(tanh))
            .hidden_layer_sizes(vec![2])
            .build()
            .unwrap()
    }

    #[test]
    fn cadence_and_epsilon_progress_continue_across_learn_calls() {
        let device = Device::Cpu;
        let mut env: VectorizedGymWrapper<FixedEnv> =
            vec![FixedEnv::new(device.clone()), FixedEnv::new(device.clone())].into();
        let online_var_map = VarMap::new();
        let mut target_var_map = VarMap::new();
        let online_network = q_network(&online_var_map, &device);
        let target_network = q_network(&target_var_map, &device);
        let variable_name = online_var_map
            .data()
            .lock()
            .unwrap()
            .keys()
            .next()
            .unwrap()
            .clone();
        let online_variable = online_var_map.data().lock().unwrap()[&variable_name].clone();
        let target_variable = target_var_map.data().lock().unwrap()[&variable_name].clone();
        let mut agent = DQNAgent::builder()
            .action_space(Discrete::new(2))
            .observation_space(env.observation_space())
            .online_q_network(Box::new(online_network))
            .target_q_network(Box::new(target_network))
            .online_vars(&online_var_map)
            .target_vars(&mut target_var_map)
            .optimizer(CountingOptimizer::with_learning_rate(1e-3))
            .epsilon_schedule(Box::new(LinearSchedule::new(1.0, 0.0)))
            .replay_capacity(8)
            .batch_size(1)
            .training_start(3)
            .update_frequency(2)
            .target_update_interval(4)
            .training_horizon(10)
            .device_strategy(QLearningDeviceStrategy::OneDevice(device.clone()))
            .build()
            .unwrap();

        agent.learn(&mut env, 2).unwrap();
        assert_eq!(agent.schedule_progress.elapsed_steps(), 2);
        assert_close(agent.current_epsilon as f32, 1.0);
        assert_eq!(agent.optimizer.steps, 0);

        let changed = Tensor::full(5.0f32, online_variable.as_tensor().shape(), &device).unwrap();
        online_variable.set(&changed).unwrap();
        agent.learn(&mut env, 2).unwrap();

        assert_eq!(agent.schedule_progress.elapsed_steps(), 4);
        assert_close(agent.current_epsilon as f32, 0.8);
        assert_eq!(agent.optimizer.steps, 1);
        assert_eq!(
            online_variable
                .as_tensor()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            target_variable
                .as_tensor()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );
    }
}

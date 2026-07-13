use bon::bon;
use candle_core::{Device, Error, IndexOp, Tensor};
use candle_nn::{Optimizer, VarMap};
use std::{marker::PhantomData, ops::Deref};

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

pub mod ddqn;
pub mod dqn;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QLearningConfigurationError {
    ZeroReplayCapacity,
    ZeroBatchSize,
    ZeroTargetUpdateInterval,
    ZeroUpdateFrequency,
    ZeroTrainingHorizon,
    ReplayCapacityBelowBatchSize,
    InvalidGamma,
    InvalidEpsilon,
}

#[derive(Debug)]
pub enum QAgentError<GE, SE>
where
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    TensorError(candle_core::Error),
    ConfigurationError(QLearningConfigurationError),
    GymError(GE),
    SpaceError(SE),
}

impl<GE, SE> From<candle_core::Error> for QAgentError<GE, SE>
where
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    fn from(err: candle_core::Error) -> Self {
        Self::TensorError(err)
    }
}

impl<GE, SE> From<QLearningConfigurationError> for QAgentError<GE, SE>
where
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    fn from(err: QLearningConfigurationError) -> Self {
        Self::ConfigurationError(err)
    }
}

pub struct QLogEntry {
    pub loss: Tensor,
    pub epsilon: f64,
    pub learning_rate: f32,
    pub q_values: Tensor,
    pub rewards: Tensor,
    pub epoch: usize,
    pub timestep: usize,
}

pub(crate) trait QLearningTarget {
    fn requires_online_next_q_values() -> bool;

    fn target_q_values(
        rewards: &Tensor,
        next_dones: &Tensor,
        online_next_q_values: Option<&Tensor>,
        target_next_q_values: &Tensor,
        gamma: f32,
    ) -> Result<Tensor, Error>;
}

#[derive(Clone)]
struct QLearningExperience {
    state: Tensor,
    next_state: Tensor,
    action: Tensor,
    reward: f32,
    next_done: f32,
}

impl experience::Experience for QLearningExperience {
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

pub(crate) struct QLearningAgent<'a, O, GE, SE, T>
where
    O: Optimizer,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
    T: QLearningTarget,
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
    experience_replay: ExperienceReplay<QLearningExperience>,
    gamma: f32,
    update_frequency: usize,
    training_start: usize,
    device_strategy: QLearningDeviceStrategy,
    _phantom: PhantomData<(GE, T)>,
}

#[bon]
impl<'a, O, GE, SE, T> QLearningAgent<'a, O, GE, SE, T>
where
    O: Optimizer,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
    T: QLearningTarget,
{
    #[builder]
    pub(crate) fn new(
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
        device_strategy: QLearningDeviceStrategy,
    ) -> Result<Self, QAgentError<GE, SE>> {
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
            target_vars,
            online_vars,
            target_update_interval,
            optimizer,
            current_epsilon: initial_epsilon,
            epsilon_schedule,
            schedule_progress: ScheduleProgress::new(training_horizon),
            action_space,
            observation_space,
            experience_replay: ExperienceReplay::new(
                replay_capacity,
                batch_size,
                device_strategy.storage_device(),
            ),
            gamma,
            update_frequency,
            training_start,
            device_strategy,
            _phantom: PhantomData,
        };
        agent.update_target_network();
        Ok(agent)
    }
}

impl<'a, O, GE, SE, T> QLearningAgent<'a, O, GE, SE, T>
where
    O: Optimizer,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
    T: QLearningTarget,
{
    pub(crate) fn get_action_space(&self) -> &Discrete {
        &self.action_space
    }

    pub(crate) fn get_observation_space(&self) -> &dyn Space<Error = SE> {
        &*self.observation_space
    }

    pub(crate) fn act(&mut self, observation: &Tensor) -> Result<Tensor, QAgentError<GE, SE>> {
        Ok(epsilon_greedy_actions(
            observation,
            self.current_epsilon,
            &self.action_space,
            |observations| self.online_q_network.forward(observations),
        )?)
    }

    fn optimize(
        &mut self,
        epoch: usize,
        timestep: usize,
        on_optimization: impl FnOnce(&mut QLogEntry),
    ) -> Result<(), Error> {
        if self.experience_replay.len() < self.experience_replay.get_batch_size() {
            return Ok(());
        }
        let training_batch = match self.experience_replay.sample() {
            Ok(batch) => batch,
            Err(ExperienceReplayError::ExperienceError(error)) => return Err(error),
            Err(ExperienceReplayError::TensorError(error)) => return Err(error),
        };
        let mut elements = training_batch.get_elements();
        for element in &mut elements {
            *element = element.to_device(&self.device_strategy.optimization_device())?;
        }
        let states = elements[0].clone();
        let next_states = elements[1].clone();
        let actions = elements[2].clone();
        let rewards = elements[3].clone();
        let next_dones = elements[4].clone();

        let target_next_q_values = self.target_q_network.forward(&next_states)?;
        let online_next_q_values = T::requires_online_next_q_values()
            .then(|| self.online_q_network.forward(&next_states))
            .transpose()?;
        let target_q_values = T::target_q_values(
            &rewards,
            &next_dones,
            online_next_q_values.as_ref(),
            &target_next_q_values,
            self.gamma,
        )?
        .reshape(&[rewards.shape().dims()[0], 1])?
        .detach();

        let state_action_q_values = selected_action_q_values(
            &self.online_q_network.forward(&states)?.squeeze(1)?,
            &actions,
        )?;
        let loss = candle_nn::loss::mse(&state_action_q_values, &target_q_values)?;
        let mut entry = QLogEntry {
            loss: loss.clone(),
            epsilon: self.current_epsilon,
            learning_rate: self.optimizer.learning_rate() as f32,
            q_values: state_action_q_values,
            rewards,
            epoch,
            timestep,
        };
        on_optimization(&mut entry);
        if !tensor_has_nan(&loss)? {
            self.optimizer.backward_step(&loss)?;
        }
        Ok(())
    }

    fn update_target_network(&mut self) {
        let online_data = self.online_vars.data().lock().unwrap();
        for (name, online_var) in online_data.deref() {
            self.target_vars.set_one(name, online_var.as_tensor()).expect(
                "failed to match var names in target and online varmaps, make sure they are the same",
            );
        }
    }

    pub(crate) fn learn(
        &mut self,
        env: &mut dyn VectorizedGym<Error = GE, SpaceError = SE>,
        num_timesteps: usize,
        mut on_optimization: impl FnMut(&mut QLogEntry),
    ) -> Result<(), QAgentError<GE, SE>> {
        let mut elapsed_timesteps = 0;
        let mut observations = env.reset().map_err(QAgentError::GymError)?;
        while elapsed_timesteps < num_timesteps {
            self.current_epsilon = validate_epsilon(
                self.schedule_progress
                    .parameter(self.epsilon_schedule.as_ref()),
            )?;
            let action = self.act(&observations)?;
            let step_info = env.step(action.clone()).map_err(QAgentError::GymError)?;
            let training_next_observations = step_info.transition_next_states()?;
            let VectorizedStepInfo {
                states: next_observations,
                rewards,
                dones,
                ..
            } = step_info;

            let rewards = rewards.chunk(env.num_envs(), 0)?;
            let actions = action.chunk(env.num_envs(), 0)?;
            let current_observations = observations.chunk(env.num_envs(), 0)?;
            observations = next_observations;
            let next_observations = training_next_observations.chunk(env.num_envs(), 0)?;
            for i in 0..env.num_envs() {
                let mut state = current_observations[i].clone().squeeze(0)?.detach();
                let mut next_state = next_observations[i].clone().squeeze(0)?.detach();
                let mut action = actions[i].clone().squeeze(0)?.detach();
                for tensor in [&mut state, &mut next_state, &mut action] {
                    *tensor = tensor.to_device(&self.device_strategy.storage_device())?;
                }
                self.experience_replay.add(QLearningExperience {
                    state,
                    next_state,
                    action,
                    reward: rewards[i].i(0)?.to_scalar::<f32>()?,
                    next_done: if dones[i] { 1.0 } else { 0.0 },
                });
            }

            let first_training_timestep = self.schedule_progress.elapsed_steps();
            for timestep_offset in 1..=env.num_envs() {
                elapsed_timesteps += 1;
                let training_timestep = first_training_timestep.saturating_add(timestep_offset);
                if training_timestep % self.update_frequency == 0
                    && training_timestep >= self.training_start
                {
                    self.optimize(0, training_timestep, |entry| on_optimization(entry))?;
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

pub(crate) fn validate_configuration(
    replay_capacity: usize,
    batch_size: usize,
    gamma: f32,
    initial_epsilon: f64,
    final_epsilon: f64,
    update_frequency: usize,
    target_update_interval: usize,
    training_horizon: usize,
) -> Result<(), QLearningConfigurationError> {
    if replay_capacity == 0 {
        return Err(QLearningConfigurationError::ZeroReplayCapacity);
    }
    if batch_size == 0 {
        return Err(QLearningConfigurationError::ZeroBatchSize);
    }
    if target_update_interval == 0 {
        return Err(QLearningConfigurationError::ZeroTargetUpdateInterval);
    }
    if update_frequency == 0 {
        return Err(QLearningConfigurationError::ZeroUpdateFrequency);
    }
    if training_horizon == 0 {
        return Err(QLearningConfigurationError::ZeroTrainingHorizon);
    }
    if replay_capacity < batch_size {
        return Err(QLearningConfigurationError::ReplayCapacityBelowBatchSize);
    }
    if !gamma.is_finite() || !(0.0..=1.0).contains(&gamma) {
        return Err(QLearningConfigurationError::InvalidGamma);
    }
    validate_epsilon(initial_epsilon)?;
    validate_epsilon(final_epsilon)?;
    Ok(())
}

pub(crate) fn validate_epsilon(epsilon: f64) -> Result<f64, QLearningConfigurationError> {
    if !epsilon.is_finite() || !(0.0..=1.0).contains(&epsilon) {
        return Err(QLearningConfigurationError::InvalidEpsilon);
    }
    Ok(epsilon)
}

pub(crate) fn epsilon_greedy_actions(
    observation: &Tensor,
    epsilon: f64,
    action_space: &Discrete,
    forward: impl FnOnce(&Tensor) -> Result<Tensor, Error>,
) -> Result<Tensor, Error> {
    let batch_size = observation.shape().dims()[0];
    let explores = Tensor::rand(0.0f64, 1.0, &[batch_size], observation.device())?
        .to_vec1::<f64>()?
        .into_iter()
        .map(|value| value < epsilon)
        .collect::<Vec<_>>();

    let greedy_indices = explores
        .iter()
        .enumerate()
        .filter_map(|(index, &explore)| (!explore).then_some(index as u32))
        .collect::<Vec<_>>();
    let greedy_count = greedy_indices.len();
    let mut greedy_actions = if greedy_indices.is_empty() {
        None
    } else {
        let greedy_indices =
            Tensor::from_vec(greedy_indices, &[greedy_count], observation.device())?;
        let greedy_observations = observation.index_select(&greedy_indices, 0)?;
        Some(
            forward(&greedy_observations)?
                .argmax(1)?
                .chunk(greedy_count, 0)?
                .into_iter(),
        )
    };

    let mut actions = Vec::with_capacity(batch_size);
    for explore in explores {
        if explore {
            actions.push(action_space.sample(observation.device())?);
        } else {
            let action = greedy_actions
                .as_mut()
                .and_then(|actions| actions.next())
                .expect("greedy actions must exist for non-exploring environments");
            actions.push(action.squeeze(0)?);
        }
    }
    Tensor::stack(&actions, 0)
}

pub(crate) fn selected_action_q_values(
    q_values: &Tensor,
    actions: &Tensor,
) -> Result<Tensor, Error> {
    let actions = actions.reshape(&[actions.shape().dims()[0], 1])?;
    q_values.gather(&actions, 1)
}

fn bellman_targets(
    rewards: &Tensor,
    next_dones: &Tensor,
    next_q_values: &Tensor,
    gamma: f32,
) -> Result<Tensor, Error> {
    let gamma_tensor = Tensor::full(gamma, next_q_values.shape(), rewards.device())?;
    Ok((rewards + (next_q_values * ((1.0 - next_dones)?.mul(&gamma_tensor)?)))?.detach())
}

/// Strategy for selecting devices used by value-based agent computations.
///
/// `OneDevice` keeps collection, replay, and optimization on one device.
/// `Hybrid` stores replay on one device and transfers sampled batches to the
/// device used for network optimization.
pub enum QLearningDeviceStrategy {
    OneDevice(Device),
    Hybrid {
        optimization_device: Device,
        storage_device: Device,
    },
}

impl QLearningDeviceStrategy {
    pub(crate) fn storage_device(&self) -> Device {
        match self {
            QLearningDeviceStrategy::OneDevice(device) => device.clone(),
            QLearningDeviceStrategy::Hybrid { storage_device, .. } => storage_device.clone(),
        }
    }

    pub(crate) fn optimization_device(&self) -> Device {
        match self {
            QLearningDeviceStrategy::OneDevice(device) => device.clone(),
            QLearningDeviceStrategy::Hybrid {
                optimization_device,
                ..
            } => optimization_device.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        QLearningAgent, QLearningConfigurationError, QLearningDeviceStrategy, QLearningTarget,
        bellman_targets, selected_action_q_values, validate_configuration, validate_epsilon,
    };
    use crate::{
        agents::test_support::{CountingOptimizer, FixedEnv},
        gym::{VectorizedGym, VectorizedGymError, VectorizedGymWrapper},
        models::MLP,
        parameter_schedule::LinearSchedule,
        spaces::Discrete,
        tensor_operations::tanh,
    };
    use candle_core::{DType, Device, Error, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    struct TestTarget;

    impl QLearningTarget for TestTarget {
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

    #[test]
    fn accepts_valid_configuration() {
        assert_eq!(
            validate_configuration(1_000, 32, 0.99, 1.0, 0.1, 4, 1_000, 10_000),
            Ok(())
        );
    }

    #[test]
    fn rejects_invalid_configuration_values() {
        assert_eq!(
            validate_configuration(0, 32, 0.99, 1.0, 0.1, 4, 1_000, 10_000),
            Err(QLearningConfigurationError::ZeroReplayCapacity)
        );
        assert_eq!(
            validate_configuration(1_000, 0, 0.99, 1.0, 0.1, 4, 1_000, 10_000),
            Err(QLearningConfigurationError::ZeroBatchSize)
        );
        assert_eq!(
            validate_configuration(1_000, 32, 0.99, 1.0, 0.1, 4, 0, 10_000),
            Err(QLearningConfigurationError::ZeroTargetUpdateInterval)
        );
        assert_eq!(
            validate_configuration(1_000, 32, 0.99, 1.0, 0.1, 0, 1_000, 10_000),
            Err(QLearningConfigurationError::ZeroUpdateFrequency)
        );
        assert_eq!(
            validate_configuration(31, 32, 0.99, 1.0, 0.1, 4, 1_000, 10_000),
            Err(QLearningConfigurationError::ReplayCapacityBelowBatchSize)
        );
        assert_eq!(
            validate_configuration(1_000, 32, 1.01, 1.0, 0.1, 4, 1_000, 10_000),
            Err(QLearningConfigurationError::InvalidGamma)
        );
        assert_eq!(
            validate_configuration(1_000, 32, 0.99, 1.0, 0.1, 4, 1_000, 0),
            Err(QLearningConfigurationError::ZeroTrainingHorizon)
        );
        assert_eq!(
            validate_epsilon(1.01),
            Err(QLearningConfigurationError::InvalidEpsilon)
        );
    }

    #[test]
    fn selected_action_q_values_uses_one_value_per_transition() {
        let device = Device::Cpu;
        let q_values =
            Tensor::from_vec(vec![1.0f32, 5.0, 2.0, 7.0, 3.0, 4.0], (2, 3), &device).unwrap();
        let actions = Tensor::from_vec(vec![1u32, 2], 2, &device).unwrap();

        let selected = selected_action_q_values(&q_values, &actions).unwrap();
        assert_eq!(
            selected.to_vec2::<f32>().unwrap(),
            vec![vec![5.0], vec![4.0]]
        );
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
        let mut agent: QLearningAgent<
            '_,
            CountingOptimizer,
            VectorizedGymError<candle_core::Error>,
            candle_core::Error,
            TestTarget,
        > = QLearningAgent::builder()
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

        agent.learn(&mut env, 2, |_| {}).unwrap();
        assert_eq!(agent.schedule_progress.elapsed_steps(), 2);
        assert_eq!(agent.current_epsilon, 1.0);
        assert_eq!(agent.optimizer.steps, 0);

        let changed = Tensor::full(5.0f32, online_variable.as_tensor().shape(), &device).unwrap();
        online_variable.set(&changed).unwrap();
        agent.learn(&mut env, 2, |_| {}).unwrap();

        assert_eq!(agent.schedule_progress.elapsed_steps(), 4);
        assert_eq!(agent.current_epsilon, 0.8);
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

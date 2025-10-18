use bon::bon;
use candle_core::{Error, IndexOp, Tensor};
use candle_nn::Optimizer;

use super::Actor;
use crate::{
    buffers::{
        experience,
        experience_replay::{ExperienceReplay, ExperienceReplayError},
    },
    gym::{VectorizedGym, VectorizedStepInfo},
    spaces::{Discrete, Space},
    tensor_operations::tensor_has_nan,
};

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
    pub epsilon: f32,
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
pub enum DQNActorError<GE, SE>
where
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    TensorError(candle_core::Error),
    GymError(GE),
    SpaceError(SE),
}

impl<GE, SE> From<candle_core::Error> for DQNActorError<GE, SE>
where
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    fn from(err: candle_core::Error) -> Self {
        DQNActorError::TensorError(err)
    }
}

#[derive(Clone)]
struct DQNActorExperience {
    state: Tensor,
    next_state: Tensor,
    action: Tensor,
    reward: f32,
    done: f32,
}

impl experience::Experience for DQNActorExperience {
    type Error = candle_core::Error;
    fn get_elements(&self) -> Result<Vec<Tensor>, Self::Error> {
        Ok(vec![
            self.state.clone(),
            self.next_state.clone(),
            self.action.clone(),
            Tensor::from_vec(vec![self.reward], [].to_vec(), self.state.device())?,
            Tensor::from_vec(vec![self.done], [].to_vec(), self.state.device())?,
        ])
    }
}

impl DQNActorExperience {
    pub fn new(state: Tensor, next_state: Tensor, action: Tensor, reward: f32, done: f32) -> Self {
        Self {
            state,
            next_state,
            action,
            reward,
            done,
        }
    }
}

/// Deep Q-Network actor.
/// The DQN actor uses a neural network to approximate the Q-values of the environment for each action.
/// The actor then picks the action with the highest Q-value. (Greedy)
/// The actor also sometimes picks a random action with probability epsilon. (Exploration)
pub struct DQNActor<'a, O, GE, SE>
where
    O: Optimizer,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    q_network: Box<dyn candle_core::Module>,
    optimizer: O,
    epsilon: f32,
    epsilon_decay: f32,
    action_space: Discrete,
    observation_space: Box<dyn Space<Error = SE>>,
    experience_replay: ExperienceReplay<DQNActorExperience>,
    gamma: f32,
    update_frequency: usize,
    logging_info: Option<DQNLoggingInfo<'a>>,
    _phantom: std::marker::PhantomData<GE>,
}

#[bon]
impl<'a, O, GE, SE> DQNActor<'a, O, GE, SE>
where
    O: Optimizer,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    #[builder]
    pub fn new(
        action_space: Discrete,
        observation_space: Box<dyn Space<Error = SE>>,
        q_network: Box<dyn candle_core::Module>,
        optimizer: O,
        #[builder(default = 0.9)] epsilon_start: f32,
        #[builder(default = 0.99)] epsilon_decay: f32,
        #[builder(default = 10000)] replay_capacity: usize,
        #[builder(default = 32)] batch_size: usize,
        #[builder(default = 0.99)] gamma: f32,
        #[builder(default = 4)] update_frequency: usize,
        logger: Option<&'a mut dyn DQNLogger>,
        device: candle_core::Device,
    ) -> Self {
        let experience_replay = ExperienceReplay::new(replay_capacity, batch_size, device);
        Self {
            q_network,
            optimizer,
            epsilon: epsilon_start,
            epsilon_decay,
            action_space,
            observation_space,
            experience_replay,
            gamma,
            update_frequency,
            logging_info: logger.map(|l| DQNLoggingInfo::new(l)),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, O, GE, SE> DQNActor<'a, O, GE, SE>
where
    O: Optimizer,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    pub fn get_action_space(&self) -> &Discrete {
        &self.action_space
    }

    pub fn get_observation_space(&self) -> &Box<dyn Space<Error = SE>> {
        &self.observation_space
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
        let elements = training_batch.get_elements();
        let states = elements[0].clone();
        let next_states = elements[1].clone();
        let actions = elements[2].clone();
        let rewards = elements[3].clone();
        let dones = elements[4].clone();

        let mut state_action_q_values = self.q_network.forward(&states)?;
        state_action_q_values = state_action_q_values.squeeze(1)?;
        let actions = actions.reshape(&[actions.shape().dims()[0], 1])?.clone();
        // Select the q-values for the actions taken.
        state_action_q_values = state_action_q_values.gather(&actions, 1)?;

        let next_q_values = self.q_network.forward(&next_states)?;
        let next_q_values = next_q_values.max(1)?;

        // Compute the target Q-values.
        let gamma_tensor = Tensor::full(self.gamma, next_q_values.shape(), states.device())?;
        let mut target_q_values =
            (rewards.clone() + (next_q_values * ((1.0 - dones)?.mul(&gamma_tensor)?)))?;

        target_q_values = target_q_values
            .reshape(&[target_q_values.shape().dims()[0], 1])?
            .detach();
        let loss = candle_nn::loss::mse(&state_action_q_values, &target_q_values)?;

        if let Some(logging_info) = &mut self.logging_info {
            let log_entry = DQNLogEntry {
                loss: loss.clone(),
                epsilon: self.epsilon,
                learning_rate: self.optimizer.learning_rate() as f32,
                q_values: state_action_q_values.clone(),
                rewards: rewards,
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
}

impl<'a, O, GE, SE> Actor for DQNActor<'a, O, GE, SE>
where
    O: Optimizer,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    type Error = DQNActorError<GE, SE>;
    type GymError = GE;
    type SpaceError = SE;

    fn act(&mut self, observation: &Tensor) -> Result<Tensor, Self::Error> {
        let rand = Tensor::rand(0.0f32, 1.0, &[], observation.device())?;

        if rand.to_vec0::<f32>()? < self.epsilon {
            let mut actions = vec![];
            let batch_size = observation.shape().dims()[0];
            for _ in 0..batch_size {
                let sample = self.action_space.sample(observation.device())?;

                actions.push(sample);
            }
            // turn that into a
            let actions = Tensor::stack(&actions, 0)?;
            Ok(actions)
        } else {
            let q_values = self.q_network.forward(observation)?;
            let actions = q_values.argmax(1)?;
            Ok(actions)
        }
    }

    fn learn(
        &mut self,
        env: &mut dyn VectorizedGym<Error = Self::GymError, SpaceError = Self::SpaceError>,
        num_timesteps: usize,
    ) -> Result<(), Self::Error> {
        let mut elapsed_timesteps = 0;
        let mut observations = env.reset().map_err(DQNActorError::GymError)?;
        while elapsed_timesteps < num_timesteps {
            let action = self.act(&observations)?;
            let VectorizedStepInfo {
                states: next_observations,
                rewards,
                dones,
                truncateds: _,
            } = env.step(action.clone()).map_err(DQNActorError::GymError)?;

            let rewards = rewards.chunk(env.num_envs(), 0)?;
            let action = action.chunk(env.num_envs(), 0)?;
            let this_observations = observations.chunk(env.num_envs(), 0)?;
            observations = next_observations.clone();
            let next_observations = next_observations.chunk(env.num_envs(), 0)?;

            for i in 0..env.num_envs() {
                let reward = rewards[i].i(0)?.to_scalar::<f32>()?;
                let done = dones[i];
                let next_observation = next_observations[i].clone().squeeze(0)?;
                let observation = this_observations[i].clone().squeeze(0)?;
                let action = action[i].clone().squeeze(0)?;
                // Add the experience to the replay buffer.
                self.experience_replay.add(DQNActorExperience::new(
                    observation.clone(),
                    next_observation,
                    action,
                    reward,
                    if done { 1.0 } else { 0.0 },
                ));
            }

            self.epsilon *= self.epsilon_decay;

            for _ in 0..env.num_envs() {
                elapsed_timesteps += 1;
                if elapsed_timesteps % self.update_frequency == 0 {
                    self.optimize()?;
                }
            }
        }

        Ok(())
    }
}

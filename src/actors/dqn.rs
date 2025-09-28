use bon::bon;
use candle_core::{Error, Tensor};
use candle_nn::Optimizer;
use rand::Rng;

use crate::{
    buffers::{experience, experience_replay::ExperienceReplay},
    gym::{Gym, StepInfo},
    spaces::{Discrete, Space},
};

use super::Actor;

#[derive(Debug)]
pub enum DQNActorError<GE>
where
    GE: std::fmt::Debug,
{
    TensorError(candle_core::Error),
    GymError(GE),
}

impl<GE> From<candle_core::Error> for DQNActorError<GE>
where
    GE: std::fmt::Debug,
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
pub struct DQNActor<O, GE>
where
    O: Optimizer,
    GE: std::fmt::Debug,
{
    q_network: Box<dyn candle_core::Module>,
    optimizer: O,
    epsilon: f32,
    epsilon_decay: f32,
    action_space: Discrete,
    observation_space: Box<dyn Space>,
    experience_replay: ExperienceReplay<DQNActorExperience>,
    gamma: f32,
    epochs: usize,
    _phantom: std::marker::PhantomData<GE>,
}

#[bon]
impl<O, GE> DQNActor<O, GE>
where
    O: Optimizer,
    GE: std::fmt::Debug,
{
    #[builder]
    pub fn new(
        action_space: Discrete,
        observation_space: Box<dyn Space>,
        q_network: Box<dyn candle_core::Module>,
        optimizer: O,
        #[builder(default = 0.9)] epsilon_start: f32,
        #[builder(default = 0.99)] epsilon_decay: f32,
        #[builder(default = 10000)] replay_capacity: usize,
        #[builder(default = 32)] batch_size: usize,
        #[builder(default = 0.99)] gamma: f32,
        #[builder(default = 10)] epochs: usize,
    ) -> Self {
        let experience_replay = ExperienceReplay::new(replay_capacity, batch_size);
        Self {
            q_network,
            optimizer,
            epsilon: epsilon_start,
            epsilon_decay,
            action_space,
            observation_space,
            experience_replay,
            gamma,
            epochs,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<O, GE> DQNActor<O, GE>
where
    O: Optimizer,
    GE: std::fmt::Debug,
{
    pub fn get_action_space(&self) -> &Discrete {
        &self.action_space
    }

    pub fn get_observation_space(&self) -> &Box<dyn Space> {
        &self.observation_space
    }

    fn optimize(&mut self) -> Result<(), Error> {
        if self.experience_replay.len() < self.experience_replay.get_batch_size() {
            return Ok(()); // Not enough samples to train.
        }
        let training_batch = self.experience_replay.sample()?;
        let elements = training_batch.get_elements();
        let states = elements[0].clone();
        let next_states = elements[1].clone();
        let actions = elements[2].clone();
        let rewards = elements[3].clone();
        let dones = elements[4].clone();

        // Make sure the tensors are of the correct type.
        let actions = actions.to_dtype(candle_core::DType::I64)?;
        let rewards = rewards.to_dtype(candle_core::DType::F32)?;
        let states = states.to_dtype(candle_core::DType::F32)?;
        let dones = dones.to_dtype(candle_core::DType::F32)?;
        let next_states = next_states.to_dtype(candle_core::DType::F32)?;

        let mut state_action_q_values = self.q_network.forward(&states)?;
        state_action_q_values = state_action_q_values.to_dtype(candle_core::DType::F32)?;
        state_action_q_values = state_action_q_values.squeeze(1)?;
        let actions = actions.reshape(&[actions.shape().dims()[0], 1])?.clone();
        // Select the q-values for the actions taken.
        state_action_q_values = state_action_q_values.gather(&actions, 1)?;

        let next_q_values = self.q_network.forward(&next_states)?;
        let next_q_values = next_q_values.to_dtype(candle_core::DType::F32)?;
        let next_q_values = next_q_values.max(1)?;

        // Compute the target Q-values.
        let gamma_tensor = Tensor::full(self.gamma, next_q_values.shape(), states.device())?
            .to_dtype(candle_core::DType::F32)?;
        let mut target_q_values =
            (rewards + (next_q_values * ((1.0 - dones)?.mul(&gamma_tensor)?)))?;

        let criterion = candle_nn::loss::mse;
        target_q_values = target_q_values.reshape(&[target_q_values.shape().dims()[0], 1])?;
        let loss = criterion(&state_action_q_values, &target_q_values)?;
        self.optimizer.backward_step(&loss)?;

        Ok(())
    }
}

impl<O, GE> Actor for DQNActor<O, GE>
where
    O: Optimizer,
    GE: std::fmt::Debug,
{
    type Error = DQNActorError<GE>;
    type GymError = GE;
    fn act(&mut self, observation: &Tensor) -> Result<Tensor, Self::Error> {
        self.epsilon *= self.epsilon_decay;
        if rand::rng().random_range(0.0..1.0) < self.epsilon {
            let mut actions = vec![];
            let batch_size = observation.shape().dims()[0];
            for _ in 0..batch_size {
                actions.push(self.action_space.sample(observation.device()));
            }
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
        env: &mut dyn Gym<Error = Self::GymError>,
        num_timesteps: usize,
    ) -> Result<(), Self::Error> {
        let mut elapsed_timesteps = 0;
        let mut episode_idx = 0;
        while elapsed_timesteps < num_timesteps {
            let mut total_reward = 0.0;
            let mut observation = env.reset().map_err(DQNActorError::GymError)?;
            loop {
                let mut new_observations_shape: Vec<usize> = vec![1];
                new_observations_shape.append(&mut self.observation_space.shape());
                observation = observation.reshape(&*new_observations_shape)?;

                let action = self.act(&observation)?;
                // outputs a tensor of shape [1, dim] so we need to squeeze it to [dim]
                let action = action.squeeze(0)?;
                let StepInfo {
                    state: next_observation,
                    reward,
                    done,
                    truncated,
                } = env.step(action.clone()).map_err(DQNActorError::GymError)?;
                total_reward += reward;
                // Add the experience to the replay buffer.
                self.experience_replay.add(DQNActorExperience::new(
                    observation,
                    next_observation.clone(),
                    action,
                    reward,
                    if done { 1.0 } else { 0.0 },
                ));
                observation = next_observation;

                if done || truncated {
                    break;
                }

                elapsed_timesteps += 1;
            }

            println!(
                "Episode {} finished with total reward: {}",
                episode_idx + 1,
                total_reward
            );

            // Optimize the Q-network.
            for _ in 0..self.epochs {
                self.optimize()?;
            }
            episode_idx += 1;
        }

        Ok(())
    }
}

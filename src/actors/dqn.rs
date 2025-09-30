use bon::bon;
use candle_core::{Error, IndexOp, Tensor};
use candle_nn::Optimizer;
use rand::Rng;

use crate::{
    buffers::{experience, experience_replay::ExperienceReplay},
    gym::{VectorizedGym, VectorizedStepInfo},
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
    update_frequency: usize,
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
        #[builder(default = 4)] update_frequency: usize,
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
            update_frequency,
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
            (rewards + (next_q_values * ((1.0 - dones)?.mul(&gamma_tensor)?)))?;

        target_q_values = target_q_values
            .reshape(&[target_q_values.shape().dims()[0], 1])?
            .detach();
        let loss = candle_nn::loss::mse(&state_action_q_values, &target_q_values)?;
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
        if rand::rng().random_range(0.0..1.0) < self.epsilon {
            let mut actions = vec![];
            let batch_size = observation.shape().dims()[0];
            for _ in 0..batch_size {
                let sample = self.action_space.sample(observation.device());

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
        env: &mut dyn VectorizedGym<Error = Self::GymError>,
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

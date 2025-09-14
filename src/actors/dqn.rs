use std::collections::HashMap;

use candle_core::{safetensors::save, Error, Tensor, Var};
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

pub struct DQNActorBuilder<O, GE>
where
    O: Optimizer,
    GE: std::fmt::Debug,
{
    // Everything other than the spaces are optional.
    q_network: Box<dyn candle_core::Module>,
    optimizer: O,
    epsilon_start: Option<f32>,
    epsilon_decay: Option<f32>,
    action_space: Discrete,
    observation_space: Box<dyn Space>,
    batch_size: Option<usize>,
    gamma: Option<f32>,
    replay_capacity: Option<usize>,
    epochs: Option<usize>,
    _phantom: std::marker::PhantomData<GE>,
}

impl<O, GE> DQNActorBuilder<O, GE>
where
    O: Optimizer,
    GE: std::fmt::Debug,
{
    pub fn new(
        action_space: Discrete,
        observation_space: Box<dyn Space>,
        q_network: Box<dyn candle_core::Module>,
        optimizer: O,
    ) -> Self {
        Self {
            q_network: q_network,
            optimizer: optimizer,
            epsilon_start: None,
            epsilon_decay: None,
            action_space,
            observation_space,
            batch_size: None,
            gamma: None,
            replay_capacity: None,
            epochs: None,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn epsilon_start(mut self, epsilon_start: f32) -> Self {
        self.epsilon_start = Some(epsilon_start);
        self
    }

    pub fn epsilon_decay(mut self, epsilon_decay: f32) -> Self {
        self.epsilon_decay = Some(epsilon_decay);
        self
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    pub fn gamma(mut self, gamma: f32) -> Self {
        self.gamma = Some(gamma);
        self
    }

    pub fn replay_capacity(mut self, replay_capacity: usize) -> Self {
        self.replay_capacity = Some(replay_capacity);
        self
    }

    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = Some(epochs);
        self
    }

    pub fn build(self) -> DQNActor<O, GE> {
        let epsilon_start = self.epsilon_start.unwrap_or(0.9);
        let epsilon_decay = self.epsilon_decay.unwrap_or(0.99);
        let batch_size = self.batch_size.unwrap_or(32);
        let gamma = self.gamma.unwrap_or(0.99);
        let replay_capacity = self.replay_capacity.unwrap_or(10000);
        let epochs = self.epochs.unwrap_or(10);

        DQNActor {
            q_network: self.q_network,
            optimizer: self.optimizer,
            epsilon: epsilon_start,
            epsilon_decay,
            action_space: self.action_space,
            observation_space: self.observation_space,
            experience_replay: ExperienceReplay::new(replay_capacity, batch_size),
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

    fn save(&self, vars: Vec<Var>, path: &str) -> Result<(), Self::Error> {
        let tensors = vars.iter().map(|v| v.as_tensor()).collect::<Vec<_>>();
        let mut hashmap = HashMap::new();
        for (i, tensor) in tensors.iter().enumerate() {
            hashmap.insert(format!("var_{i}"), (*tensor).clone());
        }
        save(&hashmap, path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use candle_nn::{AdamW, ParamsAdamW, VarBuilder, VarMap};

    use super::*;
    use crate::gym::common_gyms::CartPoleV1;
    use crate::gym::Gym;
    use crate::models::MLPBuilder;

    // Test the DQN actor by training it on the CartPole environment.
    #[test]
    fn dqn_cartpole() {
        let mut env = CartPoleV1::new(&candle_core::Device::Cpu);
        let observation_space = env.observation_space();
        let var_map = VarMap::new();
        let vb =
            VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &candle_core::Device::Cpu);

        let mlp = MLPBuilder::new(observation_space.shape().iter().sum(), 2, vb)
            .hidden_layer_sizes(vec![64, 64])
            .build()
            .expect("Failed to create MLP");

        let optimizer =
            AdamW::new(var_map.all_vars(), ParamsAdamW::default()).expect("Failed to create AdamW");

        let mut actor = DQNActorBuilder::new(
            Discrete::new(2, 0), // had to hardcode this :(
            observation_space,
            Box::new(mlp),
            optimizer,
        )
        .replay_capacity(128)
        .batch_size(32) // So that it actually reaches the optimization step.
        .build();

        actor.learn(&mut env, 500000).expect("Failed to learn");
    }
}

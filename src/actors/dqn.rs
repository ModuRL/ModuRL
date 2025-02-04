use std::collections::HashMap;

use candle_core::{safetensors::save, Error, Tensor, Var};
use candle_nn::Optimizer;
use rand::Rng;

use crate::{Discrete, Experience, ExperienceReplay, Gym, Space};

use super::Actor;

/// Deep Q-Network actor.
/// The DQN actor uses a neural network to approximate the Q-values of the environment for each action.
/// The actor then picks the action with the highest Q-value. (Greedy)
/// The actor also sometimes picks a random action with probability epsilon. (Exploration)
pub struct DQNActor<O>
where
    O: Optimizer,
{
    q_network: Box<dyn candle_core::Module>,
    optimizer: O,
    epsilon: f32,
    epsilon_decay: f32,
    action_space: Discrete,
    observation_space: Box<dyn Space>,
    experience_replay: ExperienceReplay,
    gamma: f32,
    epochs: usize,
}

impl<O> DQNActor<O>
where
    O: Optimizer,
{
    pub fn new(
        q_network: Box<dyn candle_core::Module>,
        optimizer: O,
        epsilon_start: f32,
        epsilon_decay: f32,
        action_space: Discrete,
        observation_space: Box<dyn Space>,
        batch_size: usize,
        gamma: f32,
        replay_capacity: usize,
        epochs: usize,
    ) -> Self {
        Self {
            q_network,
            optimizer: optimizer,
            epsilon: epsilon_start,
            epsilon_decay,
            action_space,
            observation_space,
            experience_replay: ExperienceReplay::new(replay_capacity, batch_size),
            gamma: gamma,
            epochs: epochs,
        }
    }

    pub fn get_action_space(&self) -> &Discrete {
        &self.action_space
    }

    pub fn get_observation_space(&self) -> &Box<dyn Space> {
        &self.observation_space
    }

    fn optimize(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.experience_replay.len() < self.experience_replay.get_batch_size() {
            return Ok(()); // Not enough samples to train.
        }
        let training_batch = self.experience_replay.sample()?;
        let actions = training_batch.actions();
        let rewards = training_batch.rewards();
        let states = training_batch.states();
        let dones = training_batch.dones();
        let next_states = training_batch.next_states();

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

        println!("Loss: {:?}", loss.to_vec0::<f32>()?);

        Ok(())
    }
}

impl<O> Actor for DQNActor<O>
where
    O: Optimizer,
{
    type Error = Error;
    fn act(&mut self, observation: &Tensor) -> Result<Tensor, Error> {
        self.epsilon *= self.epsilon_decay;
        if rand::rng().random_range(0.0..1.0) < self.epsilon {
            let mut actions = vec![];
            let batch_size = observation.shape().dims()[0];
            for _ in 0..batch_size {
                actions.push(self.action_space.sample(observation.device()));
            }
            let actions = Tensor::stack(&actions, 0);
            actions
        } else {
            let q_values = self.q_network.forward(observation)?;
            let actions = q_values.argmax(1)?;
            Ok(actions)
        }
    }

    fn learn(
        &mut self,
        env: &mut dyn Gym<Error = Self::Error>,
        num_episodes: usize,
    ) -> Result<(), Error> {
        for episode_idx in 0..num_episodes {
            let mut total_reward = 0.0;
            let mut observation = env.reset()?;
            loop {
                let previous_observation_shape = observation.shape().clone().into_dims();
                observation = observation.reshape(&[1, previous_observation_shape[0]])?;

                let action = self.act(&observation)?;
                // outputs a tensor of shape [1, dim] so we need to squeeze it to [dim]
                let action = action.squeeze(0)?;
                let (next_observation, reward, done) = env.step(action.clone())?;
                total_reward += reward;
                // Add the experience to the replay buffer.
                self.experience_replay.add(Experience::new(
                    observation,
                    next_observation.clone(),
                    action,
                    reward,
                    done,
                ));
                observation = next_observation;

                if done {
                    break;
                }
            }

            println!(
                "Episode {} finished with total reward: {}",
                episode_idx + 1,
                total_reward
            );

            // Optimize the Q-network.
            for _ in 0..self.epochs {
                let result = self.optimize();
                if let Err(e) = result {
                    println!("Error optimizing the Q-network: {:?}", e);
                }
            }
        }

        Ok(())
    }

    fn save(&self, vars: Vec<Var>, path: &str) -> Result<(), Error> {
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
    use crate::gym::CartPole;
    use crate::{Gym, MLP};

    // Test the DQN actor by training it on the CartPole environment.
    #[test]
    fn test_dqn_actor() {
        let mut env = CartPole::new(&candle_core::Device::Cpu);
        let observation_space = env.observation_space();
        let var_map = VarMap::new();
        let mut vb =
            VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &candle_core::Device::Cpu);
        let mlp = MLP::new(
            observation_space
                .sample(&candle_core::Device::Cpu)
                .shape()
                .elem_count(),
            vec![32, 32, 32],
            2,
            candle_nn::Activation::Gelu,
            None,
            &mut vb,
        )
        .expect("Failed to create MLP");

        let mut actor = DQNActor::new(
            Box::new(mlp),
            AdamW::new(var_map.all_vars(), ParamsAdamW::default()).expect("Failed to create AdamW"),
            0.9,
            0.99,
            Discrete::new(2, 0), // had to hardcode this puppy in sadly.
            observation_space,
            128,
            0.99,
            1000,
            1,
        );

        actor.learn(&mut env, 600).expect("Failed to learn");
    }
}

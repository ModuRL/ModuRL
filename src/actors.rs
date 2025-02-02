use std::{collections::HashMap};

use candle_core::{safetensors::save, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder};
use rand::Rng;

use crate::{Discrete, Experience, ExperienceReplay, Gym, Space};

pub trait Actor {
    fn act(&mut self, observation: &Tensor) -> Tensor;
    fn learn(&mut self, env: &mut dyn Gym, num_episodes: usize);
    fn save(&self, vars: Vec<Var>, path: &str);
}

/// Q-learning actor
/// The most basic actor, which just takes the action with the highest Q-value.
pub struct DQNActor<O>
where O: Optimizer,{
    q_network: Box<dyn candle_core::Module>,
    optimizer: O,
    epsilon: f32,
    action_space: Discrete,
    observation_space: Box<dyn Space>,
    experience_replay: ExperienceReplay,
    gamma: f32,
}

impl<O> DQNActor<O> 
where O: Optimizer {
    pub fn new(
        q_network: Box<dyn candle_core::Module>,
        optimizer: O,
        epsilon: f32,
        action_space: Discrete,
        observation_space: Box<dyn Space>,
        batch_size: usize,
        gamma: f32,
    ) -> Self {
        Self {
            q_network,
            optimizer: optimizer,
            epsilon,
            action_space,
            observation_space,
            experience_replay: ExperienceReplay::new(10000, batch_size),
            gamma: gamma,
        }
    }

    pub fn set_epsilon(&mut self, epsilon: f32) {
        self.epsilon = epsilon;
    }

    pub fn get_epsilon(&self) -> f32 {
        self.epsilon
    }

    pub fn get_action_space(&self) -> &Discrete {
        &self.action_space
    }

    pub fn get_observation_space(&self) -> &Box<dyn Space> {
        &self.observation_space
    }

    fn optimize(&mut self) -> Result<(), Box<dyn std::error::Error>>{
        if self.experience_replay.len() < self.experience_replay.get_batch_size() {
            return Ok(()); // Not enough samples to train.
        }
        let training_batch = self.experience_replay.sample();
        let actions = training_batch.actions();
        let rewards = training_batch.rewards();
        let states = training_batch.states();
        let dones = training_batch.dones();
        let next_states = training_batch.next_states();

        let mut state_action_q_values = self.q_network.forward(&states)?;
        let mut state_action_q_values = state_action_q_values
            .to_dtype(candle_core::DType::F32)?;
        // Select the q-values for the actions taken.
        let state_action_q_values = state_action_q_values
            .gather(actions, 1)?;

        let next_q_values = self.q_network.forward(&next_states)?;
        let next_q_values = next_q_values
            .to_dtype(candle_core::DType::F32)?;
        let next_q_values = next_q_values.max(1)?;

        // Compute the target Q-values.
        let gamma_tensor = Tensor::full(self.gamma, next_q_values.shape(), states.device())?;
        let target_q_values = (rewards + (next_q_values * ((1.0 - dones)?.mul(&gamma_tensor)?)))?;

        let criterion = candle_nn::loss::mse;
        let loss = criterion(&state_action_q_values, &target_q_values)?;
        self.optimizer.backward_step(&loss)?;

        println!("Loss: {:?}", loss.to_vec0::<f32>()?);
        
        Ok(())
    }
}

impl<O> Actor for DQNActor<O> 
where O: Optimizer {
    fn act(&mut self, observation: &Tensor) -> Tensor {
        if rand::rng().random_range(0.0..1.0) < self.epsilon {
            self.action_space.sample(observation.device())
        } else {
            let q_values = self.q_network.forward(observation).unwrap();
            assert!(q_values.dims() == vec![self.action_space.get_possible_values()]);
            let max_q_value = q_values.max(1).unwrap();
            let action = q_values
                .eq(&max_q_value)
                .unwrap()
                .to_dtype(candle_core::DType::I64)
                .unwrap();
            action
        }
    }

    fn learn(&mut self, env: &mut dyn Gym, num_episodes: usize) {
        for episode_idx in 0..num_episodes {
            let mut average_reward = 0.0;
            let mut step_count = 0;
            let mut observation = env.reset();
            loop {
                let action = self.act(&observation);
                let (next_observation, reward, done) = env.step(action.clone());
                average_reward += reward;
                step_count += 1;
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
            average_reward /= step_count as f32;
            println!("Episode {} finished with average reward: {}", episode_idx, average_reward);

            // Optimize the Q-network.
            let _ = self.optimize();
        }
    }

    fn save(&self, vars: Vec<Var>, path: &str) {
        let tensors = vars.iter().map(|v| v.as_tensor()).collect::<Vec<_>>();
        let mut hashmap = HashMap::new();
        for (i, tensor) in tensors.iter().enumerate() {
            hashmap.insert(format!("var_{i}"), (*tensor).clone());
        }
        save(&hashmap, path).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use candle_nn::VarMap;

    use super::*;
    use crate::gym::CartPole;
    use crate::{Gym, MLP};

    // Test the DQN actor by training it on the CartPole environment.
    #[test]
    fn test_dqn_actor()
    {
        let mut env = CartPole::new(&candle_core::Device::Cpu);
        let observation_space = env.observation_space();
        let action_space = env.action_space();
        let var_map = VarMap::new();
        let mut vb = VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &candle_core::Device::Cpu);
        let mlp = MLP::new(
            observation_space.sample(&candle_core::Device::Cpu).shape().elem_count(),
            vec![32,32,32],
            action_space.sample(&candle_core::Device::Cpu).shape().elem_count(),
            candle_nn::Activation::Gelu,
            None,
            &mut vb,
        ).unwrap();

        let mut actor = DQNActor::new(
            Box::new(mlp),
            AdamW::new(var_map.all_vars(), ParamsAdamW::default()).unwrap(),
            1.0,
            Discrete::new(2, 0), // had to hardcode this puppy in sadly.
            observation_space,
            32,
            0.99,
        );

        actor.learn(&mut env, 10);
    }
}
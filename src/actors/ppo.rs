use std::collections::HashMap;

use crate::tensor_operations::torch_like_min;
use candle_core::{safetensors::save, Error};
use candle_nn::Optimizer;

use crate::{
    buffers::{rollout_buffer::RolloutBuffer, Experience},
    models::probabilistic_model::ProbabilisticActor,
    spaces,
};

use super::Actor;

pub struct PPOActor<O>
where
    O: Optimizer,
{
    clipped: bool,
    gamma: f32,
    gae_lambda: f32,
    clip_range: f32,
    normalize_advantage: bool,
    vf_coef: f32,
    ent_coef: f32,
    critic_optimizer: O,
    actor_optimizer: O,
    observation_space: Box<dyn spaces::Space>,
    action_space: Box<dyn spaces::Space>,
    critic_network: Box<dyn candle_core::Module>,
    actor_network: Box<dyn ProbabilisticActor<Error = Error>>,
    rollout_buffer: RolloutBuffer,
    num_epochs: usize,
    batch_size: usize,
}

pub struct PPOActorBuilder<O>
where
    O: Optimizer,
{
    // Neccesary hyperparameters
    critic_optimizer: O,
    actor_optimizer: O,
    observation_space: Box<dyn spaces::Space>,
    action_space: Box<dyn spaces::Space>,
    critic_network: Box<dyn candle_core::Module>,
    actor_network: Box<dyn ProbabilisticActor<Error = Error>>,
    // Has default values so optional
    clipped: Option<bool>,
    gamma: Option<f32>,
    gae_lambda: Option<f32>,
    clip_range: Option<f32>,
    normalize_advantage: Option<bool>,
    vf_coef: Option<f32>,
    ent_coef: Option<f32>,
    batch_size: Option<usize>,
    mini_batch_size: Option<usize>,
    num_epochs: Option<usize>,
}

impl<O> PPOActorBuilder<O>
where
    O: Optimizer,
{
    pub fn new(
        observation_space: Box<dyn spaces::Space>,
        action_space: Box<dyn spaces::Space>,
        actor_network: Box<dyn ProbabilisticActor<Error = Error>>,
        critic_network: Box<dyn candle_core::Module>,
        critic_optimizer: O,
        actor_optimizer: O,
    ) -> Self {
        Self {
            critic_optimizer,
            actor_optimizer,
            observation_space,
            action_space,
            actor_network,
            critic_network,
            clipped: None,
            gamma: None,
            gae_lambda: None,
            clip_range: None,
            normalize_advantage: None,
            vf_coef: None,
            ent_coef: None,
            batch_size: None,
            mini_batch_size: None,
            num_epochs: None,
        }
    }

    pub fn clipped(mut self, clipped: bool) -> Self {
        self.clipped = Some(clipped);
        self
    }

    pub fn gamma(mut self, gamma: f32) -> Self {
        self.gamma = Some(gamma);
        self
    }

    pub fn gae_lambda(mut self, gae_lambda: f32) -> Self {
        self.gae_lambda = Some(gae_lambda);
        self
    }

    pub fn clip_range(mut self, clip_range: f32) -> Self {
        self.clip_range = Some(clip_range);
        self
    }

    pub fn normalize_advantage(mut self, normalize_advantage: bool) -> Self {
        self.normalize_advantage = Some(normalize_advantage);
        self
    }

    pub fn vf_coef(mut self, vf_coef: f32) -> Self {
        self.vf_coef = Some(vf_coef);
        self
    }

    pub fn ent_coef(mut self, ent_coef: f32) -> Self {
        self.ent_coef = Some(ent_coef);
        self
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    pub fn mini_batch_size(mut self, mini_batch_size: usize) -> Self {
        self.mini_batch_size = Some(mini_batch_size);
        self
    }

    pub fn num_epochs(mut self, num_epochs: usize) -> Self {
        self.num_epochs = Some(num_epochs);
        self
    }

    pub fn build(self) -> PPOActor<O> {
        if let Some(batch_size) = self.batch_size {
            assert!(batch_size > 0);
        }

        PPOActor {
            clipped: self.clipped.unwrap_or(true),
            gamma: self.gamma.unwrap_or(0.99),
            gae_lambda: self.gae_lambda.unwrap_or(0.95),
            clip_range: self.clip_range.unwrap_or(0.2),
            normalize_advantage: self.normalize_advantage.unwrap_or(true),
            vf_coef: self.vf_coef.unwrap_or(0.5),
            ent_coef: self.ent_coef.unwrap_or(0.01),
            critic_optimizer: self.critic_optimizer,
            actor_optimizer: self.actor_optimizer,
            observation_space: self.observation_space,
            action_space: self.action_space,
            critic_network: self.critic_network,
            actor_network: self.actor_network,
            rollout_buffer: RolloutBuffer::new(self.mini_batch_size.unwrap_or(128)),
            num_epochs: self.num_epochs.unwrap_or(1),
            batch_size: self.batch_size.unwrap_or(1024),
        }
    }
}

impl<O> PPOActor<O>
where
    O: Optimizer,
{
    fn optimize(&mut self) -> Result<(), Error> {
        for epoch in 0..self.num_epochs {
            for sample in self.rollout_buffer.get_all_shuffled() {
                let actions = sample.actions();
                let states = sample.states();
                let rewards = sample.rewards();
                let dones = sample.dones();

                let values = self.critic_network.forward(states).unwrap();

                let advantages = self
                    .compute_advantages(rewards, values.clone(), dones)
                    .unwrap();

                let (old_log_probs, entropy) = self
                    .actor_network
                    .log_prob_and_entropy(states, actions)
                    .unwrap();

                // Compute the loss and backpropagate
                let (actor_loss, critic_loss) = self
                    .compute_loss(states, actions, old_log_probs, rewards, advantages)
                    .unwrap();

                println!("Actor Loss: {}, Critic Loss: {}", actor_loss, critic_loss);
            }
        }

        Ok(())
    }

    // Uses GAE to compute the advantages
    fn compute_advantages(
        &self,
        rewards: &candle_core::Tensor,
        values: candle_core::Tensor,
        dones: &candle_core::Tensor,
    ) -> Result<candle_core::Tensor, Error> {
        let mut advantages = vec![];
        let mut last_advantage = 0.0;
        let values = values.squeeze(1).unwrap();
        let values = values.squeeze(1).unwrap();

        let mut last_value = values
            .get(values.dims()[0] - 1)
            .unwrap()
            .get(0)
            .unwrap()
            .to_scalar()
            .unwrap();

        for i in (0..rewards.dims1().unwrap()).rev() {
            if dones.get(i).unwrap().to_scalar::<f32>().unwrap() > 0.5 {
                last_advantage = 0.0;
                last_value = 0.0;
            }
            let reward = rewards.get(i).unwrap().to_scalar::<f32>().unwrap();
            let value = values.get(i).unwrap().to_scalar().unwrap();

            let delta = reward + self.gamma * last_value - value;
            last_advantage = delta + self.gamma * self.gae_lambda * last_advantage;
            advantages.push(last_advantage);
            last_value = value;
        }

        let advantages =
            candle_core::Tensor::from_vec(advantages, rewards.shape(), rewards.device()).unwrap();

        let advantages = advantages.clone();
        let advantages = if self.normalize_advantage {
            let advantages_mean = advantages.mean(0).unwrap().to_scalar::<f32>().unwrap();
            let advantage_diff = (advantages.clone() - advantages_mean.clone() as f64).unwrap();

            let epsilon = f32::EPSILON;
            let advantages_std_sqrt = (advantage_diff.clone() * advantage_diff)
                .unwrap()
                .mean(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                .max(epsilon);
            ((advantages.clone() - advantages_mean.clone() as f64).unwrap()
                / (advantages_std_sqrt as f64))
                .unwrap()
        } else {
            advantages
        };

        Ok(advantages)
    }

    /// Compute the loss for the actor and critic networks
    /// Then backpropagate the loss
    fn compute_loss(
        &mut self,
        states: &candle_core::Tensor,
        actions: &candle_core::Tensor,
        old_log_probs: candle_core::Tensor,
        rewards: &candle_core::Tensor,
        advantages: candle_core::Tensor,
    ) -> Result<(f64, f64), Error> {
        let values = self.critic_network.forward(states).unwrap();
        let values = values.squeeze(1).unwrap();

        println!(
            "Advantages mean: {}",
            advantages.mean(0).unwrap().to_scalar::<f32>().unwrap()
        );
        let values = values.squeeze(1).unwrap();
        let target_values = (advantages.clone() + values.clone()).unwrap();
        println!(
            "Values mean: {}",
            values.mean(0).unwrap().to_scalar::<f32>().unwrap()
        );

        let (log_probs, entropy) = self
            .actor_network
            .log_prob_and_entropy(states, actions)
            .unwrap();

        println!(
            "Log probs: {}",
            log_probs.mean_all().unwrap().to_scalar::<f32>().unwrap()
        );

        let ratio = (log_probs - old_log_probs).unwrap().exp().unwrap();

        let advantages = advantages.unsqueeze(1).unwrap();
        let advantages = advantages.expand(ratio.dims()).unwrap();

        let actor_loss = match self.clipped {
            true => {
                let clip_range = self.clip_range;
                let clipped_ratio = ratio.clamp(1.0 - clip_range, 1.0 + clip_range).unwrap();

                let actor_loss = (-1.0
                    * torch_like_min(
                        &(ratio.clone() * advantages.clone()).unwrap(),
                        &(clipped_ratio.clone() * advantages.clone()).unwrap(),
                    )
                    .unwrap()
                    .mean(0)
                    .unwrap())
                .unwrap();
                actor_loss
            }
            false => {
                let actor_loss = (-1.0
                    * (ratio.clone() * advantages.clone())
                        .unwrap()
                        .mean(0)
                        .unwrap())
                .unwrap();
                actor_loss
            }
        };

        let values_minus_targets = (values.clone() - target_values.clone()).unwrap();
        let critic_loss = (values_minus_targets.clone() * values_minus_targets).unwrap();

        // Use mse for now
        let critic_loss = critic_loss.mean(0).unwrap().to_scalar::<f32>().unwrap() as f64;

        let entropy_loss = (-1.0 * entropy.mean(0).unwrap()).unwrap();

        let loss = ((actor_loss.clone() + ((self.vf_coef as f64) * critic_loss.clone())).unwrap()
            + ((-self.ent_coef as f64) * entropy_loss))
            .unwrap();

        self.actor_optimizer.backward_step(&loss.clone()).unwrap();
        self.critic_optimizer.backward_step(&loss).unwrap();

        let actor_loss_scalar = actor_loss.mean_all().unwrap().to_scalar::<f32>().unwrap() as f64;

        Ok((actor_loss_scalar, critic_loss))
    }
}

impl<O> Actor for PPOActor<O>
where
    O: Optimizer,
{
    type Error = Error;
    fn act(&mut self, observation: &candle_core::Tensor) -> Result<candle_core::Tensor, Error> {
        self.actor_network.sample(observation)
    }

    fn learn(
        &mut self,
        env: &mut dyn crate::gym::Gym<Error = Self::Error>,
        num_episodes: usize,
    ) -> Result<(), Self::Error> {
        let mut last_optimization_step = 0;
        for episode_idx in 0..num_episodes {
            let mut done = false;
            let mut obs = env.reset().unwrap();
            let (mut next_obs, mut reward);

            while !done {
                let mut new_observations_shape: Vec<usize> = vec![1];
                new_observations_shape.append(&mut self.observation_space.shape());
                obs = obs.reshape(&*new_observations_shape).unwrap();

                let action = self.act(&obs).unwrap();
                // Now: the action is a tensor of shape [1, dim] so we need to squeeze it to [dim]
                let actual_action = self.action_space.from_neurons(&action);
                let actual_action = actual_action.squeeze(0).unwrap();

                (next_obs, reward, done) = env.step(actual_action).unwrap();
                self.rollout_buffer.add(Experience::new(
                    obs.clone(),
                    next_obs.clone(),
                    action,
                    reward,
                    done,
                ));
                obs = next_obs;
            }
            if self.rollout_buffer.len() >= self.batch_size {
                println!(
                    "Average episode length: {}",
                    self.rollout_buffer.len() / (episode_idx + 1 - last_optimization_step)
                );
                last_optimization_step = episode_idx;
                self.optimize().unwrap();
            }
        }
        Ok(())
    }

    fn save(&self, vars: Vec<candle_core::Var>, path: &str) -> Result<(), Self::Error> {
        // TODO! Reevaluate this is there ever a case where anything different happens.unwrap()
        // This is a copy paste from the DQN actor but I guess it shouldn't be needed to be implemented for every actor
        // It might be universal for all actors but load is different for each actor

        let tensors = vars.iter().map(|v| v.as_tensor()).collect::<Vec<_>>();
        let mut hashmap = HashMap::new();
        for (i, tensor) in tensors.iter().enumerate() {
            hashmap.insert(format!("var_{i}"), (*tensor).clone());
        }
        save(&hashmap, path).unwrap();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gym::{common_gyms::CartPole, Gym},
        models::{probabilistic_model::MLPProbabilisticActor, MLPBuilder},
    };
    use candle_nn::{AdamW, ParamsAdamW, VarBuilder, VarMap};

    #[test]
    fn test_ppo_actor_cartpole() {
        let mut env = CartPole::new(&candle_core::Device::Cpu);
        let observation_space = env.observation_space();
        let action_space = env.action_space();
        let var_map = VarMap::new();
        let vb =
            VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &candle_core::Device::Cpu);

        let actor_network = MLPBuilder::new(
            observation_space.shape().iter().sum(),
            action_space.shape().iter().sum::<usize>() * 2,
            vb.clone(),
        )
        .activation(candle_nn::Activation::Relu)
        .output_activation(candle_nn::Activation::Sigmoid)
        .build()
        .unwrap();

        let mut config = ParamsAdamW::default();
        config.lr = 1e-4;

        let actor_optimizer =
            AdamW::new(var_map.all_vars(), config.clone()).expect("Failed to create AdamW");

        let var_map = VarMap::new();
        let vb =
            VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &candle_core::Device::Cpu);

        let critic_network = MLPBuilder::new(observation_space.shape().iter().sum(), 1, vb)
            .activation(candle_nn::Activation::Relu)
            .build()
            .unwrap();

        let critic_optimizer =
            AdamW::new(var_map.all_vars(), config).expect("Failed to create AdamW");

        let mut actor = PPOActorBuilder::new(
            observation_space,
            action_space,
            Box::new(MLPProbabilisticActor::new(actor_network)),
            Box::new(critic_network),
            critic_optimizer,
            actor_optimizer,
        )
        .batch_size(128)
        .normalize_advantage(false)
        .build();
        actor.learn(&mut env, 100000).unwrap();
    }
}

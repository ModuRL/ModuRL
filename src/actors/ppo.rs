use candle_core::{safetensors::save, Error, IndexOp, Tensor};
use candle_nn::Optimizer;
use std::collections::{HashMap, VecDeque};

use crate::{
    buffers::{rollout_buffer::RolloutBuffer, experience},
    models::probabilistic_model::ProbabilisticActor,
    spaces,
    tensor_operations::torch_like_min,
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
        let samples = self.rollout_buffer.clear_and_get_all();
        for epoch in 0..self.num_epochs {
            let mut old_log_probs = vec![];
            let mut returns = vec![];
            let mut advantages = vec![];
            for sample in &samples {
                let actions = sample.actions();
                let states = sample.states();
                let rewards = sample.rewards();
                let dones = sample.dones();

                let (log_probs, entropy) = self
                    .actor_network
                    .log_prob_and_entropy(states, actions)
                    .unwrap();

                let values = self.critic_network.forward(states).unwrap();

                let advantage = self
                    .compute_advantages(rewards, values.clone(), dones)
                    .unwrap();

                let this_return = self.compute_returns(rewards, &values, dones).unwrap();

                old_log_probs.push(log_probs);
                returns.push(this_return);
                advantages.push(advantage);
            }

            let mut average_actor_loss = 0.0;
            let mut average_critic_loss = 0.0;

            for (i, sample) in samples.iter().enumerate() {
                let actions = sample.actions();
                let states = sample.states();

                let returns = returns[i].clone();
                let advantages = advantages[i].clone();
                let batch_old_log_probs = old_log_probs[i].clone();

                // Compute the loss and backpropagate
                let (actor_loss, critic_loss) = self
                    .compute_loss(states, actions, batch_old_log_probs, advantages, returns)
                    .unwrap();

                average_actor_loss += actor_loss;
                average_critic_loss += critic_loss;
            }

            average_actor_loss /= samples.len() as f64;
            average_critic_loss /= samples.len() as f64;

            println!("Epoch: {}", epoch + 1);
            println!(
                "Average Actor Loss: {}, Average Critic Loss: {}",
                average_actor_loss, average_critic_loss
            );
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
        let mut last_advantage = Tensor::from_vec(vec![0.0f32], &[], values.device())?;
        let values = values.squeeze(1).unwrap();
        let values = values.squeeze(1).unwrap();

        let mut last_value = values.get(values.dims()[0] - 1).unwrap().get(0).unwrap();

        for i in (0..rewards.dims1().unwrap()).rev() {
            if dones.get(i).unwrap().to_scalar::<f32>().unwrap() > 0.5 {
                last_advantage = Tensor::from_vec(vec![0.0f32], &[], values.device())?;
                last_value = Tensor::from_vec(vec![0.0f32], &[], values.device()).unwrap();
            }
            let reward = rewards.get(i).unwrap();
            let value = values.get(i).unwrap();

            let delta = reward + self.gamma as f64 * last_value - value.clone();
            last_advantage = (delta.unwrap()
                + self.gamma as f64 * self.gae_lambda as f64 * last_advantage)
                .unwrap();
            advantages.push(last_advantage.clone());
            last_value = value;
        }

        advantages.reverse();
        let advantages = Tensor::stack(&advantages, 0).unwrap();

        Ok(advantages)
    }

    // Computes returns as the discounted sum of rewards.
    // Assumes rewards and dones are 1D tensors.
    fn compute_returns(
        &self,
        rewards: &candle_core::Tensor,
        values: &candle_core::Tensor,
        dones: &candle_core::Tensor,
    ) -> Result<candle_core::Tensor, candle_core::Error> {
        let seq_len = rewards.dims()[0];

        // Create next_values by shifting values tensor
        let next_values = values.i(1..seq_len).unwrap();
        let next_values = next_values.squeeze(1).unwrap();
        let next_values = next_values.squeeze(1).unwrap();

        // put an extra 0 at the end of next_values
        let last_value = Tensor::from_vec(vec![0.0f32], &[1], values.device())?;
        let next_values = Tensor::cat(&[next_values, last_value], 0).unwrap();

        // Calculate returns: rewards + gamma * (1-dones) * next_values
        let gamma_tensor = Tensor::full(self.gamma, rewards.shape(), rewards.device())?;
        let next_value_contribution = ((1.0 - dones) * gamma_tensor * next_values).unwrap();
        let returns = (rewards + next_value_contribution)?;

        Ok(returns)
    }

    /// Compute the loss for the actor and critic networks
    /// Then backpropagate the loss
    fn compute_loss(
        &mut self,
        states: &candle_core::Tensor,
        actions: &candle_core::Tensor,
        old_log_probs: candle_core::Tensor,
        advantages: candle_core::Tensor,
        returns: candle_core::Tensor,
    ) -> Result<(f64, f64), Error> {
        let advantages = if self.normalize_advantage {
            let advantages_mean = advantages.mean(0).unwrap().to_scalar::<f32>().unwrap();
            let advantage_diff = (advantages.clone() - advantages_mean as f64).unwrap();

            let epsilon = f32::EPSILON;
            let advantages_std_sqrt = (advantage_diff.clone() * advantage_diff)
                .unwrap()
                .mean(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                .sqrt()
                .max(epsilon);
            ((advantages.clone() - advantages_mean.clone() as f64).unwrap()
                / (advantages_std_sqrt as f64))
                .unwrap()
        } else {
            advantages
        };

        let (log_probs, entropy) = self
            .actor_network
            .log_prob_and_entropy(states, actions)
            .unwrap();

        println!(
            "KL Divergence: {}",
            (log_probs.clone() - old_log_probs.clone())
                .unwrap()
                .mean_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
        );

        let ratio = (log_probs - old_log_probs)
            .unwrap()
            .clamp(-5.0, 5.0)
            .unwrap()
            .exp()
            .unwrap();

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
                    .unwrap())
                .unwrap();
                actor_loss
            }
            false => {
                let actor_loss = (-1.0 * (ratio.clone() * advantages.clone()).unwrap()).unwrap();
                actor_loss
            }
        };

        let values = self.critic_network.forward(states).unwrap();
        let values = values.squeeze(1).unwrap();
        let values = values.squeeze(1).unwrap();

        //println!("Returns: {:?}", returns.mean_all().unwrap());
        //println!("Values: {:?}", values.mean_all().unwrap());

        let values_minus_targets = (values.clone() - returns.clone()).unwrap();

        // Use mse for now
        let critic_loss = (values_minus_targets.clone() * values_minus_targets).unwrap();

        let entropy_loss = (-1.0 * entropy).unwrap();

        let final_actor_loss =
            (actor_loss.clone() + ((self.ent_coef as f64) * entropy_loss.clone()))?;

        let final_critic_loss = ((self.vf_coef as f64) * critic_loss.clone()).unwrap();

        self.actor_optimizer
            .backward_step(&final_actor_loss)
            .unwrap();
        self.critic_optimizer
            .backward_step(&final_critic_loss)
            .unwrap();

        let actor_loss_scalar = actor_loss.mean_all().unwrap().to_scalar::<f32>().unwrap() as f64;

        Ok((
            actor_loss_scalar,
            critic_loss.mean_all().unwrap().to_scalar::<f32>().unwrap() as f64,
        ))
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
                self.rollout_buffer.add(experience::new(
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
    fn ppo_cartpole() {
        let tracer = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .finish();
        tracing::subscriber::set_global_default(tracer).unwrap();
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
        .hidden_layer_sizes(vec![64, 64])
        .build()
        .unwrap();

        let mut config = ParamsAdamW::default();
        config.lr = 2.5e-4;

        let actor_optimizer =
            AdamW::new(var_map.all_vars(), config.clone()).expect("Failed to create AdamW");

        let var_map = VarMap::new();
        let vb =
            VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &candle_core::Device::Cpu);

        let critic_network = MLPBuilder::new(observation_space.shape().iter().sum(), 1, vb)
            .activation(candle_nn::Activation::Relu)
            .hidden_layer_sizes(vec![64, 64])
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
        .batch_size(512)
        .mini_batch_size(64)
        .normalize_advantage(true)
        .ent_coef(0.01)
        .clipped(true)
        .build();
        actor.learn(&mut env, 1000).unwrap();
    }
}

use candle_core::{safetensors::save, Error, IndexOp, Tensor};
use candle_nn::Optimizer;
use std::collections::{HashMap, VecDeque};

use crate::{
    buffers::{experience, rollout_buffer::RolloutBuffer},
    models::probabilistic_model::ProbabilisticActor,
    spaces,
    tensor_operations::torch_like_min,
};

use super::Actor;

#[derive(Debug, Clone)]
struct PPOExperience {
    state: Tensor,
    next_state: Tensor,
    action: Tensor,
    reward: f32,
    done: bool,
    log_prob: Tensor,
}

impl PPOExperience {
    pub fn new(
        state: Tensor,
        next_state: Tensor,
        action: Tensor,
        reward: f32,
        done: bool,
        log_prob: Tensor,
    ) -> Self {
        Self {
            state,
            next_state,
            action,
            reward,
            done,
            log_prob,
        }
    }
}

impl experience::Experience for PPOExperience {
    fn get_elements(&self) -> Vec<Tensor> {
        vec![
            self.state.clone(),
            self.next_state.clone(),
            self.action.clone(),
            Tensor::from_vec(vec![self.reward], &[], self.state.device()).unwrap(),
            Tensor::from_vec(
                vec![if self.done { 1.0f32 } else { 0.0f32 }],
                &[],
                self.state.device(),
            )
            .unwrap(),
            self.log_prob.clone(),
        ]
    }
}

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
    rollout_buffer: RolloutBuffer<PPOExperience>,
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
        let batches = self.rollout_buffer.clear_and_get_all();
        for epoch in 0..self.num_epochs {
            let mut old_log_probs = vec![];
            let mut returns = vec![];
            let mut advantages = vec![];
            for batch in &batches {
                let elements = batch.get_elements();
                let states = elements[0].clone();
                let rewards = elements[3].clone();
                let dones = elements[4].clone();
                let log_probs = elements[5].clone();

                let values = self.critic_network.forward(&states).unwrap();

                let this_return = self.compute_returns(&rewards, &dones).unwrap();
                let this_advantage = self.compute_gae(&rewards, &values, &dones).unwrap();

                returns.push(this_return.detach());
                advantages.push(this_advantage.detach());
                old_log_probs.push(log_probs.detach().clone());
            }

            let mut average_actor_loss = 0.0;
            let mut average_critic_loss = 0.0;

            for (i, batch) in batches.iter().enumerate() {
                let actions = batch.get_elements()[2].clone();
                let states = batch.get_elements()[0].clone();

                let returns = returns[i].clone();
                let advantages = advantages[i].clone();
                let batch_old_log_probs = old_log_probs[i].clone();

                // Compute the loss and backpropagate
                let (actor_loss, critic_loss) = self
                    .compute_loss(&states, &actions, batch_old_log_probs, advantages, returns)
                    .unwrap();

                average_actor_loss += actor_loss;
                average_critic_loss += critic_loss;
            }

            average_actor_loss /= batches.len() as f64;
            average_critic_loss /= batches.len() as f64;

            println!("Epoch: {}", epoch + 1);
            println!(
                "Average Actor Loss: {}, Average Critic Loss: {}",
                average_actor_loss, average_critic_loss
            );
        }

        Ok(())
    }

    // Computes returns as the discounted sum of rewards.
    // Assumes rewards and dones are 1D tensors.
    fn compute_returns(
        &self,
        rewards: &candle_core::Tensor,
        dones: &candle_core::Tensor,
    ) -> Result<candle_core::Tensor, candle_core::Error> {
        let mut running_return = 0.0;
        let mut returns = vec![];

        for i in (0..rewards.dims1().unwrap()).rev() {
            let reward = rewards.i(i).unwrap().to_scalar::<f32>().unwrap() as f64;
            let done = dones.i(i).unwrap().to_scalar::<f32>().unwrap() as f64 > 0.5;
            if done {
                running_return = 0.0;
            }

            running_return = reward + self.gamma as f64 * running_return;
            returns.push(running_return as f32);
        }

        returns.reverse();
        let returns_tensor =
            candle_core::Tensor::from_vec(returns, rewards.shape(), rewards.device())?;

        Ok(returns_tensor)
    }

    fn compute_gae(
        &self,
        rewards: &candle_core::Tensor,
        values: &candle_core::Tensor,
        dones: &candle_core::Tensor,
    ) -> Result<candle_core::Tensor, candle_core::Error> {
        let mut advantages: Vec<f32> = vec![];
        let mut last_gae = 0.0;

        for i in (0..rewards.dims1().unwrap()).rev() {
            let reward = rewards
                .i(i)
                .unwrap()
                .reshape(&[])
                .unwrap()
                .to_scalar::<f32>()
                .unwrap() as f64;
            let value = values
                .i(i)
                .unwrap()
                .reshape(&[])
                .unwrap()
                .to_scalar::<f32>()
                .unwrap() as f64;
            let done = dones
                .i(i)
                .unwrap()
                .reshape(&[])
                .unwrap()
                .to_scalar::<f32>()
                .unwrap() as f64
                > 0.5;
            let next_value = if i == rewards.dims1().unwrap() - 1 || done {
                0.0
            } else {
                values
                    .i(i + 1)
                    .unwrap()
                    .reshape(&[])
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap() as f64
            };

            //delta = rewards[t] + gamma * next_value - values[t]
            //last_gae = delta + gamma * gae_lambda * last_gae * (1 - dones[t])
            //advantages[t] = last_gae

            let delta = reward + self.gamma as f64 * next_value - value;
            last_gae = delta
                + self.gamma as f64 * self.gae_lambda as f64 * last_gae * (1.0 - f64::from(done));
            advantages.push(last_gae as f32);
        }

        advantages.reverse();

        let advantages_tensor =
            candle_core::Tensor::from_vec(advantages, rewards.shape(), rewards.device()).unwrap();

        return Ok(advantages_tensor);
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

            let advantages_std_sqrt = (advantage_diff.clone() * advantage_diff)
                .unwrap()
                .mean(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                .sqrt()
                .max(f32::EPSILON);
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

        println!("Returns: {:?}", returns.mean_all().unwrap());
        println!("Values: {:?}", values.mean_all().unwrap());

        let values_minus_targets = (values.clone() - returns.clone()).unwrap();

        // Use mse for now
        let critic_loss = (values_minus_targets.clone() * values_minus_targets).unwrap();

        let entropy_loss = entropy;

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

                let (mut log_prob, _) = self
                    .actor_network
                    .log_prob_and_entropy(&obs, &action)
                    .unwrap();
                log_prob = log_prob.squeeze(0).unwrap();

                (next_obs, reward, done) = env.step(actual_action).unwrap();
                self.rollout_buffer.add(PPOExperience::new(
                    obs.clone(),
                    next_obs.clone(),
                    action,
                    reward,
                    done,
                    log_prob,
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
        // TODO! Reevaluate this is there ever a case where anything different happens?
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
        config.lr = 3e-4;

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
        .batch_size(128)
        .mini_batch_size(32)
        .normalize_advantage(true)
        .ent_coef(0.01)
        .gamma(0.99)
        .vf_coef(1.0)
        .clip_range(0.2)
        .clipped(true)
        .num_epochs(3)
        .build();
        actor.learn(&mut env, 1000).unwrap();
    }
}

use candle_core::{safetensors::save, IndexOp, Tensor};
use candle_nn::{loss, Optimizer};
use std::collections::HashMap;
use std::marker::PhantomData;

use crate::{
    buffers::{experience, rollout_buffer::RolloutBuffer},
    gym::StepInfo,
    lr_scheduler::LrScheduler,
    models::probabilistic_model::ProbabilisticActor,
    spaces,
    tensor_operations::torch_like_min,
};

#[derive(Debug)]
pub enum PPOError<AE, GE>
where
    AE: std::fmt::Debug,
    GE: std::fmt::Debug,
{
    ActorError(AE),
    GymError(GE),
    TensorError(candle_core::Error),
}

impl<AE, GE> From<candle_core::Error> for PPOError<AE, GE>
where
    AE: std::fmt::Debug,
    GE: std::fmt::Debug,
{
    fn from(err: candle_core::Error) -> Self {
        PPOError::TensorError(err)
    }
}

use super::Actor;

#[derive(Debug, Clone)]
struct PPOExperience {
    state: Tensor,
    next_state: Tensor,
    action: Tensor,
    reward: f32,
    done: bool,
    log_prob: Tensor,
    advantage: Option<Tensor>,
    this_return: Option<Tensor>,
}

impl PPOExperience {
    pub fn new(
        state: Tensor,
        next_state: Tensor,
        action: Tensor,
        reward: f32,
        done: bool,
        log_prob: Tensor,
        advantage: Option<Tensor>,
        this_return: Option<Tensor>,
    ) -> Self {
        Self {
            state,
            next_state,
            action,
            reward,
            done,
            log_prob,
            advantage,
            this_return,
        }
    }
}

impl experience::Experience for PPOExperience {
    type Error = candle_core::Error;
    fn get_elements(&self) -> Result<Vec<Tensor>, candle_core::Error> {
        Ok(vec![
            self.state.clone(),
            self.next_state.clone(),
            self.action.clone(),
            Tensor::from_vec(vec![self.reward], &[], self.state.device())?,
            Tensor::from_vec(
                vec![if self.done { 1.0f32 } else { 0.0f32 }],
                &[],
                self.state.device(),
            )?,
            self.log_prob.clone(),
            self.advantage
                .clone()
                .unwrap_or_else(|| panic!("Advantage not set")),
            self.this_return
                .clone()
                .unwrap_or_else(|| panic!("Return not set")),
        ])
    }
}

#[allow(unused)]
enum PPOElement {
    State = 0,
    NextState = 1,
    Action = 2,
    Reward = 3,
    Done = 4,
    LogProb = 5,
    Advantage = 6,
    Return = 7,
}

pub struct PPOActor<O1, O2, AE, GE>
where
    O1: Optimizer,
    O2: Optimizer,
    AE: std::fmt::Debug,
    GE: std::fmt::Debug,
{
    clipped: bool,
    gamma: f32,
    gae_lambda: f32,
    clip_range: f32,
    normalize_advantage: bool,
    vf_coef: f32,
    ent_coef: f32,
    critic_optimizer: O1,
    actor_optimizer: O2,
    observation_space: Box<dyn spaces::Space>,
    action_space: Box<dyn spaces::Space>,
    critic_network: Box<dyn candle_core::Module>,
    actor_network: Box<dyn ProbabilisticActor<Error = AE>>,
    rollout_buffer: RolloutBuffer<PPOExperience>,
    num_epochs: usize,
    batch_size: usize,
    actor_lr_scheduler: Option<Box<dyn LrScheduler>>,
    critic_lr_scheduler: Option<Box<dyn LrScheduler>>,
    _phantom: PhantomData<GE>,
}

pub struct PPOActorBuilder<O1, O2, AE, GE>
where
    O1: Optimizer,
    O2: Optimizer,
    AE: std::fmt::Debug,
    GE: std::fmt::Debug,
{
    // Neccesary hyperparameters
    critic_optimizer: O1,
    actor_optimizer: O2,
    observation_space: Box<dyn spaces::Space>,
    action_space: Box<dyn spaces::Space>,
    critic_network: Box<dyn candle_core::Module>,
    actor_network: Box<dyn ProbabilisticActor<Error = AE>>,
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
    actor_lr_scheduler: Option<Box<dyn LrScheduler>>,
    critic_lr_scheduler: Option<Box<dyn LrScheduler>>,
    _phantom: PhantomData<GE>,
}

impl<O1, O2, AE, GE> PPOActorBuilder<O1, O2, AE, GE>
where
    O1: Optimizer,
    O2: Optimizer,
    AE: std::fmt::Debug,
    GE: std::fmt::Debug,
{
    pub fn new(
        observation_space: Box<dyn spaces::Space>,
        action_space: Box<dyn spaces::Space>,
        actor_network: Box<dyn ProbabilisticActor<Error = AE>>,
        critic_network: Box<dyn candle_core::Module>,
        critic_optimizer: O1,
        actor_optimizer: O2,
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
            actor_lr_scheduler: None,
            critic_lr_scheduler: None,
            _phantom: PhantomData,
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

    /// returns a percent of t
    pub fn actor_lr_scheduler(mut self, lr_scheduler: Box<dyn LrScheduler>) -> Self {
        self.actor_lr_scheduler = Some(lr_scheduler);
        self
    }

    pub fn critic_lr_scheduler(mut self, lr_scheduler: Box<dyn LrScheduler>) -> Self {
        self.critic_lr_scheduler = Some(lr_scheduler);
        self
    }

    pub fn build(self) -> PPOActor<O1, O2, AE, GE> {
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
            actor_lr_scheduler: self.actor_lr_scheduler,
            critic_lr_scheduler: self.critic_lr_scheduler,
            _phantom: PhantomData,
        }
    }
}

impl<O1, O2, AE, GE> PPOActor<O1, O2, AE, GE>
where
    O1: Optimizer,
    O2: Optimizer,
    AE: std::fmt::Debug,
    GE: std::fmt::Debug,
{
    fn optimize(&mut self) -> Result<(), PPOError<AE, GE>> {
        self.add_advantages_and_returns()?;
        for epoch in 0..self.num_epochs {
            let batches = self.rollout_buffer.get_all_shuffled()?;
            let mut average_abs_actor_loss = 0.0;
            let mut average_abs_critic_loss = 0.0;

            for batch in batches.iter() {
                let actions = batch.get_elements()[PPOElement::Action as usize].clone();
                let states = batch.get_elements()[PPOElement::State as usize].clone();
                let batch_old_log_probs =
                    batch.get_elements()[PPOElement::LogProb as usize].clone();
                let advantages = batch.get_elements()[PPOElement::Advantage as usize].clone();
                let returns = batch.get_elements()[PPOElement::Return as usize].clone();

                // Compute the loss and backpropagate
                let (actor_loss, critic_loss) = self.compute_loss(
                    &states,
                    &actions.detach(),
                    batch_old_log_probs.detach(),
                    advantages,
                    returns,
                )?;

                average_abs_actor_loss += actor_loss;
                average_abs_critic_loss += critic_loss;
            }

            average_abs_actor_loss /= batches.len() as f64;
            average_abs_critic_loss /= batches.len() as f64;

            println!("Epoch: {}", epoch + 1);
            println!(
                "Average Abs Actor Loss: {}, Average Abs Critic Loss: {}",
                average_abs_actor_loss, average_abs_critic_loss
            );
        }

        self.rollout_buffer.clear();
        Ok(())
    }

    fn add_advantages_and_returns(&mut self) -> Result<(), PPOError<AE, GE>> {
        let experiences = self.rollout_buffer.get_raw();
        let mut rewards = vec![];
        let mut dones = vec![];
        let mut states = vec![];
        let mut next_states = vec![];

        for experience in experiences.into_iter() {
            rewards.push(experience.reward);
            dones.push(if experience.done { 1.0f32 } else { 0.0f32 });
            states.push(experience.state.clone());
            next_states.push(experience.next_state.clone());
        }

        let values_tensor = self.critic_network.forward(&Tensor::stack(&states, 0)?)?;

        let next_values_tensor = self
            .critic_network
            .forward(&Tensor::stack(&next_states, 0)?)?;

        let device = experiences[0].log_prob.device();

        let rewards_length = rewards.len();
        let dones_length = dones.len();
        let rewards_tensor = candle_core::Tensor::from_vec(rewards, &[rewards_length], device)?;
        let dones_tensor = candle_core::Tensor::from_vec(dones, &[dones_length], device)?;

        let advantages = self.compute_gae(
            &rewards_tensor,
            &values_tensor,
            &next_values_tensor,
            &dones_tensor,
        )?;
        let returns = (&advantages + &values_tensor.flatten_all()?)?.clamp(-1e3, 1e3)?;

        let experiences = self.rollout_buffer.get_raw_mut();

        for (i, experience) in experiences.iter_mut().enumerate() {
            experience.advantage = Some(advantages.i(i)?.clone());
            experience.this_return = Some(returns.i(i)?.clone());
        }

        Ok(())
    }

    fn compute_gae(
        &self,
        rewards: &candle_core::Tensor,
        values: &candle_core::Tensor,
        next_values: &candle_core::Tensor,
        dones: &candle_core::Tensor,
    ) -> Result<candle_core::Tensor, candle_core::Error> {
        let shape = rewards.shape();
        let device = rewards.device();

        let rewards: Vec<f32> = rewards.flatten_all()?.to_vec1()?;
        let dones: Vec<f32> = dones.flatten_all()?.to_vec1()?;
        let values: Vec<f32> = values.flatten_all()?.to_vec1()?;
        let next_values: Vec<f32> = next_values.flatten_all()?.to_vec1()?;

        let mut advantages = vec![0.0; rewards.len()];
        let mut gae = 0.0;

        // Compute GAE backwards through the trajectory
        for i in (0..rewards.len()).rev() {
            let done = dones[i] > 0.5;
            let next_non_terminal = if done { 0.0 } else { 1.0 };

            // Use actual next state value (zero for terminal states due to next_non_terminal)
            let next_value = next_values[i] * next_non_terminal;

            // TD error: δ = r + γ * V(s') - V(s)
            let delta = rewards[i] + self.gamma * next_value - values[i];

            // GAE: A = δ + γ * λ * next_gae * (1 - done)
            gae = delta + self.gamma * self.gae_lambda * gae * next_non_terminal;
            advantages[i] = gae;
        }

        let advantages_tensor = candle_core::Tensor::from_vec(advantages, shape, device)?;

        Ok(advantages_tensor)
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
    ) -> Result<(f64, f64), PPOError<AE, GE>> {
        let advantages = if self.normalize_advantage {
            let advantages_mean = advantages.mean(0)?.to_scalar::<f32>()?;
            let advantage_diff = (advantages.clone() - advantages_mean as f64)?;

            let advantages_std_sqrt = advantage_diff
                .sqr()?
                .mean(0)?
                .to_scalar::<f32>()?
                .sqrt()
                .max(1e-6);
            ((advantages.clone() - advantages_mean.clone() as f64)? / (advantages_std_sqrt as f64))?
        } else {
            advantages
        };

        let (log_probs, entropy) = self
            .actor_network
            .log_prob_and_entropy(states, actions)
            .map_err(PPOError::ActorError)?;

        let ratio = (&log_probs - &old_log_probs)?.clamp(-20.0, 10.0)?.exp()?;

        let actor_loss = match self.clipped {
            true => {
                let clip_range = self.clip_range;
                let clipped_ratio = ratio.clamp(1.0 - clip_range, 1.0 + clip_range)?;

                let surrogate1 = (ratio.clone() * advantages.clone())?;
                let surrogate2 = (clipped_ratio.clone() * advantages.clone())?;
                let surrogate = torch_like_min(&surrogate1, &surrogate2)?;
                let actor_loss = (-1.0 * surrogate.mean(0)?)?;

                actor_loss
            }
            false => {
                let surrogate = (ratio * &advantages)?;
                (-1.0 * surrogate.mean(0)?)?
            }
        };

        let values = self.critic_network.forward(states)?;
        let values = values.squeeze(1)?;
        let values = values.squeeze(1)?;

        let returns = {
            let returns_mean = returns.mean(0)?.to_scalar::<f32>()?;
            let returns_diff = (returns.clone() - returns_mean as f64)?;
            let returns_std_sqrt = returns_diff
                .sqr()?
                .mean(0)?
                .to_scalar::<f32>()?
                .sqrt()
                .max(1e-6);
            ((returns.clone() - returns_mean.clone() as f64)? / (returns_std_sqrt as f64))?
        };

        // Use mse for now
        let critic_loss = loss::mse(&values, &returns)?;

        let entropy_loss = entropy.mean(0)?;

        let final_actor_loss =
            (actor_loss.clone() - ((self.ent_coef as f64) * entropy_loss.clone()))?;

        let final_critic_loss = ((self.vf_coef as f64) * critic_loss.clone())?;

        // clip the gradients and step the optimizers
        let actor_grad = &mut final_actor_loss.backward()?;
        let _actor_grad_norm = crate::tensor_operations::clip_gradients(actor_grad, 1.0)?;
        self.actor_optimizer.step(actor_grad)?;

        let critic_grad = &mut final_critic_loss.backward()?;
        let _critic_grad_norm = crate::tensor_operations::clip_gradients(critic_grad, 1.0)?;
        self.critic_optimizer.step(critic_grad)?;

        let actor_loss_scalar = actor_loss.abs()?.mean_all()?.to_scalar::<f32>()? as f64;
        let critic_loss_scalar = critic_loss.abs()?.mean_all()?.to_scalar::<f32>()? as f64;
        Ok((actor_loss_scalar, critic_loss_scalar))
    }
}

impl<O1, O2, AE, GE> Actor for PPOActor<O1, O2, AE, GE>
where
    O1: Optimizer,
    O2: Optimizer,
    AE: std::fmt::Debug,
    GE: std::fmt::Debug,
{
    type Error = PPOError<AE, GE>;
    type GymError = GE;
    fn act(
        &mut self,
        observation: &candle_core::Tensor,
    ) -> Result<candle_core::Tensor, PPOError<AE, GE>> {
        self.actor_network
            .sample(observation)
            .map_err(PPOError::ActorError)
    }

    fn learn(
        &mut self,
        env: &mut dyn crate::gym::Gym<Error = Self::GymError>,
        num_timesteps: usize,
    ) -> Result<(), PPOError<AE, GE>> {
        let mut last_optimization_step = 0;
        let mut elapsed_timesteps = 0;
        let mut episode_idx = 0;
        while elapsed_timesteps < num_timesteps {
            let mut done = false;
            let mut obs = env.reset().map_err(PPOError::GymError)?;
            let (mut next_obs, mut reward);

            while !done && self.rollout_buffer.len() < self.batch_size {
                let mut new_observations_shape: Vec<usize> = vec![1];
                new_observations_shape.append(&mut self.observation_space.shape());
                obs = obs.reshape(&*new_observations_shape)?;

                let action = self.act(&obs)?;
                // Now: the action is a tensor of shape [1, dim] so we need to squeeze it to [dim]
                let actual_action = self.action_space.from_neurons(&action);
                let actual_action = actual_action.squeeze(0)?;

                let (mut log_prob, _) = self
                    .actor_network
                    .log_prob_and_entropy(&obs, &action)
                    .map_err(PPOError::ActorError)?;
                log_prob = log_prob.squeeze(0)?;

                let truncated;
                StepInfo {
                    state: next_obs,
                    reward,
                    done,
                    truncated,
                } = env.step(actual_action).map_err(PPOError::GymError)?;
                self.rollout_buffer.add(PPOExperience::new(
                    obs.clone(),
                    next_obs.clone(),
                    action,
                    reward,
                    done,
                    log_prob,
                    None,
                    None,
                ));
                obs = next_obs;
                done = done || truncated;
            }

            episode_idx += 1;

            if self.rollout_buffer.len() >= self.batch_size {
                elapsed_timesteps += self.rollout_buffer.len();

                println!("Episode: {}", episode_idx + 1);

                println!(
                    "Average episode length: {}",
                    self.rollout_buffer.len() / (episode_idx + 1 - last_optimization_step)
                );
                last_optimization_step = episode_idx;
                if let Some(scheduler) = &self.actor_lr_scheduler {
                    let progress = (elapsed_timesteps as f64) / (num_timesteps as f64);
                    let new_lr = scheduler.get_lr(progress);
                    self.actor_optimizer.set_learning_rate(new_lr);
                }
                if let Some(scheduler) = &self.critic_lr_scheduler {
                    let progress = (elapsed_timesteps as f64) / (num_timesteps as f64);
                    let new_lr = scheduler.get_lr(progress);
                    self.critic_optimizer.set_learning_rate(new_lr);
                }
                self.optimize()?;
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
        save(&hashmap, path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        distributions::CategoricalDistribution,
        gym::{common_gyms::CartPole, Gym},
        models::{probabilistic_model::MLPProbabilisticActor, MLPBuilder},
        tensor_operations::tanh,
    };
    use candle_nn::{VarBuilder, VarMap};
    use candle_optimisers::adam::{Adam, ParamsAdam};

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

        // Actor network: 2x64, tanh activation
        let actor_network = MLPBuilder::new(
            observation_space.shape().iter().sum(),
            action_space.shape().iter().sum::<usize>(),
            vb.clone(),
        )
        .activation(Box::new(tanh))
        .hidden_layer_sizes(vec![64, 64])
        .name("actor_network")
        .build()
        .unwrap();
        let mut config = ParamsAdam::default();
        // Optimizers: both with lr=3e-4
        config.lr = 3e-4;
        let actor_optimizer =
            Adam::new(var_map.all_vars(), config.clone()).expect("Failed to create Adam");

        let critic_var_map = VarMap::new();
        let critic_vb = VarBuilder::from_varmap(
            &critic_var_map,
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        );

        // Critic network: 2x64, tanh activation
        let critic_network = MLPBuilder::new(observation_space.shape().iter().sum(), 1, critic_vb)
            .activation(Box::new(tanh))
            .hidden_layer_sizes(vec![64, 64])
            .name("critic_network")
            .build()
            .unwrap();

        config.lr = 3e-4;
        let critic_optimizer =
            Adam::new(critic_var_map.all_vars(), config.clone()).expect("Failed to create Adam");

        // PPO config
        // Stable baselines3 config:
        let mut actor = PPOActorBuilder::new(
            observation_space,
            action_space,
            Box::new(MLPProbabilisticActor::<CategoricalDistribution>::new(
                actor_network,
            )),
            Box::new(critic_network),
            critic_optimizer,
            actor_optimizer,
        )
        .batch_size(2048)
        .mini_batch_size(64)
        .normalize_advantage(true)
        .ent_coef(0.01)
        .gamma(0.99)
        .vf_coef(0.5)
        .clip_range(0.2)
        .clipped(true)
        .gae_lambda(0.95)
        .num_epochs(10)
        .actor_lr_scheduler(Box::new(|progress: f64| 3e-4 * (1.0 - progress)))
        .critic_lr_scheduler(Box::new(|progress: f64| 6e-4 * (1.0 - progress)))
        .build();

        actor.learn(&mut env, 500000).unwrap();
    }
}

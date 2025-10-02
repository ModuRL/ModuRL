use bon::bon;
use candle_core::{D, IndexOp, Tensor};
use candle_nn::{Optimizer, loss};
use std::marker::PhantomData;

use crate::{
    buffers::{experience, rollout_buffer::RolloutBuffer},
    gym::{VectorizedGym, VectorizedStepInfo},
    lr_scheduler::LrScheduler,
    models::probabilistic_model::ProbabilisticActor,
    spaces,
    tensor_operations::{tensor_has_nan, torch_like_min},
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
    states: Tensor,
    next_states: Tensor,
    actions: Tensor,
    rewards: Tensor,
    dones: Vec<bool>,
    truncateds: Vec<bool>,
    log_probs: Tensor,
    advantages: Option<Tensor>,
    this_returns: Option<Tensor>,
}

impl PPOExperience {
    pub fn new(
        states: Tensor,
        next_states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        dones: Vec<bool>,
        truncateds: Vec<bool>,
        log_probs: Tensor,
        advantages: Option<Tensor>,
        this_returns: Option<Tensor>,
    ) -> Self {
        Self {
            states,
            next_states,
            actions,
            rewards,
            dones,
            truncateds,
            log_probs,
            advantages,
            this_returns,
        }
    }
}

impl experience::Experience for PPOExperience {
    type Error = candle_core::Error;
    fn get_elements(&self) -> Result<Vec<Tensor>, candle_core::Error> {
        Ok(vec![
            self.states.clone(),
            self.next_states.clone(),
            self.actions.clone(),
            self.rewards.clone(),
            Tensor::from_vec(
                self.dones
                    .iter()
                    .map(|&d| if d { 1.0 } else { 0.0 })
                    .collect::<Vec<f32>>(),
                &[self.dones.len()],
                self.states.device(),
            )?,
            Tensor::from_vec(
                self.truncateds
                    .iter()
                    .map(|&d| if d { 1.0 } else { 0.0 })
                    .collect::<Vec<f32>>(),
                &[self.truncateds.len()],
                self.states.device(),
            )?,
            self.log_probs.clone(),
            self.advantages
                .clone()
                .unwrap_or_else(|| panic!("Advantage not set")),
            self.this_returns
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
    Truncated = 5,
    LogProb = 6,
    Advantage = 7,
    Return = 8,
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

#[bon]
impl<O1, O2, AE, GE> PPOActor<O1, O2, AE, GE>
where
    O1: Optimizer,
    O2: Optimizer,
    AE: std::fmt::Debug,
    GE: std::fmt::Debug,
{
    #[builder]
    pub fn new(
        action_space: Box<dyn spaces::Space>,
        actor_network: Box<dyn ProbabilisticActor<Error = AE>>,
        critic_network: Box<dyn candle_core::Module>,
        critic_optimizer: O1,
        actor_optimizer: O2,
        #[builder(default = true)] clipped: bool,
        #[builder(default = 0.99)] gamma: f32,
        #[builder(default = 0.95)] gae_lambda: f32,
        #[builder(default = 0.2)] clip_range: f32,
        #[builder(default = true)] normalize_advantage: bool,
        #[builder(default = 0.5)] vf_coef: f32,
        #[builder(default = 0.01)] ent_coef: f32,
        #[builder(default = 1024)] batch_size: usize,
        #[builder(default = 128)] mini_batch_size: usize,
        #[builder(default = 1)] num_epochs: usize,
        actor_lr_scheduler: Option<Box<dyn LrScheduler>>,
        critic_lr_scheduler: Option<Box<dyn LrScheduler>>,
    ) -> Self {
        if batch_size > 0 {
            assert!(batch_size > 0);
        }

        Self {
            clipped,
            gamma,
            gae_lambda,
            clip_range,
            normalize_advantage,
            vf_coef,
            ent_coef,
            critic_optimizer,
            actor_optimizer,
            action_space,
            critic_network,
            actor_network,
            rollout_buffer: RolloutBuffer::new(mini_batch_size),
            num_epochs,
            batch_size,
            actor_lr_scheduler,
            critic_lr_scheduler,
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
            let mut count = 0;

            for batch in batches.iter() {
                let elements = batch.get_elements();

                let mut actions = elements[PPOElement::Action as usize].clone();
                let mut states = elements[PPOElement::State as usize].clone();
                let mut batch_old_log_probs = elements[PPOElement::LogProb as usize].clone();
                let mut advantages = elements[PPOElement::Advantage as usize].clone();
                let mut returns = elements[PPOElement::Return as usize].clone();
                // each of these tensors has shape [batch_size, env_count, ...]

                let action_shape = actions.shape().dims();
                let env_count = action_shape[1];
                let batch_size = action_shape[0];
                for tensor in [
                    &mut actions,
                    &mut states,
                    &mut batch_old_log_probs,
                    &mut advantages,
                    &mut returns,
                ] {
                    let mut shape = tensor.shape().dims().to_vec();
                    *tensor = tensor.flatten(0, 1)?;
                    // now the envs are interwoven and split so we have proper batch sizes
                    shape[0] = env_count;
                    shape[1] = batch_size;
                    tensor.reshape(shape)?;
                }

                let actions = actions.chunk(env_count, 0)?;
                let states = states.chunk(env_count, 0)?;
                let advantages = advantages.chunk(env_count, 0)?;
                let returns = returns.chunk(env_count, 0)?;
                let batch_old_log_probs = batch_old_log_probs.chunk(env_count, 0)?;

                // right now we have [batch_size, env_count, ...]
                // we chunk along the env_count dimension
                for env_idx in 0..actions.len() {
                    let actions = actions[env_idx].clone();
                    let states = states[env_idx].clone();
                    let advantages = advantages[env_idx].clone();
                    let returns = returns[env_idx].clone();
                    let batch_old_log_probs = batch_old_log_probs[env_idx].clone();

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
                    count += 1;
                }
            }

            // we could calculate this without keeping the count variable but then we need env count
            // I would rather not pass it around
            average_abs_actor_loss /= count as f64;
            average_abs_critic_loss /= count as f64;

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
        // For advantage return calculation:
        let mut rewards = vec![];
        let mut dones = vec![];
        let mut states = vec![];
        let mut next_states = vec![];
        // For splitting the envs
        let mut log_probs = vec![];
        let mut actions = vec![];

        for experience in experiences.into_iter() {
            rewards.push(experience.rewards.clone());
            dones.push(
                experience
                    .dones
                    .iter()
                    .map(|&d| if d { 1.0 } else { 0.0 })
                    .collect::<Vec<f32>>(),
            );
            states.push(experience.states.clone());
            next_states.push(experience.next_states.clone());
            log_probs.push(experience.log_probs.clone());
            actions.push(experience.actions.clone());
        }

        let values_tensor = self.critic_network.forward(&Tensor::stack(&states, 0)?)?;

        // TODO! there is no need to run everything through the critic again
        let next_values_tensor = self
            .critic_network
            .forward(&Tensor::stack(&next_states, 0)?)?;

        let device = values_tensor.device();

        let dones_shape = &[dones.len(), dones[0].len()];
        let rewards_tensor = candle_core::Tensor::stack(&rewards, 0)?;
        let dones_tensor = candle_core::Tensor::from_vec(
            dones.into_iter().flatten().collect(),
            dones_shape,
            device,
        )?;

        let advantages = self.compute_gae(
            &rewards_tensor,
            &values_tensor,
            &next_values_tensor,
            &dones_tensor,
        )?;

        let values_tensor = values_tensor.squeeze(D::Minus1)?;

        let returns = (&values_tensor + &advantages)?;
        let returns = returns.clamp(-1e3, 1e3)?;

        let experiences = self.rollout_buffer.get_raw_mut();

        for (i, experience) in experiences.iter_mut().enumerate() {
            experience.advantages = Some(advantages.i(i)?.clone());
            experience.this_returns = Some(returns.i(i)?.clone());
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

        let rewards: Vec<Vec<f32>> = rewards.to_vec2()?;
        let dones: Vec<Vec<f32>> = dones.to_vec2()?;
        let values: Vec<Vec<f32>> = values.squeeze(D::Minus1)?.to_vec2()?;
        let next_values: Vec<Vec<f32>> = next_values.squeeze(D::Minus1)?.to_vec2()?;

        let mut advantages = vec![0.0; rewards.len() * rewards[0].len()];

        for env_idx in 0..rewards[0].len() {
            let env_rewards: Vec<f32> = rewards.iter().map(|r| r[env_idx]).collect();
            let env_dones: Vec<f32> = dones.iter().map(|d| d[env_idx]).collect();
            let env_values: Vec<f32> = values.iter().map(|v| v[env_idx]).collect();
            let env_next_values: Vec<f32> = next_values.iter().map(|nv| nv[env_idx]).collect();

            let mut gae = 0.0;
            // Compute GAE backwards through the trajectory
            for i in (0..env_rewards.len()).rev() {
                let done = env_dones[i] > 0.5;
                let next_non_terminal = if done { 0.0 } else { 1.0 };

                // Use actual next state value (zero for terminal states due to next_non_terminal)
                let next_value = env_next_values[i] * next_non_terminal;

                // TD error: δ = r + γ * V(s') - V(s)
                let delta = env_rewards[i] + self.gamma * next_value - env_values[i];

                // GAE: A = δ + γ * λ * next_gae * (1 - done)
                gae = delta + self.gamma * self.gae_lambda * gae * next_non_terminal;
                advantages[i * rewards[0].len() + env_idx] = gae;
            }
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
            let advantages_mean = advantages.mean_all()?.to_scalar::<f32>()?;
            let advantage_diff = (advantages.clone() - advantages_mean as f64)?;

            let advantages_std_sqrt = advantage_diff
                .sqr()?
                .mean_all()?
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
                let actor_loss = (-1.0 * surrogate.mean_all()?)?;

                actor_loss
            }
            false => {
                let surrogate = (ratio * &advantages)?;
                (-1.0 * surrogate.mean_all()?)?
            }
        };

        let values = self.critic_network.forward(states)?;
        let values = values.squeeze(D::Minus1)?;

        let entropy_loss = entropy.mean_all()?;

        let final_actor_loss =
            (actor_loss.clone() - ((self.ent_coef as f64) * entropy_loss.clone()))?;

        // check for NaNs
        if !tensor_has_nan(&final_actor_loss)? {
            // clip the gradients and step the optimizers
            let actor_grad = &mut final_actor_loss.backward()?;
            let _actor_grad_norm = crate::tensor_operations::clip_gradients(actor_grad, 1.0)?;
            self.actor_optimizer.step(actor_grad)?;
        }

        let returns = returns.detach();
        let returns = {
            let returns_mean = returns.mean_all()?.to_scalar::<f32>()?;
            let returns_diff = (returns.clone() - returns_mean as f64)?;
            let returns_std_sqrt = returns_diff
                .sqr()?
                .mean_all()?
                .to_scalar::<f32>()?
                .sqrt()
                .max(1e-6);
            ((returns.clone() - returns_mean.clone() as f64)? / (returns_std_sqrt as f64))?
        };

        let critic_loss = loss::mse(&values, &returns)?;
        let final_critic_loss = ((self.vf_coef as f64) * critic_loss.clone())?;
        if !tensor_has_nan(&final_critic_loss)? {
            // clip the gradients and step the optimizers
            let critic_grad = &mut final_critic_loss.backward()?;
            let _critic_grad_norm = crate::tensor_operations::clip_gradients(critic_grad, 1.0)?;
            self.critic_optimizer.step(critic_grad)?;
        }

        let actor_loss_scalar = actor_loss.abs()?.mean_all()?.to_scalar::<f32>()? as f64;
        let critic_loss_scalar = critic_loss.abs()?.mean_all()?.to_scalar::<f32>()? as f64;
        Ok((actor_loss_scalar, critic_loss_scalar))
    }

    fn act_neurons(
        &mut self,
        observation: &candle_core::Tensor,
    ) -> Result<candle_core::Tensor, PPOError<AE, GE>> {
        self.actor_network
            .sample(observation)
            .map_err(PPOError::ActorError)
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

    fn act(&mut self, observation: &Tensor) -> Result<Tensor, Self::Error> {
        let neurons = self.act_neurons(observation)?;
        let actions = self.action_space.from_neurons(&neurons);
        Ok(actions)
    }

    fn learn(
        &mut self,
        env: &mut dyn VectorizedGym<Error = Self::GymError>,
        num_timesteps: usize,
    ) -> Result<(), PPOError<AE, GE>> {
        let mut elapsed_timesteps = 0;
        let mut next_states: Tensor;
        let mut rewards;

        next_states = env.reset().map_err(PPOError::GymError)?;

        while elapsed_timesteps < num_timesteps {
            while self.rollout_buffer.len() * env.num_envs() < self.batch_size {
                let states = next_states.clone();
                let action = self.act_neurons(&states)?;
                let actual_action = self.action_space.from_neurons(&action);

                let (log_probs, _) = self
                    .actor_network
                    .log_prob_and_entropy(&states, &action)
                    .map_err(PPOError::ActorError)?;

                let truncateds;
                let dones;
                VectorizedStepInfo {
                    states: next_states,
                    rewards,
                    dones,
                    truncateds,
                } = env.step(actual_action).map_err(PPOError::GymError)?;
                self.rollout_buffer.add(PPOExperience::new(
                    states.clone(),
                    next_states.clone(),
                    action,
                    rewards.clone(),
                    dones,
                    truncateds,
                    log_probs,
                    None,
                    None,
                ));
            }

            elapsed_timesteps += self.rollout_buffer.len() * env.num_envs();

            if let Some(scheduler) = &self.actor_lr_scheduler {
                let progress =
                    ((elapsed_timesteps as f64) / (num_timesteps as f64)).clamp(0.0, 1.0);

                let new_lr = scheduler.get_lr(progress);
                self.actor_optimizer.set_learning_rate(new_lr);
            }
            if let Some(scheduler) = &self.critic_lr_scheduler {
                let progress =
                    ((elapsed_timesteps as f64) / (num_timesteps as f64)).clamp(0.0, 1.0);
                let new_lr = scheduler.get_lr(progress);
                self.critic_optimizer.set_learning_rate(new_lr);
            }
            self.optimize()?;
        }
        Ok(())
    }
}

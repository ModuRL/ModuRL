use bon::bon;
use candle_core::{D, IndexOp, Tensor};
use candle_nn::{Optimizer, loss};
use std::marker::PhantomData;

use crate::{
    buffers::{
        experience,
        rollout_buffer::{RolloutBuffer, RolloutBufferError},
    },
    gym::{VectorizedGym, VectorizedStepInfo},
    lr_scheduler::LrScheduler,
    models::probabilistic_model::ProbabilisticActor,
    spaces,
    tensor_operations::{normalize_tensor, tensor_has_nan, torch_like_min},
};

#[derive(Debug)]
pub enum PPOError<AE, GE, SE>
where
    AE: std::fmt::Debug,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    ActorError(AE),
    GymError(GE),
    TensorError(candle_core::Error),
    SpaceError(SE),
}

impl<AE, GE, SE> From<candle_core::Error> for PPOError<AE, GE, SE>
where
    AE: std::fmt::Debug,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
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

struct PPOLoggingInfo<'a> {
    logger: &'a mut dyn PPOLogger,
    epoch: usize,
    timestep: usize,
}

impl<'a> PPOLoggingInfo<'a> {
    fn new(logger: &'a mut dyn PPOLogger) -> Self {
        Self {
            logger,
            epoch: 0,
            timestep: 0,
        }
    }
}

pub struct PPOLogEntry {
    pub actor_loss: Tensor,
    pub critic_loss: Tensor,
    pub entropy: Tensor,
    pub kl_divergence: Tensor,
    pub explained_variance: Tensor,
    pub rewards: Tensor,
    pub epoch: usize,
    pub timestep: usize,
    pub returns: Tensor,
    pub advantages: Tensor,
}

pub trait PPOLogger {
    fn log(&mut self, info: &PPOLogEntry);
}

#[derive(Clone)]
struct PPOLosses {
    actor_loss: Tensor,
    critic_loss: Tensor,
}

pub struct FakeOptimizer(());

impl Optimizer for FakeOptimizer {
    type Config = ();
    fn new(_vars: Vec<candle_core::Var>, _config: Self::Config) -> candle_core::Result<Self> {
        panic!("FakeOptimizer should never be used");
    }

    fn step(&mut self, _grads: &candle_core::backprop::GradStore) -> candle_core::Result<()> {
        panic!("FakeOptimizer should never be used");
    }

    fn learning_rate(&self) -> f64 {
        panic!("FakeOptimizer should never be used");
    }

    fn set_learning_rate(&mut self, _lr: f64) {
        panic!("FakeOptimizer should never be used");
    }
}

pub enum PPONetworkInfo<O1, E, O2 = FakeOptimizer> {
    Shared(SharedPPONetwork<O1, E>),
    Separate(SeparatePPONetwork<O1, O2, E>),
}

/// Shared network architecture for PPO
/// The output of the shared network is fed into both the actor and critic heads
pub struct SharedPPONetwork<O, E> {
    optimizer: O,
    shared_network: Box<dyn candle_core::Module>,
    critic_head: Box<dyn candle_core::Module>,
    actor_head: Box<dyn ProbabilisticActor<Error = E>>,
    lr_scheduler: Option<Box<dyn LrScheduler>>,
}

#[bon]
impl<O, E> SharedPPONetwork<O, E>
where
    O: Optimizer,
    E: std::fmt::Debug,
{
    #[builder]
    pub fn new(
        optimizer: O,
        shared_network: Box<dyn candle_core::Module>,
        critic_head: Box<dyn candle_core::Module>,
        actor_head: Box<dyn ProbabilisticActor<Error = E>>,
        lr_scheduler: Option<Box<dyn LrScheduler>>,
    ) -> Self {
        Self {
            optimizer,
            shared_network,
            critic_head,
            actor_head,
            lr_scheduler,
        }
    }
}

pub struct SeparatePPONetwork<O1, O2, E> {
    actor_optimizer: O1,
    critic_optimizer: O2,
    actor_network: Box<dyn ProbabilisticActor<Error = E>>,
    critic_network: Box<dyn candle_core::Module>,
    actor_lr_scheduler: Option<Box<dyn LrScheduler>>,
    critic_lr_scheduler: Option<Box<dyn LrScheduler>>,
    combined_loss: bool,
}

#[bon]
impl<O1, O2, E> SeparatePPONetwork<O1, O2, E>
where
    O1: Optimizer,
    O2: Optimizer,
    E: std::fmt::Debug,
{
    #[builder]
    pub fn new(
        actor_optimizer: O1,
        critic_optimizer: O2,
        actor_network: Box<dyn ProbabilisticActor<Error = E>>,
        critic_network: Box<dyn candle_core::Module>,
        actor_lr_scheduler: Option<Box<dyn LrScheduler>>,
        critic_lr_scheduler: Option<Box<dyn LrScheduler>>,
        #[builder(default = false)] combined_loss: bool,
    ) -> Self {
        Self {
            actor_optimizer,
            critic_optimizer,
            actor_network,
            critic_network,
            actor_lr_scheduler,
            critic_lr_scheduler,
            combined_loss,
        }
    }
}

pub struct PPOActor<'a, O1, O2, AE, GE, SE>
where
    O1: Optimizer,
    O2: Optimizer,
    AE: std::fmt::Debug,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    clipped: bool,
    gamma: f32,
    gae_lambda: f32,
    clip_range: f32,
    normalize_advantage: bool,
    normalize_returns: bool,
    vf_coef: f32,
    ent_coef: f32,
    action_space: Box<dyn spaces::Space<Error = SE>>,
    network_info: PPONetworkInfo<O1, AE, O2>,
    rollout_buffer: RolloutBuffer<PPOExperience>,
    mini_batch_size: usize,
    num_epochs: usize,
    batch_size: usize,
    logging_info: Option<PPOLoggingInfo<'a>>,
    gradient_clip: f32,
    _phantom: PhantomData<GE>,
}

#[bon]
impl<'a, O1, O2, AE, GE, SE> PPOActor<'a, O1, O2, AE, GE, SE>
where
    O1: Optimizer,
    O2: Optimizer,
    AE: std::fmt::Debug,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    #[builder]
    pub fn new(
        action_space: Box<dyn spaces::Space<Error = SE>>,
        network_info: PPONetworkInfo<O1, AE, O2>,
        #[builder(default = true)] clipped: bool,
        #[builder(default = 0.99)] gamma: f32,
        #[builder(default = 0.95)] gae_lambda: f32,
        #[builder(default = 0.2)] clip_range: f32,
        #[builder(default = true)] normalize_advantage: bool,
        #[builder(default = true)] normalize_returns: bool,
        #[builder(default = 0.5)] vf_coef: f32,
        #[builder(default = 0.01)] ent_coef: f32,
        #[builder(default = 1024)] batch_size: usize,
        #[builder(default = 128)] mini_batch_size: usize,
        #[builder(default = 1)] num_epochs: usize,
        #[builder(default = 0.5)] gradient_clip: f32,
        logging_info: Option<&'a mut dyn PPOLogger>,
        device: candle_core::Device,
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
            normalize_returns,
            vf_coef,
            ent_coef,
            network_info,
            action_space,
            rollout_buffer: RolloutBuffer::new(0, device), // placeholder, will set when we know num envs
            num_epochs,
            batch_size,
            mini_batch_size,
            logging_info: logging_info.map(|logger| PPOLoggingInfo::new(logger)),
            gradient_clip,
            _phantom: PhantomData,
        }
    }
}

impl<'a, O1, O2, AE, GE, SE> PPOActor<'a, O1, O2, AE, GE, SE>
where
    O1: Optimizer,
    O2: Optimizer,
    AE: std::fmt::Debug,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    fn optimize(&mut self) -> Result<(), PPOError<AE, GE, SE>> {
        self.add_advantages_and_returns()?;
        for epoch in 0..self.num_epochs {
            if let Some(logging_info) = &mut self.logging_info {
                logging_info.epoch = epoch;
            }

            let batches = self.rollout_buffer.get_all_shuffled();
            let batches = match batches {
                Ok(b) => b,
                Err(e) => match e {
                    RolloutBufferError::TensorError(te) => return Err(PPOError::TensorError(te)),
                    RolloutBufferError::ExperienceError(ee) => {
                        return Err(PPOError::TensorError(ee));
                    }
                },
            };

            for batch in batches.iter() {
                let elements = batch.get_elements();
                let mut actions = elements[PPOElement::Action as usize].clone();
                let mut states = elements[PPOElement::State as usize].clone();
                let mut batch_old_log_probs = elements[PPOElement::LogProb as usize].clone();
                let mut advantages = elements[PPOElement::Advantage as usize].clone();
                let mut returns = elements[PPOElement::Return as usize].clone();
                let mut rewards = elements[PPOElement::Reward as usize].clone();
                // flatten the env dimension
                actions = actions.flatten(0, 1)?;
                states = states.flatten(0, 1)?;
                batch_old_log_probs = batch_old_log_probs.flatten(0, 1)?;
                advantages = advantages.flatten(0, 1)?;
                returns = returns.flatten(0, 1)?;
                rewards = rewards.flatten(0, 1)?;

                // Compute the loss and backpropagate
                let ppo_losses = self.compute_loss(
                    &states,
                    &actions,
                    batch_old_log_probs.detach(),
                    advantages,
                    returns,
                    rewards,
                )?;

                self.backpropagate_loss(ppo_losses.clone())?;
            }
        }

        self.rollout_buffer.clear();
        Ok(())
    }

    fn add_advantages_and_returns(&mut self) -> Result<(), PPOError<AE, GE, SE>> {
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

        let states = Tensor::stack(&states, 0)?; // shape [batch_size, env_count, ...]
        let (batch_size, env_count) = (states.dims()[0], states.dims()[1]);
        // Flatten temporally to feed into networks
        let states = states.flatten(0, 1)?; // shape [batch_size * env_count, ...]

        let latent_states = match self.network_info {
            PPONetworkInfo::Shared(ref mut shared_info) => {
                shared_info.shared_network.forward(&states)?
            }
            PPONetworkInfo::Separate(ref mut _separate_info) => states,
        };

        let values_tensor = self.critic_network_forward(&latent_states)?.detach();

        // unflatten back to [batch_size, env_count, ...]
        let values_tensor = values_tensor.reshape((batch_size, env_count, ()))?; // shape [batch_size, env_count, ...]

        // the last step for every env is needed for bootstrapping
        let next_states_tensor = Tensor::stack(&next_states, 0)?;
        let bootstrapped_states = next_states_tensor.i(next_states_tensor.shape().dims()[0] - 1)?; // shape [env_count, ...]
        let latent_bootstrapped_states = match self.network_info {
            PPONetworkInfo::Shared(ref mut shared_info) => {
                shared_info.shared_network.forward(&bootstrapped_states)?
            }
            PPONetworkInfo::Separate(ref mut _separate_info) => bootstrapped_states,
        };

        let bootstrapped_values = self
            .critic_network_forward(&latent_bootstrapped_states)?
            .flatten_all()?
            .detach(); // shape [env_count]

        let device = values_tensor.device();

        let dones_shape = &[dones.len(), dones[0].len()];
        let rewards_tensor = candle_core::Tensor::stack(&rewards, 0)?;
        let dones_tensor = candle_core::Tensor::from_vec(
            dones.into_iter().flatten().collect(),
            dones_shape,
            device,
        )?;

        let advantages = self
            .compute_gae(
                &rewards_tensor,
                &values_tensor,
                &dones_tensor,
                &bootstrapped_values,
            )?
            .detach();

        let values_tensor = values_tensor.squeeze(D::Minus1)?;

        let returns = (&values_tensor + &advantages)?;
        let returns = returns.clamp(-1e5, 1e5)?;

        let experiences = self.rollout_buffer.get_raw_mut();

        for (i, experience) in experiences.iter_mut().enumerate() {
            // The detaches here should be redundant but just to be safe
            experience.advantages = Some(advantages.i(i)?.clone().detach());
            experience.this_returns = Some(returns.i(i)?.clone().detach());
        }

        Ok(())
    }

    fn compute_gae(
        &self,
        rewards: &candle_core::Tensor,
        values: &candle_core::Tensor,
        dones: &candle_core::Tensor,
        bootstrapped_values: &candle_core::Tensor,
    ) -> Result<candle_core::Tensor, candle_core::Error> {
        let device = rewards.device();
        let gamma_tensor = Tensor::new(self.gamma, &device)?.to_dtype(values.dtype())?;
        let gae_lambda_tensor = Tensor::new(self.gae_lambda, &device)?.to_dtype(values.dtype())?;

        let values = values.squeeze(D::Minus1)?;
        let mut advantages = vec![];

        for env_idx in 0..rewards.shape().dims()[1] {
            let env_rewards = rewards.i((.., env_idx))?.detach();
            let env_dones = dones.i((.., env_idx))?.detach();
            let env_values = values.i((.., env_idx))?.detach();

            let last_done = env_dones.i(env_dones.shape().dims()[0] - 1)?;
            let mut next_value = ((1.0 - last_done)? * bootstrapped_values.i(env_idx)?.detach())?;
            let mut env_advantages = vec![];

            let mut gae = Tensor::new(0.0f32, device)?;
            // Compute GAE backwards through the trajectory
            for i in (0..env_rewards.shape().dims()[0]).rev() {
                let non_terminal = (1.0 - env_dones.i(i)?)?; // 0 if done, 1 if not done

                // TD error: δ = r + γ * V(s') - V(s)
                let delta = (env_rewards.i(i)?
                    + next_value * non_terminal.clone() * gamma_tensor.clone()
                    - env_values.i(i)?)?;

                // GAE: A = δ + γ * λ * next_gae * (1 - done)
                gae = (delta
                    + gamma_tensor.clone() * gae_lambda_tensor.clone() * gae * non_terminal)?;
                env_advantages.push(gae.clone());
                next_value = env_values.i(i)?;
            }

            // Reverse because our loop went backwards
            let env_advantages_tensor = Tensor::stack(
                &env_advantages.into_iter().rev().collect::<Vec<Tensor>>(),
                0,
            )?;
            advantages.push(env_advantages_tensor);
        }

        let advantages_tensor = Tensor::stack(&advantages, 1)?; // shape [time_steps, env_count]

        Ok(advantages_tensor)
    }

    /// Forward pass through the critic network
    /// Expects states as latent if shared network is used
    fn critic_network_forward(
        &mut self,
        states: &candle_core::Tensor,
    ) -> Result<candle_core::Tensor, PPOError<AE, GE, SE>> {
        match self.network_info {
            PPONetworkInfo::Shared(ref mut shared_info) => {
                let values = shared_info.critic_head.forward(states)?;
                Ok(values)
            }
            PPONetworkInfo::Separate(ref mut separate_info) => {
                let values = separate_info.critic_network.forward(states)?;
                Ok(values)
            }
        }
    }

    /// Log prob and entropy from the actor network
    /// Expects states as latent if shared network is used
    fn actor_network_log_prob_and_entropy(
        &mut self,
        states: &candle_core::Tensor,
        actions: &candle_core::Tensor,
    ) -> Result<(candle_core::Tensor, candle_core::Tensor), PPOError<AE, GE, SE>> {
        match self.network_info {
            PPONetworkInfo::Shared(ref mut shared_info) => {
                let (log_probs, entropy) = shared_info
                    .actor_head
                    .log_prob_and_entropy(states, actions)
                    .map_err(PPOError::ActorError)?;
                Ok((log_probs, entropy))
            }
            PPONetworkInfo::Separate(ref mut separate_info) => {
                let (log_probs, entropy) = separate_info
                    .actor_network
                    .log_prob_and_entropy(states, actions)
                    .map_err(PPOError::ActorError)?;
                Ok((log_probs, entropy))
            }
        }
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
        rewards: candle_core::Tensor,
    ) -> Result<PPOLosses, PPOError<AE, GE, SE>> {
        let advantages = if self.normalize_advantage {
            normalize_tensor(&advantages)?
        } else {
            advantages
        }
        .detach();

        // if the networks are shared, we need to extract the latent state
        // if it's seperate we just say this is the state as is
        let latent_state = match self.network_info {
            PPONetworkInfo::Shared(ref mut shared_info) => {
                shared_info.shared_network.forward(states)?
            }
            PPONetworkInfo::Separate(ref mut _separate_info) => states.clone(),
        };

        let (log_probs, entropy) =
            self.actor_network_log_prob_and_entropy(&latent_state, actions)?;

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
                let surrogate = (ratio.clone() * &advantages)?;
                (-1.0 * surrogate.mean_all()?)?
            }
        };

        let values = self.critic_network_forward(&latent_state)?;
        let values = values.squeeze(D::Minus1)?;

        let entropy_loss = entropy.mean_all()?;

        let final_actor_loss =
            (actor_loss.clone() - ((self.ent_coef as f64) * entropy_loss.clone()))?;

        let returns = returns.detach();
        let returns = if self.normalize_returns {
            normalize_tensor(&returns)?
        } else {
            returns
        }
        .detach();

        let critic_loss = loss::mse(&values, &returns)?;
        let final_critic_loss = ((self.vf_coef as f64) * critic_loss.clone())?;

        if let Some(logging_info) = &mut self.logging_info {
            let explained_variance = {
                let var_y = returns.var(D::Minus1)?;
                let diff = (&returns - &values)?;
                let var_diff = diff.var(D::Minus1)?;

                1.0 + (-1.0 * var_diff / (var_y + 1e-8)?)?
            }?;

            let log_entry = PPOLogEntry {
                actor_loss: final_actor_loss.clone(),
                critic_loss: final_critic_loss.clone(),
                entropy: entropy_loss,
                kl_divergence: ratio,
                explained_variance,
                rewards: rewards,
                epoch: logging_info.epoch,
                timestep: logging_info.timestep,
                returns,
                advantages,
            };
            logging_info.logger.log(&log_entry);
        }

        Ok(PPOLosses {
            actor_loss: final_actor_loss,
            critic_loss: final_critic_loss,
        })
    }

    fn backpropagate_loss(&mut self, losses: PPOLosses) -> Result<(), PPOError<AE, GE, SE>> {
        let actor_loss = losses.actor_loss;
        let critic_loss = losses.critic_loss;

        match self.network_info {
            PPONetworkInfo::Shared(ref mut shared_info) => {
                let total_loss = (&actor_loss + &critic_loss)?;
                if !tensor_has_nan(&total_loss)? {
                    let total_grad = &mut total_loss.backward()?;
                    let _total_grad_norm =
                        crate::tensor_operations::clip_gradients(total_grad, self.gradient_clip)?;
                    shared_info.optimizer.step(total_grad)?;
                }
            }
            PPONetworkInfo::Separate(ref mut separate_info) => match separate_info.combined_loss {
                true => {
                    let total_loss = (&actor_loss + &critic_loss)?;
                    if !tensor_has_nan(&total_loss)? {
                        let total_grad = &mut total_loss.backward()?;
                        let _total_grad_norm = crate::tensor_operations::clip_gradients(
                            total_grad,
                            self.gradient_clip,
                        )?;
                        separate_info.actor_optimizer.step(total_grad)?;
                        separate_info.critic_optimizer.step(total_grad)?;
                    }
                }
                false => {
                    if !tensor_has_nan(&actor_loss)? {
                        let actor_grad = &mut actor_loss.backward()?;
                        let _actor_grad_norm = crate::tensor_operations::clip_gradients(
                            actor_grad,
                            self.gradient_clip,
                        )?;
                        separate_info.actor_optimizer.step(actor_grad)?;
                    }

                    if !tensor_has_nan(&critic_loss)? {
                        let critic_grad = &mut critic_loss.backward()?;
                        let _critic_grad_norm = crate::tensor_operations::clip_gradients(
                            critic_grad,
                            self.gradient_clip,
                        )?;
                        separate_info.critic_optimizer.step(critic_grad)?;
                    }
                }
            },
        }

        Ok(())
    }

    /// Expects states as latent
    fn act_neurons(
        &mut self,
        states: &candle_core::Tensor,
    ) -> Result<candle_core::Tensor, PPOError<AE, GE, SE>> {
        let network = match self.network_info {
            PPONetworkInfo::Shared(ref mut shared_info) => &shared_info.actor_head,
            PPONetworkInfo::Separate(ref mut separate_info) => &separate_info.actor_network,
        };

        network.sample(&states).map_err(PPOError::ActorError)
    }

    fn update_learning_rates(&mut self, progress: f64) {
        match self.network_info {
            PPONetworkInfo::Shared(ref mut shared_info) => {
                if let Some(scheduler) = &shared_info.lr_scheduler {
                    let new_lr = scheduler.get_lr(progress);
                    shared_info.optimizer.set_learning_rate(new_lr);
                }
            }
            PPONetworkInfo::Separate(ref mut separate_info) => {
                if let Some(scheduler) = &separate_info.actor_lr_scheduler {
                    let new_lr = scheduler.get_lr(progress);
                    separate_info.actor_optimizer.set_learning_rate(new_lr);
                }
                if let Some(scheduler) = &separate_info.critic_lr_scheduler {
                    let new_lr = scheduler.get_lr(progress);
                    separate_info.critic_optimizer.set_learning_rate(new_lr);
                }
            }
        }
    }
}

impl<'a, O1, O2, AE, GE, SE> Actor for PPOActor<'a, O1, O2, AE, GE, SE>
where
    O1: Optimizer,
    O2: Optimizer,
    AE: std::fmt::Debug,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    type Error = PPOError<AE, GE, SE>;
    type GymError = GE;
    type SpaceError = SE;

    fn act(&mut self, observation: &Tensor) -> Result<Tensor, Self::Error> {
        let latent_states = match self.network_info {
            PPONetworkInfo::Shared(ref mut shared_info) => {
                shared_info.shared_network.forward(observation)?
            }
            PPONetworkInfo::Separate(ref mut _separate_info) => observation.clone(),
        };
        let neurons = self.act_neurons(&latent_states)?;
        let actions = self
            .action_space
            .from_neurons(&neurons)
            .map_err(PPOError::SpaceError)?;
        Ok(actions)
    }

    fn learn(
        &mut self,
        env: &mut dyn VectorizedGym<Error = Self::GymError, SpaceError = Self::SpaceError>,
        num_timesteps: usize,
    ) -> Result<(), PPOError<AE, GE, SE>> {
        let mut elapsed_timesteps = 0;
        let mut next_states: Tensor;
        let mut rewards;

        next_states = env.reset().map_err(PPOError::GymError)?;
        self.rollout_buffer = RolloutBuffer::new(
            self.mini_batch_size / env.num_envs(),
            next_states.device().clone(),
        );

        while elapsed_timesteps < num_timesteps {
            while self.rollout_buffer.len() * env.num_envs() < self.batch_size {
                let states = next_states.clone();
                let latent_states = match self.network_info {
                    PPONetworkInfo::Shared(ref mut shared_info) => {
                        shared_info.shared_network.forward(&states)?
                    }
                    PPONetworkInfo::Separate(ref mut _separate_info) => states.clone(),
                };

                let action = self.act_neurons(&latent_states)?;
                let actual_action = self
                    .action_space
                    .from_neurons(&action)
                    .map_err(PPOError::SpaceError)?;

                let (log_probs, _) =
                    self.actor_network_log_prob_and_entropy(&latent_states, &action)?;

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

            self.update_learning_rates((elapsed_timesteps as f64) / (num_timesteps as f64));

            if let Some(logging_info) = &mut self.logging_info {
                logging_info.timestep = elapsed_timesteps;
            }
            self.optimize()?;
        }
        Ok(())
    }
}

#[cfg(any(feature = "cuda", feature = "metal"))]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        distributions::CategoricalDistribution,
        gym::{Gym, StepInfo, VectorizedGymWrapper},
        models::{MLP, probabilistic_model::MLPProbabilisticActor},
        spaces::Discrete,
        tensor_operations::tanh,
    };
    use candle_core::Device;
    use candle_nn::{VarBuilder, VarMap};
    use candle_optimisers::adam::{Adam, ParamsAdam};

    // Simple dummy environment for testing
    struct DummyEnv {
        step_count: usize,
        device: Device,
    }

    impl DummyEnv {
        fn new(device: Device) -> Self {
            Self {
                step_count: 0,
                device,
            }
        }
    }

    impl Gym for DummyEnv {
        type Error = candle_core::Error;
        type SpaceError = candle_core::Error;

        fn step(&mut self, _action: Tensor) -> Result<StepInfo, Self::Error> {
            self.step_count += 1;
            let done = self.step_count >= 5;
            Ok(StepInfo {
                state: Tensor::rand(0.0f32, 1.0, &[4], &self.device)?,
                reward: self.step_count as f32,
                done,
                truncated: false,
            })
        }

        fn reset(&mut self) -> Result<Tensor, Self::Error> {
            self.step_count = 0;
            Tensor::rand(0.0f32, 1.0, &[4], &self.device)
        }

        fn observation_space(&self) -> Box<dyn crate::spaces::Space<Error = Self::SpaceError>> {
            Box::new(crate::spaces::BoxSpace::new(
                Tensor::full(0.0f32, &[4], &self.device).unwrap(),
                Tensor::full(1.0f32, &[4], &self.device).unwrap(),
            ))
        }

        fn action_space(&self) -> Box<dyn crate::spaces::Space<Error = Self::SpaceError>> {
            Box::new(Discrete::new(2))
        }
    }

    #[test]
    fn test_ppo_determinism() {
        #[cfg(feature = "cuda")]
        let device = Device::new_cuda(0).unwrap();
        #[cfg(feature = "metal")]
        let device = Device::new_metal(0).unwrap();

        const SAMPLE_COUNT: usize = 25;
        let mut last_actions: Option<Vec<Tensor>> = None;

        for i in 0..5 {
            println!("PPO Determinism Test Iteration {}", i + 1);

            // Reset seed before each iteration
            device.set_seed(42).unwrap();

            // Create fresh PPO actor
            let mut envs = vec![];
            for _ in 0..8 {
                envs.push(DummyEnv::new(device.clone()));
            }
            let mut vec_env: VectorizedGymWrapper<DummyEnv> = envs.into();

            let observation_space = vec_env.observation_space();
            let action_space = vec_env.action_space();

            // Actor network
            let actor_var_map = VarMap::new();
            let actor_vb =
                VarBuilder::from_varmap(&actor_var_map, candle_core::DType::F32, &device);
            let actor_network = MLP::builder()
                .input_size(observation_space.shape().iter().product())
                .output_size(action_space.shape().iter().product::<usize>())
                .vb(actor_vb)
                .activation(Box::new(tanh))
                .hidden_layer_sizes(vec![8, 8])
                .name("actor_network".to_string())
                .build()
                .unwrap();

            let mut config = ParamsAdam::default();
            config.lr = 3e-4;
            let actor_optimizer = Adam::new(actor_var_map.all_vars(), config.clone()).unwrap();

            // Critic network
            let critic_var_map = VarMap::new();
            let critic_vb =
                VarBuilder::from_varmap(&critic_var_map, candle_core::DType::F32, &device);
            let critic_network = MLP::builder()
                .input_size(observation_space.shape().iter().product())
                .output_size(1)
                .vb(critic_vb)
                .activation(Box::new(tanh))
                .hidden_layer_sizes(vec![8, 8])
                .name("critic_network".to_string())
                .build()
                .unwrap();

            let critic_optimizer = Adam::new(critic_var_map.all_vars(), config.clone()).unwrap();

            // Create PPO actor
            let mut actor = PPOActor::builder()
                .action_space(action_space)
                .actor_network(Box::new(
                    MLPProbabilisticActor::<CategoricalDistribution>::new(actor_network),
                ))
                .critic_network(Box::new(critic_network))
                .critic_optimizer(critic_optimizer)
                .actor_optimizer(actor_optimizer)
                .batch_size(2048)
                .mini_batch_size(64)
                .ent_coef(0.01)
                .vf_coef(0.5)
                .clip_range(0.2)
                .gae_lambda(0.95)
                .num_epochs(10)
                .device(device.clone())
                .build();

            // train for some timesteps
            actor
                .learn(&mut vec_env, 10000)
                .expect("PPO learning failed");

            let mut actions = vec![];
            let mut state = vec_env.reset().unwrap();
            for _ in 0..SAMPLE_COUNT {
                let action_neurons = actor.act_neurons(&state).unwrap();
                actions.push(action_neurons.clone());
                let action = actor.action_space.from_neurons(&action_neurons).unwrap();
                let step_info = vec_env.step(action).unwrap();
                state = step_info.states;
            }

            // Compare with previous iteration
            if let Some(last_actions) = &last_actions {
                for (last_action, current_action) in last_actions.iter().zip(actions.iter()) {
                    let max_diff = last_action
                        .sub(current_action)
                        .unwrap()
                        .abs()
                        .unwrap()
                        .max_all()
                        .unwrap()
                        .to_scalar::<f32>()
                        .unwrap();

                    assert!(
                        max_diff == 0.0,
                        "PPO actions differ at iteration {} by {}",
                        i,
                        max_diff
                    );
                }
            }
            last_actions = Some(actions);
        }
    }
}

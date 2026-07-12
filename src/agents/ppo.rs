use bon::bon;
use candle_core::{D, IndexOp, Tensor};
use candle_nn::{Optimizer, loss};
use std::marker::PhantomData;

use crate::{
    buffers::{experience, rollout_buffer::RolloutBuffer},
    gym::{VectorizedGym, VectorizedStepInfo},
    models::probabilistic_model::ProbabilisticPolicy,
    parameter_schedule::{ConstantSchedule, ParameterSchedule},
    spaces,
    tensor_operations::{normalize_tensor, tensor_has_nan, torch_like_max, torch_like_min},
};

#[derive(Debug)]
pub enum PPOError<AE, GE, SE>
where
    AE: std::fmt::Debug,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    PolicyError(AE),
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

use super::Agent;

#[derive(Debug, Clone)]
struct PPOExperience {
    states: Tensor,
    next_states: Tensor,
    actions: Tensor,
    rewards: Tensor,
    next_dones: Vec<bool>,
    truncateds: Vec<bool>,
    log_probs: Tensor,
    advantages: Option<Tensor>,
    this_returns: Option<Tensor>,
    /// V(s) under the rollout-collection network, for PPO2-style value-loss clipping.
    old_values: Option<Tensor>,
}

#[bon]
impl PPOExperience {
    #[builder]
    pub fn new(
        states: Tensor,
        next_states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_dones: Vec<bool>,
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
            next_dones,
            truncateds,
            log_probs,
            advantages,
            this_returns,
            old_values: None,
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
                self.next_dones
                    .iter()
                    .map(|&d| if d { 1.0 } else { 0.0 })
                    .collect::<Vec<f32>>(),
                &[self.next_dones.len()],
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
    NextDone = 4,
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
    actor_head: Box<dyn ProbabilisticPolicy<Error = E>>,
    lr_scheduler: Option<Box<dyn ParameterSchedule>>,
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
        actor_head: Box<dyn ProbabilisticPolicy<Error = E>>,
        lr_scheduler: Option<Box<dyn ParameterSchedule>>,
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
    actor_network: Box<dyn ProbabilisticPolicy<Error = E>>,
    critic_network: Box<dyn candle_core::Module>,
    actor_lr_scheduler: Option<Box<dyn ParameterSchedule>>,
    critic_lr_scheduler: Option<Box<dyn ParameterSchedule>>,
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
        actor_network: Box<dyn ProbabilisticPolicy<Error = E>>,
        critic_network: Box<dyn candle_core::Module>,
        actor_lr_scheduler: Option<Box<dyn ParameterSchedule>>,
        critic_lr_scheduler: Option<Box<dyn ParameterSchedule>>,
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

pub struct PPOAgent<'a, O1, O2, AE, GE, SE>
where
    O1: Optimizer,
    O2: Optimizer,
    AE: std::fmt::Debug,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    clipped: bool,
    clip_value_loss: bool,
    gamma: f32,
    gae_lambda: f32,
    clip_range: Box<dyn ParameterSchedule>,
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
    current_states: Option<Tensor>,
    _phantom: PhantomData<GE>,
}

#[bon]
impl<'a, O1, O2, AE, GE, SE> PPOAgent<'a, O1, O2, AE, GE, SE>
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
        // PPO2/CleanRL-style value-loss clipping. Disabled by default because
        // its usefulness is task- and reward-scale-dependent.
        // research has shown it's often harmful
        #[builder(default = false)] clip_value_loss: bool,
        #[builder(default = 0.99)] gamma: f32,
        #[builder(default = 0.95)] gae_lambda: f32,
        #[builder(default = Box::new(ConstantSchedule::new(0.1)))] clip_range: Box<
            dyn ParameterSchedule,
        >,
        #[builder(default = true)] normalize_advantage: bool,
        #[builder(default = false)] normalize_returns: bool,
        #[builder(default = 0.5)] vf_coef: f32,
        #[builder(default = 0.01)] ent_coef: f32,
        #[builder(default = 1024)] batch_size: usize,
        #[builder(default = 128)] mini_batch_size: usize,
        #[builder(default = 4)] num_epochs: usize,
        #[builder(default = 0.5)] gradient_clip: f32,
        logging_info: Option<&'a mut dyn PPOLogger>,
        device: candle_core::Device,
    ) -> Self {
        if batch_size > 0 {
            assert!(batch_size > 0);
        }

        Self {
            clipped,
            clip_value_loss,
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
            logging_info: logging_info.map(PPOLoggingInfo::new),
            gradient_clip,
            current_states: None,
            _phantom: PhantomData,
        }
    }
}

impl<'a, O1, O2, AE, GE, SE> PPOAgent<'a, O1, O2, AE, GE, SE>
where
    O1: Optimizer,
    O2: Optimizer,
    AE: std::fmt::Debug,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    fn optimize(&mut self, progress: f64) -> Result<(), PPOError<AE, GE, SE>> {
        self.add_advantages_and_returns()?;

        let experiences = self.rollout_buffer.get_raw();
        let mut all_states = vec![];
        let mut all_actions = vec![];
        let mut all_log_probs = vec![];
        let mut all_advantages = vec![];
        let mut all_returns = vec![];
        let mut all_rewards = vec![];
        let mut all_old_values = vec![];

        for experience in experiences {
            all_states.push(experience.states.clone());
            all_actions.push(experience.actions.clone());
            all_log_probs.push(experience.log_probs.clone());
            all_advantages.push(experience.advantages.clone().expect("Advantage not set"));
            all_returns.push(experience.this_returns.clone().expect("Return not set"));
            all_rewards.push(experience.rewards.clone());
            all_old_values.push(experience.old_values.clone().expect("Old value not set"));
        }

        let all_states = Tensor::stack(&all_states, 0)?.flatten(0, 1)?;
        let all_actions = Tensor::stack(&all_actions, 0)?.flatten(0, 1)?;
        let all_log_probs = Tensor::stack(&all_log_probs, 0)?.flatten(0, 1)?;
        let all_advantages = Tensor::stack(&all_advantages, 0)?.flatten(0, 1)?;
        let all_returns = Tensor::stack(&all_returns, 0)?.flatten(0, 1)?;
        let all_rewards = Tensor::stack(&all_rewards, 0)?.flatten(0, 1)?;
        let all_old_values = Tensor::stack(&all_old_values, 0)?.flatten(0, 1)?;

        let total_samples = all_states.dims()[0];
        let device = all_states.device();
        let clip_range = self.clip_range.value(progress) as f32;

        for epoch in 0..self.num_epochs {
            if let Some(logging_info) = &mut self.logging_info {
                logging_info.epoch = epoch;
            }

            let mut indices: Vec<u32> = (0..total_samples as u32).collect();
            crate::tensor_operations::fisher_yates_shuffle(&mut indices, device);
            let indices_tensor = Tensor::from_vec(indices, &[total_samples], device)?;

            for start in (0..total_samples).step_by(self.mini_batch_size) {
                let end = (start + self.mini_batch_size).min(total_samples);
                let batch_size = end - start;

                let batch_indices = indices_tensor.narrow(0, start, batch_size)?;

                let batch_states = all_states.index_select(&batch_indices, 0)?;
                let batch_actions = all_actions.index_select(&batch_indices, 0)?;
                let batch_log_probs = all_log_probs.index_select(&batch_indices, 0)?;
                let batch_advantages = all_advantages.index_select(&batch_indices, 0)?;
                let batch_returns = all_returns.index_select(&batch_indices, 0)?;
                let batch_rewards = all_rewards.index_select(&batch_indices, 0)?;
                let batch_old_values = all_old_values.index_select(&batch_indices, 0)?;

                let ppo_losses = self.compute_loss(
                    &batch_states,
                    &batch_actions,
                    batch_log_probs.detach(),
                    batch_advantages,
                    batch_returns,
                    batch_rewards,
                    batch_old_values.detach(),
                    clip_range,
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
        let mut next_dones = vec![];
        let mut next_truncateds = vec![];
        let mut states = vec![];
        let mut next_states = vec![];
        // For splitting the envs
        let mut log_probs = vec![];
        let mut actions = vec![];

        for experience in experiences.iter() {
            rewards.push(experience.rewards.clone());
            next_dones.push(
                experience
                    .next_dones
                    .iter()
                    .map(|&d| if d { 1.0 } else { 0.0 })
                    .collect::<Vec<f32>>(),
            );
            next_truncateds.push(
                experience
                    .truncateds
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
        // Flatten temporarily to feed into networks
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

        let next_dones_shape = &[next_dones.len(), next_dones[0].len()];
        let next_truncateds_shape = &[next_truncateds.len(), next_truncateds[0].len()];
        let rewards_tensor = candle_core::Tensor::stack(&rewards, 0)?;
        let next_dones_tensor = candle_core::Tensor::from_vec(
            next_dones.into_iter().flatten().collect(),
            next_dones_shape,
            device,
        )?;
        let next_truncateds_tensor = candle_core::Tensor::from_vec(
            next_truncateds.into_iter().flatten().collect(),
            next_truncateds_shape,
            device,
        )?;

        let advantages = self
            .compute_gae(
                &rewards_tensor,
                &values_tensor,
                &next_dones_tensor,
                &next_truncateds_tensor,
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
            // values_tensor comes from the pre-update network, i.e. the
            // collection-time values PPO2 clips the new values against.
            experience.old_values = Some(values_tensor.i(i)?.clone().detach());
        }

        Ok(())
    }

    fn compute_gae(
        &self,
        rewards: &candle_core::Tensor,
        values: &candle_core::Tensor,
        next_dones: &candle_core::Tensor,
        next_truncateds: &candle_core::Tensor,
        bootstrapped_values: &candle_core::Tensor,
    ) -> Result<candle_core::Tensor, candle_core::Error> {
        let device = rewards.device();
        let gamma_tensor = Tensor::new(self.gamma, device)?.to_dtype(values.dtype())?;
        let gae_lambda_tensor = Tensor::new(self.gae_lambda, device)?.to_dtype(values.dtype())?;

        let values = values.squeeze(D::Minus1)?;
        let mut advantages = vec![];

        for env_idx in 0..rewards.shape().dims()[1] {
            let env_rewards = rewards.i((.., env_idx))?.detach();
            let env_next_dones = next_dones.i((.., env_idx))?.detach();
            let env_next_truncateds = next_truncateds.i((.., env_idx))?.detach();
            let env_values = values.i((.., env_idx))?.detach();
            let mut env_advantages = vec![];

            let mut next_value = bootstrapped_values.i(env_idx)?.detach();
            let mut gae = Tensor::new(0.0f32, device)?;
            // Compute GAE backwards through the trajectory
            for i in (0..env_rewards.shape().dims()[0]).rev() {
                let same_episode =
                    ((1.0 - env_next_dones.i(i)?)? * (1.0 - env_next_truncateds.i(i)?)?)?;

                // TD error: δ = r + γ * V(s') - V(s)
                let delta = (env_rewards.i(i)?
                    + next_value.clone() * same_episode.clone() * gamma_tensor.clone()
                    - env_values.i(i)?)?;

                // GAE: A = δ + γ * λ * next_gae * (1 - next_done)
                gae = (delta
                    + gamma_tensor.clone() * gae_lambda_tensor.clone() * gae * same_episode)?;
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
                    .map_err(PPOError::PolicyError)?;
                Ok((log_probs, entropy))
            }
            PPONetworkInfo::Separate(ref mut separate_info) => {
                let (log_probs, entropy) = separate_info
                    .actor_network
                    .log_prob_and_entropy(states, actions)
                    .map_err(PPOError::PolicyError)?;
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
        old_values: candle_core::Tensor,
        clip_range: f32,
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

        let log_ratio = (&log_probs - &old_log_probs)?;
        let ratio = log_ratio.exp()?;
        let approx_kl = ((&ratio - 1.0)? - &log_ratio)?;

        let actor_loss = match self.clipped {
            true => {
                let clipped_ratio = ratio.clamp(1.0 - clip_range, 1.0 + clip_range)?;

                let surrogate1 = (ratio.clone() * advantages.clone())?;
                let surrogate2 = (clipped_ratio.clone() * advantages.clone())?;
                let surrogate = torch_like_min(&surrogate1, &surrogate2)?;

                (-1.0 * surrogate.mean_all()?)?
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

        let critic_loss = if self.clip_value_loss {
            // PPO2/CleanRL value-loss clipping: bound how far the value estimate
            // may move from its rollout-time value within one update.
            let clip_range = clip_range as f64;
            let value_delta = (&values - &old_values)?.clamp(-clip_range, clip_range)?;
            let values_clipped = (&old_values + value_delta)?;
            let loss_unclipped = (&values - &returns)?.sqr()?;
            let loss_clipped = (&values_clipped - &returns)?.sqr()?;
            torch_like_max(&loss_unclipped, &loss_clipped)?.mean_all()?
        } else {
            loss::mse(&values, &returns)?
        };
        // The 0.5 factor is from the original PPO paper
        let final_critic_loss = (((self.vf_coef * 0.5) as f64) * critic_loss.clone())?;

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
                kl_divergence: approx_kl,
                explained_variance,
                rewards,
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

        network.sample(states).map_err(PPOError::PolicyError)
    }

    fn update_learning_rates(&mut self, progress: f64) {
        match self.network_info {
            PPONetworkInfo::Shared(ref mut shared_info) => {
                if let Some(lr_scheduler) = &shared_info.lr_scheduler {
                    let new_lr = lr_scheduler.value(progress);
                    shared_info.optimizer.set_learning_rate(new_lr);
                }
            }
            PPONetworkInfo::Separate(ref mut separate_info) => {
                if let Some(actor_lr_scheduler) = &separate_info.actor_lr_scheduler {
                    let new_lr = actor_lr_scheduler.value(progress);
                    separate_info.actor_optimizer.set_learning_rate(new_lr);
                }
                if let Some(critic_lr_scheduler) = &separate_info.critic_lr_scheduler {
                    let new_lr = critic_lr_scheduler.value(progress);
                    separate_info.critic_optimizer.set_learning_rate(new_lr);
                }
            }
        }
    }

    pub fn set_learning_rate(&mut self, lr: f64) {
        match self.network_info {
            PPONetworkInfo::Shared(ref mut shared_info) => {
                shared_info.optimizer.set_learning_rate(lr);
            }
            PPONetworkInfo::Separate(ref mut separate_info) => {
                separate_info.actor_optimizer.set_learning_rate(lr);
                separate_info.critic_optimizer.set_learning_rate(lr);
            }
        }
    }

    pub fn reset_current_states(&mut self) {
        self.current_states = None;
    }
}

impl<'a, O1, O2, AE, GE, SE> Agent for PPOAgent<'a, O1, O2, AE, GE, SE>
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
            .tensor_from_neurons(&neurons)
            .map_err(PPOError::SpaceError)?;
        Ok(actions)
    }

    fn learn(
        &mut self,
        env: &mut dyn VectorizedGym<Error = Self::GymError, SpaceError = Self::SpaceError>,
        num_timesteps: usize,
    ) -> Result<(), PPOError<AE, GE, SE>> {
        let mut elapsed_timesteps = 0;
        let mut next_states = if let Some(states) = self.current_states.take() {
            states
        } else {
            env.reset().map_err(PPOError::GymError)?
        };
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
                    .tensor_from_neurons(&action)
                    .map_err(PPOError::SpaceError)?;

                let (log_probs, _) =
                    self.actor_network_log_prob_and_entropy(&latent_states, &action)?;
                // Detach so the rollout buffer does not keep the whole forward
                // graph (CNN activations) alive for every collected step.
                let log_probs = log_probs.detach();

                let step_info = env
                    .step(actual_action.clone())
                    .map_err(PPOError::GymError)?;
                let training_next_states = step_info.transition_next_states()?;
                let VectorizedStepInfo {
                    states: reset_or_next_states,
                    rewards,
                    dones: next_dones,
                    truncateds,
                    terminal_states: _,
                } = step_info;
                self.rollout_buffer.add(
                    PPOExperience::builder()
                        .states(states.clone())
                        .next_states(training_next_states)
                        .actions(action)
                        .rewards(rewards.clone())
                        .next_dones(next_dones)
                        .truncateds(truncateds)
                        .log_probs(log_probs)
                        .build(),
                );
                next_states = reset_or_next_states;
            }

            elapsed_timesteps += self.rollout_buffer.len() * env.num_envs();

            let progress = (elapsed_timesteps as f64) / (num_timesteps as f64);
            self.update_learning_rates(progress);

            if let Some(logging_info) = &mut self.logging_info {
                logging_info.timestep = elapsed_timesteps;
            }
            self.optimize(progress)?;
        }
        self.current_states = Some(next_states);
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
        models::{MLP, probabilistic_model::ProbabilisticPolicyModel},
        spaces::Discrete,
        tensor_operations::tanh,
    };
    use candle_core::Device;
    use candle_nn::{AdamW, ParamsAdamW, VarBuilder, VarMap};

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
            let next_done = self.step_count >= 5;
            Ok(StepInfo {
                state: Tensor::rand(0.0f32, 1.0, &[4], &self.device)?,
                reward: self.step_count as f32,
                done: next_done,
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

            // Create fresh PPO agent
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
                .input_size(observation_space.shape()[0])
                .output_size(action_space.shape()[0])
                .vb(actor_vb)
                .activation(Box::new(tanh))
                .hidden_layer_sizes(vec![8, 8])
                .name("actor_network".to_string())
                .build()
                .unwrap();

            let config = ParamsAdamW {
                lr: 3e-4,
                ..Default::default()
            };
            let actor_optimizer = AdamW::new(actor_var_map.all_vars(), config.clone()).unwrap();

            // Critic network
            let critic_var_map = VarMap::new();
            let critic_vb =
                VarBuilder::from_varmap(&critic_var_map, candle_core::DType::F32, &device);
            let critic_network = MLP::builder()
                .input_size(observation_space.shape()[0])
                .output_size(1)
                .vb(critic_vb)
                .activation(Box::new(tanh))
                .hidden_layer_sizes(vec![8, 8])
                .name("critic_network".to_string())
                .build()
                .unwrap();

            let critic_optimizer = AdamW::new(critic_var_map.all_vars(), config.clone()).unwrap();

            // Create PPO agent
            let network_info = PPONetworkInfo::Separate(
                SeparatePPONetwork::builder()
                    .actor_optimizer(actor_optimizer)
                    .critic_optimizer(critic_optimizer)
                    .actor_network(Box::new(
                        ProbabilisticPolicyModel::<CategoricalDistribution>::new(Box::new(
                            actor_network,
                        )),
                    ))
                    .critic_network(Box::new(critic_network))
                    .build(),
            );

            let mut agent = PPOAgent::builder()
                .action_space(action_space)
                .network_info(network_info)
                .batch_size(2048)
                .mini_batch_size(64)
                .ent_coef(0.01)
                .vf_coef(0.5)
                .clip_range(Box::new(ConstantSchedule::new(0.2)))
                .gae_lambda(0.95)
                .num_epochs(10)
                .device(device.clone())
                .build();

            // train for some timesteps
            agent
                .learn(&mut vec_env, 10000)
                .expect("PPO learning failed");

            let mut actions = vec![];
            let mut state = vec_env.reset().unwrap();
            for _ in 0..SAMPLE_COUNT {
                let action_neurons = agent.act_neurons(&state).unwrap();
                actions.push(action_neurons.clone());
                let action = agent
                    .action_space
                    .tensor_from_neurons(&action_neurons)
                    .unwrap();
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

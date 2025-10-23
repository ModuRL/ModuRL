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
    pub learning_rate: f32,
    pub rewards: Tensor,
    pub epoch: usize,
    pub timestep: usize,
}

pub trait PPOLogger {
    fn log(&mut self, info: &PPOLogEntry);
}

#[derive(Clone)]
struct PPOLosses {
    actor_loss: Tensor,
    critic_loss: Tensor,
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
    vf_coef: f32,
    ent_coef: f32,
    critic_optimizer: O1,
    actor_optimizer: O2,
    action_space: Box<dyn spaces::Space<Error = SE>>,
    critic_network: Box<dyn candle_core::Module>,
    actor_network: Box<dyn ProbabilisticActor<Error = AE>>,
    rollout_buffer: RolloutBuffer<PPOExperience>,
    num_epochs: usize,
    batch_size: usize,
    actor_lr_scheduler: Option<Box<dyn LrScheduler>>,
    critic_lr_scheduler: Option<Box<dyn LrScheduler>>,
    logging_info: Option<PPOLoggingInfo<'a>>,
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
        logging_info: Option<&'a mut dyn PPOLogger>,
        device: candle_core::Device,
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
            rollout_buffer: RolloutBuffer::new(mini_batch_size, device),
            num_epochs,
            batch_size,
            actor_lr_scheduler,
            critic_lr_scheduler,
            logging_info: logging_info.map(|logger| PPOLoggingInfo::new(logger)),
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
                    &mut rewards,
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
                let rewards = rewards.chunk(env_count, 0)?;

                // right now we have [batch_size, env_count, ...]
                // we chunk along the env_count dimension
                for env_idx in 0..actions.len() {
                    let actions = actions[env_idx].clone();
                    let states = states[env_idx].clone();
                    let advantages = advantages[env_idx].clone();
                    let returns = returns[env_idx].clone();
                    let batch_old_log_probs = batch_old_log_probs[env_idx].clone();
                    let batch_rewards = rewards[env_idx].clone();

                    // Compute the loss and backpropagate
                    let ppo_losses = self.compute_loss(
                        &states,
                        &actions.detach(),
                        batch_old_log_probs.detach(),
                        advantages,
                        returns,
                        batch_rewards,
                    )?;

                    self.backpropagate_loss(ppo_losses.clone())?;
                }
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

        let values_tensor = self.critic_network.forward(&Tensor::stack(&states, 0)?)?;

        // the last step for every env is needed for bootstrapping
        let next_states_tensor = Tensor::stack(&next_states, 0)?;
        let bootstrapped_states = next_states_tensor.i(next_states_tensor.shape().dims()[0] - 1)?; // shape [env_count, ...]
        let bootstrapped_values = self
            .critic_network
            .forward(&bootstrapped_states)?
            .flatten_all()?; // shape [env_count]

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
            &dones_tensor,
            &bootstrapped_values,
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
                let surrogate = (ratio.clone() * &advantages)?;
                (-1.0 * surrogate.mean_all()?)?
            }
        };

        let values = self.critic_network.forward(states)?;
        let values = values.squeeze(D::Minus1)?;

        let entropy_loss = entropy.mean_all()?;

        let final_actor_loss =
            (actor_loss.clone() - ((self.ent_coef as f64) * entropy_loss.clone()))?;

        let returns = returns.detach();
        let returns = normalize_tensor(&returns)?;

        let critic_loss = loss::mse(&values, &returns)?;
        let final_critic_loss = ((self.vf_coef as f64) * critic_loss.clone())?;

        if let Some(logging_info) = &mut self.logging_info {
            let explained_variance = {
                let var_y = returns.var(D::Minus1)?;
                let diff = (&returns - &values)?;
                let var_diff = diff.var(D::Minus1)?;

                1.0 + (-1.0 * (var_diff + 1e-8)? / (var_y + 1e-8)?)?
            }?;

            let log_entry = PPOLogEntry {
                actor_loss: final_actor_loss.clone(),
                critic_loss: final_critic_loss.clone(),
                entropy: entropy_loss,
                kl_divergence: ratio,
                explained_variance,
                learning_rate: self.actor_optimizer.learning_rate() as f32,
                rewards: rewards,
                epoch: logging_info.epoch,
                timestep: logging_info.timestep,
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

        if !tensor_has_nan(&actor_loss)? {
            let actor_grad = &mut actor_loss.backward()?;
            let _actor_grad_norm = crate::tensor_operations::clip_gradients(actor_grad, 1.0)?;
            self.actor_optimizer.step(actor_grad)?;
        }

        if !tensor_has_nan(&critic_loss)? {
            let critic_grad = &mut critic_loss.backward()?;
            let _critic_grad_norm = crate::tensor_operations::clip_gradients(critic_grad, 1.0)?;
            self.critic_optimizer.step(critic_grad)?;
        }

        Ok(())
    }

    fn act_neurons(
        &mut self,
        observation: &candle_core::Tensor,
    ) -> Result<candle_core::Tensor, PPOError<AE, GE, SE>> {
        self.actor_network
            .sample(observation)
            .map_err(PPOError::ActorError)
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
        let neurons = self.act_neurons(observation)?;
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

        while elapsed_timesteps < num_timesteps {
            while self.rollout_buffer.len() * env.num_envs() < self.batch_size {
                let states = next_states.clone();
                let action = self.act_neurons(&states)?;
                let actual_action = self
                    .action_space
                    .from_neurons(&action)
                    .map_err(PPOError::SpaceError)?;

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
        models::{
            MLP,
            probabilistic_model::{MLPProbabilisticActor, MLPProbabilisticActorError},
        },
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
                state: Tensor::from_vec(vec![0.1f32, 0.2, 0.3, 0.4], &[4], &self.device)?,
                reward: 1.0,
                done,
                truncated: false,
            })
        }

        fn reset(&mut self) -> Result<Tensor, Self::Error> {
            self.step_count = 0;
            Tensor::from_vec(vec![0.1f32, 0.2, 0.3, 0.4], &[4], &self.device)
        }

        fn observation_space(&self) -> Box<dyn crate::spaces::Space<Error = Self::SpaceError>> {
            Box::new(crate::spaces::BoxSpace::new(
                Tensor::full(-10.0, &[4], &self.device).unwrap(),
                Tensor::full(10.0, &[4], &self.device).unwrap(),
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

        let mut last_action: Option<Tensor> = None;

        for i in 0..5 {
            // Reset seed before each iteration
            device.set_seed(42).unwrap();

            // Create fresh PPO actor
            let mut envs = vec![DummyEnv::new(device.clone())];
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
                .batch_size(64)
                .mini_batch_size(32)
                .normalize_advantage(true)
                .ent_coef(0.01)
                .gamma(0.99)
                .vf_coef(0.5)
                .clip_range(0.2)
                .clipped(true)
                .gae_lambda(0.95)
                .num_epochs(1)
                .device(device.clone())
                .build();

            // train for some timesteps
            actor.learn(&mut vec_env, 256).expect("PPO learning failed");

            // Get initial state and take one action
            let initial_state = vec_env.reset().unwrap();
            let current_action = actor.act(&initial_state).unwrap();

            // Compare with previous iteration
            if let Some(last_action) = &last_action {
                let max_diff = last_action
                    .sub(&current_action)
                    .unwrap()
                    .max_all()
                    .unwrap()
                    .to_scalar::<u32>()
                    .unwrap();

                assert!(
                    max_diff == 0,
                    "PPO actions differ at iteration {i} by {max_diff}"
                );
            }
            last_action = Some(current_action);
        }
    }
}

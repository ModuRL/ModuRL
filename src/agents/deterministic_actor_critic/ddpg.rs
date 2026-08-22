//! Deep Deterministic Policy Gradient.
//!
//! [`DDPGAgent`] trains one deterministic actor and one state-action critic
//! from replay. [`DDPGLogger`] receives replay-update and collection metrics.

use std::fmt::Debug;

use bon::bon;
use candle_core::Tensor;
use candle_nn::{Optimizer, VarMap};

use super::{
    CriticCountRequirement, DeterministicActorCriticAgent,
    DeterministicActorCriticCollectionLogEntry, DeterministicActorCriticError,
    DeterministicActorCriticLogEntry, DeterministicActorCriticLogger,
    DeterministicActorCriticResult, DeterministicActorCriticStrategy, DeterministicCritic,
};
use crate::agents::sac::SACCriticError;
use crate::{
    agents::{Agent, ReplayDeviceStrategy},
    gym::MultiGym,
    spaces::{BoxSpace, Space},
};

/// Receives DDPG replay-update and collection metrics.
pub trait DDPGLogger<I = ()> {
    /// Records metrics from one replay optimization.
    fn log(&mut self, entry: &DeterministicActorCriticLogEntry);

    /// Records metrics from one vectorized environment step.
    fn log_collection(&mut self, _entry: &DeterministicActorCriticCollectionLogEntry<I>) {}
}

struct DDPGLoggingInfo<'a, I> {
    logger: &'a mut dyn DDPGLogger<I>,
}

impl<I> DeterministicActorCriticLogger<I> for Option<DDPGLoggingInfo<'_, I>> {
    fn log_update(&mut self, entry: &DeterministicActorCriticLogEntry) {
        if let Some(info) = self {
            info.logger.log(entry);
        }
    }

    fn log_collection(&mut self, entry: &DeterministicActorCriticCollectionLogEntry<I>) {
        if let Some(info) = self {
            info.logger.log_collection(entry);
        }
    }
}

pub(crate) struct DDPGStrategy;

impl DeterministicActorCriticStrategy for DDPGStrategy {
    fn critic_count_requirement(&self) -> CriticCountRequirement {
        CriticCountRequirement::Exact(1)
    }

    fn actor_update_interval(&self) -> usize {
        1
    }

    fn target_policy_noise(&self) -> f64 {
        0.0
    }

    fn target_noise_clip(&self) -> f64 {
        0.0
    }

    /// Returns the sole critic target tensor shaped `[batch]`.
    fn aggregate_target_values(&self, values: Vec<Tensor>) -> Result<Tensor, SACCriticError> {
        values
            .into_iter()
            .next()
            .ok_or(SACCriticError::NoCriticValues)
    }

    /// Returns the sole actor-objective Q tensor shaped `[batch_size]`.
    fn aggregate_actor_values(&self, values: Vec<Tensor>) -> Result<Tensor, SACCriticError> {
        values
            .into_iter()
            .next()
            .ok_or(SACCriticError::NoCriticValues)
    }
}

/// Deep Deterministic Policy Gradient agent for bounded continuous actions.
///
/// The agent collects transitions with a deterministic actor plus Gaussian
/// exploration noise, trains one critic from replay, and Polyak-updates target
/// networks after every replay optimization.
pub struct DDPGAgent<'a, AO, CO, GE, SE, I = ()>
where
    AO: Optimizer,
    CO: Optimizer,
    GE: Debug,
    SE: Debug,
{
    inner: DeterministicActorCriticAgent<'a, AO, CO, GE, SE, DDPGStrategy>,
    logging_info: Option<DDPGLoggingInfo<'a, I>>,
}

#[bon]
impl<'a, AO, CO, GE, SE, I> DDPGAgent<'a, AO, CO, GE, SE, I>
where
    AO: Optimizer,
    CO: Optimizer,
    GE: Debug,
    SE: Debug,
{
    #[builder]
    /// Builds a DDPG agent and initializes target parameters from the online
    /// actor and critic parameters.
    pub fn new(
        /// Actor optimized during replay updates.
        #[builder(with = |actor: impl candle_core::Module + 'static| Box::new(actor))]
        online_actor: Box<dyn candle_core::Module>,
        /// Actor used to calculate next-state Bellman targets.
        #[builder(with = |actor: impl candle_core::Module + 'static| Box::new(actor))]
        target_actor: Box<dyn candle_core::Module>,
        /// Parameters belonging to `online_actor`.
        online_actor_vars: &'a VarMap,
        /// Parameters belonging to `target_actor`.
        target_actor_vars: &'a mut VarMap,
        /// Optimizer for the online actor parameters.
        actor_optimizer: AO,
        /// The sole online/target critic pair.
        critic: DeterministicCritic<'a, CO>,
        /// Bounded action space matching the actor output and environment.
        action_space: BoxSpace,
        /// Observation-space contract expected by the actor and critic.
        observation_space: Box<dyn Space<Error = SE>>,
        /// Devices used for replay storage and optimization.
        device_strategy: ReplayDeviceStrategy,
        /// Optional replay-update and collection metric sink.
        logger: Option<&'a mut dyn DDPGLogger<I>>,
        /// Bellman discount factor.
        #[builder(default = 0.99)]
        gamma: f64,
        /// Polyak coefficient for actor and critic target updates.
        #[builder(default = 0.005)]
        tau: f64,
        /// Standard deviation of Gaussian collection-action noise.
        #[builder(default = 0.1)]
        exploration_noise: f64,
        /// Maximum number of transitions retained in replay.
        #[builder(default = 1_000_000)]
        replay_capacity: usize,
        /// Number of replay transitions sampled per optimization.
        #[builder(default = 256)]
        batch_size: usize,
        /// Collected-transition interval between replay optimizations.
        #[builder(default = 1)]
        update_frequency: usize,
        /// Number of random-action transitions collected before optimization.
        #[builder(default = 1_000)]
        training_start: usize,
        /// Global collected-transition horizon.
        training_horizon: usize,
        #[builder(default = candle_core::DType::F32)] dtype: candle_core::DType,
    ) -> DeterministicActorCriticResult<Self, GE, SE> {
        let inner = DeterministicActorCriticAgent::builder()
            .online_actor(online_actor)
            .target_actor(target_actor)
            .online_actor_vars(online_actor_vars)
            .target_actor_vars(target_actor_vars)
            .actor_optimizer(actor_optimizer)
            .critics(vec![critic])
            .strategy(DDPGStrategy)
            .action_space(action_space)
            .observation_space(observation_space)
            .device_strategy(device_strategy)
            .gamma(gamma)
            .tau(tau)
            .exploration_noise(exploration_noise)
            .replay_capacity(replay_capacity)
            .batch_size(batch_size)
            .update_frequency(update_frequency)
            .training_start(training_start)
            .training_horizon(training_horizon)
            .dtype(dtype)
            .build()?;
        Ok(Self {
            inner,
            logging_info: logger.map(|logger| DDPGLoggingInfo { logger }),
        })
    }

    /// Returns the bounded continuous action space used by the agent.
    pub fn get_action_space(&self) -> &BoxSpace {
        self.inner.get_action_space()
    }

    /// Returns the observation-space contract used by the agent.
    pub fn get_observation_space(&self) -> &dyn Space<Error = SE> {
        self.inner.get_observation_space()
    }

    /// Maps observations `[batch, ...observation_shape]` to deterministic
    /// actions `[batch, ...action_shape]`.
    pub fn act_deterministic(
        &self,
        observation: &Tensor,
    ) -> Result<Tensor, DeterministicActorCriticError<GE, SE>> {
        self.inner.act_deterministic(observation)
    }
}

impl<'a, AO, CO, GE, SE, I> Agent<I> for DDPGAgent<'a, AO, CO, GE, SE, I>
where
    AO: Optimizer,
    CO: Optimizer,
    GE: Debug,
    SE: Debug,
{
    type Error = DeterministicActorCriticError<GE, SE>;
    type GymError = GE;
    type SpaceError = SE;

    /// Maps observations `[batch, ...observation_shape]` to exploration-noised
    /// actions `[batch, ...action_shape]`.
    fn act(&mut self, observation: &Tensor) -> Result<Tensor, Self::Error> {
        self.inner.act(observation)
    }

    fn learn(
        &mut self,
        env: &mut dyn MultiGym<I, Error = GE, SpaceError = SE>,
        num_timesteps: usize,
    ) -> Result<(), Self::Error> {
        self.inner.learn(env, num_timesteps, &mut self.logging_info)
    }
}

#[cfg(test)]
mod tests {
    use super::{DDPGAgent, DDPGLogger};
    use crate::{
        agents::{
            Agent, ReplayDeviceStrategy,
            deterministic_actor_critic::DeterministicActorCriticLogEntry,
            sac::{SACCritic, ScalarStateActionCritic},
            test_support::{CountingOptimizer, FixedContinuousEnv},
        },
        gym::{MultiGym, VectorizedGymWrapper},
        models::MLP,
        spaces::{BoxSpace, Space},
    };
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    #[derive(Default)]
    struct RecordingLogger {
        updates: usize,
    }

    impl DDPGLogger for RecordingLogger {
        fn log(&mut self, entry: &DeterministicActorCriticLogEntry) {
            self.updates += 1;
            assert_eq!(entry.critic_losses.len(), 1);
            assert_eq!(entry.critic_q_values.len(), 1);
            assert_eq!(entry.critic_q_values[0].dims(), &[1]);
            assert_eq!(entry.bellman_targets.dims(), &[1]);
            assert_eq!(entry.replay_rewards.dims(), &[1]);
            assert_eq!(entry.replay_actions.dims(), &[1, 1]);
            assert_eq!(entry.policy_q_values.as_ref().unwrap().dims(), &[1]);
            assert_eq!(entry.policy_actions.as_ref().unwrap().dims(), &[1, 1]);
            assert_eq!(entry.actor_learning_rate, 1e-3);
            assert_eq!(entry.critic_learning_rates, vec![1e-3]);
            assert_eq!(entry.exploration_noise_standard_deviation, 0.0);
            assert!(entry.actor_updated);
        }
    }

    fn actor(vars: &VarMap, device: &Device) -> MLP {
        MLP::builder()
            .input_size(4)
            .output_size(1)
            .hidden_layer_sizes(vec![4])
            .output_activation(Tensor::tanh)
            .vb(VarBuilder::from_varmap(vars, DType::F32, device))
            .build()
            .unwrap()
    }

    fn critic<'a>(
        online_vars: &'a VarMap,
        target_vars: &'a mut VarMap,
        device: &Device,
    ) -> SACCritic<'a, CountingOptimizer> {
        let network = |vars| {
            MLP::builder()
                .input_size(5)
                .output_size(1)
                .hidden_layer_sizes(vec![4])
                .vb(VarBuilder::from_varmap(vars, DType::F32, device))
                .build()
                .unwrap()
        };
        SACCritic::builder()
            .online_network(ScalarStateActionCritic::new(network(online_vars)))
            .target_network(ScalarStateActionCritic::new(network(target_vars)))
            .online_vars(online_vars)
            .target_vars(target_vars)
            .optimizer(CountingOptimizer::with_learning_rate(1e-3))
            .build()
            .unwrap()
    }

    #[test]
    fn public_agent_uses_shared_collection_and_updates_actor_every_step() {
        let device = Device::Cpu;
        let mut env: VectorizedGymWrapper<FixedContinuousEnv> = vec![
            FixedContinuousEnv::new(device.clone()),
            FixedContinuousEnv::new(device.clone()),
        ]
        .into();
        let online_actor_vars = VarMap::new();
        let mut target_actor_vars = VarMap::new();
        let online_critic_vars = VarMap::new();
        let mut target_critic_vars = VarMap::new();
        let critic = critic(&online_critic_vars, &mut target_critic_vars, &device);
        let mut logger = RecordingLogger::default();
        let mut agent = DDPGAgent::builder()
            .online_actor(actor(&online_actor_vars, &device))
            .target_actor(actor(&target_actor_vars, &device))
            .online_actor_vars(&online_actor_vars)
            .target_actor_vars(&mut target_actor_vars)
            .actor_optimizer(CountingOptimizer::with_learning_rate(1e-3))
            .critic(critic)
            .action_space(BoxSpace::new_with_universal_bounds(
                vec![1],
                -1.0,
                1.0,
                &device,
            ))
            .observation_space(env.observation_space())
            .device_strategy(ReplayDeviceStrategy::OneDevice(device.clone()))
            .exploration_noise(0.0)
            .replay_capacity(4)
            .batch_size(1)
            .training_start(1)
            .training_horizon(2)
            .logger(&mut logger)
            .build()
            .unwrap();

        let action = agent
            .act_deterministic(&Tensor::zeros((2, 4), DType::F32, &device).unwrap())
            .unwrap();
        assert_eq!(action.dims(), &[2, 1]);
        assert!(
            action
                .chunk(2, 0)
                .unwrap()
                .iter()
                .all(|row| { agent.get_action_space().contains(&row.squeeze(0).unwrap()) })
        );

        agent.learn(&mut env, 2).unwrap();
        assert_eq!(agent.inner.actor_optimizer.steps, 2);
        assert_eq!(agent.inner.critics[0].optimizer().steps, 2);
        assert_eq!(logger.updates, 2);
    }
}

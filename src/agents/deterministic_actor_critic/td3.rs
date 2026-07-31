//! Twin Delayed Deep Deterministic Policy Gradient.
//!
//! [`TD3Agent`] extends deterministic replay training with a critic ensemble,
//! target-policy smoothing, and delayed actor updates. [`TD3Logger`] receives
//! replay-update and collection metrics.

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
use crate::{
    agents::{
        Agent, ReplayDeviceStrategy,
        sac::{SACCriticAggregationMode, SACCriticError, aggregate_critic_values},
    },
    gym::VectorizedGym,
    spaces::{BoxSpace, Space},
};

/// Receives TD3 replay-update and collection metrics.
pub trait TD3Logger<I = ()> {
    /// Records metrics from one replay optimization.
    fn log(&mut self, entry: &DeterministicActorCriticLogEntry);

    /// Records metrics from one vectorized environment step.
    fn log_collection(&mut self, _entry: &DeterministicActorCriticCollectionLogEntry<I>) {}
}

struct TD3LoggingInfo<'a, I> {
    logger: &'a mut dyn TD3Logger<I>,
}

impl<I> DeterministicActorCriticLogger<I> for Option<TD3LoggingInfo<'_, I>> {
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

pub(crate) struct TD3Strategy {
    target_policy_noise: f64,
    target_noise_clip: f64,
    actor_update_interval: usize,
    target_aggregation_mode: SACCriticAggregationMode,
    actor_aggregation_mode: Option<SACCriticAggregationMode>,
}

impl DeterministicActorCriticStrategy for TD3Strategy {
    fn critic_count_requirement(&self) -> CriticCountRequirement {
        CriticCountRequirement::NonEmpty
    }

    fn actor_update_interval(&self) -> usize {
        self.actor_update_interval
    }

    fn target_policy_noise(&self) -> f64 {
        self.target_policy_noise
    }

    fn target_noise_clip(&self) -> f64 {
        self.target_noise_clip
    }

    /// Takes the elementwise minimum of two target tensors shaped `[batch]`.
    fn aggregate_target_values(&self, values: Vec<Tensor>) -> Result<Tensor, SACCriticError> {
        aggregate_critic_values(&values, self.target_aggregation_mode)
    }

    /// Combines actor-objective Q tensors shaped `[batch_size]` using the
    /// configured mode, or selects the first tensor for canonical TD3.
    fn aggregate_actor_values(&self, values: Vec<Tensor>) -> Result<Tensor, SACCriticError> {
        match self.actor_aggregation_mode {
            Some(mode) => aggregate_critic_values(&values, mode),
            None => values
                .into_iter()
                .next()
                .ok_or(SACCriticError::NoCriticValues),
        }
    }
}

/// Twin Delayed Deep Deterministic Policy Gradient agent.
///
/// The canonical configuration uses two critics and
/// [`SACCriticAggregationMode::Min`] for targets, while optimizing the actor
/// against the first critic. Any non-empty critic ensemble is supported, and
/// target and actor aggregation can be configured independently for
/// experimentation.
pub struct TD3Agent<'a, AO, CO, GE, SE, I = ()>
where
    AO: Optimizer,
    CO: Optimizer,
    GE: Debug,
    SE: Debug,
{
    inner: DeterministicActorCriticAgent<'a, AO, CO, GE, SE, TD3Strategy>,
    logging_info: Option<TD3LoggingInfo<'a, I>>,
}

#[bon]
impl<'a, AO, CO, GE, SE, I> TD3Agent<'a, AO, CO, GE, SE, I>
where
    AO: Optimizer,
    CO: Optimizer,
    GE: Debug,
    SE: Debug,
{
    #[builder]
    /// Builds a TD3 agent and initializes every target network from its online
    /// counterpart.
    pub fn new(
        /// Actor optimized on delayed replay updates.
        online_actor: Box<dyn candle_core::Module>,
        /// Actor used to calculate smoothed next-state Bellman targets.
        target_actor: Box<dyn candle_core::Module>,
        /// Parameters belonging to `online_actor`.
        online_actor_vars: &'a VarMap,
        /// Parameters belonging to `target_actor`.
        target_actor_vars: &'a mut VarMap,
        /// Optimizer for the online actor parameters.
        actor_optimizer: AO,
        /// Non-empty independently optimized critic ensemble.
        critics: Vec<DeterministicCritic<'a, CO>>,
        /// Bounded action space matching the actor output and environment.
        action_space: BoxSpace,
        /// Observation-space contract expected by the actor and critics.
        observation_space: Box<dyn Space<Error = SE>>,
        /// Devices used for replay storage and optimization.
        device_strategy: ReplayDeviceStrategy,
        /// Optional replay-update and collection metric sink.
        logger: Option<&'a mut dyn TD3Logger<I>>,
        /// Bellman discount factor.
        #[builder(default = 0.99)]
        gamma: f64,
        /// Polyak coefficient for actor and critic target updates.
        #[builder(default = 0.005)]
        tau: f64,
        /// Standard deviation of Gaussian collection-action noise.
        #[builder(default = 0.1)]
        exploration_noise: f64,
        /// Standard deviation of Gaussian target-policy noise.
        #[builder(default = 0.2)]
        target_policy_noise: f64,
        /// Componentwise absolute limit for target-policy noise.
        #[builder(default = 0.5)]
        target_noise_clip: f64,
        /// Replay-optimization interval between actor and target updates.
        #[builder(default = 2)]
        actor_update_interval: usize,
        /// Aggregates the target critics' next-state Q estimates. Canonical
        /// TD3 uses [`SACCriticAggregationMode::Min`].
        #[builder(default = SACCriticAggregationMode::Min)]
        target_aggregation_mode: SACCriticAggregationMode,
        /// Optionally aggregates all online critics for the actor objective.
        /// `None` preserves canonical TD3 behavior by using the first critic.
        actor_aggregation_mode: Option<SACCriticAggregationMode>,
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
        let strategy = TD3Strategy {
            target_policy_noise,
            target_noise_clip,
            actor_update_interval,
            target_aggregation_mode,
            actor_aggregation_mode,
        };
        let inner = DeterministicActorCriticAgent::builder()
            .online_actor(online_actor)
            .target_actor(target_actor)
            .online_actor_vars(online_actor_vars)
            .target_actor_vars(target_actor_vars)
            .actor_optimizer(actor_optimizer)
            .critics(critics)
            .strategy(strategy)
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
            logging_info: logger.map(|logger| TD3LoggingInfo { logger }),
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

impl<'a, AO, CO, GE, SE, I> Agent<I> for TD3Agent<'a, AO, CO, GE, SE, I>
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
        env: &mut dyn VectorizedGym<I, Error = GE, SpaceError = SE>,
        num_timesteps: usize,
    ) -> Result<(), Self::Error> {
        self.inner.learn(env, num_timesteps, &mut self.logging_info)
    }
}

#[cfg(test)]
mod tests {
    use super::{TD3Agent, TD3Logger, TD3Strategy};
    use crate::{
        agents::{
            Agent, ReplayDeviceStrategy,
            deterministic_actor_critic::{
                DeterministicActorCriticLogEntry, DeterministicActorCriticStrategy,
            },
            sac::{SACCritic, SACCriticAggregationMode, ScalarStateActionCritic},
            test_support::{CountingOptimizer, FixedContinuousEnv},
        },
        gym::{VectorizedGym, VectorizedGymWrapper},
        models::MLP,
        spaces::BoxSpace,
    };
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn strategy_uses_the_configured_critic_aggregation() {
        let strategy = TD3Strategy {
            target_policy_noise: 0.2,
            target_noise_clip: 0.5,
            actor_update_interval: 2,
            target_aggregation_mode: SACCriticAggregationMode::Mean,
            actor_aggregation_mode: Some(SACCriticAggregationMode::Mean),
        };
        let values = || {
            vec![
                Tensor::new(&[1.0f32, 3.0], &Device::Cpu).unwrap(),
                Tensor::new(&[3.0f32, 1.0], &Device::Cpu).unwrap(),
                Tensor::new(&[2.0f32, 2.0], &Device::Cpu).unwrap(),
            ]
        };

        assert_eq!(
            strategy
                .aggregate_target_values(values())
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![2.0, 2.0]
        );
        assert_eq!(
            strategy
                .aggregate_actor_values(values())
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![2.0, 2.0]
        );
    }

    #[derive(Default)]
    struct RecordingLogger {
        actor_updates: Vec<bool>,
    }

    impl TD3Logger for RecordingLogger {
        fn log(&mut self, entry: &DeterministicActorCriticLogEntry) {
            self.actor_updates.push(entry.actor_updated);
            assert_eq!(entry.critic_losses.len(), 3);
            assert_eq!(entry.critic_q_values.len(), 3);
            assert!(
                entry
                    .critic_q_values
                    .iter()
                    .all(|values| values.dims() == [1])
            );
            assert_eq!(entry.bellman_targets.dims(), &[1]);
            assert_eq!(entry.replay_actions.dims(), &[1, 1]);
            assert_eq!(entry.critic_learning_rates, vec![1e-3, 1e-3, 1e-3]);
            assert_eq!(entry.actor_loss.is_some(), entry.actor_updated);
            assert_eq!(entry.policy_q_values.is_some(), entry.actor_updated);
            assert_eq!(entry.policy_actions.is_some(), entry.actor_updated);
        }
    }

    fn actor(vars: &VarMap, device: &Device) -> MLP {
        MLP::builder()
            .input_size(4)
            .output_size(1)
            .hidden_layer_sizes(vec![4])
            .output_activation(Box::new(Tensor::tanh))
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
            .online_network(Box::new(ScalarStateActionCritic::new(Box::new(network(
                online_vars,
            )))))
            .target_network(Box::new(ScalarStateActionCritic::new(Box::new(network(
                target_vars,
            )))))
            .online_vars(online_vars)
            .target_vars(target_vars)
            .optimizer(CountingOptimizer::with_learning_rate(1e-3))
            .build()
            .unwrap()
    }

    #[test]
    fn public_agent_supports_generalized_critic_ensembles_and_delays_actor_updates() {
        let device = Device::Cpu;
        let mut env: VectorizedGymWrapper<FixedContinuousEnv> = vec![
            FixedContinuousEnv::new(device.clone()),
            FixedContinuousEnv::new(device.clone()),
        ]
        .into();
        let online_actor_vars = VarMap::new();
        let mut target_actor_vars = VarMap::new();
        let online_critic_vars_1 = VarMap::new();
        let mut target_critic_vars_1 = VarMap::new();
        let online_critic_vars_2 = VarMap::new();
        let mut target_critic_vars_2 = VarMap::new();
        let online_critic_vars_3 = VarMap::new();
        let mut target_critic_vars_3 = VarMap::new();
        let critic_1 = critic(&online_critic_vars_1, &mut target_critic_vars_1, &device);
        let critic_2 = critic(&online_critic_vars_2, &mut target_critic_vars_2, &device);
        let critic_3 = critic(&online_critic_vars_3, &mut target_critic_vars_3, &device);
        let mut logger = RecordingLogger::default();
        let mut agent = TD3Agent::builder()
            .online_actor(Box::new(actor(&online_actor_vars, &device)))
            .target_actor(Box::new(actor(&target_actor_vars, &device)))
            .online_actor_vars(&online_actor_vars)
            .target_actor_vars(&mut target_actor_vars)
            .actor_optimizer(CountingOptimizer::with_learning_rate(1e-3))
            .critics(vec![critic_1, critic_2, critic_3])
            .action_space(BoxSpace::new_with_universal_bounds(
                vec![1],
                -1.0,
                1.0,
                &device,
            ))
            .observation_space(env.observation_space())
            .device_strategy(ReplayDeviceStrategy::OneDevice(device))
            .exploration_noise(0.0)
            .target_policy_noise(0.0)
            .target_aggregation_mode(SACCriticAggregationMode::Mean)
            .actor_aggregation_mode(SACCriticAggregationMode::Median)
            .replay_capacity(4)
            .batch_size(1)
            .training_start(1)
            .training_horizon(2)
            .logger(&mut logger)
            .build()
            .unwrap();

        agent.learn(&mut env, 2).unwrap();
        assert_eq!(agent.inner.actor_optimizer.steps, 1);
        assert_eq!(agent.inner.critics[0].optimizer().steps, 2);
        assert_eq!(agent.inner.critics[1].optimizer().steps, 2);
        assert_eq!(agent.inner.critics[2].optimizer().steps, 2);
        assert_eq!(logger.actor_updates, vec![false, true]);
    }
}

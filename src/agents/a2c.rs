//! Advantage Actor-Critic implemented as a constrained PPO configuration.
//!
//! A2C uses the PPO training implementation with policy and value clipping
//! disabled, one optimization epoch, and the entire rollout as one minibatch.
//! The public wrappers in this module keep PPO-specific types out of A2C
//! training programs.

use std::fmt::Debug;

use bon::bon;
use candle_core::Tensor;
use candle_nn::Optimizer;

use super::{
    Agent,
    ppo::{
        FakeOptimizer, PPOAgent, PPOCollectionLogEntry, PPOEpisodeLogEntry, PPOError, PPOLogEntry,
        PPOLogger, PPONetworkInfo, SeparatePPONetwork, SharedPPONetwork,
    },
};
use crate::{
    gym::VectorizedGym, models::probabilistic_model::ProbabilisticPolicy,
    parameter_schedule::ParameterSchedule, spaces::Space,
};

/// An error returned by the PPO implementation backing an A2C agent.
#[derive(Debug)]
pub struct A2CError<AE, GE, SE>
where
    AE: Debug,
    GE: Debug,
    SE: Debug,
{
    pub inner: PPOError<AE, GE, SE>,
}

impl<AE, GE, SE> From<PPOError<AE, GE, SE>> for A2CError<AE, GE, SE>
where
    AE: Debug,
    GE: Debug,
    SE: Debug,
{
    fn from(inner: PPOError<AE, GE, SE>) -> Self {
        Self { inner }
    }
}

/// A borrowed A2C view of one optimization log entry.
pub struct A2CLogEntry<'a> {
    pub inner: &'a PPOLogEntry,
}

/// A borrowed A2C view of one rollout-collection log entry.
pub struct A2CCollectionLogEntry<'a, I = ()> {
    pub inner: &'a PPOCollectionLogEntry<I>,
}

impl<'a, I> A2CCollectionLogEntry<'a, I> {
    /// Returns A2C-typed views of the episodes completed during collection.
    pub fn completed_episodes(&self) -> impl ExactSizeIterator<Item = A2CEpisodeLogEntry<'_>> {
        self.inner
            .completed_episodes
            .iter()
            .map(|inner| A2CEpisodeLogEntry { inner })
    }
}

/// A borrowed A2C view of one completed-episode log entry.
pub struct A2CEpisodeLogEntry<'a> {
    pub inner: &'a PPOEpisodeLogEntry,
}

/// Receives A2C optimization and rollout-collection metrics.
pub trait A2CLogger<I = ()> {
    fn log(&mut self, info: &A2CLogEntry<'_>);

    fn log_collection(&mut self, _info: &A2CCollectionLogEntry<'_, I>) {}
}

/// Adapts an [`A2CLogger`] to the logger stored by the inner PPO agent.
///
/// The adapter must outlive the [`A2CAgent`] that borrows it.
pub struct A2CLoggerAdapter<'a, I = ()> {
    inner: &'a mut dyn A2CLogger<I>,
}

impl<'a, I> A2CLoggerAdapter<'a, I> {
    pub fn new(inner: &'a mut dyn A2CLogger<I>) -> Self {
        Self { inner }
    }
}

impl<I> PPOLogger<I> for A2CLoggerAdapter<'_, I> {
    fn log(&mut self, inner: &PPOLogEntry) {
        self.inner.log(&A2CLogEntry { inner });
    }

    fn log_collection(&mut self, inner: &PPOCollectionLogEntry<I>) {
        self.inner.log_collection(&A2CCollectionLogEntry { inner });
    }
}

/// Shared actor-critic network configuration for A2C.
pub struct SharedA2CNetwork<O, E> {
    inner: SharedPPONetwork<O, E>,
}

#[bon]
impl<O, E> SharedA2CNetwork<O, E>
where
    O: Optimizer,
    E: Debug,
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
            inner: SharedPPONetwork::builder()
                .optimizer(optimizer)
                .shared_network(shared_network)
                .critic_head(critic_head)
                .actor_head(actor_head)
                .maybe_lr_scheduler(lr_scheduler)
                .build(),
        }
    }
}

/// Separate actor and critic network configuration for A2C.
pub struct SeparateA2CNetwork<O1, O2, E> {
    inner: SeparatePPONetwork<O1, O2, E>,
}

#[bon]
impl<O1, O2, E> SeparateA2CNetwork<O1, O2, E>
where
    O1: Optimizer,
    O2: Optimizer,
    E: Debug,
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
            inner: SeparatePPONetwork::builder()
                .actor_optimizer(actor_optimizer)
                .critic_optimizer(critic_optimizer)
                .actor_network(actor_network)
                .critic_network(critic_network)
                .maybe_actor_lr_scheduler(actor_lr_scheduler)
                .maybe_critic_lr_scheduler(critic_lr_scheduler)
                .combined_loss(combined_loss)
                .build(),
        }
    }
}

/// An A2C-typed shared or separate network configuration.
pub struct A2CNetworkInfo<O1, E, O2 = FakeOptimizer> {
    inner: PPONetworkInfo<O1, E, O2>,
}

impl<O, E> A2CNetworkInfo<O, E, FakeOptimizer> {
    pub fn shared(network: SharedA2CNetwork<O, E>) -> Self {
        Self {
            inner: PPONetworkInfo::Shared(network.inner),
        }
    }
}

impl<O1, E, O2> A2CNetworkInfo<O1, E, O2> {
    pub fn separate(network: SeparateA2CNetwork<O1, O2, E>) -> Self {
        Self {
            inner: PPONetworkInfo::Separate(network.inner),
        }
    }
}

impl<O, E> From<SharedA2CNetwork<O, E>> for A2CNetworkInfo<O, E, FakeOptimizer> {
    fn from(network: SharedA2CNetwork<O, E>) -> Self {
        Self::shared(network)
    }
}

impl<O1, O2, E> From<SeparateA2CNetwork<O1, O2, E>> for A2CNetworkInfo<O1, E, O2> {
    fn from(network: SeparateA2CNetwork<O1, O2, E>) -> Self {
        Self::separate(network)
    }
}

/// Advantage Actor-Critic agent backed by a constrained PPO agent.
pub struct A2CAgent<'a, O1, O2, AE, GE, SE, I = ()>
where
    O1: Optimizer,
    O2: Optimizer,
    AE: Debug,
    GE: Debug,
    SE: Debug,
{
    inner: PPOAgent<'a, O1, O2, AE, GE, SE, I>,
}

#[bon]
impl<'a, O1, O2, AE, GE, SE, I> A2CAgent<'a, O1, O2, AE, GE, SE, I>
where
    O1: Optimizer,
    O2: Optimizer,
    AE: Debug,
    GE: Debug,
    SE: Debug,
{
    /// Builds A2C as unclipped PPO with one full-rollout optimization epoch.
    ///
    /// `batch_size` should be divisible by the number of vectorized
    /// environments so the collected rollout contains exactly `batch_size`
    /// transitions.
    #[builder]
    pub fn new<'logger>(
        action_space: Box<dyn Space<Error = SE>>,
        network_info: A2CNetworkInfo<O1, AE, O2>,
        #[builder(default = 0.99)] gamma: f32,
        #[builder(default = 1.0)] gae_lambda: f32,
        #[builder(default = false)] normalize_advantage: bool,
        #[builder(default = false)] normalize_returns: bool,
        #[builder(default = 0.5)] vf_coef: f32,
        #[builder(default = 0.0)] ent_coef: f32,
        #[builder(default = 1024)] batch_size: usize,
        #[builder(default = 0.5)] gradient_clip: f32,
        training_horizon: usize,
        logging_info: Option<&'a mut A2CLoggerAdapter<'logger, I>>,
        device: candle_core::Device,
    ) -> Self
    where
        'logger: 'a,
    {
        assert!(batch_size > 0, "A2C batch_size must be greater than zero");

        let logging_info = logging_info.map(|logger| logger as &mut dyn PPOLogger<I>);
        let inner = PPOAgent::builder()
            .action_space(action_space)
            .network_info(network_info.inner)
            .clipped(false)
            .clip_value_loss(false)
            .gamma(gamma)
            .gae_lambda(gae_lambda)
            .normalize_advantage(normalize_advantage)
            .normalize_returns(normalize_returns)
            .vf_coef(vf_coef)
            .ent_coef(ent_coef)
            .batch_size(batch_size)
            .mini_batch_size(batch_size)
            .num_epochs(1)
            .gradient_clip(gradient_clip)
            .training_horizon(training_horizon)
            .maybe_logging_info(logging_info)
            .device(device)
            .build();

        Self { inner }
    }

    pub fn set_learning_rate(&mut self, lr: f64) {
        self.inner.set_learning_rate(lr);
    }

    /// Selects deterministic environment actions for observations shaped
    /// `[batch, ...observation_shape]`.
    pub fn act_deterministic(
        &mut self,
        observation: &Tensor,
    ) -> Result<Tensor, A2CError<AE, GE, SE>> {
        self.inner
            .act_deterministic(observation)
            .map_err(A2CError::from)
    }

    pub fn reset_current_states(&mut self) {
        self.inner.reset_current_states();
    }
}

impl<'a, O1, O2, AE, GE, SE, I> Agent<I> for A2CAgent<'a, O1, O2, AE, GE, SE, I>
where
    O1: Optimizer,
    O2: Optimizer,
    AE: Debug,
    GE: Debug,
    SE: Debug,
{
    type Error = A2CError<AE, GE, SE>;
    type GymError = GE;
    type SpaceError = SE;

    /// Selects stochastic environment actions for observations shaped
    /// `[batch, ...observation_shape]`.
    fn act(&mut self, observation: &Tensor) -> Result<Tensor, Self::Error> {
        self.inner.act(observation).map_err(A2CError::from)
    }

    fn learn(
        &mut self,
        env: &mut dyn VectorizedGym<I, Error = GE, SpaceError = SE>,
        num_timesteps: usize,
    ) -> Result<(), Self::Error> {
        self.inner.learn(env, num_timesteps).map_err(A2CError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        agents::test_support::{CountingOptimizer, FixedEnv},
        distributions::{CategoricalDistribution, CategoricalDistributionError},
        gym::VectorizedGymWrapper,
        models::{
            MLP,
            probabilistic_model::{ProbabilisticPolicyModel, ProbabilisticPolicyModelError},
        },
        spaces::Discrete,
    };
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};

    #[derive(Default)]
    struct RecordingLogger {
        updates: usize,
        collections: usize,
        completed_episodes: usize,
        advantages: Vec<Vec<f32>>,
    }

    impl A2CLogger for RecordingLogger {
        fn log(&mut self, entry: &A2CLogEntry<'_>) {
            self.updates += 1;
            assert_eq!(entry.inner.epoch, 0);
            assert_eq!(entry.inner.advantages.dims(), &[2]);
            self.advantages
                .push(entry.inner.advantages.to_vec1::<f32>().unwrap());
        }

        fn log_collection(&mut self, entry: &A2CCollectionLogEntry<'_>) {
            self.collections += 1;
            self.completed_episodes += entry.completed_episodes().count();
        }
    }

    fn separate_network(
        device: &Device,
    ) -> A2CNetworkInfo<
        CountingOptimizer,
        ProbabilisticPolicyModelError<CategoricalDistributionError>,
        CountingOptimizer,
    > {
        let actor_vars = VarMap::new();
        let critic_vars = VarMap::new();
        let actor = MLP::builder()
            .input_size(4)
            .output_size(2)
            .vb(VarBuilder::from_varmap(&actor_vars, DType::F32, device))
            .hidden_layer_sizes(vec![2])
            .build()
            .unwrap();
        let critic = MLP::builder()
            .input_size(4)
            .output_size(1)
            .vb(VarBuilder::from_varmap(&critic_vars, DType::F32, device))
            .hidden_layer_sizes(vec![2])
            .build()
            .unwrap();

        SeparateA2CNetwork::builder()
            .actor_optimizer(CountingOptimizer::with_learning_rate(1e-3))
            .critic_optimizer(CountingOptimizer::with_learning_rate(1e-3))
            .actor_network(Box::new(
                ProbabilisticPolicyModel::<CategoricalDistribution>::new(Box::new(actor)),
            ))
            .critic_network(Box::new(critic))
            .build()
            .into()
    }

    #[test]
    fn public_agent_runs_one_full_rollout_update_and_forwards_actions() {
        let device = Device::Cpu;
        let mut env: VectorizedGymWrapper<FixedEnv> =
            vec![FixedEnv::new(device.clone()), FixedEnv::new(device.clone())].into();
        let mut logger = RecordingLogger::default();
        let mut adapter = A2CLoggerAdapter::new(&mut logger);
        let mut agent = A2CAgent::builder()
            .action_space(Box::new(Discrete::new(2)))
            .network_info(separate_network(&device))
            .batch_size(2)
            .training_horizon(4)
            .logging_info(&mut adapter)
            .device(device.clone())
            .build();

        agent.learn(&mut env, 2).unwrap();
        let observations = Tensor::zeros((2, 4), DType::F32, &device).unwrap();
        assert_eq!(agent.act(&observations).unwrap().dims(), &[2]);
        assert_eq!(agent.act_deterministic(&observations).unwrap().dims(), &[2]);
        agent.set_learning_rate(0.25);
        agent.reset_current_states();
        drop(agent);
        drop(adapter);

        assert_eq!(logger.updates, 1);
        assert_eq!(logger.collections, 1);
        assert_eq!(logger.completed_episodes, 0);
        assert!(
            logger.advantages[0]
                .iter()
                .any(|advantage| advantage.abs() > 1e-6),
            "A2C should not normalize advantages by default"
        );
    }

    #[test]
    fn network_configuration_is_a2c_typed_until_it_is_wrapped() {
        let network = separate_network(&Device::Cpu);
        assert!(matches!(network.inner, PPONetworkInfo::Separate(_)));
    }
}

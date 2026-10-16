//! Advantage Actor-Critic implemented as a constrained PPO configuration.
//!
//! A2C disables policy and value clipping, uses one optimization epoch,
//! and trains on the complete rollout as one minibatch.

use super::{
    Agent,
    ppo::{
        PPOAgent, PPOConfigurationError, PPOError, PPOModelGroup, PPOModelTypes, PPONetwork,
        SeparatePPONetwork, SharedPPONetwork,
    },
};
use crate::{buffers::experience::ExperienceBatchError, sampling::SamplingError};
use crate::{
    gym::MultiGym,
    parameter_schedule::ParameterSchedule,
    spaces::{ActionMap, ObservationSpace},
    tensor_rank::{NextRank, PrevRank},
};
use bon::bon;
use burn::{
    module::Module,
    optim::ModuleOptimizer,
    tensor::{DType, Device, Tensor, TensorReadError, kind::Autodiff},
};
use std::sync::Arc;

pub use super::ppo::{
    PPOCollectionLogEntry as A2CCollectionLogEntry, PPOEpisodeLogEntry as A2CEpisodeLogEntry,
    PPOLogEntry as A2CLogEntry, PPOLogger as A2CLogger,
};

#[derive(Debug, thiserror::Error)]
pub enum A2CError<AE, GE, SE>
where
    AE: std::fmt::Debug,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    #[error("critic dtype {critic:?} must match environment reward dtype {reward:?}")]
    CriticRewardDTypeMismatch { critic: DType, reward: DType },
    #[error("network failed: {0}")]
    NetworkError(#[source] AE),
    #[error("gym failed: {0}")]
    GymError(#[source] GE),
    #[error("reading A2C tensor data failed: {0}")]
    TensorError(#[from] TensorReadError),
    #[error("rollout batching failed: {0}")]
    ExperienceBatch(#[from] ExperienceBatchError),
    #[error("minibatch sampling failed: {0}")]
    Sampling(#[from] SamplingError),
    #[error("space operation failed: {0}")]
    SpaceError(#[source] SE),
    #[error("invalid A2C configuration: {0}")]
    ConfigurationError(#[source] PPOConfigurationError),
    #[error("termination batches differ in length: {dones} dones and {truncateds} truncations")]
    MismatchedTerminationBatch { dones: usize, truncateds: usize },
    #[error("no prepared rollout is available")]
    MissingPreparedRollout,
    #[error("old value estimates are missing")]
    MissingOldValues,
}

impl<AE: std::fmt::Debug, GE: std::fmt::Debug, SE: std::fmt::Debug> From<PPOError<AE, GE, SE>>
    for A2CError<AE, GE, SE>
{
    fn from(error: PPOError<AE, GE, SE>) -> Self {
        match error {
            PPOError::CriticRewardDTypeMismatch { critic, reward } => {
                Self::CriticRewardDTypeMismatch { critic, reward }
            }
            PPOError::NetworkError(error) => Self::NetworkError(error),
            PPOError::GymError(error) => Self::GymError(error),
            PPOError::TensorError(error) => Self::TensorError(error),
            PPOError::ExperienceBatch(error) => Self::ExperienceBatch(error),
            PPOError::Sampling(error) => Self::Sampling(error),
            PPOError::SpaceError(error) => Self::SpaceError(error),
            PPOError::ConfigurationError(error) => Self::ConfigurationError(error),
            PPOError::MismatchedTerminationBatch { dones, truncateds } => {
                Self::MismatchedTerminationBatch { dones, truncateds }
            }
            PPOError::MissingPreparedRollout => Self::MissingPreparedRollout,
            PPOError::MissingOldValues => Self::MissingOldValues,
        }
    }
}

/// Owns a shared model and separate actor and critic heads for A2C.
pub struct SharedA2CNetwork<T: Module, A: Module, C: Module> {
    inner: SharedPPONetwork<T, A, C>,
}

#[bon]
impl<T: Module, A: Module, C: Module> SharedA2CNetwork<T, A, C> {
    /// Owns the models and optimizer. Model shapes, dtype, and device must meet the PPO network contract.
    /// Rejects nonpositive or non-finite learning rates. Learning-rate schedules must implement `Send + Sync`.
    #[builder]
    pub fn new(
        #[builder(into)] optimizer: ModuleOptimizer,
        learning_rate: f64,
        shared_network: T,
        critic_head: C,
        actor_head: A,
        #[builder(with = |schedule: impl ParameterSchedule + Send + Sync + 'static| Arc::new(schedule))]
        lr_scheduler: Option<Arc<dyn ParameterSchedule + Send + Sync>>,
    ) -> Result<Self, PPOConfigurationError> {
        Ok(Self {
            inner: SharedPPONetwork::builder()
                .optimizer(optimizer)
                .learning_rate(learning_rate)
                .shared_network(shared_network)
                .critic_head(critic_head)
                .actor_head(actor_head)
                .maybe_lr_scheduler(
                    lr_scheduler.map(|schedule| move |progress| schedule.value(progress)),
                )
                .build()?,
        })
    }
}

/// Owns separate actor and critic models for A2C.
pub struct SeparateA2CNetwork<A: Module, C: Module> {
    inner: SeparatePPONetwork<A, C>,
}

#[bon]
impl<A: Module, C: Module> SeparateA2CNetwork<A, C> {
    /// Owns the models and optimizers. Models must meet the separate PPO network contract.
    /// Rejects nonpositive or non-finite learning rates. Learning-rate schedules must implement `Send + Sync`.
    #[builder]
    pub fn new(
        #[builder(into)] actor_optimizer: ModuleOptimizer,
        #[builder(into)] critic_optimizer: ModuleOptimizer,
        actor_learning_rate: f64,
        critic_learning_rate: f64,
        actor_network: A,
        critic_network: C,
        #[builder(with = |schedule: impl ParameterSchedule + Send + Sync + 'static| Arc::new(schedule))]
        actor_lr_scheduler: Option<Arc<dyn ParameterSchedule + Send + Sync>>,
        #[builder(with = |schedule: impl ParameterSchedule + Send + Sync + 'static| Arc::new(schedule))]
        critic_lr_scheduler: Option<Arc<dyn ParameterSchedule + Send + Sync>>,
        #[builder(default = false)] combined_loss: bool,
    ) -> Result<Self, PPOConfigurationError> {
        Ok(Self {
            inner: SeparatePPONetwork::builder()
                .actor_optimizer(actor_optimizer)
                .critic_optimizer(critic_optimizer)
                .actor_learning_rate(actor_learning_rate)
                .critic_learning_rate(critic_learning_rate)
                .actor_network(actor_network)
                .critic_network(critic_network)
                .maybe_actor_lr_scheduler(
                    actor_lr_scheduler.map(|schedule| move |progress| schedule.value(progress)),
                )
                .maybe_critic_lr_scheduler(
                    critic_lr_scheduler.map(|schedule| move |progress| schedule.value(progress)),
                )
                .combined_loss(combined_loss)
                .build()?,
        })
    }
}

/// Contains an owned A2C network configuration.
pub struct A2CNetworkInfo<N> {
    inner: N,
}

impl<T: Module, A: Module, C: Module> A2CNetworkInfo<SharedPPONetwork<T, A, C>> {
    pub fn shared(network: SharedA2CNetwork<T, A, C>) -> Self {
        Self {
            inner: network.inner,
        }
    }
}

impl<A: Module, C: Module> A2CNetworkInfo<SeparatePPONetwork<A, C>> {
    pub fn separate(network: SeparateA2CNetwork<A, C>) -> Self {
        Self {
            inner: network.inner,
        }
    }
}

impl<T: Module, A: Module, C: Module> From<SharedA2CNetwork<T, A, C>>
    for A2CNetworkInfo<SharedPPONetwork<T, A, C>>
{
    fn from(network: SharedA2CNetwork<T, A, C>) -> Self {
        Self::shared(network)
    }
}

impl<A: Module, C: Module> From<SeparateA2CNetwork<A, C>>
    for A2CNetworkInfo<SeparatePPONetwork<A, C>>
{
    fn from(network: SeparateA2CNetwork<A, C>) -> Self {
        Self::separate(network)
    }
}

/// Advantage Actor-Critic agent with one unclipped full-rollout update.
pub struct A2CAgent<'a, G: PPOModelTypes, OS, AS, GE, I = (), const A: usize = 2> {
    inner: PPOAgent<'a, G, OS, AS, GE, I, A>,
}

#[bon]
impl<'a, G: PPOModelTypes, OS, AS, GE, I, const A: usize> A2CAgent<'a, G, OS, AS, GE, I, A> {
    /// Builds A2C with one unclipped optimization epoch over the complete rollout.
    /// Observation batches have rank O; policy actions have rank L and environment actions have rank A.
    /// Models must share a device. Critic outputs must use the environment reward dtype.
    /// Batch size must be divisible by the environment count.
    #[builder]
    pub fn new<N, const O: usize, const H: usize, const L: usize>(
        observation_space: OS,
        action_space: AS,
        network_info: A2CNetworkInfo<N>,
        #[builder(default = 0.99)] gamma: f32,
        #[builder(default = 1.0)] gae_lambda: f32,
        #[builder(default = false)] normalize_advantage: bool,
        #[builder(default = false)] normalize_returns: bool,
        #[builder(default = 0.5)] vf_coef: f32,
        #[builder(default = 0.0)] ent_coef: f32,
        #[builder(default = 1024)] batch_size: usize,
        #[builder(default = 0.5)] gradient_clip: f32,
        training_horizon: usize,
        logging_info: Option<&'a mut dyn A2CLogger<I>>,
        device: Device,
    ) -> Result<Self, PPOConfigurationError>
    where
        N: PPONetwork<O, H, L, InputKind = OS::Kind, ModelGroup = G>,
        G: PPOModelTypes<Network = N, Observation = OS::Batch, Action = Tensor<L>>,
        OS: ObservationSpace<O>,
        AS: ActionMap<L, A>,
        AS::Error: std::fmt::Debug,
        Tensor<O, OS::Kind>: NextRank,
        Tensor<L>: NextRank,
        GE: std::fmt::Debug,
    {
        let inner = PPOAgent::builder()
            .observation_space(observation_space)
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
            .build()?;
        Ok(Self { inner })
    }
}

impl<'a, N, OS, AS, GE, I, const O: usize, const H: usize, const L: usize, const A: usize>
    A2CAgent<
        'a,
        PPOModelGroup<N, Tensor<O, OS::Kind>, Tensor<H, N::LatentKind>, Tensor<L>>,
        OS,
        AS,
        GE,
        I,
        A,
    >
where
    N: PPONetwork<O, H, L, InputKind = OS::Kind>,
    OS: ObservationSpace<O>,
    AS: ActionMap<L, A>,
    AS::Error: std::fmt::Debug,
    OS::Kind: Autodiff,
    Tensor<O, OS::Kind>: PrevRank + NextRank,
    Tensor<L>: NextRank,
    GE: std::fmt::Debug,
{
    pub fn set_learning_rate(&mut self, learning_rate: f64) {
        self.inner.set_learning_rate(learning_rate);
    }

    /// Maps rank-O observation batches to rank-A deterministic environment actions, preserving the batch axis.
    /// Inputs have shape `[batch_size, ...observation_shape]` and must meet the model's dtype and device contract.
    pub fn act_deterministic(
        &mut self,
        observations: &OS::Batch,
    ) -> Result<AS::Batch, A2CError<N::Error, GE, AS::Error>> {
        self.inner
            .act_deterministic(observations)
            .map_err(A2CError::from)
    }

    pub fn reset_current_observations(&mut self) {
        self.inner.reset_current_observations();
    }
}

impl<
    'a,
    N,
    OS,
    AS,
    GE,
    I,
    const O: usize,
    const H: usize,
    const L: usize,
    const A: usize,
    const BO: usize,
    const BL: usize,
    const U: usize,
> Agent<I, O, A>
    for A2CAgent<
        'a,
        PPOModelGroup<N, Tensor<O, OS::Kind>, Tensor<H, N::LatentKind>, Tensor<L>>,
        OS,
        AS,
        GE,
        I,
        A,
    >
where
    N: PPONetwork<O, H, L, InputKind = OS::Kind>,
    OS: ObservationSpace<O, Item = Tensor<U, <OS as crate::tensor_rank::TensorBatchTypes>::Kind>>,
    AS: ActionMap<L, A>,
    AS::Error: std::fmt::Debug,
    OS::Kind: Autodiff,
    Tensor<O, OS::Kind>: PrevRank + NextRank,
    Tensor<L>: NextRank,
    Tensor<O, OS::Kind>: NextRank<Next = Tensor<BO, OS::Kind>>,
    Tensor<L>: NextRank<Next = Tensor<BL>>,
    Tensor<U, OS::Kind>: NextRank<Next = Tensor<O, OS::Kind>>,
    GE: std::fmt::Debug,
{
    type Error = A2CError<N::Error, GE, AS::Error>;
    type GymError = GE;
    type SpaceError = AS::Error;
    type ObservationSpace = OS;
    type ActionSpace = AS;

    /// Maps rank-O observations `[batch_size, ...observation_shape]` to rank-A actions `[batch_size, ...action_shape]`, preserving native kinds and device.
    fn act(&mut self, observations: &OS::Batch) -> Result<AS::Batch, Self::Error> {
        self.inner.act(observations).map_err(A2CError::from)
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            name = "a2c.learn",
            target = "modurl::performance",
            skip_all,
            fields(num_timesteps)
        )
    )]
    /// Trains with rank-O observation batches `[num_envs, ...observation_shape]`.
    /// Terminal observations omit the batch axis and preserve the environment observation kind, dtype, and device.
    fn learn(
        &mut self,
        env: &mut dyn MultiGym<I, O, A, Error = GE, ObservationSpace = OS, ActionSpace = AS>,
        num_timesteps: usize,
    ) -> Result<(), Self::Error> {
        self.inner.learn(env, num_timesteps).map_err(A2CError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        agents::test_support::FixedEnv,
        distributions::CategoricalDistribution,
        gym::VectorizedGymWrapper,
        models::{MLP, probabilistic_model::ProbabilisticPolicyModel},
    };
    use burn::{optim::SgdConfig, tensor::DType};
    use std::sync::Mutex;

    #[derive(Default)]
    struct RecordingLogger {
        updates: usize,
        collections: usize,
        advantages: Vec<f32>,
    }

    impl A2CLogger for RecordingLogger {
        fn log(&mut self, entry: &A2CLogEntry) {
            self.updates += 1;
            assert_eq!(entry.epoch, 0);
            assert_eq!(entry.advantages.dims(), [2, 1]);
            self.advantages = entry.advantages.to_data().try_to_vec().unwrap();
        }

        fn log_collection(&mut self, _entry: &A2CCollectionLogEntry) {
            self.collections += 1;
        }
    }

    fn model(output: usize, device: &Device) -> MLP {
        MLP::builder()
            .input_size(4)
            .output_size(output)
            .hidden_layer_sizes(vec![2])
            .options((device, DType::F32))
            .build()
            .unwrap()
    }

    #[test]
    fn public_agent_runs_one_full_rollout_update_and_forwards_actions() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        let mut env = VectorizedGymWrapper::new(vec![
            FixedEnv::new(device.clone()),
            FixedEnv::new(device.clone()),
        ])
        .unwrap();
        let network = SeparateA2CNetwork::builder()
            .actor_network(ProbabilisticPolicyModel::new(
                model(2, &device),
                CategoricalDistribution,
            ))
            .critic_network(model(1, &device))
            .actor_optimizer(SgdConfig::new().init())
            .critic_optimizer(SgdConfig::new().init())
            .actor_learning_rate(1e-3)
            .critic_learning_rate(1e-3)
            .build()
            .unwrap();
        let mut logger = RecordingLogger::default();
        let mut agent = A2CAgent::builder()
            .observation_space(env.observation_space())
            .action_space(env.action_space())
            .network_info(network.into())
            .batch_size(2)
            .training_horizon(4)
            .logging_info(&mut logger)
            .device(device.clone())
            .build()
            .unwrap();
        agent.learn(&mut env, 2).unwrap();
        let observations = Tensor::zeros([2, 4], (&device, DType::F32));
        assert_eq!(agent.act(&observations).unwrap().dims(), [2, 1]);
        assert_eq!(
            agent.act_deterministic(&observations).unwrap().dims(),
            [2, 1]
        );
        agent.set_learning_rate(0.25);
        agent.reset_current_observations();
        assert_eq!(logger.updates, 1);
        assert_eq!(logger.collections, 1);
        assert!(logger.advantages.iter().any(|value| value.abs() > 1e-6));
    }

    #[test]
    fn shared_configuration_runs_one_full_rollout_update() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        let mut env = VectorizedGymWrapper::new(vec![
            FixedEnv::new(device.clone()),
            FixedEnv::new(device.clone()),
        ])
        .unwrap();
        let network = SharedA2CNetwork::builder()
            .shared_network(model(4, &device))
            .actor_head(ProbabilisticPolicyModel::new(
                model(2, &device),
                CategoricalDistribution,
            ))
            .critic_head(model(1, &device))
            .optimizer(SgdConfig::new().init())
            .learning_rate(1e-3)
            .build()
            .unwrap();
        let mut logger = RecordingLogger::default();
        let mut agent = A2CAgent::builder()
            .observation_space(env.observation_space())
            .action_space(env.action_space())
            .network_info(A2CNetworkInfo::shared(network))
            .batch_size(2)
            .training_horizon(4)
            .logging_info(&mut logger)
            .device(device)
            .build()
            .unwrap();
        agent.learn(&mut env, 2).unwrap();
        assert_eq!(logger.updates, 1);
        assert_eq!(logger.collections, 1);
    }

    #[test]
    fn network_builders_forward_custom_schedules_to_ppo() {
        struct RecordingSchedule {
            progress: Arc<Mutex<Vec<f64>>>,
        }

        impl ParameterSchedule for RecordingSchedule {
            fn value(&self, progress: f64) -> f64 {
                self.progress.lock().unwrap().push(progress);
                1e-3 * (1.0 - progress)
            }
        }

        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        let shared_progress = Arc::new(Mutex::new(Vec::new()));
        let actor_progress = Arc::new(Mutex::new(Vec::new()));
        let critic_progress = Arc::new(Mutex::new(Vec::new()));
        let mut shared = SharedA2CNetwork::builder()
            .shared_network(model(4, &device))
            .actor_head(ProbabilisticPolicyModel::new(
                model(2, &device),
                CategoricalDistribution,
            ))
            .critic_head(model(1, &device))
            .optimizer(SgdConfig::new().init())
            .learning_rate(1e-3)
            .lr_scheduler(RecordingSchedule {
                progress: shared_progress.clone(),
            })
            .build()
            .unwrap();
        let mut separate = SeparateA2CNetwork::builder()
            .actor_network(ProbabilisticPolicyModel::new(
                model(2, &device),
                CategoricalDistribution,
            ))
            .critic_network(model(1, &device))
            .actor_optimizer(SgdConfig::new().init())
            .critic_optimizer(SgdConfig::new().init())
            .actor_learning_rate(1e-3)
            .critic_learning_rate(1e-3)
            .actor_lr_scheduler(RecordingSchedule {
                progress: actor_progress.clone(),
            })
            .critic_lr_scheduler(RecordingSchedule {
                progress: critic_progress.clone(),
            })
            .build()
            .unwrap();
        let mut progress = crate::parameter_schedule::ScheduleProgress::new(8);
        progress.advance_steps(2);
        PPONetwork::<2, 2, 2>::update_learning_rates(&mut shared.inner, &progress);
        PPONetwork::<2, 2, 2>::update_learning_rates(&mut separate.inner, &progress);
        assert_eq!(*shared_progress.lock().unwrap(), [0.25]);
        assert_eq!(*actor_progress.lock().unwrap(), [0.25]);
        assert_eq!(*critic_progress.lock().unwrap(), [0.25]);
    }
}

//! Advantage Actor-Critic implemented as a constrained PPO configuration.
//!
//! A2C disables policy and value clipping, uses one optimization epoch,
//! and trains on the complete rollout as one minibatch.

use super::{
    Agent,
    ppo::{
        PPOAgent, PPOConfigurationError, PPOError, PPOModelGroup, PPOModelTypes, PPONetwork,
        PolicyActions, SeparatePPONetwork, SharedPPONetwork,
    },
};
use crate::{buffers::experience::ExperienceBatchError, sampling::SamplingError};
use crate::{
    gym::{MultiGym, MultiGymStepInfo},
    parameter_schedule::ParameterSchedule,
    spaces::{ActionMap, ObservationSpace},
    tensor_rank::{NextRank, PrevRank},
};
use bon::bon;
use burn::{
    module::Module,
    optim::ModuleOptimizer,
    tensor::{DType, Tensor, TensorReadError, kind::Autodiff},
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
    #[error("batch length mismatch for {field}: expected {expected}, got {actual}")]
    BatchLengthMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
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
            PPOError::BatchLengthMismatch {
                field,
                expected,
                actual,
            } => Self::BatchLengthMismatch {
                field,
                expected,
                actual,
            },
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
        #[builder(with = |schedule: impl ParameterSchedule + Send + Sync + 'static| Arc::new(schedule))]
        learning_rate: Arc<dyn ParameterSchedule + Send + Sync>,
        shared_network: T,
        critic_head: C,
        actor_head: A,
    ) -> Result<Self, PPOConfigurationError> {
        Ok(Self {
            inner: SharedPPONetwork::builder()
                .optimizer(optimizer)
                .learning_rate(move |progress| learning_rate.value(progress))
                .shared_network(shared_network)
                .critic_head(critic_head)
                .actor_head(actor_head)
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
        #[builder(with = |schedule: impl ParameterSchedule + Send + Sync + 'static| Arc::new(schedule))]
        actor_learning_rate: Arc<dyn ParameterSchedule + Send + Sync>,
        #[builder(with = |schedule: impl ParameterSchedule + Send + Sync + 'static| Arc::new(schedule))]
        critic_learning_rate: Arc<dyn ParameterSchedule + Send + Sync>,
        actor_network: A,
        critic_network: C,
        #[builder(default = false)] combined_loss: bool,
    ) -> Result<Self, PPOConfigurationError> {
        Ok(Self {
            inner: SeparatePPONetwork::builder()
                .actor_optimizer(actor_optimizer)
                .critic_optimizer(critic_optimizer)
                .actor_learning_rate(move |progress| actor_learning_rate.value(progress))
                .critic_learning_rate(move |progress| critic_learning_rate.value(progress))
                .actor_network(actor_network)
                .critic_network(critic_network)
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
        training_horizon: Option<usize>,
        logging_info: Option<&'a mut dyn A2CLogger<I>>,
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
            .maybe_training_horizon(training_horizon)
            .maybe_logging_info(logging_info)
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
    /// Replaces the learning-rate schedule with an `f64` or any `ParameterSchedule`, continuing from the current training progress.
    /// Rejects a nonpositive or non-finite rate at the current progress.
    pub fn set_learning_rate(
        &mut self,
        learning_rate: impl ParameterSchedule + Send + Sync + 'static,
    ) -> Result<(), PPOConfigurationError> {
        self.inner.set_learning_rate(learning_rate)
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
    AS: ActionMap<L, A> + Clone,
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
    type ObservationSpace = OS;
    type ActionSpace = AS;
    type RawAction = PolicyActions<AS, A2CError<N::Error, GE, AS::Error>, L, A>;

    /// Selects policy samples `[batch_size, ...policy_shape]` from rank-O observations `[batch_size, ...observation_shape]`.
    fn act_raw(&mut self, observations: &OS::Batch) -> Result<Self::RawAction, Self::Error> {
        Ok(self
            .inner
            .act_raw(observations)
            .map_err(A2CError::from)?
            .with_error(A2CError::SpaceError))
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

    /// Stores a caller-collected batch with observation shape `[num_envs, ...observation_shape]`.
    fn learn_from_timesteps(
        &mut self,
        observations: &OS::Batch,
        raw_actions: Self::RawAction,
        step: MultiGymStepInfo<I, Tensor<U, OS::Kind>>,
    ) -> Result<(), Self::Error> {
        self.inner
            .learn_from_timesteps(
                observations,
                raw_actions.with_error(PPOError::SpaceError),
                step,
            )
            .map_err(A2CError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        agents::{RawAction, test_support::FixedEnv},
        distributions::CategoricalDistribution,
        gym::{VectorizedGymError, VectorizedGymWrapper},
        models::{MLP, probabilistic_model::ProbabilisticPolicyModel},
    };
    use burn::{
        module::{Module, ModuleVisitor, Param},
        optim::SgdConfig,
        tensor::{DType, Device},
    };
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
        let mut agent =
            A2CAgent::<_, _, _, VectorizedGymError<std::convert::Infallible>>::builder()
                .observation_space(env.observation_space())
                .action_space(env.action_space())
                .network_info(network.into())
                .batch_size(2)
                .training_horizon(4)
                .logging_info(&mut logger)
                .build()
                .unwrap();
        agent.learn(&mut env, 2).unwrap();
        let observations = Tensor::zeros([2, 4], (&device, DType::F32));
        assert_eq!(agent.act(&observations).unwrap().dims(), [2, 1]);
        assert_eq!(
            agent.act_deterministic(&observations).unwrap().dims(),
            [2, 1]
        );
        agent.set_learning_rate(0.25).unwrap();
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
            .learning_rate(RecordingSchedule {
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
            .actor_learning_rate(RecordingSchedule {
                progress: actor_progress.clone(),
            })
            .critic_learning_rate(RecordingSchedule {
                progress: critic_progress.clone(),
            })
            .build()
            .unwrap();
        let mut progress = crate::parameter_schedule::ScheduleProgress::new(Some(8));
        progress.advance_steps(2);
        PPONetwork::<2, 2, 2>::update_learning_rates(&mut shared.inner, &progress);
        PPONetwork::<2, 2, 2>::update_learning_rates(&mut separate.inner, &progress);
        assert_eq!(*shared_progress.lock().unwrap(), [0.0, 0.25]);
        assert_eq!(*actor_progress.lock().unwrap(), [0.0, 0.25]);
        assert_eq!(*critic_progress.lock().unwrap(), [0.0, 0.25]);
    }

    #[test]
    fn caller_collected_rollout_rejects_environment_count_changes() {
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
        let mut agent =
            A2CAgent::<_, _, _, VectorizedGymError<std::convert::Infallible>>::builder()
                .observation_space(env.observation_space())
                .action_space(env.action_space())
                .network_info(network.into())
                .batch_size(4)
                .training_horizon(8)
                .build()
                .unwrap();

        let observations = env.reset().unwrap();
        let actions = agent.act_raw(&observations).unwrap();
        let bad_step = env.step(actions.environment().unwrap()).unwrap();
        let bad_actions = PolicyActions::new(
            Tensor::zeros([1, 1], &device),
            env.action_space(),
            A2CError::SpaceError,
        );
        assert!(matches!(
            agent.learn_from_timesteps(&observations, bad_actions, bad_step),
            Err(A2CError::BatchLengthMismatch {
                field: "raw actions",
                expected: 2,
                actual: 1,
            })
        ));

        let observations = env.reset().unwrap();
        let actions = agent.act_raw(&observations).unwrap();
        let step = env.step(actions.environment().unwrap()).unwrap();
        let next_observations = step.observations.clone();
        agent
            .learn_from_timesteps(&observations, actions, step)
            .unwrap();

        let mut single_env = VectorizedGymWrapper::from(FixedEnv::new(device.clone()));
        let single_observations = single_env.reset().unwrap();
        let single_actions = agent.act_raw(&single_observations).unwrap();
        let single_step = single_env
            .step(single_actions.environment().unwrap())
            .unwrap();
        assert!(matches!(
            agent.learn_from_timesteps(&single_observations, single_actions, single_step),
            Err(A2CError::BatchLengthMismatch {
                field: "environment count",
                expected: 2,
                actual: 1,
            })
        ));

        let actions = agent.act_raw(&next_observations).unwrap();
        let step = env.step(actions.environment().unwrap()).unwrap();
        agent
            .learn_from_timesteps(&next_observations, actions, step)
            .unwrap();
    }

    #[test]
    fn caller_collected_rollout_matches_learn() {
        struct Parameters(Vec<f32>);

        impl ModuleVisitor for Parameters {
            /// Visits parameter values while preserving their tensor shape.
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
                self.0
                    .extend(param.val().into_data().try_to_vec::<f32>().unwrap());
            }
        }

        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        let build_env = || {
            VectorizedGymWrapper::new(vec![
                FixedEnv::new(device.clone()),
                FixedEnv::new(device.clone()),
            ])
            .unwrap()
        };
        let build_agent = |observation_space, action_space| {
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
            A2CAgent::<_, _, _, VectorizedGymError<std::convert::Infallible>>::builder()
                .observation_space(observation_space)
                .action_space(action_space)
                .network_info(network.into())
                .batch_size(4)
                .training_horizon(8)
                .build()
                .unwrap()
        };

        device.seed(153);
        let mut learned_env = build_env();
        let mut learned_agent =
            build_agent(learned_env.observation_space(), learned_env.action_space());
        learned_agent.learn(&mut learned_env, 4).unwrap();

        device.seed(153);
        let mut caller_env = build_env();
        let mut caller_agent =
            build_agent(caller_env.observation_space(), caller_env.action_space());
        let mut observations = caller_env.reset().unwrap();
        for _ in 0..2 {
            let actions = caller_agent.act_raw(&observations).unwrap();
            let step = caller_env.step(actions.environment().unwrap()).unwrap();
            let next_observations = step.observations.clone();
            caller_agent
                .learn_from_timesteps(&observations, actions, step)
                .unwrap();
            observations = next_observations;
        }

        let learned_network = learned_agent.inner.network();
        let mut learned_parameters = Parameters(Vec::new());
        learned_network.visit(&mut learned_parameters);
        let learned_latent = learned_network
            .forward_observations(Tensor::zeros([2, 4], &device))
            .unwrap();
        let learned_snapshot = (
            learned_network
                .forward_values(learned_latent.clone())
                .unwrap()
                .into_data(),
            learned_network.mode(learned_latent).unwrap().into_data(),
        );
        let caller_network = caller_agent.inner.network();
        let caller_latent = caller_network
            .forward_observations(Tensor::zeros([2, 4], &device))
            .unwrap();
        let caller_snapshot = (
            caller_network
                .forward_values(caller_latent.clone())
                .unwrap()
                .into_data(),
            caller_network.mode(caller_latent).unwrap().into_data(),
        );
        assert_eq!(learned_snapshot, caller_snapshot);

        let mut caller_parameters = Parameters(Vec::new());
        caller_network.visit(&mut caller_parameters);
        assert_eq!(learned_parameters.0, caller_parameters.0);
    }
}

//! Deep Deterministic Policy Gradient with owned models and replay.

use super::{
    CriticCountRequirement, DeterministicActorCriticAgent,
    DeterministicActorCriticCollectionLogEntry, DeterministicActorCriticConfigurationError,
    DeterministicActorCriticError, DeterministicActorCriticLogEntry,
    DeterministicActorCriticLogger, DeterministicActorCriticModelFor,
    DeterministicActorCriticModelGroup, DeterministicActorCriticModelTypes,
    DeterministicActorCriticStrategy,
};
use crate::{
    agents::{
        Agent, ReplayStorageConfig,
        action_value_critic::{
            ActionValueCritic, ActionValueCriticError, ActionValueCriticNetwork,
        },
    },
    gym::MultiGym,
    models::Forward,
    spaces::{ActionMap, ObservationSpace},
    tensor_rank::{NextRank, PrevRank},
};
use bon::bon;
use burn::{
    module::Module,
    optim::ModuleOptimizer,
    tensor::{DType, Float, Tensor, kind::Autodiff},
};
use std::fmt::Debug;

/// Receives DDPG replay-update and collection metrics.
pub trait DDPGLogger<I = (), const A: usize = 2> {
    /// Records metrics from one replay optimization with rank-A action batches.
    fn log(&mut self, entry: &DeterministicActorCriticLogEntry<A>);

    /// Records metrics from one vectorized environment step.
    fn log_collection(&mut self, _entry: &DeterministicActorCriticCollectionLogEntry<I>) {}
}

struct DDPGLoggingInfo<'a, I, const A: usize> {
    logger: &'a mut dyn DDPGLogger<I, A>,
}

impl<I, const A: usize> DeterministicActorCriticLogger<I, A> for Option<DDPGLoggingInfo<'_, I, A>> {
    fn log_update(&mut self, entry: &DeterministicActorCriticLogEntry<A>) {
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

    /// Returns the sole critic target tensor shaped `[batch, 1]`.
    fn aggregate_target_values(
        &self,
        values: Vec<Tensor<2>>,
    ) -> Result<Tensor<2>, ActionValueCriticError> {
        values
            .into_iter()
            .next()
            .ok_or(ActionValueCriticError::NoCriticValues)
    }

    /// Returns the sole actor-objective Q tensor shaped `[batch_size, 1]`.
    fn aggregate_actor_values(
        &self,
        values: Vec<Tensor<2>>,
    ) -> Result<Tensor<2>, ActionValueCriticError> {
        values
            .into_iter()
            .next()
            .ok_or(ActionValueCriticError::NoCriticValues)
    }
}

/// Trains one deterministic actor and one critic with bounded continuous actions.
/// Native observations keep their kind and dtype. Actions and training statistics use the configured floating-point dtype.
/// Rank O includes observation batch and event axes; rank A includes action batch and event axes.
pub struct DDPGAgent<
    'a,
    G: DeterministicActorCriticModelTypes,
    OS,
    AS,
    GE,
    I = (),
    const A: usize = 2,
> {
    inner: DeterministicActorCriticAgent<G, OS, AS, DDPGStrategy, GE, A>,
    logging_info: Option<DDPGLoggingInfo<'a, I, A>>,
}

#[bon]
impl<'a, G: DeterministicActorCriticModelTypes, OS, AS, GE, I, const A: usize>
    DDPGAgent<'a, G, OS, AS, GE, I, A>
{
    /// Builds an agent with detached target copies of the online actor and critic.
    /// Observations use [batch_size, ...observation_shape]; Float actions use [batch_size, ...action_shape].
    /// Models must use the optimization device. Models perform any observation dtype conversion explicitly.
    /// Returns typed errors for invalid configuration, space shapes, or model devices.
    #[builder]
    pub fn new<M, N, AE: Debug, CE: Debug, SE: Debug, const O: usize, const C: usize>(
        online_actor: M,
        #[builder(into)] actor_optimizer: ModuleOptimizer,
        actor_learning_rate: f64,
        critic: ActionValueCritic<N>,
        action_space: AS,
        observation_space: OS,
        replay_storage_config: ReplayStorageConfig,
        logger: Option<&'a mut dyn DDPGLogger<I, A>>,
        #[builder(default = 0.99)] gamma: f64,
        #[builder(default = 0.005)] tau: f64,
        #[builder(default = 0.1)] exploration_noise: f64,
        #[builder(default = 1_000_000)] replay_capacity: usize,
        #[builder(default = 256)] batch_size: usize,
        #[builder(default = 1)] update_frequency: usize,
        #[builder(default = 1_000)] training_start: usize,
        training_horizon: usize,
        #[builder(default = DType::F32)] dtype: DType,
    ) -> Result<Self, DeterministicActorCriticConfigurationError>
    where
        OS: ObservationSpace<O>,
        OS::Kind: Autodiff,
        AS: ActionMap<A, A, Kind = Float, Error = SE>,
        Tensor<A>: NextRank<Next = Tensor<C>>,
        M: Module
            + Forward<O, A, InputKind = OS::Kind, OutputKind = Float, Error = AE>
            + DeterministicActorCriticModelFor<
                N,
                Tensor<O, OS::Kind>,
                Tensor<A>,
                Tensor<C>,
                ModelGroup = G,
            >,
        N: ActionValueCriticNetwork<
                O,
                A,
                C,
                ObservationKind = OS::Kind,
                ActionKind = Float,
                Error = CE,
            >,
        G: DeterministicActorCriticModelTypes<Actor = M, Critic = N>,
        GE: Debug,
    {
        let inner = DeterministicActorCriticAgent::builder()
            .online_actor(online_actor)
            .actor_optimizer(actor_optimizer)
            .actor_learning_rate(actor_learning_rate)
            .critics(vec![critic])
            .strategy(DDPGStrategy)
            .action_space(action_space)
            .observation_space(observation_space)
            .replay_storage_config(replay_storage_config)
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
}

impl<
    M,
    N,
    OS,
    AS,
    GE,
    I,
    AE: Debug,
    CE: Debug,
    SE: Debug,
    const O: usize,
    const A: usize,
    const C: usize,
>
    DDPGAgent<
        '_,
        DeterministicActorCriticModelGroup<M, N, Tensor<O, OS::Kind>, Tensor<A>, Tensor<C>>,
        OS,
        AS,
        GE,
        I,
        A,
    >
where
    OS: ObservationSpace<O>,
    OS::Kind: Autodiff,
    AS: ActionMap<A, A, Kind = Float, Error = SE>,
    Tensor<A>: NextRank<Next = Tensor<C>>,
    M: Module + Forward<O, A, InputKind = OS::Kind, OutputKind = Float, Error = AE>,
    N: ActionValueCriticNetwork<
            O,
            A,
            C,
            ObservationKind = OS::Kind,
            ActionKind = Float,
            Error = CE,
        >,
    GE: Debug,
{
    /// Returns the bounded action-space contract.
    pub fn get_action_space(&self) -> &AS {
        self.inner.get_action_space()
    }

    /// Returns the native observation-space contract.
    pub fn get_observation_space(&self) -> &OS {
        self.inner.get_observation_space()
    }

    /// Maps native observations [batch_size, ...observation_shape] to Float actions [batch_size, ...action_shape].
    /// Preserves the batch axis and actor gradients. Transfers inputs to the optimization device and applies the action transform without noise.
    pub fn act_deterministic(
        &self,
        observations: &Tensor<O, OS::Kind>,
    ) -> Result<Tensor<A>, DeterministicActorCriticError<AE, GE, SE, CE>> {
        self.inner.act_deterministic(observations)
    }
}

impl<
    M,
    N,
    OS,
    AS,
    GE,
    I,
    K: Autodiff,
    AE: Debug,
    CE: Debug,
    SE: Debug,
    const O: usize,
    const A: usize,
    const C: usize,
    const U: usize,
> Agent<I, O, A>
    for DDPGAgent<
        '_,
        DeterministicActorCriticModelGroup<M, N, Tensor<O, K>, Tensor<A>, Tensor<C>>,
        OS,
        AS,
        GE,
        I,
        A,
    >
where
    OS: ObservationSpace<O, Kind = K, Item = Tensor<U, K>>,
    AS: ActionMap<A, A, Kind = Float, Error = SE>,
    Tensor<A>: NextRank<Next = Tensor<C>>,
    Tensor<U, K>: NextRank<Next = Tensor<O, K>>,
    Tensor<O, K>: PrevRank<Prev = Tensor<U, K>>,
    M: Module + Forward<O, A, InputKind = K, OutputKind = Float, Error = AE>,
    N: ActionValueCriticNetwork<O, A, C, ObservationKind = K, ActionKind = Float, Error = CE>,
    GE: Debug,
{
    type Error = DeterministicActorCriticError<AE, GE, SE, CE>;
    type GymError = GE;
    type SpaceError = SE;
    type ObservationSpace = OS;
    type ActionSpace = AS;

    /// Maps native observations [batch_size, ...observation_shape] to Float exploration actions [batch_size, ...action_shape].
    /// Preserves input kind and dtype; returned actions use the configured dtype and optimization device.
    fn act(&mut self, observations: &Tensor<O, K>) -> Result<Tensor<A>, Self::Error> {
        self.inner.act(observations)
    }

    /// Trains from native observation batches, Float action batches, and rewards [num_envs, 1].
    /// Terminal observations omit the batch axis. Replay and progress persist across calls; truncations bootstrap from terminal observations.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            name = "ddpg.learn",
            target = "modurl::performance",
            skip_all,
            fields(num_timesteps)
        )
    )]
    fn learn(
        &mut self,
        env: &mut dyn MultiGym<I, O, A, Error = GE, ObservationSpace = OS, ActionSpace = AS>,
        num_timesteps: usize,
    ) -> Result<(), Self::Error> {
        self.inner.learn(env, num_timesteps, &mut self.logging_info)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        agents::ReplayDeviceStrategy, agents::action_value_critic::ScalarStateActionCritic,
        agents::test_support::FixedContinuousEnv, gym::VectorizedGymWrapper, models::MLP,
        spaces::BoxSpace,
    };
    use burn::{nn::Tanh, optim::SgdConfig, tensor::Device};

    struct DTypeEnv {
        inner: FixedContinuousEnv,
        dtype: DType,
    }

    impl crate::gym::Gym for DTypeEnv {
        type Error = std::convert::Infallible;
        type ObservationSpace = BoxSpace<2>;
        type ActionSpace = BoxSpace<2>;

        /// Accepts Float actions [1] and returns observations [4] in the fixture dtype; rewards remain F32.
        fn step(&mut self, action: Tensor<1>) -> Result<crate::gym::StepInfo, Self::Error> {
            let mut step = self.inner.step(action)?;
            step.observation = step.observation.cast(self.dtype);
            Ok(step)
        }

        /// Returns native Float observations [4] in the fixture dtype.
        fn reset(&mut self) -> Result<crate::gym::ResetInfo, Self::Error> {
            let mut reset = self.inner.reset()?;
            reset.observation = reset.observation.cast(self.dtype);
            Ok(reset)
        }

        fn observation_space(&self) -> Self::ObservationSpace {
            self.inner.observation_space()
        }

        fn action_space(&self) -> Self::ActionSpace {
            self.inner.action_space()
        }
    }

    #[derive(Default)]
    struct Logger {
        updates: usize,
        collections: usize,
    }

    impl DDPGLogger for Logger {
        fn log(&mut self, entry: &DeterministicActorCriticLogEntry) {
            self.updates += 1;
            assert!(entry.actor_updated);
            assert_eq!(entry.critic_losses.len(), 1);
            assert_eq!(entry.critic_q_values[0].dims(), [1, 1]);
            assert_eq!(entry.bellman_targets.dims(), [1, 1]);
            assert_eq!(entry.policy_actions.as_ref().unwrap().dims(), [1, 1]);
            assert!(!entry.actor_loss.as_ref().unwrap().is_require_grad());
            assert_eq!(entry.actor_learning_rate, 0.01);
        }

        fn log_collection(&mut self, entry: &DeterministicActorCriticCollectionLogEntry) {
            self.collections += 1;
            assert_eq!(entry.collection_rewards.dims(), [2, 1]);
        }
    }

    #[test]
    fn public_builder_and_agent_train_one_critic_every_update() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        for dtype in [DType::F32, DType::F64] {
            let device = Device::flex().autodiff();
            let actor = MLP::builder()
                .input_size(4)
                .output_size(1)
                .hidden_layer_sizes(vec![4])
                .output_activation(Tanh)
                .options((&device, dtype))
                .build()
                .unwrap();
            let critic = ActionValueCritic::builder()
                .online_network(ScalarStateActionCritic::new(
                    MLP::builder()
                        .input_size(5)
                        .output_size(1)
                        .hidden_layer_sizes(vec![4])
                        .options((&device, dtype))
                        .build()
                        .unwrap(),
                ))
                .optimizer(SgdConfig::new().init())
                .learning_rate(0.01)
                .build()
                .unwrap();
            let mut env = VectorizedGymWrapper::new(vec![
                DTypeEnv {
                    inner: FixedContinuousEnv::new(device.clone()),
                    dtype,
                },
                DTypeEnv {
                    inner: FixedContinuousEnv::new(device.clone()),
                    dtype,
                },
            ])
            .unwrap();
            let mut logger = Logger::default();
            let mut agent = DDPGAgent::builder()
                .online_actor(actor)
                .critic(critic)
                .actor_optimizer(SgdConfig::new().init())
                .actor_learning_rate(0.01)
                .action_space(BoxSpace::new_with_universal_bounds(
                    [1, 1],
                    -1.0,
                    1.0,
                    &device,
                ))
                .observation_space(env.observation_space())
                .replay_storage_config(ReplayStorageConfig::new(ReplayDeviceStrategy::OneDevice(
                    device.clone(),
                )))
                .dtype(dtype)
                .exploration_noise(0.0)
                .replay_capacity(4)
                .batch_size(1)
                .training_start(0)
                .training_horizon(4)
                .logger(&mut logger)
                .build()
                .unwrap();
            let observations = Tensor::zeros([2, 4], (&device, dtype));
            let actions = agent.act_deterministic(&observations).unwrap();
            assert_eq!(actions.dims(), [2, 1]);
            assert_eq!(actions.dtype(), dtype);
            assert!(!actions.is_require_grad());
            assert!(matches!(
                agent.act_deterministic(&Tensor::zeros([2, 3], (&device, dtype))),
                Err(DeterministicActorCriticError::Shape { .. })
            ));
            if dtype == DType::F64 {
                assert!(matches!(
                    agent.act_deterministic(&Tensor::zeros([2, 4], (&device, DType::F32))),
                    Err(DeterministicActorCriticError::ActorError(
                        crate::models::ModelError::InputDType { .. }
                    ))
                ));
            }
            agent.learn(&mut env, 2).unwrap();
            agent.learn(&mut env, 2).unwrap();
            assert_eq!(agent.inner.optimization_steps, 4);
            assert_eq!(logger.updates, 4);
            assert_eq!(logger.collections, 2);
        }
    }
}

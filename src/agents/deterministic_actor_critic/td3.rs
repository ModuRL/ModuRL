//! Twin Delayed Deep Deterministic Policy Gradient with a critic ensemble.

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
            ActionValueCritic, ActionValueCriticAggregationMode, ActionValueCriticError,
            ActionValueCriticNetwork, aggregate_critic_values,
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

/// Receives TD3 replay-update and collection metrics.
pub trait TD3Logger<I = (), const A: usize = 2> {
    /// Records metrics from one replay optimization with rank-A action batches.
    fn log(&mut self, entry: &DeterministicActorCriticLogEntry<A>);

    /// Records metrics from one vectorized environment step.
    fn log_collection(&mut self, _entry: &DeterministicActorCriticCollectionLogEntry<I>) {}
}

struct TD3LoggingInfo<'a, I, const A: usize> {
    logger: &'a mut dyn TD3Logger<I, A>,
}

impl<I, const A: usize> DeterministicActorCriticLogger<I, A> for Option<TD3LoggingInfo<'_, I, A>> {
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

pub(crate) struct TD3Strategy {
    target_policy_noise: f64,
    target_noise_clip: f64,
    actor_update_interval: usize,
    target_aggregation_mode: ActionValueCriticAggregationMode,
    actor_aggregation_mode: Option<ActionValueCriticAggregationMode>,
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

    /// Combines critic targets [batch_size, 1] with the configured elementwise aggregation; preserves dtype and device.
    fn aggregate_target_values(
        &self,
        values: Vec<Tensor<2>>,
    ) -> Result<Tensor<2>, ActionValueCriticError> {
        aggregate_critic_values::<2, 3>(&values, self.target_aggregation_mode)
    }

    /// Combines actor-objective Q tensors shaped `[batch_size, 1]` using the
    /// configured mode, or selects the first tensor for canonical TD3.
    fn aggregate_actor_values(
        &self,
        values: Vec<Tensor<2>>,
    ) -> Result<Tensor<2>, ActionValueCriticError> {
        match self.actor_aggregation_mode {
            Some(mode) => aggregate_critic_values::<2, 3>(&values, mode),
            None => values
                .into_iter()
                .next()
                .ok_or(ActionValueCriticError::NoCriticValues),
        }
    }
}

/// Trains one deterministic actor and a nonempty critic ensemble with bounded continuous actions.
/// Targets use the minimum Q value by default; the actor uses the first critic.
/// Target-policy noise defaults to 0.2 with a 0.5 componentwise clip. Actor and target updates occur every second replay update.
/// Native observations keep their kind and dtype. Actions and training statistics use the configured floating-point dtype.
/// Rank O includes observation batch and event axes; rank A includes action batch and event axes.
pub struct TD3Agent<
    'a,
    G: DeterministicActorCriticModelTypes,
    OS,
    AS,
    GE,
    I = (),
    const A: usize = 2,
> {
    inner: DeterministicActorCriticAgent<G, OS, AS, TD3Strategy, GE, A>,
    logging_info: Option<TD3LoggingInfo<'a, I, A>>,
}

#[bon]
impl<'a, G: DeterministicActorCriticModelTypes, OS, AS, GE, I, const A: usize>
    TD3Agent<'a, G, OS, AS, GE, I, A>
{
    /// Builds an agent with detached target copies of the online actor and each critic.
    /// Observations use [batch_size, ...observation_shape]; Float actions use [batch_size, ...action_shape].
    /// Models must use the optimization device. Models perform any observation dtype conversion explicitly.
    /// Returns typed errors for invalid configuration, space shapes, or model devices.
    #[builder]
    pub fn builder<M, N, AE: Debug, CE: Debug, SE: Debug, const O: usize, const C: usize>(
        online_actor: M,
        #[builder(into)] actor_optimizer: ModuleOptimizer,
        actor_learning_rate: f64,
        critics: Vec<ActionValueCritic<N>>,
        action_space: AS,
        observation_space: OS,
        replay_storage_config: ReplayStorageConfig,
        logger: Option<&'a mut dyn TD3Logger<I, A>>,
        #[builder(default = 0.99)] gamma: f64,
        #[builder(default = 0.005)] tau: f64,
        #[builder(default = 0.1)] exploration_noise: f64,
        /// Standard deviation of Gaussian noise applied only to target policy outputs.
        #[builder(default = 0.2)]
        target_policy_noise: f64,
        /// Componentwise absolute limit for target-policy noise.
        #[builder(default = 0.5)]
        target_noise_clip: f64,
        /// Replay-update interval between actor and target updates.
        #[builder(default = 2)]
        actor_update_interval: usize,
        /// Elementwise combination of target Q values [batch_size, 1].
        #[builder(default = ActionValueCriticAggregationMode::Min)]
        target_aggregation_mode: ActionValueCriticAggregationMode,
        /// Optional combination of actor-objective Q values [batch_size, 1]; omission selects the first critic.
        actor_aggregation_mode: Option<ActionValueCriticAggregationMode>,
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
        let strategy = TD3Strategy {
            target_policy_noise,
            target_noise_clip,
            actor_update_interval,
            target_aggregation_mode,
            actor_aggregation_mode,
        };

        let inner = DeterministicActorCriticAgent::builder()
            .online_actor(online_actor)
            .actor_optimizer(actor_optimizer)
            .actor_learning_rate(actor_learning_rate)
            .critics(critics)
            .strategy(strategy)
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
            logging_info: logger.map(|logger| TD3LoggingInfo { logger }),
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
    TD3Agent<
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
    for TD3Agent<
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
            step.observation = step.observation.cast(self.dtype) + 1.0;
            Ok(step)
        }

        /// Returns native Float observations [4] in the fixture dtype.
        fn reset(&mut self) -> Result<crate::gym::ResetInfo, Self::Error> {
            let mut reset = self.inner.reset()?;
            reset.observation = reset.observation.cast(self.dtype) + 1.0;
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
        actor_updates: Vec<bool>,
    }

    impl TD3Logger for Logger {
        fn log(&mut self, entry: &DeterministicActorCriticLogEntry) {
            self.updates += 1;
            self.actor_updates.push(entry.actor_updated);
            assert_eq!(entry.critic_losses.len(), 3);
            assert_eq!(entry.critic_q_values[0].dims(), [1, 1]);
            assert_eq!(entry.bellman_targets.dims(), [1, 1]);
            assert_eq!(entry.policy_actions.is_some(), entry.actor_updated);
            assert_eq!(entry.actor_loss.is_some(), entry.actor_updated);
            assert_eq!(entry.policy_q_values.is_some(), entry.actor_updated);
            assert_eq!(entry.actor_learning_rate, 0.01);
        }

        fn log_collection(&mut self, entry: &DeterministicActorCriticCollectionLogEntry) {
            self.collections += 1;
            assert_eq!(entry.collection_rewards.dims(), [2, 1]);
        }
    }

    #[test]
    fn builder_rejects_empty_ensembles_and_invalid_target_settings() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        let actor = || {
            MLP::builder()
                .input_size(4)
                .output_size(1)
                .hidden_layer_sizes(vec![4])
                .options((&device, DType::F32))
                .build()
                .unwrap()
        };
        let critic = || {
            ActionValueCritic::builder()
                .online_network(ScalarStateActionCritic::new(
                    MLP::builder()
                        .input_size(5)
                        .output_size(1)
                        .hidden_layer_sizes(vec![4])
                        .options((&device, DType::F32))
                        .build()
                        .unwrap(),
                ))
                .optimizer(SgdConfig::new().init())
                .learning_rate(0.01)
                .build()
                .unwrap()
        };
        let build = |critics, interval, noise, clip| {
            TD3Agent::<_, _, _, std::convert::Infallible>::builder()
                .online_actor(actor())
                .actor_optimizer(SgdConfig::new().init())
                .actor_learning_rate(0.01)
                .critics(critics)
                .actor_update_interval(interval)
                .target_policy_noise(noise)
                .target_noise_clip(clip)
                .action_space(BoxSpace::new_with_universal_bounds(
                    [1, 1],
                    -1.0,
                    1.0,
                    &device,
                ))
                .observation_space(BoxSpace::new_unbounded([1, 4], &device))
                .replay_storage_config(ReplayStorageConfig::new(ReplayDeviceStrategy::OneDevice(
                    device.clone(),
                )))
                .training_horizon(4)
                .build()
        };
        assert!(matches!(
            build(vec![], 2, 0.2, 0.5),
            Err(DeterministicActorCriticConfigurationError::NoCritics)
        ));
        assert!(matches!(
            build(vec![critic()], 0, 0.2, 0.5),
            Err(DeterministicActorCriticConfigurationError::ZeroActorUpdateInterval)
        ));
        assert!(matches!(
            build(vec![critic()], 2, f64::NAN, 0.5),
            Err(DeterministicActorCriticConfigurationError::InvalidTargetPolicyNoise)
        ));
        assert!(matches!(
            build(vec![critic()], 2, 0.2, -0.5),
            Err(DeterministicActorCriticConfigurationError::InvalidTargetNoiseClip)
        ));
    }

    #[test]
    fn aggregation_preserves_modes_and_canonical_actor_selection() {
        let device = Device::flex();
        for dtype in [DType::F32, DType::F64] {
            let values = || {
                vec![1.0, 3.0, 2.0]
                    .into_iter()
                    .map(|value| Tensor::full([2, 1], value, (&device, dtype)))
                    .collect()
            };
            for (mode, expected) in [
                (ActionValueCriticAggregationMode::Min, 1.0),
                (ActionValueCriticAggregationMode::Mean, 2.0),
                (ActionValueCriticAggregationMode::Median, 2.0),
                (ActionValueCriticAggregationMode::Max, 3.0),
            ] {
                let strategy = TD3Strategy {
                    target_policy_noise: 0.2,
                    target_noise_clip: 0.5,
                    actor_update_interval: 2,
                    target_aggregation_mode: mode,
                    actor_aggregation_mode: Some(mode),
                };
                for result in [
                    strategy.aggregate_target_values(values()).unwrap(),
                    strategy.aggregate_actor_values(values()).unwrap(),
                ] {
                    assert_eq!(result.dims(), [2, 1]);
                    assert_eq!(result.dtype(), dtype);
                    assert_eq!(
                        result
                            .into_data()
                            .convert::<f64>()
                            .try_to_vec::<f64>()
                            .unwrap(),
                        [expected, expected]
                    );
                }
                assert!(matches!(
                    strategy.aggregate_target_values(vec![]),
                    Err(ActionValueCriticError::NoCriticValues)
                ));
            }
            let strategy = TD3Strategy {
                target_policy_noise: 0.2,
                target_noise_clip: 0.5,
                actor_update_interval: 2,
                target_aggregation_mode: ActionValueCriticAggregationMode::Min,
                actor_aggregation_mode: None,
            };
            assert_eq!(
                strategy
                    .aggregate_actor_values(values())
                    .unwrap()
                    .into_data()
                    .convert::<f64>()
                    .try_to_vec::<f64>()
                    .unwrap(),
                [1.0, 1.0]
            );
            assert!(matches!(
                strategy.aggregate_actor_values(vec![]),
                Err(ActionValueCriticError::NoCriticValues)
            ));
        }
    }

    #[test]
    fn public_builder_trains_ensembles_and_delays_actor_and_targets() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        for dtype in [DType::F32, DType::F64] {
            let device = Device::flex().autodiff();
            let actor = MLP::builder()
                .input_size(4)
                .output_size(1)
                .hidden_layer_sizes(vec![4])
                .hidden_initializer(burn::nn::Initializer::Constant { value: 0.1 })
                .output_initializer(burn::nn::Initializer::Constant { value: 0.1 })
                .output_activation(Tanh)
                .options((&device, dtype))
                .build()
                .unwrap();
            let critic = || {
                ActionValueCritic::builder()
                    .online_network(ScalarStateActionCritic::new(
                        MLP::builder()
                            .input_size(5)
                            .output_size(1)
                            .hidden_layer_sizes(vec![4])
                            .hidden_initializer(burn::nn::Initializer::Constant { value: 0.1 })
                            .output_initializer(burn::nn::Initializer::Constant { value: 0.1 })
                            .options((&device, dtype))
                            .build()
                            .unwrap(),
                    ))
                    .optimizer(SgdConfig::new().init())
                    .learning_rate(0.01)
                    .build()
                    .unwrap()
            };
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
            let mut agent = TD3Agent::builder()
                .online_actor(actor)
                .critics(vec![critic(), critic(), critic()])
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
                .actor_update_interval(3)
                .target_aggregation_mode(ActionValueCriticAggregationMode::Mean)
                .actor_aggregation_mode(ActionValueCriticAggregationMode::Median)
                .logger(&mut logger)
                .build()
                .unwrap();
            let observations = Tensor::ones([2, 4], (&device, dtype));
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
            let target_before = agent
                .inner
                .target_actor
                .forward(observations.clone())
                .unwrap()
                .into_data();
            agent.learn(&mut env, 2).unwrap();
            assert_eq!(
                agent
                    .inner
                    .target_actor
                    .forward(observations.clone())
                    .unwrap()
                    .into_data(),
                target_before
            );
            agent.learn(&mut env, 2).unwrap();
            assert_ne!(
                agent
                    .inner
                    .target_actor
                    .forward(observations.clone())
                    .unwrap()
                    .into_data(),
                target_before
            );
            assert_eq!(agent.inner.optimization_steps, 4);
            assert_eq!(logger.updates, 4);
            assert_eq!(logger.collections, 2);
            assert_eq!(logger.actor_updates, [false, false, true, false]);
        }
    }
}

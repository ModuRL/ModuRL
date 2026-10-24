use super::*;
use crate::{
    agents::{
        ReplayDeviceStrategy,
        action_value_critic::{
            ActionValueCriticAggregationMode, ScalarStateActionCritic, aggregate_critic_values,
        },
    },
    buffers::experience_replay::ReplayStorage,
    gym::{Gym, ResetInfo, StepInfo, VectorizedGymWrapper},
    spaces::BoxSpace,
};
use burn::{
    module::Param,
    optim::SgdConfig,
    tensor::{Bool, Device, Float},
};
use std::convert::Infallible;

#[test]
fn validates_shared_and_strategy_configuration() {
    assert_eq!(
        DeterministicActorCriticConfigurationValidator::validate_configuration()
            .actor_learning_rate(0.01)
            .dtype(DType::F32)
            .critic_count_requirement(CriticCountRequirement::Exact(2))
            .actual_critic_count(1)
            .replay_capacity(10)
            .batch_size(2)
            .gamma(0.99)
            .tau(0.005)
            .exploration_noise(0.1)
            .target_policy_noise(0.2)
            .target_noise_clip(0.5)
            .actor_update_interval(2)
            .update_frequency(1)
            .training_horizon(10)
            .call(),
        Err(
            DeterministicActorCriticConfigurationError::IncorrectCriticCount {
                expected: 2,
                actual: 1,
            }
        )
    );
    assert_eq!(
        DeterministicActorCriticConfigurationValidator::validate_configuration()
            .actor_learning_rate(0.01)
            .dtype(DType::F32)
            .critic_count_requirement(CriticCountRequirement::NonEmpty)
            .actual_critic_count(0)
            .replay_capacity(10)
            .batch_size(2)
            .gamma(0.99)
            .tau(0.005)
            .exploration_noise(0.1)
            .target_policy_noise(0.0)
            .target_noise_clip(0.0)
            .actor_update_interval(1)
            .update_frequency(1)
            .training_horizon(10)
            .call(),
        Err(DeterministicActorCriticConfigurationError::NoCritics)
    );
    assert_eq!(
        DeterministicActorCriticConfigurationValidator::validate_configuration()
            .actor_learning_rate(0.01)
            .dtype(DType::F32)
            .critic_count_requirement(CriticCountRequirement::Exact(1))
            .actual_critic_count(1)
            .replay_capacity(10)
            .batch_size(2)
            .gamma(0.99)
            .tau(0.005)
            .exploration_noise(-0.1)
            .target_policy_noise(0.0)
            .target_noise_clip(0.0)
            .actor_update_interval(1)
            .update_frequency(1)
            .training_horizon(10)
            .call(),
        Err(DeterministicActorCriticConfigurationError::InvalidExplorationNoise)
    );
    assert_eq!(
        DeterministicActorCriticConfigurationValidator::validate_configuration()
            .actor_learning_rate(0.01)
            .dtype(DType::F32)
            .critic_count_requirement(CriticCountRequirement::NonEmpty)
            .actual_critic_count(3)
            .replay_capacity(10)
            .batch_size(2)
            .gamma(0.99)
            .tau(0.005)
            .exploration_noise(0.1)
            .target_policy_noise(0.0)
            .target_noise_clip(0.0)
            .actor_update_interval(1)
            .update_frequency(1)
            .training_horizon(10)
            .call(),
        Ok(())
    );
}

#[test]
fn zero_noise_preserves_values() {
    let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
    let values = Tensor::<2>::from_data([[1.0f32, 2.0]], &Device::flex());
    assert_eq!(
        add_gaussian_noise(&values, 0.0, Some(0.0))
            .into_data()
            .try_to_vec::<f32>()
            .unwrap(),
        [1.0, 2.0]
    );
}

#[derive(Module, Debug)]
struct LinearModel {
    weight: Param<Tensor<2>>,
}

impl Forward<2, 2> for LinearModel {
    type InputKind = Float;
    type OutputKind = Float;
    type Error = Infallible;

    /// Maps Float observations [batch_size, features] through weights [features, outputs] to Float [batch_size, outputs].
    fn forward(&self, input: Tensor<2>) -> Result<Tensor<2>, Self::Error> {
        Ok(input.matmul(self.weight.val()))
    }
}

struct TestStrategy {
    interval: usize,
    noise: f64,
    clip: f64,
}

impl DeterministicActorCriticStrategy for TestStrategy {
    fn critic_count_requirement(&self) -> CriticCountRequirement {
        CriticCountRequirement::NonEmpty
    }

    fn actor_update_interval(&self) -> usize {
        self.interval
    }

    fn target_policy_noise(&self) -> f64 {
        self.noise
    }

    fn target_noise_clip(&self) -> f64 {
        self.clip
    }

    /// Combines target Q values [batch_size, 1] by minimum, preserving the scalar axis and dtype.
    fn aggregate_target_values(
        &self,
        values: Vec<Tensor<2>>,
    ) -> Result<Tensor<2>, ActionValueCriticError> {
        aggregate_critic_values::<2, 3>(&values, ActionValueCriticAggregationMode::Min)
    }

    /// Uses the first critic's Q values [batch_size, 1] for the actor loss, preserving gradients.
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

fn linear_model(weights: &[f64], device: &Device, dtype: DType) -> LinearModel {
    LinearModel {
        weight: Param::from_tensor(Tensor::from_data(
            TensorData::new(weights.to_vec(), [weights.len(), 1]),
            (device, dtype),
        )),
    }
}

fn critic(
    device: &Device,
    dtype: DType,
) -> ActionValueCritic<ScalarStateActionCritic<LinearModel>> {
    ActionValueCritic::builder()
        .online_network(ScalarStateActionCritic::new(linear_model(
            &[0.5, 1.0],
            device,
            dtype,
        )))
        .optimizer(SgdConfig::new().init())
        .learning_rate(0.01)
        .build()
        .unwrap()
}

fn agent(
    device: &Device,
    dtype: DType,
    interval: usize,
) -> DeterministicActorCriticAgent<
    DeterministicActorCriticModelGroup<LinearModel, ScalarStateActionCritic<LinearModel>>,
    BoxSpace<2>,
    BoxSpace<2>,
    TestStrategy,
    crate::gym::VectorizedGymError<Infallible>,
> {
    DeterministicActorCriticAgent::builder()
        .online_actor(linear_model(&[0.25], device, dtype))
        .actor_optimizer(SgdConfig::new().init())
        .actor_learning_rate(0.01)
        .critics(vec![critic(device, dtype), critic(device, dtype)])
        .strategy(TestStrategy {
            interval,
            noise: 0.0,
            clip: 0.0,
        })
        .action_space(BoxSpace::new_with_universal_bounds(
            [1, 1],
            -1.0,
            1.0,
            device,
        ))
        .observation_space(BoxSpace::new_unbounded([1, 1], device))
        .replay_storage_config(ReplayStorageConfig::new(ReplayDeviceStrategy::OneDevice(
            device.clone(),
        )))
        .gamma(0.9)
        .tau(0.25)
        .exploration_noise(0.0)
        .training_start(0)
        .replay_capacity(8)
        .batch_size(2)
        .update_frequency(2)
        .training_horizon(20)
        .dtype(dtype)
        .build()
        .unwrap()
}

/// Reads rank-D Float tensor values with any shape into flattened F64 host values.
fn numbers<const D: usize>(values: Tensor<D>) -> Vec<f64> {
    values
        .into_data()
        .convert::<f64>()
        .try_to_vec::<f64>()
        .unwrap()
}

#[derive(Default)]
struct Logger {
    updates: Vec<usize>,
    actor_updated: Vec<bool>,
    rewards: Vec<Vec<f64>>,
    episodes: Vec<(f32, usize, bool, bool, usize)>,
    replay_lengths: Vec<usize>,
}

impl DeterministicActorCriticLogger for Logger {
    fn log_update(&mut self, entry: &DeterministicActorCriticLogEntry) {
        self.updates.push(entry.collection_timestep);
        self.actor_updated.push(entry.actor_updated);
        assert!(
            entry
                .critic_losses
                .iter()
                .all(|loss| loss.dims() == [1] && !loss.is_require_grad())
        );
        assert!(
            entry
                .critic_q_values
                .iter()
                .all(|values| values.dims()[1] == 1 && !values.is_require_grad())
        );
        assert_eq!(entry.actor_loss.is_some(), entry.actor_updated);
        assert_eq!(entry.policy_q_values.is_some(), entry.actor_updated);
        assert_eq!(entry.policy_actions.is_some(), entry.actor_updated);
        assert_eq!(entry.actor_learning_rate, 0.01);
        assert_eq!(entry.critic_learning_rates, [0.01, 0.01]);
        assert_eq!(entry.exploration_noise_standard_deviation, 0.0);
        assert_eq!(entry.update_index + 1, self.updates.len());
        assert_eq!(entry.replay_actions.dims(), [2, 1]);
        assert_eq!(entry.replay_rewards.dims(), [2, 1]);
        assert_eq!(entry.bellman_targets.dims(), [2, 1]);
        assert!(!entry.bellman_targets.is_require_grad());
    }

    fn log_collection(&mut self, entry: &DeterministicActorCriticCollectionLogEntry) {
        self.rewards.push(numbers(entry.collection_rewards.clone()));
        self.replay_lengths.push(entry.replay_len);
        assert_eq!(entry.infos.len(), 2);
        for episode in &entry.completed_episodes {
            self.episodes.push((
                episode.episode_return,
                episode.episode_length,
                episode.terminated,
                episode.truncated,
                episode.collection_timestep,
            ));
            assert!(episode.environment_index < 2);
        }
    }
}

struct EpisodeEnv {
    device: Device,
    truncate: bool,
    steps: usize,
}

impl Gym for EpisodeEnv {
    type Error = Infallible;
    type ObservationSpace = BoxSpace<2>;
    type ActionSpace = BoxSpace<2>;

    /// Accepts Float action [1] and returns Float terminal-capable observation [1] on the CPU device.
    fn step(&mut self, action: Tensor<1>) -> Result<StepInfo, Self::Error> {
        assert_eq!(action.dims(), [1]);
        assert!(!action.is_autodiff());
        self.steps += 1;
        Ok(StepInfo {
            observation: Tensor::from_data([self.steps as f64], (&self.device, DType::F64)),
            reward: 2.0,
            done: self.steps == 2 && !self.truncate,
            truncated: self.steps == 2 && self.truncate,
            info: (),
        })
    }

    /// Returns Float observation [1] after resetting the episode counter on the CPU device.
    fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
        self.steps = 0;
        Ok(ResetInfo {
            observation: Tensor::zeros([1], (&self.device, DType::F64)),
            info: (),
        })
    }

    fn observation_space(&self) -> BoxSpace<2> {
        BoxSpace::new_unbounded([1, 1], &self.device)
    }

    fn action_space(&self) -> BoxSpace<2> {
        BoxSpace::new_with_universal_bounds([1, 1], -1.0, 1.0, &self.device)
    }
}

#[test]
fn full_mse_targets_and_actor_gradient_boundaries_are_preserved() {
    let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
    for dtype in [DType::F32, DType::F64] {
        let device = Device::flex().autodiff();
        let mut agent = agent(&device, dtype, 1);
        let batch = DeterministicBatch {
            observations: Tensor::from_data([[1.0f64], [2.0]], (&device, dtype)),
            next_observations: Tensor::from_data([[2.0f64], [3.0]], (&device, dtype)),
            actions: Tensor::from_data([[0.1f64], [0.2]], (&device, dtype)),
            rewards: Tensor::from_data([[1.0f64], [2.0]], (&device, dtype)),
            terminated: Tensor::from_data([[false], [true]], &device),
        };
        let targets = agent.bellman_targets(&batch).unwrap();
        assert_eq!(targets.dtype(), dtype);
        assert!(!targets.is_require_grad());
        for (actual, expected) in numbers(targets.clone()).into_iter().zip([2.35, 2.0]) {
            assert!((actual - expected).abs() < 1e-6);
        }
        let actor_before = agent.online_actor.weight.val().into_data();
        let critic_before = numbers(
            agent.critics[0]
                .online_replay_values::<2, 2, 3>(&batch.observations, &batch.actions)
                .unwrap(),
        );
        let critic_update = agent.optimize_critics(&batch, &targets).unwrap();
        let expected_loss =
            ((critic_before[0] - 2.35).powi(2) + (critic_before[1] - 2.0).powi(2)) / 2.0;
        assert!((numbers(critic_update.losses[0].clone())[0] - expected_loss).abs() < 1e-6);
        assert_eq!(agent.online_actor.weight.val().into_data(), actor_before);
        let before = numbers(
            agent.critics[0]
                .online_replay_values::<2, 2, 3>(&batch.observations, &batch.actions)
                .unwrap(),
        );
        let actor_update = agent.optimize_actor(&batch.observations).unwrap();
        assert_eq!(actor_update.loss.dims(), [1]);
        assert_ne!(agent.online_actor.weight.val().into_data(), actor_before);
        assert_eq!(
            numbers(
                agent.critics[0]
                    .online_replay_values::<2, 2, 3>(&batch.observations, &batch.actions)
                    .unwrap()
            ),
            before
        );
        assert_eq!(agent.target_actor.weight.val().into_data(), actor_before);
        agent.update_targets().unwrap();
        assert_eq!(agent.target_actor.weight.id, agent.online_actor.weight.id);
        assert!(!agent.target_actor.weight.val().is_require_grad());
        let expected = 0.25 * numbers(agent.online_actor.weight.val())[0] + 0.75 * 0.25;
        assert!((numbers(agent.target_actor.weight.val())[0] - expected).abs() < 1e-6);
    }
}

#[test]
fn collection_keeps_delay_cadence_terminal_targets_and_episode_metrics() {
    let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
    let device = Device::flex().autodiff();
    let mut agent = agent(&device, DType::F64, 2);
    let env_device = device.clone().without_autodiff();
    let mut env = VectorizedGymWrapper::new(vec![
        EpisodeEnv {
            device: env_device.clone(),
            truncate: false,
            steps: 0,
        },
        EpisodeEnv {
            device: env_device.clone(),
            truncate: true,
            steps: 0,
        },
    ])
    .unwrap();
    let initial = agent.target_actor.weight.val().into_data();
    let mut logger = Logger::default();
    agent.learn(&mut env, 4, &mut logger).unwrap();
    assert_eq!(agent.schedule_progress.elapsed_steps(), 4);
    assert_eq!(logger.updates, [2, 4]);
    assert_eq!(logger.actor_updated, [false, true]);
    assert_eq!(logger.rewards, [vec![2.0, 2.0], vec![2.0, 2.0]]);
    assert_eq!(
        logger.episodes,
        [(4.0, 2, true, false, 3), (4.0, 2, false, true, 4)]
    );
    assert_eq!(logger.replay_lengths, [2, 4]);
    assert_ne!(agent.target_actor.weight.val().into_data(), initial);
    assert_eq!(agent.get_action_space().shape(), [1]);
    assert_eq!(agent.get_observation_space().shape(), [1]);
    let batch = agent
        .replay
        .as_mut()
        .unwrap()
        .storage_mut()
        .gather(&[2, 3])
        .unwrap();
    assert_eq!(numbers(batch.next_observations), [2.0, 2.0]);
    assert_eq!(
        batch
            .terminated
            .into_data()
            .iter::<bool>()
            .collect::<Vec<_>>(),
        [true, false]
    );
    agent.learn(&mut env, 2, &mut logger).unwrap();
    assert_eq!(logger.updates, [2, 4, 6]);
    assert_eq!(logger.actor_updated, [false, true, false]);
    assert_eq!(agent.schedule_progress.elapsed_steps(), 6);
    let len = agent.replay.as_ref().unwrap().len();
    let mut single = VectorizedGymWrapper::from(EpisodeEnv {
        device: env_device,
        truncate: false,
        steps: 0,
    });
    assert!(matches!(
        agent.learn(&mut single, 1, &mut logger),
        Err(DeterministicActorCriticError::ReplayStorageError(
            DeterministicReplayError::Storage(ReplayStorageError::EnvironmentCountMismatch {
                expected: 2,
                actual: 1
            })
        ))
    ));
    assert_eq!(agent.replay.as_ref().unwrap().len(), len);
}

#[test]
fn noise_is_seeded_clipped_and_keeps_float_dtypes() {
    let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
    let device = Device::flex().autodiff();
    for dtype in [DType::F32, DType::F64] {
        let values = Tensor::<2>::zeros([8, 2], (&device, dtype)).require_grad();
        device.seed(18);
        let first = add_gaussian_noise(&values, 0.5, Some(0.1));
        device.seed(18);
        assert_eq!(
            first.clone().into_data(),
            add_gaussian_noise(&values, 0.5, Some(0.1)).into_data()
        );
        assert!(
            numbers(first.clone())
                .iter()
                .all(|value| value.abs() <= 0.100001)
        );
        assert_eq!(first.dtype(), dtype);
        assert_eq!(
            numbers(values.grad(&first.sum().backward()).unwrap()),
            [1.0; 16]
        );
    }
}

struct BooleanSpace;

impl crate::tensor_rank::TensorBatchTypes for BooleanSpace {
    type Kind = Bool;
    type Item = Tensor<2, Bool>;
    type Batch = Tensor<3, Bool>;
}

impl ObservationSpace<3> for BooleanSpace {
    /// Checks native Bool observations [batch_size, 1, 1] without changing axes or dtype.
    fn contains(&self, values: &Tensor<3, Bool>) -> bool {
        values.dims()[1..] == [1, 1]
    }

    fn shape(&self) -> Vec<usize> {
        vec![1, 1]
    }
}

#[derive(Module, Debug)]
struct BooleanActor {
    weight: Param<Tensor<2>>,
}

impl Forward<3, 2> for BooleanActor {
    type InputKind = Bool;
    type OutputKind = Float;
    type Error = Infallible;

    /// Converts native Bool observations [batch_size, 1, 1] explicitly to Float features [batch_size, 1] and returns Float actions [batch_size, 1].
    fn forward(&self, input: Tensor<3, Bool>) -> Result<Tensor<2>, Self::Error> {
        let count = input.dims()[0];
        Ok(input
            .float()
            .cast(self.weight.val().dtype())
            .reshape([count, 1])
            .matmul(self.weight.val()))
    }
}

#[derive(Module, Debug)]
struct BooleanCritic {
    weight: Param<Tensor<2>>,
}

impl ActionValueCriticNetwork<3, 2, 3> for BooleanCritic {
    type ObservationKind = Bool;
    type ActionKind = Float;
    type Error = Infallible;

    /// Converts native Bool observations [batch_size, 1, 1] to Float and joins actions [batch_size, 1], returning Q values [batch_size, 1].
    fn replay_values(
        &self,
        observations: &Tensor<3, Bool>,
        actions: &Tensor<2>,
    ) -> Result<Tensor<2>, Self::Error> {
        let count = observations.dims()[0];
        Ok(Tensor::cat(
            vec![
                observations
                    .clone()
                    .float()
                    .cast(self.weight.val().dtype())
                    .reshape([count, 1]),
                actions.clone(),
            ],
            1,
        )
        .matmul(self.weight.val()))
    }

    /// Broadcasts native Bool observations [batch_size, 1, 1] over Float candidates [batch_size, candidate_count, 1].
    /// Returns Q values [batch_size, candidate_count, 1] in the parameter dtype, preserving gradients.
    fn policy_values(
        &self,
        observations: &Tensor<3, Bool>,
        actions: &Tensor<3>,
    ) -> Result<Tensor<3>, Self::Error> {
        let [count, candidates, _] = actions.dims();
        let features = observations
            .clone()
            .float()
            .cast(self.weight.val().dtype())
            .reshape([count, 1, 1])
            .expand([count, candidates, 1]);
        Ok(Tensor::cat(vec![features, actions.clone()], 2)
            .reshape([count * candidates, 2])
            .matmul(self.weight.val())
            .reshape([count, candidates, 1]))
    }

    /// Returns Q values [batch_size, candidate_count, 1] from native Bool observations and Float candidates with the policy layouts.
    /// Freezes critic parameters while retaining candidate-action gradients.
    fn actor_values(
        &self,
        observations: &Tensor<3, Bool>,
        actions: &Tensor<3>,
    ) -> Result<Tensor<3>, Self::Error> {
        self.clone()
            .set_require_grad(false)
            .policy_values(observations, actions)
    }
}

struct BooleanEnv {
    device: Device,
    malformed: bool,
}

impl MultiGym<(), 3, 2> for BooleanEnv {
    type Error = Infallible;
    type ObservationSpace = BooleanSpace;
    type ActionSpace = BoxSpace<2>;

    /// Accepts Float actions [2, 1] and returns Bool reset observations [2, 1, 1] with unbatched terminal observations [1, 1].
    fn step(
        &mut self,
        actions: Tensor<2>,
    ) -> Result<MultiGymStepInfo<(), Tensor<2, Bool>>, Self::Error> {
        assert_eq!(actions.dims(), [2, 1]);
        assert!(!actions.is_autodiff());
        let terminal = if self.malformed {
            Tensor::from_data([[true, false]], &self.device)
        } else {
            Tensor::from_data([[true]], &self.device)
        };
        Ok(MultiGymStepInfo {
            observations: Tensor::from_data([[[false]], [[true]]], &self.device),
            rewards: Tensor::ones([2, 1], &self.device),
            infos: vec![(), ()],
            dones: vec![false, true],
            truncateds: vec![true, false],
            terminal_observations: vec![
                Some(terminal),
                Some(Tensor::from_data([[false]], &self.device)),
            ],
        })
    }

    fn observation_space(&self) -> BooleanSpace {
        BooleanSpace
    }

    fn action_space(&self) -> BoxSpace<2> {
        BoxSpace::new_with_universal_bounds([1, 1], -1.0, 1.0, &self.device)
    }

    fn num_envs(&self) -> usize {
        2
    }

    /// Returns native Bool observations [2, 1, 1] on the CPU environment device.
    fn reset(&mut self) -> Result<Tensor<3, Bool>, Self::Error> {
        Ok(Tensor::from_data([[[true]], [[false]]], &self.device))
    }
}

#[test]
fn native_high_rank_boolean_collection_trains_and_rejects_invalid_terminals() {
    let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
    let device = Device::flex().autodiff();
    let mut env = BooleanEnv {
        device: device.clone().without_autodiff(),
        malformed: false,
    };
    let mut agent = DeterministicActorCriticAgent::<_, _, _, _, Infallible>::builder()
        .online_actor(BooleanActor {
            weight: Param::from_tensor(Tensor::from_data([[0.25f64]], (&device, DType::F64))),
        })
        .actor_optimizer(SgdConfig::new().init())
        .actor_learning_rate(0.01)
        .critics(vec![
            ActionValueCritic::builder()
                .online_network(BooleanCritic {
                    weight: Param::from_tensor(Tensor::from_data(
                        [[0.5f64], [1.0]],
                        (&device, DType::F64),
                    )),
                })
                .optimizer(SgdConfig::new().init())
                .learning_rate(0.01)
                .build()
                .unwrap(),
        ])
        .strategy(TestStrategy {
            interval: 1,
            noise: 0.0,
            clip: 0.0,
        })
        .action_space(env.action_space())
        .observation_space(BooleanSpace)
        .replay_storage_config(ReplayStorageConfig::new(ReplayDeviceStrategy::Hybrid {
            optimization_device: device.clone(),
            storage_device: env.device.clone(),
        }))
        .dtype(DType::F64)
        .replay_capacity(8)
        .batch_size(2)
        .training_horizon(10)
        .training_start(0)
        .exploration_noise(0.0)
        .build()
        .unwrap();
    let observations = env.reset().unwrap();
    assert_eq!(
        numbers(agent.act_deterministic(&observations).unwrap()),
        [0.25, 0.0]
    );
    let actor_before = agent.online_actor.weight.val().into_data();
    struct Silent;
    impl DeterministicActorCriticLogger for Silent {
        fn log_update(&mut self, _entry: &DeterministicActorCriticLogEntry) {}

        fn log_collection(&mut self, _entry: &DeterministicActorCriticCollectionLogEntry) {}
    }
    agent.learn(&mut env, 2, &mut Silent).unwrap();
    assert_ne!(agent.online_actor.weight.val().into_data(), actor_before);
    let batch = agent
        .replay
        .as_mut()
        .unwrap()
        .storage_mut()
        .gather(&[0, 1])
        .unwrap();
    assert_eq!(batch.observations.dims(), [2, 1, 1]);
    assert_eq!(
        batch
            .observations
            .into_data()
            .iter::<bool>()
            .collect::<Vec<_>>(),
        [true, false]
    );
    assert_eq!(
        batch
            .next_observations
            .into_data()
            .iter::<bool>()
            .collect::<Vec<_>>(),
        [true, false]
    );
    assert_eq!(batch.actions.dtype(), DType::F64);
    assert_eq!(batch.rewards.dtype(), DType::F64);
    let len = agent.replay.as_ref().unwrap().len();
    let elapsed = agent.schedule_progress.elapsed_steps();
    env.malformed = true;
    assert!(matches!(
        agent.learn(&mut env, 2, &mut Silent),
        Err(DeterministicActorCriticError::Shape {
            field: "terminal observations",
            ..
        })
    ));
    assert_eq!(agent.replay.as_ref().unwrap().len(), len);
    assert_eq!(agent.schedule_progress.elapsed_steps(), elapsed);
}

#[test]
fn builder_and_actions_reject_invalid_contracts() {
    let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
    let device = Device::flex().autodiff();
    let build = |rate, dtype| {
        DeterministicActorCriticAgent::<_, _, _, _, Infallible>::builder()
            .online_actor(linear_model(&[0.25], &device, DType::F32))
            .actor_optimizer(SgdConfig::new().init())
            .actor_learning_rate(rate)
            .critics(vec![critic(&device, DType::F32)])
            .strategy(TestStrategy {
                interval: 1,
                noise: 0.0,
                clip: 0.0,
            })
            .action_space(BoxSpace::new_with_universal_bounds(
                [1, 1],
                -1.0,
                1.0,
                &device,
            ))
            .observation_space(BoxSpace::new_unbounded([1, 1], &device))
            .replay_storage_config(ReplayStorageConfig::new(ReplayDeviceStrategy::OneDevice(
                device.clone(),
            )))
            .dtype(dtype)
            .training_horizon(10)
            .build()
    };
    for rate in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            build(rate, DType::F32),
            Err(DeterministicActorCriticConfigurationError::InvalidActorLearningRate)
        ));
    }
    assert!(matches!(
        build(0.01, DType::U32),
        Err(DeterministicActorCriticConfigurationError::InvalidDType(
            DType::U32
        ))
    ));
    assert!(matches!(
        space_storage_shape::<3>("observations", vec![1], 8),
        Err(DeterministicActorCriticConfigurationError::SpaceRank {
            expected: 2,
            actual: 1,
            ..
        })
    ));
    assert!(matches!(
        space_storage_shape::<2>("actions", vec![0], 8),
        Err(DeterministicActorCriticConfigurationError::ZeroItemDimension { .. })
    ));
    let agent = agent(&device, DType::F32, 1);
    assert!(matches!(
        agent.act(&Tensor::zeros([2, 2], &device)),
        Err(DeterministicActorCriticError::Shape {
            field: "observations",
            ..
        })
    ));
    assert!(matches!(
        agent.act(&Tensor::zeros([0, 1], &device)),
        Err(DeterministicActorCriticError::EmptyBatch)
    ));
    assert!(matches!(
        agent.decode_actions(Tensor::zeros([2, 1], (&device, DType::F64)), 2),
        Err(DeterministicActorCriticError::DType {
            field: "policy actions",
            ..
        })
    ));
    assert!(matches!(
        agent.decode_actions(Tensor::zeros([2, 2], &device), 2),
        Err(DeterministicActorCriticError::Shape {
            field: "policy actions",
            ..
        })
    ));
}

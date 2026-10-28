use super::*;
use crate::tensor_rank::TensorBatchTypes;
use crate::{
    gym::{Gym, ResetInfo, StepInfo, VectorizedGymError, VectorizedGymWrapper},
    objectives::bellman_targets,
    spaces::BoxSpace,
};
use burn::{module::Param, optim::SgdConfig};
use std::{
    convert::Infallible,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

struct TestTarget;

impl QLearningTarget for TestTarget {
    fn requires_online_next_q_values() -> bool {
        false
    }

    /// Builds Float targets [batch_size, 1] from rewards and Bool termination flags [batch_size, 1] and Q values [batch_size, action_count].
    fn target_q_values(
        rewards: &Tensor<2>,
        next_dones: &Tensor<2, Bool>,
        _online: Option<&Tensor<2>>,
        target: &Tensor<2>,
        gamma: f32,
    ) -> Tensor<2> {
        bellman_targets(
            rewards.clone(),
            next_dones.clone(),
            target.clone().max_dim(1),
            f64::from(gamma),
        )
    }
}

#[derive(Module, Debug)]
pub(super) struct TestNetwork<K: Basic = Float> {
    pub(super) values: Param<Tensor<2>>,
    #[module(skip)]
    kind: PhantomData<K>,
}

impl<const R: usize, K: Basic> Forward<R, 2> for TestNetwork<K> {
    type InputKind = K;
    type OutputKind = Float;
    type Error = Infallible;

    /// Returns Float Q values [batch_size, 2] for native rank-R observations [batch_size, ...observation_shape].
    fn forward(&self, input: Tensor<R, K>) -> Result<Tensor<2>, Self::Error> {
        Ok(self.values.val().expand([input.dims()[0], 2]))
    }
}

pub(super) fn network<K: Basic>(device: &Device, dtype: DType) -> TestNetwork<K> {
    TestNetwork {
        values: Param::from_tensor(Tensor::from_data([[0.0f64, 1.0]], (device, dtype))),
        kind: PhantomData,
    }
}

struct EpisodeEnv {
    device: Device,
    steps: usize,
    truncate: bool,
}

impl Gym for EpisodeEnv {
    type Error = Infallible;
    type ObservationSpace = BoxSpace<2>;
    type ActionSpace = Discrete;

    /// Accepts an Int scalar action [1] and returns a Float observation [1] on the environment device.
    fn step(&mut self, _action: Tensor<1, Int>) -> Result<StepInfo, Self::Error> {
        self.steps += 1;
        Ok(StepInfo {
            observation: Tensor::from_data([self.steps as f32], &self.device),
            reward: 2.0,
            done: self.steps == 2 && !self.truncate,
            truncated: self.steps == 2 && self.truncate,
            info: (),
        })
    }

    /// Resets to Float observations [1] on the environment device.
    fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
        self.steps = 0;
        Ok(ResetInfo {
            observation: Tensor::zeros([1], &self.device),
            info: (),
        })
    }

    fn observation_space(&self) -> Self::ObservationSpace {
        BoxSpace::new_unbounded([1, 1], &self.device)
    }

    fn action_space(&self) -> Self::ActionSpace {
        Discrete::new(2).unwrap()
    }
}

#[derive(Default)]
struct GymCallCounts {
    resets: AtomicUsize,
    steps: AtomicUsize,
}

struct CountingEpisodeEnv {
    device: Device,
    calls: Arc<GymCallCounts>,
}

impl Gym for CountingEpisodeEnv {
    type Error = Infallible;
    type ObservationSpace = BoxSpace<2>;
    type ActionSpace = Discrete;

    /// Accepts an Int scalar action [1] and returns a Float observation [1] on the environment device.
    fn step(&mut self, _action: Tensor<1, Int>) -> Result<StepInfo, Self::Error> {
        self.calls.steps.fetch_add(1, Ordering::Relaxed);
        Ok(StepInfo {
            observation: Tensor::zeros([1], &self.device),
            reward: 0.0,
            done: false,
            truncated: false,
            info: (),
        })
    }

    fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
        self.calls.resets.fetch_add(1, Ordering::Relaxed);
        Ok(ResetInfo {
            observation: Tensor::zeros([1], &self.device),
            info: (),
        })
    }

    fn observation_space(&self) -> Self::ObservationSpace {
        BoxSpace::new_unbounded([1, 1], &self.device)
    }

    fn action_space(&self) -> Self::ActionSpace {
        Discrete::new(2).unwrap()
    }
}

fn counting_envs(
    count: usize,
    device: &Device,
    calls: Arc<GymCallCounts>,
) -> VectorizedGymWrapper<CountingEpisodeEnv> {
    VectorizedGymWrapper::new(
        (0..count)
            .map(|_| CountingEpisodeEnv {
                device: device.clone(),
                calls: calls.clone(),
            })
            .collect(),
    )
    .unwrap()
}

#[derive(Default)]
struct Logger {
    updates: Vec<usize>,
    rewards: Vec<Vec<f32>>,
    episodes: Vec<QEpisodeLogEntry>,
}

impl QLearningLogger for Logger {
    fn log_update(&mut self, entry: &QLogEntry) {
        assert_eq!(entry.loss.dims(), [1]);
        assert_eq!(entry.q_values.dims()[1], 1);
        assert_eq!(entry.q_values.dims(), entry.replay_rewards.dims());
        assert_eq!(entry.loss.dtype(), entry.q_values.dtype());
        assert_eq!(entry.update_index, self.updates.len());
        self.updates.push(entry.collection_timestep);
    }

    fn log_collection(&mut self, entry: &QCollectionLogEntry) {
        self.rewards.push(
            entry
                .collection_rewards
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
        );
        self.episodes.extend(
            entry
                .completed_episodes
                .iter()
                .map(|entry| QEpisodeLogEntry {
                    environment_index: entry.environment_index,
                    episode_return: entry.episode_return,
                    episode_length: entry.episode_length,
                    terminated: entry.terminated,
                    truncated: entry.truncated,
                    collection_timestep: entry.collection_timestep,
                }),
        );
    }
}

fn float_agent(
    device: &Device,
) -> QLearningAgent<TestNetwork, BoxSpace<2>, VectorizedGymError<Infallible>, TestTarget> {
    QLearningAgent::builder()
        .action_space(Discrete::new(2).unwrap())
        .observation_space(BoxSpace::new_unbounded([1, 1], device))
        .online_q_network(network(&device.clone().autodiff(), DType::F32))
        .optimizer(SgdConfig::new().init())
        .learning_rate(0.01)
        .epsilon_schedule(LinearSchedule::new(1.0, 0.0))
        .replay_capacity(8)
        .batch_size(1)
        .training_start(3)
        .update_frequency(2)
        .target_update_interval(4)
        .training_horizon(10)
        .build()
        .unwrap()
}

#[test]
fn epsilon_greedy_preserves_layout_kind_and_seeded_host_randomness() {
    let device = Device::flex();
    let observations = Tensor::<2>::from_data(
        [[0.0f32, 2.0, 1.0], [3.0, 1.0, 2.0], [0.0, 1.0, 4.0]],
        &device,
    );
    let greedy = epsilon_greedy_actions(
        &observations,
        0.0,
        &Discrete::new(3).unwrap(),
        &mut StdRng::seed_from_u64(1),
        |input| Ok::<_, Infallible>(input.clone()),
    )
    .unwrap();
    assert_eq!(greedy.dims(), [3, 1]);
    assert_eq!(greedy.dtype(), DType::U32);
    assert_eq!(greedy.into_data().try_to_vec::<u32>().unwrap(), [1, 0, 2]);
    let input =
        Tensor::<3, Bool>::from_data(TensorData::new(vec![true; 128], [128, 1, 1]), &device);
    let sample = || {
        epsilon_greedy_actions(
            &input,
            1.0,
            &Discrete::new(3).unwrap(),
            &mut StdRng::seed_from_u64(2),
            |_| -> Result<Tensor<2>, Infallible> {
                panic!("full exploration must not evaluate the model")
            },
        )
        .unwrap()
    };
    assert_eq!(sample().dims(), [128, 1]);
    let first = sample().into_data().try_to_vec::<u32>().unwrap();
    assert_eq!(first, sample().into_data().try_to_vec::<u32>().unwrap());
    assert!(first.into_iter().all(|action| action < 3));
}

#[test]
fn mixed_exploration_forwards_only_greedy_native_observations() {
    let device = Device::flex();
    let observations = Tensor::<2, Int>::from_data(
        TensorData::new((0i64..128).collect(), [128, 1]),
        (&device, DType::I64),
    );
    let mut forwarded = 0;
    let actions = epsilon_greedy_actions(
        &observations,
        0.5,
        &Discrete::new(2).unwrap(),
        &mut StdRng::seed_from_u64(3),
        |input| {
            assert_eq!(input.dtype(), DType::I64);
            forwarded = input.dims()[0];
            let input = input.clone().float();
            Ok::<_, Infallible>(Tensor::cat(vec![input.clone(), input.neg()], 1))
        },
    )
    .unwrap();
    assert!(forwarded > 0 && forwarded < 128);
    assert_eq!(actions.dims(), [128, 1]);
    assert!(
        actions
            .into_data()
            .try_to_vec::<u32>()
            .unwrap()
            .into_iter()
            .all(|action| action < 2)
    );
}

#[test]
fn selected_action_q_values_retains_shape_precision_and_gradient_flow() {
    let device = Device::flex().autodiff();
    let q_values =
        Tensor::<2>::from_data([[1.0f64, 5.0, 2.0], [7.0, 3.0, 4.0]], (&device, DType::F64))
            .require_grad();
    let actions = Tensor::<2, Int>::from_data([[1u32], [2]], (&device, DType::U32));
    let selected = selected_action_q_values(&q_values, &actions);
    assert_eq!(selected.dims(), [2, 1]);
    assert_eq!(selected.dtype(), DType::F64);
    assert_eq!(
        selected.clone().into_data().try_to_vec::<f64>().unwrap(),
        [5.0, 4.0]
    );
    let gradients = selected.sum().backward();
    assert_eq!(
        q_values
            .grad(&gradients)
            .unwrap()
            .into_data()
            .try_to_vec::<f64>()
            .unwrap(),
        [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    );
}

/// Creates Float replay rows [batch_size, 1] with scalar rewards and Bool flags [batch_size, 1].
fn replay_insert(values: &[f64], device: &Device) -> QLearningInsert<2, Float> {
    let count = values.len();
    QLearningInsert {
        observations: Tensor::from_data(
            TensorData::new(values.to_vec(), [count, 1]),
            (device, DType::F64),
        ),
        next_observations: Tensor::from_data(
            TensorData::new(
                values.iter().map(|value| value + 10.0).collect::<Vec<_>>(),
                [count, 1],
            ),
            (device, DType::F64),
        ),
        actions: Tensor::zeros([count, 1], (device, DType::U32)),
        rewards: Tensor::ones([count, 1], device),
        next_dones: Tensor::from_data(TensorData::new(vec![false; count], [count, 1]), device),
        truncateds: vec![false; count],
    }
}

#[test]
fn replay_matches_observation_dtype_and_does_not_alias_gathered_observations() {
    let device = Device::flex();
    let mut storage = QLearningReplayStorage::<2, Float>::new([4, 1], device.clone());
    storage
        .insert(0, replay_insert(&[1.0, 2.0], &device))
        .unwrap();
    let before = storage.gather(&[0]).unwrap();
    storage
        .insert(2, replay_insert(&[3.0, 4.0], &device))
        .unwrap();
    storage
        .insert(0, replay_insert(&[5.0, 6.0], &device))
        .unwrap();
    assert_eq!(
        before.observations.into_data().try_to_vec::<f64>().unwrap(),
        [1.0]
    );
    let batch = storage.gather(&[0, 1, 2, 3]).unwrap();
    assert_eq!(
        batch.observations.into_data().try_to_vec::<f64>().unwrap(),
        [5.0, 6.0, 15.0, 16.0]
    );
    assert_eq!(batch.actions.dims(), [4, 1]);
    assert_eq!(batch.next_dones.dims(), [4, 1]);
    assert_eq!(batch.rewards.dims(), [4, 1]);
}

#[test]
fn replay_preserves_native_integer_and_boolean_observations() {
    let device = Device::flex();
    let mut integer = QLearningReplayStorage::<2, Int>::new([4, 1], device.clone());
    let values = [u64::MAX - 1, u64::MAX - 2];
    integer
        .insert(
            0,
            QLearningInsert {
                observations: Tensor::from_data(
                    TensorData::new(values.to_vec(), [2, 1]),
                    (&device, DType::U64),
                ),
                next_observations: Tensor::from_data(
                    TensorData::new(values.to_vec(), [2, 1]),
                    (&device, DType::U64),
                ),
                actions: Tensor::zeros([2, 1], (&device, DType::U32)),
                rewards: Tensor::zeros([2, 1], &device),
                next_dones: Tensor::from_data([[false], [true]], &device),
                truncateds: vec![false; 2],
            },
        )
        .unwrap();
    let batch = integer.gather(&[0, 1]).unwrap();
    assert_eq!(batch.observations.dtype(), DType::U64);
    assert_eq!(
        batch.observations.into_data().try_to_vec::<u64>().unwrap(),
        values
    );
    let mut boolean = QLearningReplayStorage::<2, Bool>::new([4, 1], device.clone());
    boolean
        .insert(
            0,
            QLearningInsert {
                observations: Tensor::from_data([[false], [true]], &device),
                next_observations: Tensor::from_data([[true], [false]], &device),
                actions: Tensor::zeros([2, 1], (&device, DType::U32)),
                rewards: Tensor::zeros([2, 1], &device),
                next_dones: Tensor::from_data([[false], [false]], &device),
                truncateds: vec![true, false],
            },
        )
        .unwrap();
    let batch = boolean.gather(&[0, 1]).unwrap();
    assert_eq!(
        batch
            .observations
            .try_into_data_as::<bool>()
            .unwrap()
            .try_to_vec::<bool>()
            .unwrap(),
        [false, true]
    );
    assert_eq!(
        batch
            .next_observations
            .try_into_data_as::<bool>()
            .unwrap()
            .try_to_vec::<bool>()
            .unwrap(),
        [true, false]
    );
}

#[derive(Clone)]
pub(super) struct ObservationFixture<const R: usize, K> {
    pub(super) _kind: PhantomData<K>,
}

impl<const R: usize, K: Basic> TensorBatchTypes for ObservationFixture<R, K>
where
    Tensor<R, K>: PrevRank,
{
    type Kind = K;
    type Item = <Tensor<R, K> as PrevRank>::Prev;
    type Batch = Tensor<R, K>;
}

impl<const R: usize, K: Basic> ObservationSpace<R> for ObservationFixture<R, K>
where
    Tensor<R, K>: PrevRank,
{
    /// Checks native rank-R observations [batch_size, 1, ...] against size-one item axes.
    fn contains(&self, observations: &Tensor<R, K>) -> bool {
        observations.dims()[1..].iter().all(|size| *size == 1)
    }

    fn shape(&self) -> Vec<usize> {
        vec![1; R - 1]
    }
}

#[test]
fn native_high_rank_boolean_model_trains_with_matching_replay() {
    let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
    let device = Device::flex().autodiff();
    let mut agent = QLearningAgent::<_, _, Infallible, TestTarget, 3>::builder()
        .action_space(Discrete::new(2).unwrap())
        .observation_space(ObservationFixture::<3, Bool> { _kind: PhantomData })
        .online_q_network(network(&device, DType::F64))
        .optimizer(SgdConfig::new().init())
        .learning_rate(0.01)
        .epsilon_schedule(LinearSchedule::new(0.0, 0.0))
        .replay_capacity(8)
        .batch_size(2)
        .training_horizon(10)
        .dtype(DType::F64)
        .build()
        .unwrap();
    let observations = Tensor::<3, Bool>::from_data([[[true]], [[false]]], &device);
    let actions = agent.act(&observations).unwrap();
    assert_eq!(actions.dims(), [2, 1]);
    assert_eq!(actions.into_data().try_to_vec::<u32>().unwrap(), [1, 1]);
    agent
        .experience_replay
        .add(QLearningInsert {
            observations: observations.clone(),
            next_observations: observations,
            actions: Tensor::ones([2, 1], (&device, DType::U32)),
            rewards: Tensor::full([2, 1], 2.0, (&device, DType::F64)),
            next_dones: Tensor::from_data([[false], [false]], &device),
            truncateds: vec![false; 2],
        })
        .unwrap();
    let before = agent.online_q_network.values.val().into_data();
    let batch = agent
        .experience_replay
        .storage_mut()
        .gather(&[0, 1])
        .unwrap();
    assert_eq!(batch.observations.dims(), [2, 1, 1]);
    assert_eq!(
        batch
            .observations
            .try_into_data_as::<bool>()
            .unwrap()
            .try_to_vec::<bool>()
            .unwrap(),
        [true, false]
    );
    agent.optimize(2, &mut Logger::default()).unwrap();
    assert_ne!(agent.online_q_network.values.val().into_data(), before);
    assert_eq!(agent.target_q_network.values.val().into_data(), before);
    assert!(!agent.target_q_network.values.val().is_require_grad());
}

#[test]
fn matching_replay_rejects_dtype_changes_without_overwriting_existing_rows() {
    let device = Device::flex();
    let mut storage = QLearningReplayStorage::<2, Float>::new([4, 1], device.clone());
    storage
        .insert(0, replay_insert(&[1.0, 2.0], &device))
        .unwrap();
    let mut changed = replay_insert(&[3.0, 4.0], &device);
    changed.observations = changed.observations.cast(DType::F32);
    changed.next_observations = changed.next_observations.cast(DType::F32);
    assert!(matches!(
        storage.insert(0, changed),
        Err(QLearningReplayError::Storage(
            ReplayStorageError::DTypeMismatch {
                field: "observations",
                expected: DType::F64,
                actual: DType::F32,
            }
        ))
    ));
    let retained = storage.gather(&[0, 1]).unwrap();
    assert_eq!(retained.observations.dtype(), DType::F64);
    assert_eq!(
        retained
            .observations
            .into_data()
            .try_to_vec::<f64>()
            .unwrap(),
        [1.0, 2.0]
    );
}

#[test]
fn learn_preserves_schedule_replay_alignment_target_snapshots_and_episode_metrics() {
    let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
    let device = Device::flex();
    let mut env = VectorizedGymWrapper::new(vec![
        EpisodeEnv {
            device: device.clone(),
            steps: 0,
            truncate: false,
        },
        EpisodeEnv {
            device: device.clone(),
            steps: 0,
            truncate: true,
        },
    ])
    .unwrap();
    let mut agent = float_agent(&device);
    assert!(!agent.target_q_network.values.val().is_require_grad());
    let initial_target = agent.target_q_network.values.val().into_data();
    let mut logger = Logger::default();
    agent.learn(&mut env, 4, &mut logger).unwrap();
    assert_eq!(agent.schedule_progress.elapsed_steps(), 4);
    assert_eq!(agent.current_epsilon, 0.8);
    assert_eq!(logger.updates, [4]);
    assert_eq!(logger.rewards, [vec![2.0, 2.0], vec![2.0, 2.0]]);
    assert_eq!(logger.episodes.len(), 2);
    assert_eq!(
        (
            logger.episodes[0].episode_return,
            logger.episodes[0].episode_length,
            logger.episodes[0].collection_timestep
        ),
        (4.0, 2, 3)
    );
    assert!(logger.episodes[0].terminated);
    assert!(logger.episodes[1].truncated);
    assert!(!logger.episodes[1].terminated);
    assert_ne!(
        agent.online_q_network.values.val().into_data(),
        initial_target
    );
    assert_eq!(
        agent.online_q_network.values.val().into_data(),
        agent.target_q_network.values.val().into_data()
    );
    let batch = agent
        .experience_replay
        .storage_mut()
        .gather(&[2, 3])
        .unwrap();
    assert_eq!(
        batch
            .next_observations
            .into_data()
            .try_to_vec::<f32>()
            .unwrap(),
        [2.0, 2.0]
    );
    assert_eq!(
        batch
            .next_dones
            .try_into_data_as::<bool>()
            .unwrap()
            .try_to_vec::<bool>()
            .unwrap(),
        [true, false]
    );
    agent.learn(&mut env, 2, &mut logger).unwrap();
    assert_eq!(agent.schedule_progress.elapsed_steps(), 6);
    assert_eq!(logger.updates, [4, 6]);
    assert_ne!(
        agent.online_q_network.values.val().into_data(),
        agent.target_q_network.values.val().into_data()
    );
    assert_eq!(agent.get_action_space().get_possible_values(), 2);
    assert_eq!(agent.get_observation_space().shape(), [1]);
    let len = agent.experience_replay.len();
    let mut incompatible = VectorizedGymWrapper::from(EpisodeEnv {
        device,
        steps: 0,
        truncate: false,
    });
    assert!(matches!(
        agent.learn(&mut incompatible, 1, &mut logger),
        Err(QAgentError::ReplayStorageError(
            QLearningReplayError::Storage(ReplayStorageError::EnvironmentCountMismatch {
                expected: 2,
                actual: 1
            })
        ))
    ));
    assert_eq!(agent.experience_replay.len(), len);
}

#[test]
fn learn_rejects_invalid_replay_alignment_before_reset_or_step() {
    let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
    let device = Device::flex();
    let calls = Arc::new(GymCallCounts::default());
    let mut env = counting_envs(3, &device, calls.clone());
    let mut agent = float_agent(&device);

    assert!(matches!(
        agent.learn(&mut env, 1, &mut Logger::default()),
        Err(QAgentError::ReplayStorageError(
            QLearningReplayError::Storage(ReplayStorageError::InvalidReplayAlignment {
                capacity: 8,
                environment_count: 3
            })
        ))
    ));
    assert_eq!(calls.resets.load(Ordering::Relaxed), 0);
    assert_eq!(calls.steps.load(Ordering::Relaxed), 0);
}

#[test]
fn learn_reuses_matching_count_and_rejects_a_new_count_before_reset() {
    let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
    let device = Device::flex();
    let calls = Arc::new(GymCallCounts::default());
    let mut env = counting_envs(2, &device, calls.clone());
    let mut agent = float_agent(&device);
    let mut logger = Logger::default();

    agent.learn(&mut env, 1, &mut logger).unwrap();
    agent.learn(&mut env, 1, &mut logger).unwrap();
    assert_eq!(calls.resets.load(Ordering::Relaxed), 4);
    assert_eq!(calls.steps.load(Ordering::Relaxed), 4);

    let changed_calls = Arc::new(GymCallCounts::default());
    let mut changed_env = counting_envs(4, &device, changed_calls.clone());
    assert!(matches!(
        agent.learn(&mut changed_env, 1, &mut logger),
        Err(QAgentError::ReplayStorageError(
            QLearningReplayError::Storage(ReplayStorageError::EnvironmentCountMismatch {
                expected: 2,
                actual: 4
            })
        ))
    ));
    assert_eq!(changed_calls.resets.load(Ordering::Relaxed), 0);
    assert_eq!(changed_calls.steps.load(Ordering::Relaxed), 0);
}

#[test]
fn builder_rejects_invalid_learning_rates_and_compute_dtypes() {
    let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
    let device = Device::flex().autodiff();
    let build = |rate, dtype| {
        QLearningAgent::<_, _, Infallible, TestTarget>::builder()
            .action_space(Discrete::new(2).unwrap())
            .observation_space(BoxSpace::new_unbounded([1, 1], &device))
            .online_q_network(network(&device, DType::F32))
            .optimizer(SgdConfig::new().init())
            .learning_rate(rate)
            .training_horizon(10)
            .dtype(dtype)
            .build()
    };
    assert!(matches!(
        build(f64::NAN, DType::F32),
        Err(QAgentError::ConfigurationError(
            QLearningConfigurationError::InvalidLearningRate
        ))
    ));
    assert!(matches!(
        build(0.01, DType::U32),
        Err(QAgentError::ConfigurationError(
            QLearningConfigurationError::InvalidDType(DType::U32)
        ))
    ));
}

#[test]
fn accepts_valid_configuration() {
    assert_eq!(
        QLearningConfigurationValidator::validate_configuration()
            .replay_capacity(1_000)
            .batch_size(32)
            .gamma(0.99)
            .initial_epsilon(1.0)
            .final_epsilon(0.1)
            .update_frequency(4)
            .target_update_interval(1_000)
            .training_horizon(10_000)
            .call(),
        Ok(())
    );
}

#[test]
fn rejects_invalid_configuration_values() {
    assert_eq!(
        QLearningConfigurationValidator::validate_configuration()
            .replay_capacity(0)
            .batch_size(32)
            .gamma(0.99)
            .initial_epsilon(1.0)
            .final_epsilon(0.1)
            .update_frequency(4)
            .target_update_interval(1_000)
            .training_horizon(10_000)
            .call(),
        Err(QLearningConfigurationError::ZeroReplayCapacity)
    );
    assert_eq!(
        QLearningConfigurationValidator::validate_configuration()
            .replay_capacity(1_000)
            .batch_size(0)
            .gamma(0.99)
            .initial_epsilon(1.0)
            .final_epsilon(0.1)
            .update_frequency(4)
            .target_update_interval(1_000)
            .training_horizon(10_000)
            .call(),
        Err(QLearningConfigurationError::ZeroBatchSize)
    );
    assert_eq!(
        QLearningConfigurationValidator::validate_configuration()
            .replay_capacity(1_000)
            .batch_size(32)
            .gamma(0.99)
            .initial_epsilon(1.0)
            .final_epsilon(0.1)
            .update_frequency(4)
            .target_update_interval(0)
            .training_horizon(10_000)
            .call(),
        Err(QLearningConfigurationError::ZeroTargetUpdateInterval)
    );
    assert_eq!(
        QLearningConfigurationValidator::validate_configuration()
            .replay_capacity(1_000)
            .batch_size(32)
            .gamma(0.99)
            .initial_epsilon(1.0)
            .final_epsilon(0.1)
            .update_frequency(0)
            .target_update_interval(1_000)
            .training_horizon(10_000)
            .call(),
        Err(QLearningConfigurationError::ZeroUpdateFrequency)
    );
    assert_eq!(
        QLearningConfigurationValidator::validate_configuration()
            .replay_capacity(31)
            .batch_size(32)
            .gamma(0.99)
            .initial_epsilon(1.0)
            .final_epsilon(0.1)
            .update_frequency(4)
            .target_update_interval(1_000)
            .training_horizon(10_000)
            .call(),
        Err(QLearningConfigurationError::ReplayCapacityBelowBatchSize)
    );
    assert_eq!(
        QLearningConfigurationValidator::validate_configuration()
            .replay_capacity(1_000)
            .batch_size(32)
            .gamma(1.01)
            .initial_epsilon(1.0)
            .final_epsilon(0.1)
            .update_frequency(4)
            .target_update_interval(1_000)
            .training_horizon(10_000)
            .call(),
        Err(QLearningConfigurationError::InvalidGamma)
    );
    assert_eq!(
        QLearningConfigurationValidator::validate_configuration()
            .replay_capacity(1_000)
            .batch_size(32)
            .gamma(0.99)
            .initial_epsilon(1.0)
            .final_epsilon(0.1)
            .update_frequency(4)
            .target_update_interval(1_000)
            .training_horizon(0)
            .call(),
        Err(QLearningConfigurationError::ZeroTrainingHorizon)
    );
    assert_eq!(
        validate_epsilon(1.01),
        Err(QLearningConfigurationError::InvalidEpsilon)
    );
}

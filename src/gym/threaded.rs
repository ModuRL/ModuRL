use super::{
    Gym, GymBatchMetadata, MultiGym, MultiGymStepInfo, StackedMultiGymError, VectorizedGymError,
    VectorizedGymWrapper, batch_metadata, batch_offsets, combine_steps,
};
use crate::spaces::{ActionSpace, ObservationSpace};
use burn::tensor::{Slice, Tensor, kind::Basic};
use std::{sync::mpsc, thread};

struct WorkerCommand<G, I, const O: usize, const A: usize, const U: usize>
where
    G: MultiGym<I, O, A, U>,
{
    action: Option<Tensor<A, <G::ActionSpace as ActionSpace<A>>::Kind>>,
}

enum WorkerResponse<I, const O: usize, K: Basic, E, const U: usize> {
    Metadata(GymBatchMetadata),
    Step(Box<Result<MultiGymStepInfo<I, O, K, U>, E>>),
    Reset(Box<Result<Tensor<O, K>, E>>),
}

struct GymWorker<G, I, const O: usize, const A: usize, const U: usize>
where
    G: MultiGym<I, O, A, U>,
{
    commands: mpsc::Sender<WorkerCommand<G, I, O, A, U>>,
    responses: mpsc::Receiver<
        WorkerResponse<I, O, <G::ObservationSpace as ObservationSpace<O>>::Kind, G::Error, U>,
    >,
}

/// Constructs and owns a MultiGym on a persistent worker thread.
/// Step commands carry actions `[num_envs, ...action_shape]` of rank `A`; responses carry rank-`O` observations and rank-1 rewards.
/// Native tensor kinds come from G's spaces. Commands preserve dtype, device, and axis order.
fn start_worker<G, F, I, const O: usize, const A: usize, const U: usize>(
    make_gym: F,
) -> GymWorker<G, I, O, A, U>
where
    G: MultiGym<I, O, A, U> + 'static,
    F: FnOnce() -> G + Send + 'static,
    G::Error: Send + 'static,
    I: Send + 'static,
{
    let (commands, command_rx) = mpsc::channel::<WorkerCommand<G, I, O, A, U>>();
    let (response_tx, responses) = mpsc::channel();
    thread::spawn(move || {
        let mut gym = make_gym();
        if response_tx
            .send(WorkerResponse::Metadata(batch_metadata(&gym)))
            .is_err()
        {
            return;
        }
        while let Ok(command) = command_rx.recv() {
            let response = match command.action {
                Some(action) => WorkerResponse::Step(Box::new(gym.step(action))),
                None => WorkerResponse::Reset(Box::new(gym.reset())),
            };
            if response_tx.send(response).is_err() {
                break;
            }
        }
    });
    GymWorker {
        commands,
        responses,
    }
}

/// Flattens homogeneous MultiGym batches, with one persistent worker per group.
/// Constructors run on their owning threads. Dispatches all groups before waiting and preserves group order.
/// Reset after an error because other groups may have advanced.
pub struct MultithreadedStackedMultiGym<
    G,
    I = (),
    const O: usize = 2,
    const A: usize = 1,
    const U: usize = 1,
> where
    G: MultiGym<I, O, A, U>,
{
    groups: Vec<GymWorker<G, I, O, A, U>>,
    group_offsets: Vec<usize>,
    observation_space: G::ObservationSpace,
    action_space: G::ActionSpace,
}

impl<G, I, const O: usize, const A: usize, const U: usize>
    MultithreadedStackedMultiGym<G, I, O, A, U>
where
    G: MultiGym<I, O, A, U> + 'static,
    G::Error: Send + 'static,
    G::ObservationSpace: Clone,
    G::ActionSpace: Clone,
    I: Send + 'static,
{
    /// Starts nonempty groups whose item shapes match the supplied observation and action spaces.
    pub fn new<F>(
        gym_constructors: Vec<F>,
        observation_space: G::ObservationSpace,
        action_space: G::ActionSpace,
    ) -> Result<Self, StackedMultiGymError<G::Error>>
    where
        F: FnOnce() -> G + Send + 'static,
    {
        const {
            assert!(
                O == U + 1 || (O == 1 && U == 1),
                "batch observation rank must be single observation rank + 1, except scalar observations"
            );
            assert!(A >= 1, "batched actions require a batch axis");
        }
        if gym_constructors.is_empty() {
            return Err(StackedMultiGymError::Empty);
        }
        let groups: Vec<_> = gym_constructors.into_iter().map(start_worker).collect();
        let mut metadata = Vec::with_capacity(groups.len());
        for (gym_index, group) in groups.iter().enumerate() {
            match group.responses.recv() {
                Ok(WorkerResponse::Metadata(info)) => metadata.push(info),
                _ => return Err(StackedMultiGymError::WorkerDisconnected { gym_index }),
            }
        }
        let group_offsets =
            batch_offsets(&metadata, &observation_space.shape(), &action_space.shape())?;
        Ok(Self {
            groups,
            group_offsets,
            observation_space,
            action_space,
        })
    }

    pub fn group_offsets(&self) -> &[usize] {
        &self.group_offsets
    }

    pub fn num_groups(&self) -> usize {
        self.groups.len()
    }
}

impl<G, I, const O: usize, const A: usize, const U: usize> MultiGym<I, O, A, U>
    for MultithreadedStackedMultiGym<G, I, O, A, U>
where
    G: MultiGym<I, O, A, U> + 'static,
    G::Error: Send + 'static,
    G::ObservationSpace: Clone,
    G::ActionSpace: Clone,
    I: Send + 'static,
{
    type Error = StackedMultiGymError<G::Error>;
    type ObservationSpace = G::ObservationSpace;
    type ActionSpace = G::ActionSpace;

    /// Splits rank-`A` actions `[total_size, ...action_shape]` into rank-`A` group batches and steps them concurrently.
    /// Returns concatenated rank-`O` observations `[total_size, ...observation_shape]` and rewards `[total_size]`.
    /// Inputs must meet the inner gyms' dtype, device, and item-shape contracts.
    fn step(
        &mut self,
        action: Tensor<A, <G::ActionSpace as ActionSpace<A>>::Kind>,
    ) -> Result<
        MultiGymStepInfo<I, O, <G::ObservationSpace as ObservationSpace<O>>::Kind, U>,
        Self::Error,
    > {
        // Prepare every action group before dispatch so a slicing failure cannot advance only part of the stack.
        let actions: Vec<_> = self
            .group_offsets
            .windows(2)
            .map(|range| action.clone().slice([Slice::from(range[0]..range[1])]))
            .collect();
        let sent: Vec<_> = self
            .groups
            .iter()
            .zip(actions)
            .map(|(group, action)| {
                group
                    .commands
                    .send(WorkerCommand {
                        action: Some(action),
                    })
                    .is_ok()
            })
            .collect();
        // Drain every dispatched response before reporting an error so the next reset cannot consume stale step replies.
        let replies: Vec<_> =
            self.groups
                .iter()
                .zip(sent)
                .enumerate()
                .map(|(gym_index, (group, sent))| {
                    if !sent {
                        return Err(StackedMultiGymError::WorkerDisconnected { gym_index });
                    }
                    match group.responses.recv() {
                        Ok(WorkerResponse::Step(result)) => (*result)
                            .map_err(|error| StackedMultiGymError::Inner { gym_index, error }),
                        _ => Err(StackedMultiGymError::WorkerDisconnected { gym_index }),
                    }
                })
                .collect();
        let steps = replies.into_iter().collect::<Result<Vec<_>, _>>()?;
        Ok(combine_steps(steps))
    }

    fn observation_space(&self) -> Self::ObservationSpace {
        self.observation_space.clone()
    }

    fn action_space(&self) -> Self::ActionSpace {
        self.action_space.clone()
    }

    fn num_envs(&self) -> usize {
        self.group_offsets[self.group_offsets.len() - 1]
    }

    /// Resets groups concurrently, concatenating rank-`O` observations into `[total_size, ...observation_shape]`.
    /// Groups must share observation kind, dtype, device, and item dimensions. Gradient paths are preserved.
    fn reset(
        &mut self,
    ) -> Result<Tensor<O, <G::ObservationSpace as ObservationSpace<O>>::Kind>, Self::Error> {
        let sent: Vec<_> = self
            .groups
            .iter()
            .map(|group| group.commands.send(WorkerCommand { action: None }).is_ok())
            .collect();
        let replies: Vec<_> =
            self.groups
                .iter()
                .zip(sent)
                .enumerate()
                .map(|(gym_index, (group, sent))| {
                    if !sent {
                        return Err(StackedMultiGymError::WorkerDisconnected { gym_index });
                    }
                    match group.responses.recv() {
                        Ok(WorkerResponse::Reset(result)) => (*result)
                            .map_err(|error| StackedMultiGymError::Inner { gym_index, error }),
                        _ => Err(StackedMultiGymError::WorkerDisconnected { gym_index }),
                    }
                })
                .collect();
        let observations = replies.into_iter().collect::<Result<Vec<_>, _>>()?;
        Ok(Tensor::cat(observations, 0))
    }
}

/// Steps independent environments concurrently while preserving environment order.
/// Each worker owns a one-environment vectorizer, which saves terminal observations before auto-reset.
/// Reset after an error because some environments may already have advanced.
pub struct MultithreadedVectorizedGymWrapper<
    G,
    I = (),
    const O: usize = 1,
    const A: usize = 1,
    const BO: usize = 2,
    const BA: usize = 1,
> where
    G: Gym<I, O, A, BO, BA>,
{
    envs: Vec<GymWorker<VectorizedGymWrapper<G, I, O, A, BO, BA>, I, BO, BA, O>>,
    observation_space: G::ObservationSpace,
    action_space: G::ActionSpace,
}

impl<G, I, const O: usize, const A: usize, const BO: usize, const BA: usize>
    MultithreadedVectorizedGymWrapper<G, I, O, A, BO, BA>
where
    G: Gym<I, O, A, BO, BA> + 'static,
    G::Error: Send + 'static,
    G::ObservationSpace: Clone,
    G::ActionSpace: Clone,
    I: Send + 'static,
{
    /// Starts nonempty environment constructors on their persistent worker threads.
    /// Supplied spaces must describe every environment.
    pub fn new<F>(
        env_constructors: Vec<F>,
        observation_space: G::ObservationSpace,
        action_space: G::ActionSpace,
    ) -> Result<Self, VectorizedGymError<G::Error>>
    where
        F: FnOnce() -> G + Send + 'static,
    {
        const {
            assert!(
                BO == O + 1 || (BO == 1 && O == 1),
                "batch observation rank must be single observation rank + 1, except scalar observations"
            );
            assert!(
                BA == A + 1 || (BA == 1 && A == 1),
                "batch action rank must be single action rank + 1, except scalar actions"
            );
        }
        if env_constructors.is_empty() {
            return Err(VectorizedGymError::Empty);
        }
        let envs: Vec<_> = env_constructors
            .into_iter()
            .map(|make_env| start_worker(move || VectorizedGymWrapper::from(make_env())))
            .collect();
        for (environment_index, env) in envs.iter().enumerate() {
            if !matches!(env.responses.recv(), Ok(WorkerResponse::Metadata(_))) {
                return Err(VectorizedGymError::WorkerDisconnected { environment_index });
            }
        }
        Ok(Self {
            envs,
            observation_space,
            action_space,
        })
    }
}

impl<G, I, const O: usize, const A: usize, const BO: usize, const BA: usize> MultiGym<I, BO, BA, O>
    for MultithreadedVectorizedGymWrapper<G, I, O, A, BO, BA>
where
    G: Gym<I, O, A, BO, BA> + 'static,
    G::Error: Send + 'static,
    G::ObservationSpace: Clone,
    G::ActionSpace: Clone,
    I: Send + 'static,
{
    type Error = VectorizedGymError<G::Error>;
    type ObservationSpace = G::ObservationSpace;
    type ActionSpace = G::ActionSpace;

    /// Sends rank-`BA` action slices `[1, ...action_shape]` to each worker's vectorizer, which removes the batch axis before Gym execution.
    /// Concatenates rank-`BO` observations `[1, ...observation_shape]` and F32 rewards `[1]` in environment order.
    /// Terminal observations remain unbatched rank `O`; scalar terminal observations use `[1]`.
    /// Inputs must match the environments' dtype and device contracts; the batch axis is preserved.
    fn step(
        &mut self,
        action: Tensor<BA, <G::ActionSpace as ActionSpace<BA>>::Kind>,
    ) -> Result<
        MultiGymStepInfo<I, BO, <G::ObservationSpace as ObservationSpace<BO>>::Kind, O>,
        Self::Error,
    > {
        let actions: Vec<_> = (0..self.envs.len())
            .map(|index| action.clone().slice([Slice::from(index..index + 1)]))
            .collect();
        let sent: Vec<_> = self
            .envs
            .iter()
            .zip(actions)
            .map(|(env, action)| {
                env.commands
                    .send(WorkerCommand {
                        action: Some(action),
                    })
                    .is_ok()
            })
            .collect();
        // Consume all responses in order, including successes after a failed environment.
        let replies: Vec<_> = self
            .envs
            .iter()
            .zip(sent)
            .enumerate()
            .map(|(environment_index, (env, sent))| {
                if !sent {
                    return Err(VectorizedGymError::WorkerDisconnected { environment_index });
                }
                match env.responses.recv() {
                    Ok(WorkerResponse::Step(result)) => *result,
                    _ => Err(VectorizedGymError::WorkerDisconnected { environment_index }),
                }
            })
            .collect();
        let steps = replies.into_iter().collect::<Result<Vec<_>, _>>()?;
        Ok(combine_steps(steps))
    }

    fn observation_space(&self) -> Self::ObservationSpace {
        self.observation_space.clone()
    }

    fn action_space(&self) -> Self::ActionSpace {
        self.action_space.clone()
    }

    fn num_envs(&self) -> usize {
        self.envs.len()
    }

    /// Resets workers concurrently and concatenates rank-`BO` observations into `[num_envs, ...observation_shape]`.
    /// Observation kind, dtype, device, and item dimensions must match across environments. Gradients are preserved.
    fn reset(
        &mut self,
    ) -> Result<Tensor<BO, <G::ObservationSpace as ObservationSpace<BO>>::Kind>, Self::Error> {
        let sent: Vec<_> = self
            .envs
            .iter()
            .map(|env| env.commands.send(WorkerCommand { action: None }).is_ok())
            .collect();
        let replies: Vec<_> = self
            .envs
            .iter()
            .zip(sent)
            .enumerate()
            .map(|(environment_index, (env, sent))| {
                if !sent {
                    return Err(VectorizedGymError::WorkerDisconnected { environment_index });
                }
                match env.responses.recv() {
                    Ok(WorkerResponse::Reset(result)) => *result,
                    _ => Err(VectorizedGymError::WorkerDisconnected { environment_index }),
                }
            })
            .collect();
        let observations = replies.into_iter().collect::<Result<Vec<_>, _>>()?;
        Ok(Tensor::cat(observations, 0))
    }
}

#[cfg(test)]
mod tests {
    use super::super::{StackedMultiGym, test_support::*};
    use super::*;
    use crate::spaces::{BoxSpace, Discrete};
    use burn::tensor::{Device, Int};

    #[test]
    fn threaded_vectorization_preserves_unbatched_continuous_actions_and_terminals() {
        let device = Device::flex();
        let mut env = MultithreadedVectorizedGymWrapper::new(
            vec![|| ContinuousEnv, || ContinuousEnv],
            BoxSpace::new_unbounded([1, 2], &device),
            BoxSpace::new_with_universal_bounds([1, 2], -10.0, 10.0, &device),
        )
        .unwrap();
        assert_eq!(env.reset().unwrap().dims(), [2, 2]);
        let step = env
            .step(Tensor::from_data(
                [[1.0f64, 2.0], [3.0, 4.0]],
                (&device, burn::tensor::DType::F64),
            ))
            .unwrap();
        assert!(
            step.terminal_observations
                .iter()
                .flatten()
                .all(|observation| observation.dims() == [2])
        );
        assert_eq!(
            step.transition_next_observations()
                .into_data()
                .try_to_vec::<f64>()
                .unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn threaded_vectorization_preserves_auto_reset_and_error_recovery() {
        let factories: Vec<_> = (0..2)
            .map(|id| {
                move || {
                    let mut env = CounterEnv::new(id);
                    env.fail_next = id == 0;
                    env
                }
            })
            .collect();
        let mut env = MultithreadedVectorizedGymWrapper::new(
            factories,
            BoxSpace::new_unbounded([1, 2], &Device::flex()),
            Discrete::new(10),
        )
        .unwrap();
        env.reset().unwrap();
        let actions = Tensor::<1, Int>::from_data([3, 4], &Device::flex());
        assert!(matches!(
            env.step(actions.clone()),
            Err(VectorizedGymError::Single(TestError::Forced))
        ));
        assert_eq!(
            env.reset()
                .unwrap()
                .into_data()
                .try_to_vec::<f64>()
                .unwrap(),
            vec![0.0, 0.0, 1.0, 0.0]
        );
        let first = env.step(actions.clone()).unwrap();
        assert_eq!(
            first.infos[1],
            TestInfo {
                id: 1,
                step: 1,
                action: 4
            }
        );
        let ended = env.step(actions).unwrap();
        assert_eq!(ended.dones, vec![true, false]);
        assert_eq!(ended.truncateds, vec![false, true]);
        assert_eq!(
            ended
                .transition_next_observations()
                .into_data()
                .try_to_vec::<f64>()
                .unwrap(),
            vec![0.0, 2.0, 1.0, 2.0]
        );
        assert_eq!(
            ended.observations.into_data().try_to_vec::<f64>().unwrap(),
            vec![0.0, 0.0, 1.0, 0.0]
        );
    }

    #[test]
    fn threaded_stacking_matches_sequential_order_and_recovers_after_errors() {
        let factories: Vec<_> = [(10, 2), (20, 3)]
            .into_iter()
            .map(|(id, count)| move || GroupEnv::new(id, count))
            .collect();
        let mut threaded = MultithreadedStackedMultiGym::new(
            factories,
            BoxSpace::new_unbounded([1, 3], &Device::flex()),
            BoxSpace::new_unbounded([1, 1], &Device::flex()),
        )
        .unwrap();
        let mut sequential =
            StackedMultiGym::new(vec![GroupEnv::new(10, 2), GroupEnv::new(20, 3)]).unwrap();
        assert_eq!(threaded.group_offsets(), &[0, 2, 5]);
        assert_eq!(threaded.num_groups(), 2);
        assert_eq!(threaded.num_envs(), 5);
        assert_eq!(
            threaded.reset().unwrap().into_data(),
            sequential.reset().unwrap().into_data()
        );
        let actions = Tensor::from_data([[0.0], [1.0], [2.0], [3.0], [4.0]], &Device::flex());
        let a = threaded.step(actions.clone()).unwrap();
        let b = sequential.step(actions).unwrap();
        assert_eq!(
            a.observations.clone().into_data(),
            b.observations.clone().into_data()
        );
        assert_eq!(a.infos, b.infos);
        assert_eq!(a.dones, b.dones);
        assert_eq!(
            a.transition_next_observations().into_data(),
            b.transition_next_observations().into_data()
        );
        let factories: Vec<_> = (0..2)
            .map(|id| {
                move || {
                    let mut env = GroupEnv::new(id, 1);
                    env.fail_next = id == 0;
                    env
                }
            })
            .collect();
        let mut failing = MultithreadedStackedMultiGym::new(
            factories,
            BoxSpace::new_unbounded([1, 3], &Device::flex()),
            BoxSpace::new_unbounded([1, 1], &Device::flex()),
        )
        .unwrap();
        assert!(matches!(
            failing.step(Tensor::zeros([2, 1], &Device::flex())),
            Err(StackedMultiGymError::Inner {
                gym_index: 0,
                error: TestError::Forced
            })
        ));
        assert_eq!(failing.reset().unwrap().dims(), [2, 3]);
        assert!(failing.step(Tensor::zeros([2, 1], &Device::flex())).is_ok());
    }

    #[test]
    fn threaded_groups_enter_step_concurrently() {
        let gate = std::sync::Arc::new((
            std::sync::Mutex::new(GateState::default()),
            std::sync::Condvar::new(),
        ));
        let factories: Vec<_> = (0..2)
            .map(|id| {
                let gate = gate.clone();
                move || {
                    let mut env = GroupEnv::new(id, 1);
                    env.gate = Some(gate);
                    env
                }
            })
            .collect();
        let mut env = MultithreadedStackedMultiGym::new(
            factories,
            BoxSpace::new_unbounded([1, 3], &Device::flex()),
            BoxSpace::new_unbounded([1, 1], &Device::flex()),
        )
        .unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(env.step(Tensor::zeros([2, 1], &Device::flex())));
        });
        let (lock, ready) = &*gate;
        let state = lock.lock().unwrap();
        let (mut state, _) = ready
            .wait_timeout_while(state, std::time::Duration::from_secs(2), |state| {
                state.entered < 2
            })
            .unwrap();
        let entered = state.entered;
        state.released = true;
        ready.notify_all();
        drop(state);
        assert_eq!(entered, 2);
        assert_eq!(
            rx.recv_timeout(std::time::Duration::from_secs(2))
                .unwrap()
                .unwrap()
                .observations
                .dims(),
            [2, 3]
        );
    }

    #[test]
    fn threaded_vectorization_recovers_after_auto_reset_failure() {
        let factories: Vec<_> = (0..2)
            .map(|id| {
                move || {
                    let mut env = CounterEnv::new(id);
                    env.fail_reset = id == 0;
                    env
                }
            })
            .collect();
        let mut env = MultithreadedVectorizedGymWrapper::new(
            factories,
            BoxSpace::new_unbounded([1, 2], &Device::flex()),
            Discrete::new(10),
        )
        .unwrap();
        env.reset().unwrap();
        let actions = Tensor::<1, Int>::from_data([3, 4], &Device::flex());
        env.step(actions.clone()).unwrap();
        assert!(matches!(
            env.step(actions.clone()),
            Err(VectorizedGymError::Single(TestError::Forced))
        ));
        assert_eq!(
            env.reset()
                .unwrap()
                .into_data()
                .try_to_vec::<f64>()
                .unwrap(),
            vec![0.0, 0.0, 1.0, 0.0]
        );
        let step = env.step(actions).unwrap();
        assert_eq!(
            step.infos,
            vec![
                TestInfo {
                    id: 0,
                    step: 1,
                    action: 3
                },
                TestInfo {
                    id: 1,
                    step: 1,
                    action: 4
                }
            ]
        );
    }

    #[test]
    fn worker_panics_are_reported_as_disconnects() {
        fn failed_constructor() -> CounterEnv {
            panic!("fixture constructor failure");
        }
        assert!(matches!(
            MultithreadedVectorizedGymWrapper::new(
                vec![failed_constructor],
                BoxSpace::new_unbounded([1, 2], &Device::flex()),
                Discrete::new(2)
            ),
            Err(VectorizedGymError::WorkerDisconnected {
                environment_index: 0
            })
        ));
    }
}

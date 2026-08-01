use crate::spaces::Space;
use candle_core::Tensor;

/// A single reinforcement-learning environment.
///
/// `I` is typed environment metadata returned by resets and transitions. It
/// defaults to `()`, so environments without extra metadata can simply write
/// `impl Gym for MyEnvironment`.
pub trait Gym<I = ()> {
    type Error;
    type SpaceError;

    /// Steps with one unbatched environment action. A [`Discrete`](crate::spaces::Discrete)
    /// action is scalar `[]`; a [`BoxSpace`](crate::spaces::BoxSpace) action is
    /// shaped `action_space().shape()`.
    fn step(&mut self, action: Tensor) -> Result<StepInfo<I>, Self::Error>;
    /// Resets the environment to its initial state.
    fn reset(&mut self) -> Result<ResetInfo<I>, Self::Error>;
    /// Returns the observation space.
    fn observation_space(&self) -> Box<dyn Space<Error = Self::SpaceError>>;
    /// Returns the action space.
    fn action_space(&self) -> Box<dyn Space<Error = Self::SpaceError>>;
}

/// A batched environment interface used by actors.
///
/// Most implementations batch independent environments. A custom
/// implementation may instead batch coupled players from one shared world, as
/// long as it accepts and returns one consistently ordered row per slot.
/// [`VectorizedGymWrapper`] provides the independent-environment form for any
/// [`Gym`]. Reset is called automatically when an environment is done.
pub trait MultiGym<I = ()> {
    type Error;
    type SpaceError;

    /// Steps every batch slot with batched environment actions. Discrete
    /// actions are `[num_envs]`; box actions are
    /// `[num_envs, ...action_space().shape()]`.
    fn step(&mut self, action: Tensor) -> Result<MultiGymStepInfo<I>, Self::Error>;
    fn observation_space(&self) -> Box<dyn Space<Error = Self::SpaceError>>;
    fn action_space(&self) -> Box<dyn Space<Error = Self::SpaceError>>;
    fn num_envs(&self) -> usize;
    /// Resets all batch slots and returns their initial states.
    /// Only needs to be called once at the start of training.
    fn reset(&mut self) -> Result<Tensor, Self::Error>;
}

#[derive(Debug)]
pub enum VectorizedGymError<E>
where
    E: std::fmt::Debug,
{
    Single(E),
    Batch(candle_core::Error),
}

impl<E> From<candle_core::Error> for VectorizedGymError<E>
where
    E: std::fmt::Debug,
{
    fn from(err: candle_core::Error) -> Self {
        VectorizedGymError::Batch(err)
    }
}

#[derive(Debug, Clone)]
/// The initial observation and environment-specific metadata from a reset.
pub struct ResetInfo<I = ()> {
    pub state: Tensor,
    pub info: I,
}

#[derive(Debug, Clone)]
/// The result of one environment transition.
pub struct StepInfo<I = ()> {
    pub state: Tensor,
    pub reward: f32,
    pub done: bool,
    pub truncated: bool,
    pub info: I,
}

#[derive(Debug, Clone)]
pub struct MultiGymStepInfo<I = ()> {
    pub states: Tensor,
    pub rewards: Tensor,
    pub infos: Vec<I>,
    // really wish candle had a bool tensor type
    pub dones: Vec<bool>,
    pub truncateds: Vec<bool>,
    pub terminal_states: Vec<Option<Tensor>>,
}

impl<I> MultiGymStepInfo<I> {
    pub fn transition_next_states(&self) -> candle_core::Result<Tensor> {
        let env_count = self.dones.len();
        let state_chunks = self.states.chunk(env_count, 0)?;
        let mut next_states = Vec::with_capacity(env_count);

        for (i, state_chunk) in state_chunks.iter().enumerate().take(env_count) {
            match &self.terminal_states[i] {
                Some(state) => next_states.push(state.clone()),
                None => next_states.push(state_chunk.clone().squeeze(0)?),
            }
        }

        Tensor::stack(&next_states, 0)
    }
}

/// An error produced while combining several [`MultiGym`] implementations.
#[derive(Debug)]
pub enum StackedMultiGymError<E>
where
    E: std::fmt::Debug,
{
    /// At least one inner gym is required.
    Empty,
    /// An inner gym reported no batch slots.
    EmptyInner { gym_index: usize },
    /// An inner gym has a different per-slot observation shape.
    IncompatibleObservationShape {
        gym_index: usize,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    /// An inner gym has a different per-slot action shape.
    IncompatibleActionShape {
        gym_index: usize,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    /// An inner gym changed its number of batch slots after construction.
    ChangedBatchSize {
        gym_index: usize,
        expected: usize,
        actual: usize,
    },
    /// The supplied action does not have the combined batch size in dimension zero.
    InvalidActionBatch {
        expected: usize,
        actual: Option<usize>,
    },
    /// An inner gym returned a tensor or vector that violates the [`MultiGym`] contract.
    InvalidOutputShape {
        gym_index: usize,
        field: &'static str,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    /// An inner gym returned an error.
    Inner { gym_index: usize, error: E },
    /// Candle could not slice or concatenate a batch.
    Batch(candle_core::Error),
}

impl<E> From<candle_core::Error> for StackedMultiGymError<E>
where
    E: std::fmt::Debug,
{
    fn from(error: candle_core::Error) -> Self {
        Self::Batch(error)
    }
}

/// Flattens the batch rows from several homogeneous [`MultiGym`] values.
///
/// If the inner gyms return states shaped `[B0, ...]`, `[B1, ...]`, and so on,
/// this wrapper returns states shaped `[B0 + B1 + ..., ...]`. Actions are
/// sliced back into those same contiguous ranges before each inner gym is
/// stepped. Inner gyms retain responsibility for their own auto-reset logic.
pub struct StackedMultiGym<G, I = ()>
where
    G: MultiGym<I>,
{
    gyms: Vec<G>,
    group_offsets: Vec<usize>,
    observation_shape: Vec<usize>,
    _info: std::marker::PhantomData<fn() -> I>,
}

impl<G, I> StackedMultiGym<G, I>
where
    G: MultiGym<I>,
    G::Error: std::fmt::Debug,
{
    /// Creates a flattened batch from one or more homogeneous inner gyms.
    pub fn try_new(gyms: Vec<G>) -> Result<Self, StackedMultiGymError<G::Error>> {
        let Some(first) = gyms.first() else {
            return Err(StackedMultiGymError::Empty);
        };
        let observation_shape = first.observation_space().shape();
        let action_shape = first.action_space().shape();
        let mut group_offsets = Vec::with_capacity(gyms.len() + 1);
        group_offsets.push(0);

        for (gym_index, gym) in gyms.iter().enumerate() {
            let batch_size = gym.num_envs();
            if batch_size == 0 {
                return Err(StackedMultiGymError::EmptyInner { gym_index });
            }

            let actual_observation_shape = gym.observation_space().shape();
            if actual_observation_shape != observation_shape {
                return Err(StackedMultiGymError::IncompatibleObservationShape {
                    gym_index,
                    expected: observation_shape,
                    actual: actual_observation_shape,
                });
            }

            let actual_action_shape = gym.action_space().shape();
            if actual_action_shape != action_shape {
                return Err(StackedMultiGymError::IncompatibleActionShape {
                    gym_index,
                    expected: action_shape,
                    actual: actual_action_shape,
                });
            }

            group_offsets.push(group_offsets.last().copied().unwrap() + batch_size);
        }

        Ok(Self {
            gyms,
            group_offsets,
            observation_shape,
            _info: std::marker::PhantomData,
        })
    }

    /// Returns the inner gyms in their batch ordering.
    pub fn gyms(&self) -> &[G] {
        &self.gyms
    }

    /// Returns cumulative row offsets, including zero and the total batch size.
    ///
    /// For inner batch sizes two and three this returns `[0, 2, 5]`.
    pub fn group_offsets(&self) -> &[usize] {
        &self.group_offsets
    }

    /// Returns the number of independently stepped inner gyms.
    pub fn num_groups(&self) -> usize {
        self.gyms.len()
    }

    fn validate_batch_sizes(&self) -> Result<(), StackedMultiGymError<G::Error>> {
        for (gym_index, gym) in self.gyms.iter().enumerate() {
            let expected = self.group_offsets[gym_index + 1] - self.group_offsets[gym_index];
            let actual = gym.num_envs();
            if actual != expected {
                return Err(StackedMultiGymError::ChangedBatchSize {
                    gym_index,
                    expected,
                    actual,
                });
            }
        }
        Ok(())
    }

    fn validate_output_shape(
        &self,
        gym_index: usize,
        field: &'static str,
        actual: &[usize],
        expected: Vec<usize>,
    ) -> Result<(), StackedMultiGymError<G::Error>> {
        if actual != expected {
            return Err(StackedMultiGymError::InvalidOutputShape {
                gym_index,
                field,
                expected,
                actual: actual.to_vec(),
            });
        }
        Ok(())
    }
}

impl<G, I> TryFrom<Vec<G>> for StackedMultiGym<G, I>
where
    G: MultiGym<I>,
    G::Error: std::fmt::Debug,
{
    type Error = StackedMultiGymError<G::Error>;

    fn try_from(gyms: Vec<G>) -> Result<Self, Self::Error> {
        Self::try_new(gyms)
    }
}

impl<G, I> MultiGym<I> for StackedMultiGym<G, I>
where
    G: MultiGym<I>,
    G::Error: std::fmt::Debug,
{
    type Error = StackedMultiGymError<G::Error>;
    type SpaceError = G::SpaceError;

    /// Steps with actions shaped `[total_batch_size, ...]`, splitting dimension
    /// zero across the inner gyms according to [`Self::group_offsets`].
    fn step(&mut self, action: Tensor) -> Result<MultiGymStepInfo<I>, Self::Error> {
        self.validate_batch_sizes()?;
        let total_batch_size = self.num_envs();
        let actual_batch_size = action.dims().first().copied();
        if actual_batch_size != Some(total_batch_size) {
            return Err(StackedMultiGymError::InvalidActionBatch {
                expected: total_batch_size,
                actual: actual_batch_size,
            });
        }

        let mut states = Vec::with_capacity(self.gyms.len());
        let mut rewards = Vec::with_capacity(self.gyms.len());
        let mut infos = Vec::with_capacity(total_batch_size);
        let mut dones = Vec::with_capacity(total_batch_size);
        let mut truncateds = Vec::with_capacity(total_batch_size);
        let mut terminal_states = Vec::with_capacity(total_batch_size);

        for gym_index in 0..self.gyms.len() {
            let start = self.group_offsets[gym_index];
            let batch_size = self.group_offsets[gym_index + 1] - start;
            let child_action = action.narrow(0, start, batch_size)?;
            let step = self.gyms[gym_index]
                .step(child_action)
                .map_err(|error| StackedMultiGymError::Inner { gym_index, error })?;

            let mut expected_state_shape = Vec::with_capacity(self.observation_shape.len() + 1);
            expected_state_shape.push(batch_size);
            expected_state_shape.extend_from_slice(&self.observation_shape);
            self.validate_output_shape(
                gym_index,
                "states",
                step.states.dims(),
                expected_state_shape,
            )?;
            self.validate_output_shape(
                gym_index,
                "rewards",
                step.rewards.dims(),
                vec![batch_size],
            )?;
            self.validate_output_shape(gym_index, "infos", &[step.infos.len()], vec![batch_size])?;
            self.validate_output_shape(gym_index, "dones", &[step.dones.len()], vec![batch_size])?;
            self.validate_output_shape(
                gym_index,
                "truncateds",
                &[step.truncateds.len()],
                vec![batch_size],
            )?;
            self.validate_output_shape(
                gym_index,
                "terminal_states",
                &[step.terminal_states.len()],
                vec![batch_size],
            )?;

            states.push(step.states);
            rewards.push(step.rewards);
            infos.extend(step.infos);
            dones.extend(step.dones);
            truncateds.extend(step.truncateds);
            terminal_states.extend(step.terminal_states);
        }

        Ok(MultiGymStepInfo {
            states: Tensor::cat(&states, 0)?,
            rewards: Tensor::cat(&rewards, 0)?,
            infos,
            dones,
            truncateds,
            terminal_states,
        })
    }

    fn observation_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
        self.gyms[0].observation_space()
    }

    fn action_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
        self.gyms[0].action_space()
    }

    fn num_envs(&self) -> usize {
        *self.group_offsets.last().unwrap()
    }

    fn reset(&mut self) -> Result<Tensor, Self::Error> {
        self.validate_batch_sizes()?;
        let mut states = Vec::with_capacity(self.gyms.len());

        for gym_index in 0..self.gyms.len() {
            let batch_size = self.group_offsets[gym_index + 1] - self.group_offsets[gym_index];
            let state = self.gyms[gym_index]
                .reset()
                .map_err(|error| StackedMultiGymError::Inner { gym_index, error })?;
            let mut expected_shape = Vec::with_capacity(self.observation_shape.len() + 1);
            expected_shape.push(batch_size);
            expected_shape.extend_from_slice(&self.observation_shape);
            self.validate_output_shape(gym_index, "reset states", state.dims(), expected_shape)?;
            states.push(state);
        }

        Tensor::cat(&states, 0).map_err(StackedMultiGymError::Batch)
    }
}

pub struct VectorizedGymWrapper<G, I = ()>
where
    G: Gym<I>,
{
    envs: Vec<G>,
    to_reset: Vec<bool>,
    _info: std::marker::PhantomData<fn() -> I>,
}

impl<G, I> VectorizedGymWrapper<G, I>
where
    G: Gym<I>,
{
    pub fn new(envs: Vec<G>) -> Self {
        Self {
            to_reset: vec![true; envs.len()],
            envs,
            _info: std::marker::PhantomData,
        }
    }

    pub fn envs(&self) -> &Vec<G> {
        &self.envs
    }

    pub fn envs_mut(&mut self) -> &mut Vec<G> {
        &mut self.envs
    }
}

impl<G, I> MultiGym<I> for VectorizedGymWrapper<G, I>
where
    G: Gym<I>,
    G::Error: std::fmt::Debug,
{
    type Error = VectorizedGymError<G::Error>;
    type SpaceError = G::SpaceError;

    /// Steps with discrete actions `[num_envs]` or box actions
    /// `[num_envs, ...action_shape]`.
    fn step(&mut self, action: Tensor) -> Result<MultiGymStepInfo<I>, Self::Error> {
        let env_count = self.envs.len();
        let actions: Vec<Tensor> = action.chunk(env_count, 0)?;
        // this keeps the dimension for env count it's just 1 now so we squeeze it later

        let mut states = Vec::with_capacity(env_count);
        let mut rewards = Vec::with_capacity(env_count);
        let mut infos = Vec::with_capacity(env_count);
        let mut dones = Vec::with_capacity(env_count);
        let mut truncateds = Vec::with_capacity(env_count);
        let mut terminal_states = Vec::with_capacity(env_count);

        for (i, mut act) in actions.iter().cloned().enumerate() {
            let env = &mut self.envs[i];
            act = act.squeeze(0)?;

            let mut step_info = env.step(act).map_err(VectorizedGymError::Single)?;
            let terminal_state = if step_info.done || step_info.truncated {
                Some(step_info.state.clone())
            } else {
                None
            };
            if step_info.done || step_info.truncated {
                step_info.state = env.reset().map_err(VectorizedGymError::Single)?.state;
            }

            states.push(step_info.state);
            rewards.push(step_info.reward);
            infos.push(step_info.info);
            dones.push(step_info.done);
            truncateds.push(step_info.truncated);
            terminal_states.push(terminal_state);
        }

        let states = Tensor::stack(&states, 0)?;
        let rewards = Tensor::from_vec(rewards, &[env_count], states.device())?;

        Ok(MultiGymStepInfo {
            states,
            rewards,
            infos,
            dones,
            truncateds,
            terminal_states,
        })
    }

    fn observation_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
        self.envs[0].observation_space()
    }

    fn action_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
        self.envs[0].action_space()
    }

    fn reset(&mut self) -> Result<Tensor, Self::Error> {
        let batch_size = self.envs.len();
        let mut states = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let env = &mut self.envs[i];
            let reset = env.reset().map_err(VectorizedGymError::Single)?;
            states.push(reset.state);
            self.to_reset[i] = false;
        }

        let states = Tensor::stack(&states, 0)?;
        Ok(states)
    }

    fn num_envs(&self) -> usize {
        self.envs.len()
    }
}

impl<G, I> From<G> for VectorizedGymWrapper<G, I>
where
    G: Gym<I> + Clone,
{
    fn from(env: G) -> Self {
        VectorizedGymWrapper::new(vec![env])
    }
}

impl<G, I> From<Vec<G>> for VectorizedGymWrapper<G, I>
where
    G: Gym<I>,
{
    fn from(envs: Vec<G>) -> Self {
        VectorizedGymWrapper::new(envs)
    }
}

#[cfg(feature = "multithreading")]
use std::{fmt::Debug, sync::mpsc, thread};

#[cfg(feature = "multithreading")]
pub struct MultithreadedVectorizedGymWrapper<G, O, A, SE, I = ()>
where
    G: Gym<I> + 'static,
    G::Error: Send + Sync + std::fmt::Debug,
    SE: Debug,
    A: Space<Error = SE> + Clone + 'static,
    O: Space<Error = SE> + Clone + 'static,
    I: Send + 'static,
{
    envs: Vec<GymHandle<G::Error, I>>,
    to_reset: Vec<bool>,
    obs_space: O,
    action_space: A,
    _phantom: std::marker::PhantomData<(SE, fn() -> I)>,
}

#[cfg(feature = "multithreading")]
impl<G, O, A, SE, I> MultithreadedVectorizedGymWrapper<G, O, A, SE, I>
where
    G: Gym<I> + 'static,
    G::Error: Send + Sync + std::fmt::Debug,
    SE: Debug,
    A: Space<Error = SE> + Clone + 'static,
    O: Space<Error = SE> + Clone + 'static,
    I: Send + 'static,
{
    pub fn new<F>(env_constructors: Vec<F>, obs_space: O, action_space: A) -> Self
    where
        F: FnOnce() -> G + Send + 'static,
    {
        assert!(
            !env_constructors.is_empty(),
            "Must provide at least one environment constructor"
        );

        let envs: Vec<GymHandle<<G as Gym<I>>::Error, I>> = env_constructors
            .into_iter()
            .map(|constructor| start_gym_thread(constructor))
            .collect();

        Self {
            to_reset: vec![true; envs.len()],
            envs,
            obs_space,
            action_space,
            _phantom: std::marker::PhantomData,
        }
    }
}

#[cfg(feature = "multithreading")]
impl<G, O, A, SE, I> MultiGym<I> for MultithreadedVectorizedGymWrapper<G, O, A, SE, I>
where
    G: Gym<I> + 'static,
    G::Error: Send + Sync + std::fmt::Debug,
    SE: Debug,
    A: Space<Error = SE> + Clone + 'static,
    O: Space<Error = SE> + Clone + 'static,
    I: Send + 'static,
{
    type Error = VectorizedGymError<G::Error>;
    type SpaceError = SE;

    /// Steps with discrete actions `[num_envs]` or box actions
    /// `[num_envs, ...action_shape]`.
    fn step(&mut self, action: Tensor) -> Result<MultiGymStepInfo<I>, Self::Error> {
        let batch_size = self.envs.len();
        let actions: Vec<Tensor> = action.chunk(batch_size, 0)?;
        let mut step_info_recievers = Vec::with_capacity(batch_size);

        for ((env, act), to_reset) in self
            .envs
            .iter()
            .zip(actions.into_iter())
            .zip(self.to_reset.iter_mut())
        {
            let act = act.clone().squeeze(0)?;
            *to_reset = false;
            let step_info = env.step(act);
            step_info_recievers.push(step_info);
        }

        let mut states = Vec::with_capacity(batch_size);
        let mut rewards = Vec::with_capacity(batch_size);
        let mut infos = Vec::with_capacity(batch_size);
        let mut dones = Vec::with_capacity(batch_size);
        let mut truncateds = Vec::with_capacity(batch_size);
        let mut terminal_states = Vec::with_capacity(batch_size);

        for reciever in step_info_recievers {
            let thread_step_info = reciever
                .recv()
                .expect("Failed to receive step info, this was probably caused by a panic in the gym thread")
                .map_err(VectorizedGymError::Single)?;
            let step_info = thread_step_info.step_info;
            states.push(step_info.state);
            rewards.push(step_info.reward);
            infos.push(step_info.info);
            dones.push(step_info.done);
            truncateds.push(step_info.truncated);
            terminal_states.push(thread_step_info.terminal_state);
        }

        let states = Tensor::stack(&states, 0)?;
        let rewards = Tensor::from_vec(rewards, &[batch_size], states.device())?;

        Ok(MultiGymStepInfo {
            states,
            rewards,
            infos,
            dones,
            truncateds,
            terminal_states,
        })
    }

    fn num_envs(&self) -> usize {
        self.envs.len()
    }

    fn reset(&mut self) -> Result<Tensor, Self::Error> {
        let batch_size = self.envs.len();

        let states: Vec<std::sync::mpsc::Receiver<Result<ResetInfo<I>, <G as Gym<I>>::Error>>> =
            self.envs.iter().map(|env| env.reset()).collect();
        let states: Vec<Result<Tensor, VectorizedGymError<G::Error>>> = states
            .into_iter()
            .map(|reciever| {
                reciever
                    .recv()
                    .expect("Failed to receive reset info, this was probably caused by a panic in the gym thread")
                    .map_err(VectorizedGymError::Single)
                    // collects only the state if the result is ok
                    .map(|info| info.state)
            })
            .collect();

        let mut state_tensors = Vec::with_capacity(batch_size);
        for state in states {
            state_tensors.push(state?);
        }

        self.to_reset = vec![false; batch_size];

        let states = Tensor::stack(&state_tensors, 0)?;
        Ok(states)
    }

    fn observation_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
        Box::new(self.obs_space.clone())
    }

    fn action_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
        Box::new(self.action_space.clone())
    }
}

#[cfg(feature = "multithreading")]
enum GymCmd<E, I>
where
    E: Send + Sync,
{
    Step(Tensor, mpsc::Sender<Result<ThreadStepInfo<I>, E>>),
    Reset(mpsc::Sender<Result<ResetInfo<I>, E>>),
}

#[cfg(feature = "multithreading")]
struct ThreadStepInfo<I> {
    step_info: StepInfo<I>,
    terminal_state: Option<Tensor>,
}

#[cfg(feature = "multithreading")]
struct GymHandle<E, I>
where
    E: Send + Sync,
{
    tx: mpsc::Sender<GymCmd<E, I>>,
}

#[cfg(feature = "multithreading")]
impl<E, I> GymHandle<E, I>
where
    E: Send + Sync,
{
    /// Sends one unbatched environment action (`[]` for discrete or
    /// `action_space.shape()` for box actions) to the worker thread.
    fn step(&self, action: Tensor) -> std::sync::mpsc::Receiver<Result<ThreadStepInfo<I>, E>> {
        let (resp_tx, resp_rx) = mpsc::channel();
        self.tx.send(GymCmd::Step(action, resp_tx)).unwrap();
        return resp_rx;
    }

    fn reset(&self) -> std::sync::mpsc::Receiver<Result<ResetInfo<I>, E>> {
        let (resp_tx, resp_rx) = mpsc::channel();
        self.tx.send(GymCmd::Reset(resp_tx)).unwrap();
        return resp_rx;
    }
}

#[cfg(feature = "multithreading")]
/// Spawns a persistent thread that constructs and owns the gym.
/// The closure is executed inside that thread to build the gym.
fn start_gym_thread<G, F, I>(make_gym: F) -> GymHandle<G::Error, I>
where
    F: FnOnce() -> G + Send + 'static,
    G: Gym<I> + 'static,
    G::Error: Send + Sync + 'static,
    I: Send + 'static,
{
    let (tx, rx) = mpsc::channel::<GymCmd<G::Error, I>>();

    thread::spawn(move || {
        let mut gym = make_gym(); // constructed *inside* the thread
        while let Ok(cmd) = rx.recv() {
            match cmd {
                GymCmd::Step(action, resp_tx) => {
                    let mut step_info = gym.step(action);
                    let mut terminal_state = None;
                    if let Ok(info) = &mut step_info
                        && (info.done || info.truncated)
                    {
                        terminal_state = Some(info.state.clone());
                        match gym.reset() {
                            Ok(reset) => {
                                info.state = reset.state;
                            }
                            Err(err) => {
                                resp_tx.send(Err(err)).expect(
                                    "Failed to send step response, this was probably caused by a panic in the gym thread",
                                );
                                continue;
                            }
                        }
                    }
                    resp_tx
                        .send(step_info.map(|info| ThreadStepInfo {
                            step_info: info,
                            terminal_state,
                        }))
                        .expect(
                            "Failed to send step response, this was probably caused by a panic in the gym thread",
                        );
                }
                GymCmd::Reset(resp_tx) => {
                    let reset_info = gym.reset();
                    resp_tx.send(reset_info).expect(
                        "Failed to send reset response, this was probably caused by a panic in the gym thread",
                    );
                }
            }
        }
    });

    GymHandle { tx }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct TestInfo {
        value: usize,
    }

    struct InfoEnv;

    impl Gym<TestInfo> for InfoEnv {
        type Error = candle_core::Error;
        type SpaceError = candle_core::Error;

        /// Steps with one scalar discrete action shaped `[]`.
        fn step(&mut self, _action: Tensor) -> Result<StepInfo<TestInfo>, Self::Error> {
            Ok(StepInfo {
                state: Tensor::zeros(1, candle_core::DType::F32, &candle_core::Device::Cpu)?,
                reward: 0.0,
                done: false,
                truncated: false,
                info: TestInfo { value: 2 },
            })
        }

        fn reset(&mut self) -> Result<ResetInfo<TestInfo>, Self::Error> {
            Ok(ResetInfo {
                state: Tensor::zeros(1, candle_core::DType::F32, &candle_core::Device::Cpu)?,
                info: TestInfo { value: 1 },
            })
        }

        fn observation_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
            Box::new(crate::spaces::BoxSpace::new_with_universal_bounds(
                vec![1],
                -1.0,
                1.0,
                &candle_core::Device::Cpu,
            ))
        }

        fn action_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
            Box::new(crate::spaces::Discrete::new(2))
        }
    }

    struct DummyEnv {
        step_count: usize,
        id: usize,
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct BatchedTestInfo {
        group: usize,
        slot: usize,
        action: f32,
    }

    #[derive(Debug)]
    enum BatchedTestError {
        Forced,
        Tensor,
    }

    impl From<candle_core::Error> for BatchedTestError {
        fn from(_error: candle_core::Error) -> Self {
            Self::Tensor
        }
    }

    struct BatchedDummyEnv {
        group: usize,
        batch_size: usize,
        observation_size: usize,
        action_size: usize,
        fail_step: bool,
        step_count: usize,
    }

    impl BatchedDummyEnv {
        fn new(group: usize, batch_size: usize) -> Self {
            Self {
                group,
                batch_size,
                observation_size: 3,
                action_size: 1,
                fail_step: false,
                step_count: 0,
            }
        }

        fn observations(&self, actions: &[f32]) -> Result<Tensor, BatchedTestError> {
            let mut values = Vec::with_capacity(self.batch_size * self.observation_size);
            for (slot, action) in actions.iter().enumerate() {
                values.extend_from_slice(&[self.group as f32, slot as f32, *action]);
            }
            Tensor::from_vec(
                values,
                &[self.batch_size, self.observation_size],
                &candle_core::Device::Cpu,
            )
            .map_err(BatchedTestError::from)
        }
    }

    impl MultiGym<BatchedTestInfo> for BatchedDummyEnv {
        type Error = BatchedTestError;
        type SpaceError = candle_core::Error;

        /// Steps with test actions shaped `[batch_size, 1]`.
        fn step(
            &mut self,
            action: Tensor,
        ) -> Result<MultiGymStepInfo<BatchedTestInfo>, Self::Error> {
            if self.fail_step {
                return Err(BatchedTestError::Forced);
            }
            self.step_count += 1;
            let actions = action.flatten_all()?.to_vec1::<f32>()?;
            let states = self.observations(&actions)?;
            let infos = actions
                .iter()
                .enumerate()
                .map(|(slot, action)| BatchedTestInfo {
                    group: self.group,
                    slot,
                    action: *action,
                })
                .collect();
            let mut dones = vec![false; self.batch_size];
            dones[self.batch_size - 1] = true;
            let terminal_states = states
                .chunk(self.batch_size, 0)?
                .into_iter()
                .enumerate()
                .map(|(slot, state)| {
                    (slot == self.batch_size - 1)
                        .then(|| state.squeeze(0))
                        .transpose()
                })
                .collect::<Result<Vec<_>, _>>()?;

            Ok(MultiGymStepInfo {
                states,
                rewards: Tensor::from_vec(actions, &[self.batch_size], &candle_core::Device::Cpu)?,
                infos,
                dones,
                truncateds: vec![false; self.batch_size],
                terminal_states,
            })
        }

        fn observation_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
            Box::new(crate::spaces::BoxSpace::new_unbounded(
                vec![self.observation_size],
                &candle_core::Device::Cpu,
            ))
        }

        fn action_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
            Box::new(crate::spaces::BoxSpace::new_with_universal_bounds(
                vec![self.action_size],
                -1.0,
                1.0,
                &candle_core::Device::Cpu,
            ))
        }

        fn num_envs(&self) -> usize {
            self.batch_size
        }

        fn reset(&mut self) -> Result<Tensor, Self::Error> {
            self.observations(&vec![-1.0; self.batch_size])
        }
    }

    impl DummyEnv {
        fn new(id: usize) -> Self {
            Self { step_count: 0, id }
        }
    }

    impl Gym for DummyEnv {
        type Error = ();
        type SpaceError = candle_core::Error;

        /// Steps with one scalar discrete action shaped `[]`.
        fn step(&mut self, _action: Tensor) -> Result<StepInfo, Self::Error> {
            self.step_count += 1;
            let done = self.step_count >= 5;
            Ok(StepInfo {
                state: Tensor::from_vec(
                    vec![self.id as f32, self.step_count as f32],
                    &[2],
                    &candle_core::Device::Cpu,
                )
                .unwrap(),
                reward: 1.0,
                done,
                truncated: false,
                info: (),
            })
        }

        fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
            self.step_count = 0;
            Ok(ResetInfo {
                state: Tensor::from_vec(vec![self.id as f32, 0.0], &[2], &candle_core::Device::Cpu)
                    .unwrap(),
                info: (),
            })
        }

        fn observation_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
            Box::new(crate::spaces::BoxSpace::new(
                Tensor::full(-1000.0, &[2], &candle_core::Device::Cpu).unwrap(),
                Tensor::full(1000.0, &[2], &candle_core::Device::Cpu).unwrap(),
            ))
        }

        fn action_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
            Box::new(crate::spaces::Discrete::new(2))
        }
    }

    #[test]
    fn test_vectorized_gym() {
        let envs: Vec<DummyEnv> = (0..3).map(DummyEnv::new).collect();
        let mut vec_env = VectorizedGymWrapper::new(envs);

        let initial_states = vec_env.reset().unwrap();
        assert_eq!(initial_states.shape().dims(), &[3, 2]);
        assert_eq!(
            initial_states.to_vec2::<f32>().unwrap(),
            vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![2.0, 0.0]]
        );

        for step in 0..7 {
            let actions = Tensor::full(1.0, &[3, 1], &candle_core::Device::Cpu).unwrap();
            let step_info = vec_env.step(actions).unwrap();
            assert_eq!(step_info.states.shape().dims(), &[3, 2]);
            assert_eq!(
                step_info.rewards.to_vec1::<f32>().unwrap(),
                vec![1.0, 1.0, 1.0]
            );

            for (i, state) in step_info
                .states
                .to_vec2::<f32>()
                .unwrap()
                .iter()
                .enumerate()
            {
                let expected_step = match step {
                    0..=3 => step + 1,
                    4 => 0,
                    _ => step - 4,
                };
                assert_eq!(*state, vec![i as f32, expected_step as f32]);
            }

            if step != 4 {
                assert_eq!(step_info.dones, vec![false, false, false]);
                assert!(
                    step_info
                        .terminal_states
                        .iter()
                        .all(|state| state.is_none())
                );
            } else {
                assert_eq!(step_info.dones, vec![true, true, true]);
                let terminal_states = step_info.transition_next_states().unwrap();
                assert_eq!(
                    terminal_states.to_vec2::<f32>().unwrap(),
                    vec![vec![0.0, 5.0], vec![1.0, 5.0], vec![2.0, 5.0]]
                );
            }
        }
    }

    #[test]
    fn stacked_multi_gym_flattens_batches_and_routes_actions() {
        let mut env = StackedMultiGym::try_new(vec![
            BatchedDummyEnv::new(10, 2),
            BatchedDummyEnv::new(20, 3),
        ])
        .unwrap();

        assert_eq!(env.num_groups(), 2);
        assert_eq!(env.num_envs(), 5);
        assert_eq!(env.group_offsets(), &[0, 2, 5]);
        assert_eq!(env.reset().unwrap().dims(), &[5, 3]);

        let actions = Tensor::from_vec(
            vec![0.0_f32, 0.1, 0.2, 0.3, 0.4],
            &[5, 1],
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let step = env.step(actions).unwrap();

        assert_eq!(step.states.dims(), &[5, 3]);
        assert_eq!(
            step.states.to_vec2::<f32>().unwrap(),
            vec![
                vec![10.0, 0.0, 0.0],
                vec![10.0, 1.0, 0.1],
                vec![20.0, 0.0, 0.2],
                vec![20.0, 1.0, 0.3],
                vec![20.0, 2.0, 0.4],
            ]
        );
        assert_eq!(
            step.rewards.to_vec1::<f32>().unwrap(),
            vec![0.0, 0.1, 0.2, 0.3, 0.4]
        );
        assert_eq!(
            step.infos,
            vec![
                BatchedTestInfo {
                    group: 10,
                    slot: 0,
                    action: 0.0,
                },
                BatchedTestInfo {
                    group: 10,
                    slot: 1,
                    action: 0.1,
                },
                BatchedTestInfo {
                    group: 20,
                    slot: 0,
                    action: 0.2,
                },
                BatchedTestInfo {
                    group: 20,
                    slot: 1,
                    action: 0.3,
                },
                BatchedTestInfo {
                    group: 20,
                    slot: 2,
                    action: 0.4,
                },
            ]
        );
        assert_eq!(step.dones, vec![false, true, false, false, true]);
        assert_eq!(step.truncateds, vec![false; 5]);
        assert!(step.terminal_states[1].is_some());
        assert!(step.terminal_states[4].is_some());
        assert_eq!(env.gyms()[0].step_count, 1);
        assert_eq!(env.gyms()[1].step_count, 1);
    }

    #[test]
    fn stacked_multi_gym_validates_construction_and_actions() {
        assert!(matches!(
            StackedMultiGym::<BatchedDummyEnv, BatchedTestInfo>::try_new(vec![]),
            Err(StackedMultiGymError::Empty)
        ));
        assert!(matches!(
            StackedMultiGym::try_new(vec![BatchedDummyEnv::new(0, 0)]),
            Err(StackedMultiGymError::EmptyInner { gym_index: 0 })
        ));

        let mut different_observation = BatchedDummyEnv::new(1, 2);
        different_observation.observation_size = 4;
        assert!(matches!(
            StackedMultiGym::try_new(vec![BatchedDummyEnv::new(0, 2), different_observation]),
            Err(StackedMultiGymError::IncompatibleObservationShape { gym_index: 1, .. })
        ));

        let mut different_action = BatchedDummyEnv::new(1, 2);
        different_action.action_size = 2;
        assert!(matches!(
            StackedMultiGym::try_new(vec![BatchedDummyEnv::new(0, 2), different_action]),
            Err(StackedMultiGymError::IncompatibleActionShape { gym_index: 1, .. })
        ));

        let mut env = StackedMultiGym::try_new(vec![BatchedDummyEnv::new(0, 2)]).unwrap();
        let wrong_batch =
            Tensor::zeros((3, 1), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();
        assert!(matches!(
            env.step(wrong_batch),
            Err(StackedMultiGymError::InvalidActionBatch {
                expected: 2,
                actual: Some(3),
            })
        ));
    }

    #[test]
    fn stacked_multi_gym_reports_the_failing_group() {
        let mut failing = BatchedDummyEnv::new(1, 2);
        failing.fail_step = true;
        let mut env = StackedMultiGym::try_new(vec![BatchedDummyEnv::new(0, 2), failing]).unwrap();
        let actions =
            Tensor::zeros((4, 1), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();

        assert!(matches!(
            env.step(actions),
            Err(StackedMultiGymError::Inner {
                gym_index: 1,
                error: BatchedTestError::Forced,
            })
        ));
    }

    #[test]
    fn vectorized_wrapper_preserves_non_unit_step_info() {
        let mut vec_env = VectorizedGymWrapper::from(vec![InfoEnv]);
        assert_eq!(vec_env.reset().unwrap().dims(), &[1, 1]);

        let actions =
            Tensor::zeros((1, 1), candle_core::DType::U32, &candle_core::Device::Cpu).unwrap();
        let step = vec_env.step(actions).unwrap();
        assert_eq!(step.states.dims(), &[1, 1]);
        assert_eq!(step.infos, vec![TestInfo { value: 2 }]);
    }

    #[cfg(feature = "multithreading")]
    #[test]
    fn test_multithreaded_vectorized_gym() {
        let env_constructors = (0..3).map(|i| move || DummyEnv::new(i)).collect();

        // We have to manually specify the observation and action spaces here
        // Because otherwise rust doesn't know it's cloneable
        let obs_space = crate::spaces::BoxSpace::new(
            Tensor::full(-1000.0, &[2], &candle_core::Device::Cpu).unwrap(),
            Tensor::full(1000.0, &[2], &candle_core::Device::Cpu).unwrap(),
        );
        let action_space = crate::spaces::Discrete::new(2);

        let mut vec_env =
            MultithreadedVectorizedGymWrapper::new(env_constructors, obs_space, action_space);

        let initial_states = vec_env.reset().unwrap();
        assert_eq!(initial_states.shape().dims(), &[3, 2]);
        assert_eq!(
            initial_states.to_vec2::<f32>().unwrap(),
            vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![2.0, 0.0]]
        );

        for step in 0..7 {
            let actions = Tensor::full(1.0, &[3, 1], &candle_core::Device::Cpu).unwrap();
            let step_info = vec_env.step(actions).unwrap();
            assert_eq!(step_info.states.shape().dims(), &[3, 2]);
            assert_eq!(
                step_info.rewards.to_vec1::<f32>().unwrap(),
                vec![1.0, 1.0, 1.0]
            );

            for (i, state) in step_info
                .states
                .to_vec2::<f32>()
                .unwrap()
                .iter()
                .enumerate()
            {
                let expected_step = match step {
                    0..=3 => step + 1,
                    4 => 0,
                    _ => step - 4,
                };
                assert_eq!(*state, vec![i as f32, expected_step as f32]);
            }

            if step != 4 {
                assert_eq!(step_info.dones, vec![false, false, false]);
                assert!(
                    step_info
                        .terminal_states
                        .iter()
                        .all(|state| state.is_none())
                );
            } else {
                assert_eq!(step_info.dones, vec![true, true, true]);
                let terminal_states = step_info.transition_next_states().unwrap();
                assert_eq!(
                    terminal_states.to_vec2::<f32>().unwrap(),
                    vec![vec![0.0, 5.0], vec![1.0, 5.0], vec![2.0, 5.0]]
                );
            }
        }
    }
}

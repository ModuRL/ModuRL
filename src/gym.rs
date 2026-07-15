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

    /// Returns the next state, reward, and done flag.
    fn step(&mut self, action: Tensor) -> Result<StepInfo<I>, Self::Error>;
    /// Resets the environment to its initial state.
    fn reset(&mut self) -> Result<ResetInfo<I>, Self::Error>;
    /// Returns the observation space.
    fn observation_space(&self) -> Box<dyn Space<Error = Self::SpaceError>>;
    /// Returns the action space.
    fn action_space(&self) -> Box<dyn Space<Error = Self::SpaceError>>;
}

/// A vectorized version of Gym that can handle multiple environments in parallel.
/// Actors take this as input.
/// This is automatically implemented for any Gym.
/// Reset is called automatically when an environment is done.
pub trait VectorizedGym {
    type Error;
    type SpaceError;

    fn step(&mut self, action: Tensor) -> Result<VectorizedStepInfo, Self::Error>;
    fn observation_space(&self) -> Box<dyn Space<Error = Self::SpaceError>>;
    fn action_space(&self) -> Box<dyn Space<Error = Self::SpaceError>>;
    fn num_envs(&self) -> usize;
    /// Resets all environments and returns the initial states.
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
pub struct VectorizedStepInfo {
    pub states: Tensor,
    pub rewards: Tensor,
    // really wish candle had a bool tensor type
    pub dones: Vec<bool>,
    pub truncateds: Vec<bool>,
    pub terminal_states: Vec<Option<Tensor>>,
}

impl VectorizedStepInfo {
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

impl<G, I> VectorizedGym for VectorizedGymWrapper<G, I>
where
    G: Gym<I>,
    G::Error: std::fmt::Debug,
{
    type Error = VectorizedGymError<G::Error>;
    type SpaceError = G::SpaceError;

    fn step(&mut self, action: Tensor) -> Result<VectorizedStepInfo, Self::Error> {
        let env_count = self.envs.len();
        let actions: Vec<Tensor> = action.chunk(env_count, 0)?;
        // this keeps the dimension for env count it's just 1 now so we squeeze it later

        let mut states = Vec::with_capacity(env_count);
        let mut rewards = Vec::with_capacity(env_count);
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
            dones.push(step_info.done);
            truncateds.push(step_info.truncated);
            terminal_states.push(terminal_state);
        }

        let states = Tensor::stack(&states, 0)?;
        let rewards = Tensor::from_vec(rewards, &[env_count], states.device())?;

        Ok(VectorizedStepInfo {
            states,
            rewards,
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
impl<G, O, A, SE, I> VectorizedGym for MultithreadedVectorizedGymWrapper<G, O, A, SE, I>
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

    fn step(&mut self, action: Tensor) -> Result<VectorizedStepInfo, Self::Error> {
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
            dones.push(step_info.done);
            truncateds.push(step_info.truncated);
            terminal_states.push(thread_step_info.terminal_state);
        }

        let states = Tensor::stack(&states, 0)?;
        let rewards = Tensor::from_vec(rewards, &[batch_size], states.device())?;

        Ok(VectorizedStepInfo {
            states,
            rewards,
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

    impl DummyEnv {
        fn new(id: usize) -> Self {
            Self { step_count: 0, id }
        }
    }

    impl Gym for DummyEnv {
        type Error = ();
        type SpaceError = candle_core::Error;

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
    fn vectorized_wrapper_infers_and_erases_non_unit_info() {
        let mut vec_env = VectorizedGymWrapper::from(vec![InfoEnv]);
        assert_eq!(vec_env.reset().unwrap().dims(), &[1, 1]);

        let actions =
            Tensor::zeros((1, 1), candle_core::DType::U32, &candle_core::Device::Cpu).unwrap();
        assert_eq!(vec_env.step(actions).unwrap().states.dims(), &[1, 1]);
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

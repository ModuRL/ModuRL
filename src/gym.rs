use crate::spaces::Space;
use burn::{
    prelude::Backend,
    tensor::{Tensor, TensorKind},
};

/// A trait representing a reinforcement learning environment.
/// The observation and action ranks do not include the batch dimension.
pub trait Gym<const OBS_RANK: usize, const ACT_RANK: usize> {
    type Error;
    type SpaceError;
    type Backend: Backend;
    type ActType: TensorKind<Self::Backend>;
    type ObsType: TensorKind<Self::Backend>;

    /// Returns the next state, reward, and done flag.
    fn step(
        &mut self,
        action: Tensor<Self::Backend, ACT_RANK, Self::ActType>,
    ) -> Result<StepInfo, Self::Error>;
    /// Resets the environment to its initial state. Returns the initial state.
    fn reset(&mut self) -> Result<Tensor<Self::Backend, OBS_RANK, Self::ObsType>, Self::Error>;
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
pub struct StepInfo {
    pub state: Tensor,
    pub reward: f32,
    pub done: bool,
    pub truncated: bool,
}

#[derive(Debug, Clone)]
pub struct VectorizedStepInfo {
    pub states: Tensor,
    pub rewards: Tensor,
    // really wish candle had a bool tensor type
    pub dones: Vec<bool>,
    pub truncateds: Vec<bool>,
}

pub struct VectorizedGymWrapper<G: Gym> {
    envs: Vec<G>,
    to_reset: Vec<bool>,
}

impl<G: Gym> VectorizedGymWrapper<G> {
    pub fn new(envs: Vec<G>) -> Self {
        Self {
            to_reset: vec![true; envs.len()],
            envs,
        }
    }

    pub fn envs(&self) -> &Vec<G> {
        &self.envs
    }

    pub fn envs_mut(&mut self) -> &mut Vec<G> {
        &mut self.envs
    }
}

impl<G> VectorizedGym for VectorizedGymWrapper<G>
where
    G: Gym,
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

        for i in 0..env_count {
            let env = &mut self.envs[i];
            let act = actions[i].clone().squeeze(0)?;

            let step_info = if self.to_reset[i] {
                self.to_reset[i] = false;
                let state = env.reset().map_err(VectorizedGymError::Single)?;
                StepInfo {
                    state,
                    reward: 0.0,
                    done: false,
                    truncated: false,
                }
            } else {
                env.step(act).map_err(VectorizedGymError::Single)?
            };

            states.push(step_info.state);
            rewards.push(step_info.reward);
            dones.push(step_info.done);
            truncateds.push(step_info.truncated);
        }

        self.to_reset
            .iter_mut()
            .zip(dones.iter().zip(truncateds.iter()))
            .for_each(|(reset_flag, (done, truncated))| {
                if *done || *truncated {
                    *reset_flag = true;
                }
            });

        let states = Tensor::stack(&states, 0)?;
        let rewards = Tensor::from_vec(rewards, &[env_count], states.device())?;

        Ok(VectorizedStepInfo {
            states,
            rewards,
            dones,
            truncateds,
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
            let state = env.reset().map_err(VectorizedGymError::Single)?;
            states.push(state);
            self.to_reset[i] = false;
        }

        let states = Tensor::stack(&states, 0)?;
        Ok(states)
    }

    fn num_envs(&self) -> usize {
        self.envs.len()
    }
}

impl<G> From<G> for VectorizedGymWrapper<G>
where
    G: Gym + Clone,
{
    fn from(env: G) -> Self {
        VectorizedGymWrapper::new(vec![env])
    }
}

impl<G> From<Vec<G>> for VectorizedGymWrapper<G>
where
    G: Gym,
{
    fn from(envs: Vec<G>) -> Self {
        VectorizedGymWrapper::new(envs)
    }
}

#[cfg(feature = "multithreading")]
use std::{sync::mpsc, thread};

#[cfg(feature = "multithreading")]
pub struct MultithreadedVectorizedGymWrapper<G, O, A>
where
    G: Gym,
    G::Error: Send + Sync,
    O: Space + Clone + 'static,
    A: Space + Clone + 'static,
{
    envs: Vec<GymHandle<G::Error>>,
    to_reset: Vec<bool>,
    obs_space: O,
    action_space: A,
}

#[cfg(feature = "multithreading")]
impl<G, O, A> MultithreadedVectorizedGymWrapper<G, O, A>
where
    G: Gym + 'static,
    G::Error: std::fmt::Debug + Send + Sync,
    O: Space + Clone + 'static,
    A: Space + Clone + 'static,
{
    pub fn new<F>(env_constructors: Vec<F>, obs_space: O, action_space: A) -> Self
    where
        F: FnOnce() -> G + Send + 'static,
    {
        assert!(
            !env_constructors.is_empty(),
            "Must provide at least one environment constructor"
        );

        let envs: Vec<GymHandle<<G as Gym>::Error>> = env_constructors
            .into_iter()
            .map(|constructor| start_gym_thread(constructor))
            .collect();

        Self {
            to_reset: vec![true; envs.len()],
            envs,
            obs_space,
            action_space,
        }
    }
}

#[cfg(feature = "multithreading")]
impl<G, O, A> VectorizedGym for MultithreadedVectorizedGymWrapper<G, O, A>
where
    G: Gym + Send,
    G::Error: Send + Sync + std::fmt::Debug,
    A: Space + Clone + 'static,
    O: Space + Clone + 'static,
{
    type Error = VectorizedGymError<G::Error>;

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
            let step_info = if *to_reset {
                *to_reset = false;
                env.reset()
            } else {
                env.step(act)
            };
            step_info_recievers.push(step_info);
        }

        let mut states = Vec::with_capacity(batch_size);
        let mut rewards = Vec::with_capacity(batch_size);
        let mut dones = Vec::with_capacity(batch_size);
        let mut truncateds = Vec::with_capacity(batch_size);

        for reciever in step_info_recievers {
            let step_info = reciever
                .recv()
                .unwrap()
                .map_err(VectorizedGymError::Single)?;
            states.push(step_info.state);
            rewards.push(step_info.reward);
            dones.push(step_info.done);
            truncateds.push(step_info.truncated);
        }

        self.to_reset
            .iter_mut()
            .zip(dones.iter().zip(truncateds.iter()))
            .for_each(|(reset_flag, (done, truncated))| {
                if *done || *truncated {
                    *reset_flag = true;
                }
            });

        let states = Tensor::stack(&states, 0)?;
        let rewards = Tensor::from_vec(rewards, &[batch_size], states.device())?;

        Ok(VectorizedStepInfo {
            states,
            rewards,
            dones,
            truncateds,
        })
    }

    fn num_envs(&self) -> usize {
        self.envs.len()
    }

    fn reset(&mut self) -> Result<Tensor, Self::Error> {
        let batch_size = self.envs.len();

        let states: Vec<std::sync::mpsc::Receiver<Result<StepInfo, <G as Gym>::Error>>> =
            self.envs.iter().map(|env| env.reset()).collect();
        let states: Vec<Result<Tensor, VectorizedGymError<G::Error>>> = states
            .into_iter()
            .map(|reciever| {
                reciever
                    .recv()
                    .unwrap()
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

    fn observation_space(&self) -> Box<dyn Space> {
        Box::new(self.obs_space.clone())
    }

    fn action_space(&self) -> Box<dyn Space> {
        Box::new(self.action_space.clone())
    }
}

#[cfg(feature = "multithreading")]
enum GymCmd<E>
where
    E: Send + Sync,
{
    Step(Tensor, mpsc::Sender<Result<StepInfo, E>>),
    Reset(mpsc::Sender<Result<StepInfo, E>>),
}

#[cfg(feature = "multithreading")]
struct GymHandle<E>
where
    E: Send + Sync,
{
    tx: mpsc::Sender<GymCmd<E>>,
}

#[cfg(feature = "multithreading")]
impl<E> GymHandle<E>
where
    E: Send + Sync,
{
    fn step(&self, action: Tensor) -> std::sync::mpsc::Receiver<Result<StepInfo, E>> {
        let (resp_tx, resp_rx) = mpsc::channel();
        self.tx.send(GymCmd::Step(action, resp_tx)).unwrap();
        return resp_rx;
    }

    fn reset(&self) -> std::sync::mpsc::Receiver<Result<StepInfo, E>> {
        let (resp_tx, resp_rx) = mpsc::channel();
        self.tx.send(GymCmd::Reset(resp_tx)).unwrap();
        return resp_rx;
    }
}

#[cfg(feature = "multithreading")]
/// Spawns a persistent thread that constructs and owns the gym.
/// The closure is executed inside that thread to build the gym.
fn start_gym_thread<G, F>(make_gym: F) -> GymHandle<G::Error>
where
    F: FnOnce() -> G + Send + 'static,
    G: Gym + 'static,
    G::Error: Send + Sync + 'static,
{
    let (tx, rx) = mpsc::channel::<GymCmd<G::Error>>();

    thread::spawn(move || {
        let mut gym = make_gym(); // constructed *inside* the thread
        while let Ok(cmd) = rx.recv() {
            match cmd {
                GymCmd::Step(action, resp_tx) => {
                    resp_tx.send(gym.step(action)).unwrap();
                }
                GymCmd::Reset(resp_tx) => {
                    let reset_info = gym.reset();
                    if let Some(state) = reset_info.as_ref().ok() {
                        resp_tx
                            .send(Ok(StepInfo {
                                state: state.clone(),
                                reward: 0.0,
                                done: false,
                                truncated: false,
                            }))
                            .unwrap();
                    } else {
                        resp_tx.send(Err(reset_info.err().unwrap())).unwrap();
                        continue;
                    }
                }
            }
        }
    });

    GymHandle { tx }
}

#[cfg(test)]
mod tests {
    use super::*;
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
            })
        }

        fn reset(&mut self) -> Result<Tensor, Self::Error> {
            self.step_count = 0;
            Ok(
                Tensor::from_vec(vec![self.id as f32, 0.0], &[2], &candle_core::Device::Cpu)
                    .unwrap(),
            )
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
            if step != 5 {
                assert_eq!(
                    step_info.rewards.to_vec1::<f32>().unwrap(),
                    vec![1.0, 1.0, 1.0]
                );
            }

            for (i, state) in step_info
                .states
                .to_vec2::<f32>()
                .unwrap()
                .iter()
                .enumerate()
            {
                let expected_step = (step + 1) % 6;
                assert_eq!(*state, vec![i as f32, expected_step as f32]);
            }

            if step != 4 {
                assert_eq!(step_info.dones, vec![false, false, false]);
            } else {
                assert_eq!(step_info.dones, vec![true, true, true]);
            }
        }
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
        let action_space = crate::spaces::Discrete::new(2, 0);

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
            if step != 5 {
                assert_eq!(
                    step_info.rewards.to_vec1::<f32>().unwrap(),
                    vec![1.0, 1.0, 1.0]
                );
            }

            for (i, state) in step_info
                .states
                .to_vec2::<f32>()
                .unwrap()
                .iter()
                .enumerate()
            {
                let expected_step = (step + 1) % 6;
                assert_eq!(*state, vec![i as f32, expected_step as f32]);
            }

            if step != 4 {
                assert_eq!(step_info.dones, vec![false, false, false]);
            } else {
                assert_eq!(step_info.dones, vec![true, true, true]);
            }
        }
    }
}

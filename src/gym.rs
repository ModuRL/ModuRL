use crate::spaces::Space;
use candle_core::Tensor;

pub trait Gym {
    type Error;
    /// Returns the next state, reward, and done flag.
    fn step(&mut self, action: Tensor) -> Result<StepInfo, Self::Error>;
    /// Resets the environment to its initial state. Returns the initial state.
    fn reset(&mut self) -> Result<Tensor, Self::Error>;
    /// Returns the observation space.
    fn observation_space(&self) -> Box<dyn Space>;
    /// Returns the action space.
    fn action_space(&self) -> Box<dyn Space>;
}

/// A vectorized version of Gym that can handle multiple environments in parallel.
/// Actors take this as input.
/// This is automatically implemented for any Gym.
/// Reset is called automatically when an environment is done.
pub trait VectorizedGym {
    type Error;
    fn step(&mut self, action: Tensor) -> Result<VectorizedStepInfo, Self::Error>;
    fn observation_space(&self) -> Box<dyn Space>;
    fn action_space(&self) -> Box<dyn Space>;
}

pub enum VectorizedGymError<E> {
    Single(E),
    Batch(candle_core::Error),
}

impl<E> From<candle_core::Error> for VectorizedGymError<E> {
    fn from(err: candle_core::Error) -> Self {
        VectorizedGymError::Batch(err)
    }
}

impl<E> VectorizedGym for dyn Gym<Error = E> {
    type Error = VectorizedGymError<E>;

    fn step(&mut self, action: Tensor) -> Result<VectorizedStepInfo, Self::Error> {
        let step_info = self.step(action).map_err(VectorizedGymError::Single)?;
        Ok(VectorizedStepInfo {
            states: step_info.state.unsqueeze(0)?,
            rewards: Tensor::from_vec(vec![step_info.reward], &[1], step_info.state.device())?,
            dones: vec![step_info.done],
            truncateds: vec![step_info.truncated],
        })
    }

    fn observation_space(&self) -> Box<dyn Space> {
        self.observation_space()
    }

    fn action_space(&self) -> Box<dyn Space> {
        self.action_space()
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
}

impl<G: Gym> VectorizedGymWrapper<G> {
    pub fn new(envs: Vec<G>) -> Self {
        Self { envs }
    }
}

impl<G> VectorizedGym for VectorizedGymWrapper<G>
where
    G: Gym,
{
    type Error = VectorizedGymError<G::Error>;

    fn step(&mut self, action: Tensor) -> Result<VectorizedStepInfo, Self::Error> {
        let batch_size = self.envs.len();
        let actions: Vec<Tensor> = action.chunk(batch_size, 0)?;

        let mut states = Vec::with_capacity(batch_size);
        let mut rewards = Vec::with_capacity(batch_size);
        let mut dones = Vec::with_capacity(batch_size);
        let mut truncateds = Vec::with_capacity(batch_size);

        for (env, act) in self.envs.iter_mut().zip(actions.into_iter()) {
            let step_info = env.step(act).map_err(VectorizedGymError::Single)?;
            states.push(step_info.state);
            rewards.push(step_info.reward);
            dones.push(step_info.done);
            truncateds.push(step_info.truncated);
        }

        let states = Tensor::stack(&states, 0)?;
        let rewards = Tensor::from_vec(rewards, &[batch_size], states.device())?;

        Ok(VectorizedStepInfo {
            states,
            rewards,
            dones,
            truncateds,
        })
    }

    fn observation_space(&self) -> Box<dyn Space> {
        self.envs[0].observation_space()
    }

    fn action_space(&self) -> Box<dyn Space> {
        self.envs[0].action_space()
    }
}

#[cfg(feature = "multithreading")]
pub struct MultithreadedVectorizedGymWrapper<G: Gym> {
    envs: Vec<G>,
}

#[cfg(feature = "multithreading")]
impl<G: Gym> MultithreadedVectorizedGymWrapper<G> {
    pub fn new(envs: Vec<G>) -> Self {
        Self { envs }
    }
}

#[cfg(feature = "multithreading")]
impl<G> VectorizedGym for MultithreadedVectorizedGymWrapper<G>
where
    G: Gym + Send,
    G::Error: Send + Sync,
{
    type Error = VectorizedGymError<G::Error>;

    fn step(&mut self, action: Tensor) -> Result<VectorizedStepInfo, Self::Error> {
        use rayon::prelude::*;
        let batch_size = self.envs.len();
        let actions: Vec<Tensor> = action.chunk(batch_size, 0)?;

        let step_infos: Vec<Result<StepInfo, VectorizedGymError<G::Error>>> = self
            .envs
            .par_iter_mut()
            .zip(actions.into_par_iter())
            .map(|(env, act)| env.step(act).map_err(VectorizedGymError::Single))
            .collect();

        let mut states = Vec::with_capacity(batch_size);
        let mut rewards = Vec::with_capacity(batch_size);
        let mut dones = Vec::with_capacity(batch_size);
        let mut truncateds = Vec::with_capacity(batch_size);

        for step_info in step_infos {
            let step_info = step_info?;
            states.push(step_info.state);
            rewards.push(step_info.reward);
            dones.push(step_info.done);
            truncateds.push(step_info.truncated);
        }

        let states = Tensor::stack(&states, 0)?;
        let rewards = Tensor::from_vec(rewards, &[batch_size], states.device())?;

        Ok(VectorizedStepInfo {
            states,
            rewards,
            dones,
            truncateds,
        })
    }

    fn observation_space(&self) -> Box<dyn Space> {
        self.envs[0].observation_space()
    }

    fn action_space(&self) -> Box<dyn Space> {
        self.envs[0].action_space()
    }
}

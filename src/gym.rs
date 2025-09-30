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
}

impl<G, GE> VectorizedGym for VectorizedGymWrapper<G>
where
    G: Gym<Error = GE>,
    GE: std::fmt::Debug,
{
    type Error = VectorizedGymError<G::Error>;

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

    fn observation_space(&self) -> Box<dyn Space> {
        self.envs[0].observation_space()
    }

    fn action_space(&self) -> Box<dyn Space> {
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

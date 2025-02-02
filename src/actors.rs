use candle_core::Tensor;
use candle_nn::Optimizer;
use rand::Rng;

use crate::{Gym, Space};

pub trait Actor {
    fn act(&mut self, observation: &Tensor) -> Tensor;
}

/// Q-learning actor
/// The most basic actor, which just takes the action with the highest Q-value.
pub struct DQNActor<I, O> {
    q_network: Box<dyn candle_core::Module>,
    epsilon: f32,
    action_space: Box<dyn Space<O>>,
    observation_space: Box<dyn Space<I>>,
}

impl<I, O> DQNActor<I, O> {
    pub fn new(q_network: Box<dyn candle_core::Module>, epsilon: f32, gym: &dyn Gym<I, O>) -> Self {
        let action_space = gym.action_space();
        let observation_space = gym.observation_space();
        Self {
            q_network,
            epsilon,
            action_space,
            observation_space,
        }
    }
}

impl Actor for DQNActor {
    fn act(&mut self, observation: &Tensor) -> Tensor {
        let q_values = self.q_network.forward(observation).unwrap();
        todo!()
    }
}

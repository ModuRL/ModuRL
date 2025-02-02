use candle_core::Tensor;
use candle_nn::{Optimizer, VarBuilder};
use rand::Rng;

use crate::{Gym, Space};

pub trait Actor {
    fn act(&mut self, observation: &Tensor) -> Tensor;
}

/// Q-learning actor
/// The most basic actor, which just takes the action with the highest Q-value.
pub struct DQNActor {
    q_network: Box<dyn candle_core::Module>,
    epsilon: f32,
    action_space: Box<dyn Space>,
    observation_space: Box<dyn Space>,
}

impl DQNActor {
    pub fn new(
        q_network: Box<dyn candle_core::Module>,
        epsilon: f32,
        action_space: Box<dyn Space>,
        observation_space: Box<dyn Space>,
    ) -> Self {
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
        if rand::rng().random_range(0.0..1.0) < self.epsilon {
            self.action_space.sample(observation.device())
        } else {
            let max_q_value = q_values.max(1).unwrap();
            let action = q_values
                .eq(&max_q_value)
                .unwrap()
                .to_dtype(candle_core::DType::I64)
                .unwrap();
            action
        }
    }
}


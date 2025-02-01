use candle_core::Tensor;
use candle_nn::Optimizer;
use ndarray::ArrayD;
use rand::Rng;

pub trait Actor {
    fn act(&mut self, observation: &Tensor) -> Tensor;
}

/// Q-learning actor
/// The most basic actor, which just takes the action with the highest Q-value.
pub struct DQNActor {
    q_network: Box<dyn candle_core::Module>,
    epsilon: f32,
}

impl DQNActor {
    pub fn new(q_network: Box<dyn candle_core::Module>, epsilon: f32) -> Self {
        Self { q_network, epsilon }
    }
}

impl Actor for DQNActor {
    fn act(&mut self, observation: &Tensor) -> Tensor {
        let q_values = self.q_network.forward(observation).unwrap();
        todo!()
    }
}

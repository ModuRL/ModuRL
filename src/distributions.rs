use candle_core::Tensor;
mod gaussian;
pub use gaussian::GuassianDistribution;

pub struct DistEval {
    log_prob: Tensor,
    entropy: Tensor,
}

impl DistEval {
    pub fn new(log_prob: Tensor, entropy: Tensor) -> Self {
        Self { log_prob, entropy }
    }

    pub fn log_prob(&self) -> &Tensor {
        &self.log_prob
    }

    pub fn entropy(&self) -> &Tensor {
        &self.entropy
    }
}

pub trait Distribution {
    type Error;
    fn sample(&self) -> Tensor;
    fn dist_eval(&self, actions: &Tensor) -> Result<DistEval, Self::Error>;
    fn from_outputs(outputs: &Tensor) -> Self;
}

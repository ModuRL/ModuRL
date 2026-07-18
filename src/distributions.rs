use candle_core::Tensor;
mod categorical;
mod gaussian;
pub use categorical::CategoricalDistribution;
pub use gaussian::GaussianDistribution;

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

/// Interprets a model output tensor as a probability distribution.
///
/// Each implementation defines its own output layout. For example,
/// [`GaussianDistribution`] expects a rank-2 `[mean, log_std]` tensor, while
/// [`CategoricalDistribution`] interprets its output as category logits.
pub trait Distribution {
    type Error;
    /// Samples one distribution value for each row of parameters.
    fn sample(&self) -> Tensor;
    /// Returns the most likely action representation without sampling.
    fn mode(&self) -> Tensor;
    /// Returns the log probability and entropy for each batch row.
    fn dist_eval(&self, actions: &Tensor) -> Result<DistEval, Self::Error>;
    /// Interprets a model output using this distribution's documented layout.
    fn from_outputs(outputs: &Tensor) -> Self;
}

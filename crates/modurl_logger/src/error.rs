/// An error produced while accepting or aggregating metrics.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A metric tensor did not have shape [1].
    #[error("metric {metric:?} must have shape [1], but has shape {shape:?}")]
    NonScalar { metric: String, shape: Vec<usize> },
    /// A metric was logged before the currently open timestep.
    #[error("received timestep {received} before current timestep {current}")]
    OutOfOrderTimestep { current: usize, received: usize },
    /// Reading a metric tensor to the host failed.
    #[error("failed to convert metric tensor: {0}")]
    Tensor(#[source] burn::tensor::TensorReadError),
}

impl From<burn::tensor::TensorReadError> for Error {
    fn from(error: burn::tensor::TensorReadError) -> Self {
        Self::Tensor(error)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

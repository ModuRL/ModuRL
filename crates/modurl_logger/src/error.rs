/// An error produced while accepting or aggregating metrics.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A metric tensor had one or more dimensions.
    #[error("metric {metric:?} must be scalar, but has shape {shape:?}")]
    NonScalar { metric: String, shape: Vec<usize> },
    /// A metric was logged before the currently open timestep.
    #[error("received timestep {received} before current timestep {current}")]
    OutOfOrderTimestep { current: usize, received: usize },
    /// Candle could not convert a scalar tensor to `f32`.
    #[error("failed to convert metric tensor: {0}")]
    Tensor(#[source] candle_core::Error),
}

impl From<candle_core::Error> for Error {
    fn from(error: candle_core::Error) -> Self {
        Self::Tensor(error)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// An error produced while accepting or aggregating metrics.
#[derive(Debug)]
pub enum Error {
    /// A metric tensor had one or more dimensions.
    NonScalar { metric: String, shape: Vec<usize> },
    /// A metric was logged before the currently open timestep.
    OutOfOrderTimestep { current: usize, received: usize },
    /// Candle could not convert a scalar tensor to `f32`.
    Tensor(candle_core::Error),
}

impl From<candle_core::Error> for Error {
    fn from(error: candle_core::Error) -> Self {
        Self::Tensor(error)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

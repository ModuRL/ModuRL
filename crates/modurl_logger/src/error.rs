use std::{error, fmt};

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

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonScalar { metric, shape } => {
                write!(
                    formatter,
                    "metric {metric:?} must be a scalar tensor, but has shape {shape:?}"
                )
            }
            Self::OutOfOrderTimestep { current, received } => write!(
                formatter,
                "received timestep {received} after timestep {current}; timesteps must be nondecreasing"
            ),
            Self::Tensor(error) => error.fmt(formatter),
        }
    }
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Tensor(error) => Some(error),
            Self::NonScalar { .. } | Self::OutOfOrderTimestep { .. } => None,
        }
    }
}

impl From<candle_core::Error> for Error {
    fn from(error: candle_core::Error) -> Self {
        Self::Tensor(error)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

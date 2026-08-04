pub mod box_2d;
pub mod classic_control;
pub(crate) mod testing;

#[derive(Debug)]
pub enum EnvironmentError {
    Tensor(candle_core::Error),
    InvalidConfiguration(&'static str),
    InvalidAction(&'static str),
    NotInitialized(&'static str),
    InvalidPhysicsState(&'static str),
    #[cfg(feature = "rendering")]
    Rendering(minifb::Error),
}

impl std::fmt::Display for EnvironmentError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tensor(error) => write!(formatter, "tensor error: {error}"),
            Self::InvalidConfiguration(message)
            | Self::InvalidAction(message)
            | Self::NotInitialized(message)
            | Self::InvalidPhysicsState(message) => formatter.write_str(message),
            #[cfg(feature = "rendering")]
            Self::Rendering(error) => write!(formatter, "rendering error: {error}"),
        }
    }
}

impl std::error::Error for EnvironmentError {}

impl From<candle_core::Error> for EnvironmentError {
    fn from(error: candle_core::Error) -> Self {
        Self::Tensor(error)
    }
}

#[cfg(feature = "rendering")]
impl From<minifb::Error> for EnvironmentError {
    fn from(error: minifb::Error) -> Self {
        Self::Rendering(error)
    }
}

#[cfg(feature = "rendering")]
pub(crate) mod rendering;

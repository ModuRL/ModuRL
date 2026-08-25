pub mod box_2d;
pub mod classic_control;
pub(crate) mod testing;

#[derive(Debug, thiserror::Error)]
pub enum EnvironmentError {
    #[error("environment tensor operation failed: {0}")]
    Tensor(#[source] candle_core::Error),
    #[error("invalid environment configuration: {0}")]
    InvalidConfiguration(&'static str),
    #[error("invalid action: {0}")]
    InvalidAction(&'static str),
    #[error("environment is not initialized: {0}")]
    NotInitialized(&'static str),
    #[error("invalid physics state: {0}")]
    InvalidPhysicsState(&'static str),
    #[cfg(feature = "rendering")]
    #[error("environment rendering failed: {0}")]
    Rendering(#[source] minifb::Error),
}

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

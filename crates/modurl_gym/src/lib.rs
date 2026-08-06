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

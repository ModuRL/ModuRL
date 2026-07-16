//! Gymnasium-compatible MuJoCo environments for [ModuRL](https://github.com/ModuRL/ModuRL).
//!
//! The environments use MuJoCo for physics through [`mujoco_rs`] and expose
//! ModuRL's [`modurl::gym::Gym`] interface.

mod core;
mod half_cheetah;
mod hopper;
mod walker2d;

pub use half_cheetah::HalfCheetahV5;
pub use hopper::HopperV5;
pub use walker2d::Walker2dV5;

/// Convenient imports for applications using this crate.
pub mod prelude {
    pub use crate::{HalfCheetahV5, HopperV5, MujocoError, Walker2dV5};
    pub use modurl::gym::{Gym, ResetInfo, StepInfo};
}

/// Errors returned while constructing or stepping an environment.
#[derive(Debug)]
pub enum MujocoError {
    /// MuJoCo could not compile an embedded model.
    Model(mujoco_rs::error::MjModelError),
    /// A Candle tensor operation failed.
    Tensor(candle_core::Error),
    /// The optional interactive viewer could not initialize or draw a frame.
    #[cfg(feature = "rendering")]
    Viewer(mujoco_rs::viewer::MjViewerError),
    /// An action or explicit simulator state had an invalid shape or value.
    InvalidInput(String),
}

impl std::fmt::Display for MujocoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Model(error) => write!(f, "failed to load MuJoCo model: {error}"),
            Self::Tensor(error) => write!(f, "tensor error: {error}"),
            #[cfg(feature = "rendering")]
            Self::Viewer(error) => write!(f, "MuJoCo viewer error: {error}"),
            Self::InvalidInput(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for MujocoError {}

impl From<mujoco_rs::error::MjModelError> for MujocoError {
    fn from(value: mujoco_rs::error::MjModelError) -> Self {
        Self::Model(value)
    }
}

impl From<candle_core::Error> for MujocoError {
    fn from(value: candle_core::Error) -> Self {
        Self::Tensor(value)
    }
}

#[cfg(feature = "rendering")]
impl From<mujoco_rs::viewer::MjViewerError> for MujocoError {
    fn from(value: mujoco_rs::viewer::MjViewerError) -> Self {
        Self::Viewer(value)
    }
}

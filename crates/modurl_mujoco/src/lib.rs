//! Gymnasium-compatible MuJoCo environments for [ModuRL](https://github.com/ModuRL/ModuRL).
//!
//! The environments use MuJoCo for physics through [`mujoco_rs`] and expose
//! ModuRL's [`modurl::gym::Gym`] and [`modurl::gym::MultiGym`] interfaces.

mod ant;
mod core;
mod half_cheetah;
mod hopper;
mod humanoid;
mod walker2d;

pub use ant::{AntV5, AntV5Info};
pub use half_cheetah::HalfCheetahV5;
pub use hopper::HopperV5;
pub use humanoid::{HumanoidV5, HumanoidV5Info};
pub use walker2d::Walker2dV5;

/// Convenient imports for applications using this crate.
pub mod prelude {
    pub use crate::{
        AntV5, AntV5Info, HalfCheetahV5, HopperV5, HumanoidV5, HumanoidV5Info, MujocoError,
        Walker2dV5,
    };
    pub use modurl::gym::{Gym, MultiGym, MultiGymStepInfo, ResetInfo, StepInfo};
}

/// Errors returned while constructing or stepping an environment.
#[derive(Debug, thiserror::Error)]
pub enum MujocoError {
    /// MuJoCo could not compile an embedded model.
    #[error("MuJoCo model compilation failed: {0}")]
    Model(#[source] mujoco_rs::error::MjModelError),
    /// A Candle tensor operation failed.
    #[error("MuJoCo tensor operation failed: {0}")]
    Tensor(#[source] candle_core::Error),
    /// The optional interactive viewer could not initialize or draw a frame.
    #[cfg(feature = "rendering")]
    #[error("MuJoCo viewer failed: {0}")]
    Viewer(#[source] mujoco_rs::viewer::MjViewerError),
    /// An action or explicit simulator state had an invalid shape or value.
    #[error("invalid MuJoCo input: {0}")]
    InvalidInput(String),
}

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

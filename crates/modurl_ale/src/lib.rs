//! Atari 2600 environments backed by the Arcade Learning Environment (ALE).
//!
//! No ROMs are included or downloaded. Supply a filesystem path to a ROM you are legally
//! entitled to use.

mod ale;
mod ale_sys;
mod atari;
mod bindings;
#[cfg(feature = "rendering")]
mod renderer;
pub mod wrappers;

pub use atari::{AtariGym, AtariGymError, AtariInfo, AtariObsType};

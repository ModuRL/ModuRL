//! Backend-independent scalar logging and terminal graphs for ModuRL.

mod aggregation;
mod error;
mod terminal;

use candle_core::Tensor;

pub use aggregation::{Aggregation, AggregationConfig};
pub use error::{Error, Result};
pub use terminal::TerminalLogger;

/// A backend that accepts named scalar tensors at a training timestep.
///
/// Backends decide how to store or output the values. Tensors must already be
/// reduced to scalars; implementations convert numeric scalar tensors to
/// `f32` before aggregating them.
///
/// Repeated calls at the same timestep are combined. Logging a larger
/// timestep completes the preceding timestep. The final timestep therefore
/// remains incomplete until the backend is explicitly finished.
pub trait Logger {
    fn log(&mut self, timestep: usize, metrics: &[(&str, &Tensor)]) -> Result<()>;
}

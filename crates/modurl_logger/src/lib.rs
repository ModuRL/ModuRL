//! Backend-independent scalar logging for ModuRL.

mod aggregation;
mod error;
mod tensorboard;
mod terminal;

use candle_core::Tensor;

pub use aggregation::{Aggregation, AggregationConfig};
pub use error::{Error, Result};
pub use tensorboard::{TensorBoardError, TensorBoardLogger, TensorBoardResult};
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
    type Error;

    fn log(
        &mut self,
        timestep: usize,
        metrics: &[(&str, &Tensor)],
    ) -> std::result::Result<(), Self::Error>;

    /// Completes the final timestep and flushes buffered output.
    ///
    /// Call this after the last log entry. The default implementation is a
    /// no-op for backends without buffered state.
    fn finish(&mut self) -> std::result::Result<(), Self::Error> {
        Ok(())
    }
}

//! Scalar metric logging for ModuRL.

mod aggregation;
mod error;
mod tensorboard;
mod terminal;

use burn::tensor::Tensor;

pub use aggregation::{Aggregation, AggregationConfig};
pub use error::{Error, Result};
pub use tensorboard::{TensorBoardError, TensorBoardLogger, TensorBoardResult};
pub use terminal::TerminalLogger;

/// A backend that accepts named scalar tensors at a training timestep.
///
/// Backends decide how to store or output the values. Tensors must already be
/// reduced to scalars; implementations convert Float scalar tensors to
/// `f32` before aggregating them. Inputs can use any device and supported floating-point dtype.
/// The logger detaches inputs and reads values to the host without preserving gradients.
///
/// Repeated calls at the same timestep are combined. Logging a larger
/// timestep completes the preceding timestep. The final timestep therefore
/// remains incomplete until the backend is explicitly finished.
pub trait Logger {
    type Error;

    /// Logs named Float metric tensors, each shaped `[1]`, on any supported device.
    /// Converts supported floating-point dtypes to F32 and reads detached values to the host.
    /// Rejects other axis sizes before changing aggregation state.
    ///
    /// ```compile_fail
    /// use burn::tensor::{Device, Int, Tensor};
    /// use modurl_logger::{Logger, TerminalLogger};
    /// let metric = Tensor::<1, Int>::from_data([1i32], &Device::flex());
    /// TerminalLogger::default().log(0, &[("count", &metric)]).unwrap();
    /// ```
    fn log(
        &mut self,
        timestep: usize,
        metrics: &[(&str, &Tensor<1>)],
    ) -> std::result::Result<(), Self::Error>;

    /// Completes the final timestep and flushes buffered output.
    ///
    /// Call this after the last log entry. The default implementation is a
    /// no-op for backends without buffered state.
    fn finish(&mut self) -> std::result::Result<(), Self::Error> {
        Ok(())
    }
}

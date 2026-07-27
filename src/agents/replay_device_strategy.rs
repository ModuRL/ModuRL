use candle_core::Device;

/// Strategy for selecting devices used by replay-based agents.
///
/// `OneDevice` keeps replay and optimization on one device.
/// `Hybrid` stores replay on one device and transfers sampled batches to the
/// device used for network optimization.
pub enum ReplayDeviceStrategy {
    /// Stores replay and runs optimization on the same device.
    OneDevice(Device),
    /// Stores replay separately and transfers sampled batches for optimization.
    Hybrid {
        /// Device holding model parameters and tensors during optimization.
        optimization_device: Device,
        /// Device holding detached transitions while they remain in replay.
        storage_device: Device,
    },
}

impl ReplayDeviceStrategy {
    pub(crate) fn storage_device(&self) -> Device {
        match self {
            ReplayDeviceStrategy::OneDevice(device) => device.clone(),
            ReplayDeviceStrategy::Hybrid { storage_device, .. } => storage_device.clone(),
        }
    }

    pub(crate) fn optimization_device(&self) -> Device {
        match self {
            ReplayDeviceStrategy::OneDevice(device) => device.clone(),
            ReplayDeviceStrategy::Hybrid {
                optimization_device,
                ..
            } => optimization_device.clone(),
        }
    }
}

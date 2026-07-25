use candle_core::Device;

/// Strategy for selecting devices used by replay-based agents.
///
/// `OneDevice` keeps replay and optimization on one device.
/// `Hybrid` stores replay on one device and transfers sampled batches to the
/// device used for network optimization.
pub enum ReplayDeviceStrategy {
    OneDevice(Device),
    Hybrid {
        optimization_device: Device,
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

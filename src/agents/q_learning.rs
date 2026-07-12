use candle_core::Device;

pub mod ddqn;
pub mod dqn;

/// Strategy for selecting devices used by value-based agent computations.
///
/// `OneDevice` keeps collection, replay, and optimization on one device.
/// `Hybrid` stores replay on one device and transfers sampled batches to the
/// device used for network optimization.
pub enum QLearningDeviceStrategy {
    OneDevice(Device),
    Hybrid {
        optimization_device: Device,
        storage_device: Device,
    },
}

impl QLearningDeviceStrategy {
    pub(crate) fn storage_device(&self) -> Device {
        match self {
            QLearningDeviceStrategy::OneDevice(device) => device.clone(),
            QLearningDeviceStrategy::Hybrid { storage_device, .. } => storage_device.clone(),
        }
    }

    pub(crate) fn optimization_device(&self) -> Device {
        match self {
            QLearningDeviceStrategy::OneDevice(device) => device.clone(),
            QLearningDeviceStrategy::Hybrid {
                optimization_device,
                ..
            } => optimization_device.clone(),
        }
    }
}

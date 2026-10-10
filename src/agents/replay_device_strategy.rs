use burn::tensor::Device;

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

/// Configuration for replay storage and optimization devices.
pub struct ReplayStorageConfig {
    device_strategy: ReplayDeviceStrategy,
}

impl ReplayStorageConfig {
    /// Selects devices for replay storage and optimization.
    pub fn new(device_strategy: ReplayDeviceStrategy) -> Self {
        Self { device_strategy }
    }

    pub(crate) fn storage_device(&self) -> Device {
        self.device_strategy.storage_device()
    }

    pub(crate) fn optimization_device(&self) -> Device {
        self.device_strategy.optimization_device()
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_keeps_storage_separate_from_autodiff_optimization() {
        let storage = Device::flex();
        let optimization = storage.clone().autodiff();
        let config = ReplayStorageConfig::new(ReplayDeviceStrategy::Hybrid {
            optimization_device: optimization.clone(),
            storage_device: storage.clone(),
        });
        assert_eq!(config.storage_device(), storage);
        assert_eq!(config.optimization_device(), optimization);
        assert!(!config.storage_device().is_autodiff());
        assert!(config.optimization_device().is_autodiff());
    }

    #[test]
    fn one_device_keeps_storage_and_optimization_together() {
        let device = Device::flex();
        let config = ReplayStorageConfig::new(ReplayDeviceStrategy::OneDevice(device.clone()));
        assert_eq!(config.storage_device(), device);
        assert_eq!(config.optimization_device(), device);
    }
}

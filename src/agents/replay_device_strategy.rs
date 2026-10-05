use burn::tensor::{DType, Device};

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

/// Configuration for the representation and placement of replay observations.
///
/// Other replay columns retain the dtype appropriate to their values. Sampled
/// observations are converted to the agent's compute dtype before optimization.
pub struct ReplayStorageConfig {
    device_strategy: ReplayDeviceStrategy,
    observation_dtype: DType,
}

impl ReplayStorageConfig {
    /// Creates replay storage using `F32` observations.
    pub fn new(device_strategy: ReplayDeviceStrategy) -> Self {
        Self {
            device_strategy,
            observation_dtype: DType::F32,
        }
    }

    /// Sets the dtype used by observations while retained in replay.
    pub fn with_observation_dtype(mut self, observation_dtype: DType) -> Self {
        self.observation_dtype = observation_dtype;
        self
    }

    pub(crate) fn storage_device(&self) -> Device {
        self.device_strategy.storage_device()
    }

    pub(crate) fn optimization_device(&self) -> Device {
        self.device_strategy.optimization_device()
    }

    pub(crate) fn observation_dtype(&self) -> DType {
        self.observation_dtype
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
        assert_eq!(config.observation_dtype(), DType::F32);
    }

    #[test]
    fn one_device_retains_placement_when_storage_dtype_changes() {
        let device = Device::flex();
        for dtype in [
            DType::F32,
            DType::F64,
            DType::F16,
            DType::BF16,
            DType::Flex32,
            DType::U8,
            DType::U32,
            DType::U64,
            DType::I32,
            DType::I64,
            DType::Bool(burn::tensor::BoolStore::Native),
        ] {
            let config = ReplayStorageConfig::new(ReplayDeviceStrategy::OneDevice(device.clone()))
                .with_observation_dtype(dtype);
            assert_eq!(config.storage_device(), device);
            assert_eq!(config.optimization_device(), device);
            assert_eq!(config.observation_dtype(), dtype);
        }
    }
}

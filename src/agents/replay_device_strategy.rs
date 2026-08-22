use candle_core::{DType, Device};

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
        assert!(
            observation_dtype.is_float() || observation_dtype == DType::U8,
            "replay observation dtype must be floating-point or u8"
        );
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

use burn::tensor::Device;

/// Replay devices: optimization runs on the models' device, and replay storage
/// defaults to it unless a separate replay device is configured.
pub(crate) struct ReplayStorageConfig {
    optimization_device: Device,
    storage_device: Device,
}

impl ReplayStorageConfig {
    /// Returns `None` unless `model_devices` contains exactly one distinct device.
    pub(crate) fn from_models(
        model_devices: impl IntoIterator<Item = Device>,
        replay_device: Option<Device>,
    ) -> Option<Self> {
        let mut model_devices = model_devices.into_iter();
        let optimization_device = model_devices.next()?;
        if model_devices.any(|device| device != optimization_device) {
            return None;
        }
        Some(Self {
            storage_device: replay_device.unwrap_or_else(|| optimization_device.clone()),
            optimization_device,
        })
    }

    pub(crate) fn storage_device(&self) -> Device {
        self.storage_device.clone()
    }

    pub(crate) fn optimization_device(&self) -> Device {
        self.optimization_device.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(deprecated)]
    fn replay_device_keeps_storage_separate_from_optimization() {
        let storage = Device::ndarray();
        let optimization = Device::flex().autodiff();
        let config =
            ReplayStorageConfig::from_models([optimization.clone()], Some(storage.clone()))
                .unwrap();
        assert_eq!(config.storage_device(), storage);
        assert_eq!(config.optimization_device(), optimization);
        assert_ne!(config.storage_device(), config.optimization_device());
        assert!(!config.storage_device().is_autodiff());
        assert!(config.optimization_device().is_autodiff());
    }

    #[test]
    fn storage_defaults_to_model_device() {
        let device = Device::flex();
        let config =
            ReplayStorageConfig::from_models([device.clone(), device.clone()], None).unwrap();
        assert_eq!(config.storage_device(), device);
        assert_eq!(config.optimization_device(), device);
    }

    #[test]
    fn models_without_parameters_have_no_device() {
        assert!(ReplayStorageConfig::from_models([], None).is_none());
    }
}

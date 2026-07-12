use candle_core::Device;

pub mod ddqn;
pub mod dqn;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QLearningConfigurationError {
    ZeroReplayCapacity,
    ZeroBatchSize,
    ZeroTargetUpdateInterval,
    ZeroUpdateFrequency,
    ReplayCapacityBelowBatchSize,
    InvalidGamma,
    InvalidEpsilon,
}

pub(crate) fn validate_configuration(
    replay_capacity: usize,
    batch_size: usize,
    gamma: f32,
    initial_epsilon: f64,
    final_epsilon: f64,
    update_frequency: usize,
    target_update_interval: usize,
) -> Result<(), QLearningConfigurationError> {
    if replay_capacity == 0 {
        return Err(QLearningConfigurationError::ZeroReplayCapacity);
    }
    if batch_size == 0 {
        return Err(QLearningConfigurationError::ZeroBatchSize);
    }
    if target_update_interval == 0 {
        return Err(QLearningConfigurationError::ZeroTargetUpdateInterval);
    }
    if update_frequency == 0 {
        return Err(QLearningConfigurationError::ZeroUpdateFrequency);
    }
    if replay_capacity < batch_size {
        return Err(QLearningConfigurationError::ReplayCapacityBelowBatchSize);
    }
    if !gamma.is_finite() || !(0.0..=1.0).contains(&gamma) {
        return Err(QLearningConfigurationError::InvalidGamma);
    }
    validate_epsilon(initial_epsilon)?;
    validate_epsilon(final_epsilon)?;
    Ok(())
}

pub(crate) fn validate_epsilon(epsilon: f64) -> Result<f64, QLearningConfigurationError> {
    if !epsilon.is_finite() || !(0.0..=1.0).contains(&epsilon) {
        return Err(QLearningConfigurationError::InvalidEpsilon);
    }
    Ok(epsilon)
}

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

#[cfg(test)]
mod tests {
    use super::{QLearningConfigurationError, validate_configuration, validate_epsilon};

    #[test]
    fn accepts_valid_configuration() {
        assert_eq!(
            validate_configuration(1_000, 32, 0.99, 1.0, 0.1, 4, 1_000),
            Ok(())
        );
    }

    #[test]
    fn rejects_invalid_configuration_values() {
        assert_eq!(
            validate_configuration(0, 32, 0.99, 1.0, 0.1, 4, 1_000),
            Err(QLearningConfigurationError::ZeroReplayCapacity)
        );
        assert_eq!(
            validate_configuration(1_000, 0, 0.99, 1.0, 0.1, 4, 1_000),
            Err(QLearningConfigurationError::ZeroBatchSize)
        );
        assert_eq!(
            validate_configuration(1_000, 32, 0.99, 1.0, 0.1, 4, 0),
            Err(QLearningConfigurationError::ZeroTargetUpdateInterval)
        );
        assert_eq!(
            validate_configuration(1_000, 32, 0.99, 1.0, 0.1, 0, 1_000),
            Err(QLearningConfigurationError::ZeroUpdateFrequency)
        );
        assert_eq!(
            validate_configuration(31, 32, 0.99, 1.0, 0.1, 4, 1_000),
            Err(QLearningConfigurationError::ReplayCapacityBelowBatchSize)
        );
        assert_eq!(
            validate_configuration(1_000, 32, 1.01, 1.0, 0.1, 4, 1_000),
            Err(QLearningConfigurationError::InvalidGamma)
        );
        assert_eq!(
            validate_epsilon(1.01),
            Err(QLearningConfigurationError::InvalidEpsilon)
        );
    }
}

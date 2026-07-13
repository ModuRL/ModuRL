use candle_core::{Device, Error, Tensor};

use crate::spaces::{Discrete, Space};

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

pub(crate) fn epsilon_greedy_actions(
    observation: &Tensor,
    epsilon: f64,
    action_space: &Discrete,
    forward: impl FnOnce(&Tensor) -> Result<Tensor, Error>,
) -> Result<Tensor, Error> {
    let batch_size = observation.shape().dims()[0];
    let explores = Tensor::rand(0.0f64, 1.0, &[batch_size], observation.device())?
        .to_vec1::<f64>()?
        .into_iter()
        .map(|value| value < epsilon)
        .collect::<Vec<_>>();

    let greedy_indices = explores
        .iter()
        .enumerate()
        .filter_map(|(index, &explore)| (!explore).then_some(index as u32))
        .collect::<Vec<_>>();
    let greedy_count = greedy_indices.len();
    let mut greedy_actions = if greedy_indices.is_empty() {
        None
    } else {
        let greedy_indices =
            Tensor::from_vec(greedy_indices, &[greedy_count], observation.device())?;
        let greedy_observations = observation.index_select(&greedy_indices, 0)?;
        Some(
            forward(&greedy_observations)?
                .argmax(1)?
                .chunk(greedy_count, 0)?
                .into_iter(),
        )
    };

    let mut actions = Vec::with_capacity(batch_size);
    for explore in explores {
        if explore {
            actions.push(action_space.sample(observation.device())?);
        } else {
            let action = greedy_actions
                .as_mut()
                .and_then(|actions| actions.next())
                .expect("greedy actions must exist for non-exploring environments");
            actions.push(action.squeeze(0)?);
        }
    }
    Tensor::stack(&actions, 0)
}

pub(crate) fn selected_action_q_values(
    q_values: &Tensor,
    actions: &Tensor,
) -> Result<Tensor, Error> {
    let actions = actions.reshape(&[actions.shape().dims()[0], 1])?;
    q_values.gather(&actions, 1)
}

fn bellman_targets(
    rewards: &Tensor,
    next_dones: &Tensor,
    next_q_values: &Tensor,
    gamma: f32,
) -> Result<Tensor, Error> {
    let gamma_tensor = Tensor::full(gamma, next_q_values.shape(), rewards.device())?;
    Ok((rewards + (next_q_values * ((1.0 - next_dones)?.mul(&gamma_tensor)?)))?.detach())
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
    use super::{
        selected_action_q_values, validate_configuration, validate_epsilon,
        QLearningConfigurationError,
    };
    use candle_core::{Device, Tensor};

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

    #[test]
    fn selected_action_q_values_uses_one_value_per_transition() {
        let device = Device::Cpu;
        let q_values =
            Tensor::from_vec(vec![1.0f32, 5.0, 2.0, 7.0, 3.0, 4.0], (2, 3), &device).unwrap();
        let actions = Tensor::from_vec(vec![1u32, 2], 2, &device).unwrap();

        let selected = selected_action_q_values(&q_values, &actions).unwrap();
        assert_eq!(
            selected.to_vec2::<f32>().unwrap(),
            vec![vec![5.0], vec![4.0]]
        );
    }
}

use burn::tensor::{Bool, Tensor};

/// Advantages and critic value targets for one rollout, each shaped `[time, env_count, 1]`.
#[derive(Debug, Clone)]
pub struct AdvantageEstimate {
    /// Per-step advantages shaped `[time, env_count, 1]`.
    pub advantages: Tensor<3>,
    /// Critic value targets shaped `[time, env_count, 1]`.
    pub value_targets: Tensor<3>,
}

/// Computes advantages and critic value targets for an on-policy rollout.
pub trait AdvantageEstimator {
    /// Returns advantages and value targets shaped `[time, env_count, 1]` from `rewards`,
    /// `values`, `next_dones`, and `next_truncateds` shaped `[time, env_count, 1]`, and
    /// `bootstrapped_values` shaped `[env_count, 1]` (critic values of the final next observations).
    ///
    /// Float inputs share the environment reward dtype and device. Both outputs must have shape
    /// `[time, env_count, 1]` and the reward dtype and device of `values`. Rewards already include
    /// `gamma * V(terminal)` for truncated transitions. Values are detached. The critic is trained
    /// toward `value_targets`.
    fn estimate(
        &self,
        rewards: &Tensor<3>,
        values: &Tensor<3>,
        next_dones: &Tensor<3, Bool>,
        next_truncateds: &Tensor<3, Bool>,
        bootstrapped_values: &Tensor<2>,
        gamma: f32,
    ) -> AdvantageEstimate;
}

impl AdvantageEstimator for Box<dyn AdvantageEstimator> {
    /// Delegates estimation for rollout tensors with their documented shapes.
    fn estimate(
        &self,
        rewards: &Tensor<3>,
        values: &Tensor<3>,
        next_dones: &Tensor<3, Bool>,
        next_truncateds: &Tensor<3, Bool>,
        bootstrapped_values: &Tensor<2>,
        gamma: f32,
    ) -> AdvantageEstimate {
        (**self).estimate(
            rewards,
            values,
            next_dones,
            next_truncateds,
            bootstrapped_values,
            gamma,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum GAEError {
    #[error("GAE lambda must be finite and in 0..=1")]
    InvalidLambda,
}

/// Generalized advantage estimation (Schulman et al., 2016).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GAE {
    lambda: f32,
}

impl GAE {
    /// Rejects a non-finite `lambda` or one outside `0..=1`.
    pub fn new(lambda: f32) -> Result<Self, GAEError> {
        if !lambda.is_finite() || !(0.0..=1.0).contains(&lambda) {
            return Err(GAEError::InvalidLambda);
        }
        Ok(Self { lambda })
    }

    pub fn lambda(&self) -> f32 {
        self.lambda
    }
}

impl AdvantageEstimator for GAE {
    /// Computes GAE advantages shaped `[time, env_count, 1]`; value targets are the λ-returns
    /// `values + advantages`.
    fn estimate(
        &self,
        rewards: &Tensor<3>,
        values: &Tensor<3>,
        next_dones: &Tensor<3, Bool>,
        next_truncateds: &Tensor<3, Bool>,
        bootstrapped_values: &Tensor<2>,
        gamma: f32,
    ) -> AdvantageEstimate {
        let device = values.device();
        let dtype = values.dtype();
        let rewards = rewards.clone();
        let next_dones = next_dones.clone().float().cast(dtype);
        let next_truncateds = next_truncateds.clone().float().cast(dtype);
        let bootstrapped_values = bootstrapped_values.clone();
        let values = values.clone().detach();
        let masks = (1.0f64 - next_dones) * (1.0f64 - next_truncateds);
        let mut next_value = bootstrapped_values.detach();
        let mut gae = Tensor::zeros(next_value.dims(), (&device, dtype));
        let mut advantages = Vec::with_capacity(rewards.dims()[0]);
        // Environments are independent; only time has a recurrence.
        for time in (0..rewards.dims()[0]).rev() {
            let mask = masks
                .clone()
                .slice_dim(0, time..time + 1)
                .squeeze_dim::<2>(0);
            let value = values
                .clone()
                .slice_dim(0, time..time + 1)
                .squeeze_dim::<2>(0);
            let reward = rewards
                .clone()
                .slice_dim(0, time..time + 1)
                .squeeze_dim::<2>(0);
            let delta = reward + next_value * mask.clone() * f64::from(gamma) - value.clone();
            gae = delta + gae * mask * (f64::from(gamma) * f64::from(self.lambda));
            advantages.push(gae.clone());
            next_value = value;
        }
        advantages.reverse();
        let advantages = Tensor::stack(advantages, 0);
        let value_targets = values.clone() + advantages.clone();
        AdvantageEstimate {
            advantages,
            value_targets,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum NStepError {
    #[error("n-step returns need at least one step")]
    ZeroSteps,
}

/// n-step returns (Mnih et al., 2016): rewards summed over up to `n` steps, then bootstrapped
/// from the critic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NStep {
    n: usize,
}

impl NStep {
    /// Rejects `n == 0`.
    pub fn new(n: usize) -> Result<Self, NStepError> {
        if n == 0 {
            return Err(NStepError::ZeroSteps);
        }
        Ok(Self { n })
    }

    /// Returns the number of steps in the return horizon.
    pub fn n(&self) -> usize {
        self.n
    }
}

impl AdvantageEstimator for NStep {
    /// Computes n-step advantages and value targets shaped `[time, env_count, 1]`.
    fn estimate(
        &self,
        rewards: &Tensor<3>,
        values: &Tensor<3>,
        next_dones: &Tensor<3, Bool>,
        next_truncateds: &Tensor<3, Bool>,
        bootstrapped_values: &Tensor<2>,
        gamma: f32,
    ) -> AdvantageEstimate {
        let device = values.device();
        let dtype = values.dtype();
        let rewards = rewards.clone();
        let values = values.clone().detach();
        let next_dones = next_dones.clone().float().cast(dtype);
        let next_truncateds = next_truncateds.clone().float().cast(dtype);
        let bootstrapped_values = bootstrapped_values.clone().detach();
        let masks = (1.0f64 - next_dones) * (1.0f64 - next_truncateds);
        let time_steps = rewards.dims()[0];
        let mut advantages = Vec::with_capacity(time_steps);
        let mut value_targets = Vec::with_capacity(time_steps);

        for time in 0..time_steps {
            let step_count = self.n.min(time_steps - time);
            let mut discounted_return = Tensor::zeros(bootstrapped_values.dims(), (&device, dtype));
            let mut discount = Tensor::ones(bootstrapped_values.dims(), (&device, dtype));
            // Masks make rewards after an episode boundary contribute zero.
            for offset in 0..step_count {
                let step = time + offset;
                let reward = rewards
                    .clone()
                    .slice_dim(0, step..step + 1)
                    .squeeze_dim::<2>(0);
                discounted_return = discounted_return + discount.clone() * reward;
                let mask = masks
                    .clone()
                    .slice_dim(0, step..step + 1)
                    .squeeze_dim::<2>(0);
                discount = discount * mask * f64::from(gamma);
            }
            let next_value = if time + step_count < time_steps {
                values
                    .clone()
                    .slice_dim(0, time + step_count..time + step_count + 1)
                    .squeeze_dim::<2>(0)
            } else {
                bootstrapped_values.clone()
            };
            let value_target = discounted_return + discount * next_value;
            let value = values
                .clone()
                .slice_dim(0, time..time + 1)
                .squeeze_dim::<2>(0);
            advantages.push(value_target.clone() - value);
            value_targets.push(value_target);
        }

        AdvantageEstimate {
            advantages: Tensor::stack(advantages, 0),
            value_targets: Tensor::stack(value_targets, 0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AdvantageEstimator, GAE, GAEError, NStep, NStepError};
    use burn::tensor::{Bool, DType, Device, Tensor, TensorData};

    const REWARDS: [[f32; 3]; 3] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    const VALUES: [[f32; 3]; 3] = [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]];
    const DONES: [[bool; 3]; 3] = [
        [false, false, false],
        [true, false, false],
        [false, true, false],
    ];
    const TRUNCATEDS: [[bool; 3]; 3] = [
        [false, false, false],
        [false, true, false],
        [false, false, true],
    ];
    const BOOTSTRAP: [f32; 3] = [1.1, 1.2, 1.3];

    fn rollout_tensors(
        device: &Device,
    ) -> (
        Tensor<3>,
        Tensor<3>,
        Tensor<3, Bool>,
        Tensor<3, Bool>,
        Tensor<2>,
    ) {
        let reward_tensor = Tensor::<3>::from_data(
            TensorData::new(REWARDS.into_iter().flatten().collect::<Vec<_>>(), [3, 3, 1]),
            device,
        );
        let value_tensor = Tensor::<3>::from_data(
            TensorData::new(VALUES.into_iter().flatten().collect::<Vec<_>>(), [3, 3, 1]),
            device,
        );
        let done_tensor = Tensor::from_data(
            TensorData::new(DONES.into_iter().flatten().collect::<Vec<_>>(), [3, 3, 1]),
            device,
        );
        let truncated_tensor = Tensor::from_data(
            TensorData::new(
                TRUNCATEDS.into_iter().flatten().collect::<Vec<_>>(),
                [3, 3, 1],
            ),
            device,
        );
        let bootstrap_tensor =
            Tensor::<2>::from_data(TensorData::new(BOOTSTRAP.to_vec(), [3, 1]), device);
        (
            reward_tensor,
            value_tensor,
            done_tensor,
            truncated_tensor,
            bootstrap_tensor,
        )
    }

    #[test]
    fn gae_matches_scalar_reference_across_environments_and_episode_boundaries() {
        let device = Device::flex().autodiff();
        let (reward_tensor, value_tensor, done_tensor, truncated_tensor, bootstrap_tensor) =
            rollout_tensors(&device);
        for dtype in [DType::F32, DType::F64] {
            let estimate = GAE::new(0.8).unwrap().estimate(
                &reward_tensor.clone().cast(dtype),
                &value_tensor.clone().cast(dtype),
                &done_tensor,
                &truncated_tensor,
                &bootstrap_tensor.clone().cast(dtype),
                0.9,
            );
            assert_eq!(estimate.advantages.dtype(), dtype);
            assert_eq!(estimate.value_targets.dtype(), dtype);
            let actual_advantages = estimate
                .advantages
                .into_data()
                .convert::<f64>()
                .try_to_vec::<f64>()
                .unwrap();
            let actual_value_targets = estimate
                .value_targets
                .into_data()
                .convert::<f64>()
                .try_to_vec::<f64>()
                .unwrap();
            for env_idx in 0..3 {
                let mut next_value = BOOTSTRAP[env_idx];
                let mut gae = 0.0_f32;
                for time in (0..3).rev() {
                    let mask = if DONES[time][env_idx] || TRUNCATEDS[time][env_idx] {
                        0.0
                    } else {
                        1.0
                    };
                    let value = VALUES[time][env_idx];
                    let delta = REWARDS[time][env_idx] + 0.9 * next_value * mask - value;
                    gae = delta + 0.9 * 0.8 * gae * mask;
                    let sample_index = time * 3 + env_idx;
                    assert!((actual_advantages[sample_index] - f64::from(gae)).abs() < 1e-5);
                    assert!(
                        (actual_value_targets[sample_index] - f64::from(value + gae)).abs() < 1e-5
                    );
                    next_value = value;
                }
            }
        }
    }

    #[test]
    fn n_step_matches_scalar_reference_across_environments_and_episode_boundaries() {
        let device = Device::flex().autodiff();
        let (reward_tensor, value_tensor, done_tensor, truncated_tensor, bootstrap_tensor) =
            rollout_tensors(&device);
        for dtype in [DType::F32, DType::F64] {
            let estimate = NStep::new(2).unwrap().estimate(
                &reward_tensor.clone().cast(dtype),
                &value_tensor.clone().cast(dtype),
                &done_tensor,
                &truncated_tensor,
                &bootstrap_tensor.clone().cast(dtype),
                0.9,
            );
            assert_eq!(estimate.advantages.dtype(), dtype);
            assert_eq!(estimate.value_targets.dtype(), dtype);
            let actual_advantages = estimate
                .advantages
                .into_data()
                .convert::<f64>()
                .try_to_vec::<f64>()
                .unwrap();
            let actual_value_targets = estimate
                .value_targets
                .into_data()
                .convert::<f64>()
                .try_to_vec::<f64>()
                .unwrap();
            for env_idx in 0..3 {
                for time in 0..3 {
                    let step_count = 2.min(3 - time);
                    let mut target = 0.0_f32;
                    let mut discount = 1.0_f32;
                    for offset in 0..step_count {
                        let step = time + offset;
                        target += discount * REWARDS[step][env_idx];
                        let mask = if DONES[step][env_idx] || TRUNCATEDS[step][env_idx] {
                            0.0
                        } else {
                            1.0
                        };
                        discount *= mask * 0.9;
                    }
                    let next_value = if time + step_count < 3 {
                        VALUES[time + step_count][env_idx]
                    } else {
                        BOOTSTRAP[env_idx]
                    };
                    target += discount * next_value;
                    let sample_index = time * 3 + env_idx;
                    assert!(
                        (actual_advantages[sample_index]
                            - f64::from(target - VALUES[time][env_idx]))
                        .abs()
                            < 1e-5
                    );
                    assert!((actual_value_targets[sample_index] - f64::from(target)).abs() < 1e-5);
                }
            }
        }
    }

    #[test]
    fn n_step_matches_gae_at_extremes() {
        let device = Device::flex().autodiff();
        let (reward_tensor, value_tensor, done_tensor, truncated_tensor, bootstrap_tensor) =
            rollout_tensors(&device);
        for dtype in [DType::F32, DType::F64] {
            let rewards = reward_tensor.clone().cast(dtype);
            let values = value_tensor.clone().cast(dtype);
            let bootstrapped_values = bootstrap_tensor.clone().cast(dtype);
            let n_step_one = NStep::new(1).unwrap().estimate(
                &rewards,
                &values,
                &done_tensor,
                &truncated_tensor,
                &bootstrapped_values,
                0.9,
            );
            let gae_zero = GAE::new(0.0).unwrap().estimate(
                &rewards,
                &values,
                &done_tensor,
                &truncated_tensor,
                &bootstrapped_values,
                0.9,
            );
            let n_step_three = NStep::new(3).unwrap().estimate(
                &rewards,
                &values,
                &done_tensor,
                &truncated_tensor,
                &bootstrapped_values,
                0.9,
            );
            let gae_one = GAE::new(1.0).unwrap().estimate(
                &rewards,
                &values,
                &done_tensor,
                &truncated_tensor,
                &bootstrapped_values,
                0.9,
            );
            for (n_step, gae) in [(n_step_one, gae_zero), (n_step_three, gae_one)] {
                let advantages = n_step
                    .advantages
                    .into_data()
                    .convert::<f64>()
                    .try_to_vec::<f64>()
                    .unwrap();
                let gae_advantages = gae
                    .advantages
                    .into_data()
                    .convert::<f64>()
                    .try_to_vec::<f64>()
                    .unwrap();
                let value_targets = n_step
                    .value_targets
                    .into_data()
                    .convert::<f64>()
                    .try_to_vec::<f64>()
                    .unwrap();
                let gae_value_targets = gae
                    .value_targets
                    .into_data()
                    .convert::<f64>()
                    .try_to_vec::<f64>()
                    .unwrap();
                for sample_index in 0..9 {
                    assert!((advantages[sample_index] - gae_advantages[sample_index]).abs() < 1e-5);
                    assert!(
                        (value_targets[sample_index] - gae_value_targets[sample_index]).abs()
                            < 1e-5
                    );
                }
            }
        }
    }

    #[test]
    fn n_step_rejects_zero_steps() {
        assert_eq!(NStep::new(0), Err(NStepError::ZeroSteps));
        assert_eq!(NStep::new(1).unwrap().n(), 1);
    }

    #[test]
    fn gae_rejects_invalid_lambda() {
        for lambda in [-0.1, 1.1, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(GAE::new(lambda), Err(GAEError::InvalidLambda));
        }
        assert_eq!(GAE::new(0.0).unwrap().lambda(), 0.0);
        assert_eq!(GAE::new(1.0).unwrap().lambda(), 1.0);
    }
}

use burn::tensor::{Bool, Tensor};

/// Computes per-step advantages for an on-policy rollout.
pub trait AdvantageEstimator {
    /// Returns advantages shaped `[time, env_count, 1]` from `rewards`, `values`, `next_dones`,
    /// and `next_truncateds` shaped `[time, env_count, 1]`, and `bootstrapped_values` shaped
    /// `[env_count, 1]` (critic values of the final next observations).
    ///
    /// Float inputs share the environment reward dtype and device; the result must too. Rewards
    /// already include `gamma * V(terminal)` for truncated transitions. Values are detached. PPO
    /// uses `values + advantages` as value targets.
    fn estimate(
        &self,
        rewards: &Tensor<3>,
        values: &Tensor<3>,
        next_dones: &Tensor<3, Bool>,
        next_truncateds: &Tensor<3, Bool>,
        bootstrapped_values: &Tensor<2>,
        gamma: f32,
    ) -> Tensor<3>;
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
    ) -> Tensor<3> {
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
    /// Computes GAE advantages shaped `[time, env_count, 1]` from rollout tensors.
    fn estimate(
        &self,
        rewards: &Tensor<3>,
        values: &Tensor<3>,
        next_dones: &Tensor<3, Bool>,
        next_truncateds: &Tensor<3, Bool>,
        bootstrapped_values: &Tensor<2>,
        gamma: f32,
    ) -> Tensor<3> {
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
        Tensor::stack(advantages, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::{AdvantageEstimator, GAE, GAEError};
    use burn::tensor::{DType, Device, Tensor, TensorData};

    #[test]
    fn gae_matches_scalar_reference_across_environments_and_episode_boundaries() {
        let device = Device::flex().autodiff();
        let rewards = [[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let values = [[0.2_f32, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]];
        let dones = [
            [false, false, false],
            [true, false, false],
            [false, true, false],
        ];
        let truncateds = [
            [false, false, false],
            [false, true, false],
            [false, false, true],
        ];
        let bootstrap = [1.1_f32, 1.2, 1.3];
        let reward_tensor = Tensor::<3>::from_data(
            TensorData::new(rewards.into_iter().flatten().collect::<Vec<_>>(), [3, 3, 1]),
            &device,
        );
        let value_tensor = Tensor::<3>::from_data(
            TensorData::new(values.into_iter().flatten().collect::<Vec<_>>(), [3, 3, 1]),
            &device,
        );
        let done_tensor = Tensor::from_data(
            TensorData::new(dones.into_iter().flatten().collect::<Vec<_>>(), [3, 3, 1]),
            &device,
        );
        let truncated_tensor = Tensor::from_data(
            TensorData::new(
                truncateds.into_iter().flatten().collect::<Vec<_>>(),
                [3, 3, 1],
            ),
            &device,
        );
        let bootstrap_tensor =
            Tensor::<2>::from_data(TensorData::new(bootstrap.to_vec(), [3, 1]), &device);
        for dtype in [DType::F32, DType::F64] {
            let advantages = GAE::new(0.8).unwrap().estimate(
                &reward_tensor.clone().cast(dtype),
                &value_tensor.clone().cast(dtype),
                &done_tensor,
                &truncated_tensor,
                &bootstrap_tensor.clone().cast(dtype),
                0.9,
            );
            assert_eq!(advantages.dtype(), dtype);
            let actual = advantages
                .into_data()
                .convert::<f64>()
                .try_to_vec::<f64>()
                .unwrap();
            for env_idx in 0..3 {
                let mut next_value = bootstrap[env_idx];
                let mut gae = 0.0_f32;
                for time in (0..3).rev() {
                    let mask = if dones[time][env_idx] || truncateds[time][env_idx] {
                        0.0
                    } else {
                        1.0
                    };
                    let value = values[time][env_idx];
                    let delta = rewards[time][env_idx] + 0.9 * next_value * mask - value;
                    gae = delta + 0.9 * 0.8 * gae * mask;
                    assert!((actual[time * 3 + env_idx] - f64::from(gae)).abs() < 1e-5);
                    next_value = value;
                }
            }
        }
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

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

/// Generalized advantage estimation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GAE {
    lambda: f32,
    value_lambda: f32,
}

impl GAE {
    /// Rejects a non-finite `lambda` or one outside `0..=1`.
    pub fn new(lambda: f32) -> Result<Self, GAEError> {
        if !lambda.is_finite() || !(0.0..=1.0).contains(&lambda) {
            return Err(GAEError::InvalidLambda);
        }
        Ok(Self {
            lambda,
            value_lambda: lambda,
        })
    }

    /// Uses `value_lambda` for the value targets' λ-returns; rejects a non-finite value or one
    /// outside `0..=1`.
    pub fn with_value_lambda(mut self, value_lambda: f32) -> Result<Self, GAEError> {
        if !value_lambda.is_finite() || !(0.0..=1.0).contains(&value_lambda) {
            return Err(GAEError::InvalidLambda);
        }
        self.value_lambda = value_lambda;
        Ok(self)
    }

    pub fn lambda(&self) -> f32 {
        self.lambda
    }

    /// Returns the lambda used for value-target λ-returns.
    pub fn value_lambda(&self) -> f32 {
        self.value_lambda
    }
}

impl AdvantageEstimator for GAE {
    /// Computes GAE advantages shaped `[time, env_count, 1]`; value targets are the
    /// `value_lambda` λ-returns.
    fn estimate(
        &self,
        rewards: &Tensor<3>,
        values: &Tensor<3>,
        next_dones: &Tensor<3, Bool>,
        next_truncateds: &Tensor<3, Bool>,
        bootstrapped_values: &Tensor<2>,
        gamma: f32,
    ) -> AdvantageEstimate {
        let dtype = values.dtype();
        let rewards = rewards.clone();
        let next_dones = next_dones.clone().float().cast(dtype);
        let next_truncateds = next_truncateds.clone().float().cast(dtype);
        let bootstrapped_values = bootstrapped_values.clone().detach();
        let values = values.clone().detach();
        let masks = (1.0f64 - next_dones) * (1.0f64 - next_truncateds);
        let time_steps = rewards.dims()[0];
        let next_values = if time_steps == 1 {
            bootstrapped_values.unsqueeze_dim::<3>(0)
        } else {
            Tensor::cat(
                vec![
                    values.clone().slice_dim(0, 1..time_steps),
                    bootstrapped_values.unsqueeze_dim::<3>(0),
                ],
                0,
            )
        };
        let deltas = rewards + next_values * masks.clone() * f64::from(gamma) - values.clone();
        let advantages = discounted_sum(&deltas, &masks, f64::from(gamma) * f64::from(self.lambda));
        let value_advantages = if self.value_lambda == self.lambda {
            advantages.clone()
        } else {
            discounted_sum(
                &deltas,
                &masks,
                f64::from(gamma) * f64::from(self.value_lambda),
            )
        };

        AdvantageEstimate {
            value_targets: values + value_advantages,
            advantages,
        }
    }
}

/// Backward recurrence `acc_t = delta_t + discount * mask_t * acc_{t+1}` over time for tensors
/// shaped `[time, env_count, 1]`.
fn discounted_sum(deltas: &Tensor<3>, masks: &Tensor<3>, discount: f64) -> Tensor<3> {
    let dims = deltas.dims();
    let device = deltas.device();
    let dtype = deltas.dtype();
    let mut accumulator = Tensor::zeros([dims[1], dims[2]], (&device, dtype));
    let mut returns = Vec::with_capacity(dims[0]);
    // Environments are independent; only time has a recurrence.
    for time in (0..dims[0]).rev() {
        let mask = masks
            .clone()
            .slice_dim(0, time..time + 1)
            .squeeze_dim::<2>(0);
        let delta = deltas
            .clone()
            .slice_dim(0, time..time + 1)
            .squeeze_dim::<2>(0);
        accumulator = delta + accumulator * mask * discount;
        returns.push(accumulator.clone());
    }
    returns.reverse();
    Tensor::stack(returns, 0)
}

#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum NStepError {
    #[error("n-step returns need at least one step")]
    ZeroSteps,
}

/// n-step returns: rewards summed over up to `n` steps, then bootstrapped from the critic.
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

/// Upgoing policy update: advantages follow rollout returns only while they beat the critic;
/// value targets come from `value_estimator`.
#[derive(Debug, Clone)]
pub struct UPGO<V> {
    value_estimator: V,
}

impl<V> UPGO<V> {
    /// Creates an estimator that uses `value_estimator` for critic value targets.
    pub fn new(value_estimator: V) -> Self {
        Self { value_estimator }
    }

    /// Returns the estimator used to produce critic value targets.
    pub fn value_estimator(&self) -> &V {
        &self.value_estimator
    }
}

impl<V: AdvantageEstimator> AdvantageEstimator for UPGO<V> {
    /// Computes upgoing advantages shaped `[time, env_count, 1]`; value targets come from
    /// `value_estimator`.
    fn estimate(
        &self,
        rewards: &Tensor<3>,
        values: &Tensor<3>,
        next_dones: &Tensor<3, Bool>,
        next_truncateds: &Tensor<3, Bool>,
        bootstrapped_values: &Tensor<2>,
        gamma: f32,
    ) -> AdvantageEstimate {
        let value_targets = self
            .value_estimator
            .estimate(
                rewards,
                values,
                next_dones,
                next_truncateds,
                bootstrapped_values,
                gamma,
            )
            .value_targets;
        let dtype = values.dtype();
        let rewards = rewards.clone();
        let values = values.clone().detach();
        let next_dones = next_dones.clone().float().cast(dtype);
        let next_truncateds = next_truncateds.clone().float().cast(dtype);
        let bootstrapped_values = bootstrapped_values.clone().detach();
        let masks = (1.0f64 - next_dones) * (1.0f64 - next_truncateds);
        let mut next_return = bootstrapped_values.clone();
        let mut next_q = bootstrapped_values.clone();
        let mut next_value = bootstrapped_values;
        let mut returns = Vec::with_capacity(rewards.dims()[0]);

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
            let continuation = next_value.clone().mask_where(
                next_q.clone().greater_equal(next_value.clone()),
                next_return.clone(),
            );
            let current_return = reward.clone() + continuation * mask.clone() * f64::from(gamma);
            next_q = reward + next_value * mask * f64::from(gamma);
            next_return = current_return.clone();
            next_value = value;
            returns.push(current_return);
        }
        returns.reverse();
        let advantages = Tensor::stack(returns, 0) - values;

        AdvantageEstimate {
            advantages,
            value_targets,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AdvantageEstimator, GAE, GAEError, NStep, NStepError, UPGO};
    use burn::tensor::{Bool, DType, Device, Tensor, TensorData};
    use proptest::{collection::vec, prelude::*};

    /// Per-step columns flattened from `[time, env]`, so the next step of row `i` is `i + envs`.
    #[derive(Debug, Clone)]
    struct Rollout {
        envs: usize,
        rewards: Vec<f32>,
        values: Vec<f32>,
        dones: Vec<bool>,
        truncateds: Vec<bool>,
        bootstrap: Vec<f32>,
    }

    impl Rollout {
        fn reward(&self, i: usize) -> f64 {
            f64::from(self.rewards[i])
        }

        fn value(&self, i: usize) -> f64 {
            f64::from(self.values[i])
        }

        fn mask(&self, i: usize) -> f64 {
            f64::from(u8::from(!(self.dones[i] || self.truncateds[i])))
        }

        fn next_value(&self, i: usize) -> f64 {
            f64::from(
                *self
                    .values
                    .get(i + self.envs)
                    .unwrap_or(&self.bootstrap[i % self.envs]),
            )
        }

        fn estimate(
            &self,
            estimator: &impl AdvantageEstimator,
            gamma: f32,
            dtype: DType,
        ) -> Estimate {
            let device = Device::flex().autodiff();
            let shape = [self.rewards.len() / self.envs, self.envs, 1];
            let float = |data: &[f32]| {
                Tensor::<3>::from_data(TensorData::new(data.to_vec(), shape), (&device, dtype))
            };
            let flags = |data: &[bool]| {
                Tensor::<3, Bool>::from_data(TensorData::new(data.to_vec(), shape), &device)
            };
            let bootstrap = TensorData::new(self.bootstrap.clone(), [self.envs, 1]);
            let estimate = estimator.estimate(
                &float(&self.rewards),
                &float(&self.values),
                &flags(&self.dones),
                &flags(&self.truncateds),
                &Tensor::from_data(bootstrap, (&device, dtype)),
                gamma,
            );
            let values = |tensor: Tensor<3>| {
                assert_eq!((tensor.dims(), tensor.dtype()), (shape, dtype));
                tensor
                    .into_data()
                    .convert::<f64>()
                    .try_to_vec::<f64>()
                    .unwrap()
            };
            (values(estimate.advantages), values(estimate.value_targets))
        }

        /// Backward recurrence `acc_i = delta_i + discount * mask_i * acc_{i + envs}`.
        fn discounted_deltas(&self, gamma: f64, discount: f64) -> Vec<f64> {
            let mut sums = vec![0.0; self.rewards.len()];
            for i in (0..sums.len()).rev() {
                let later = sums.get(i + self.envs).copied().unwrap_or(0.0);
                sums[i] = self.reward(i)
                    + self.mask(i) * (gamma * self.next_value(i) + discount * later)
                    - self.value(i);
            }
            sums
        }

        fn gae(&self, gamma: f32, lambda: f32, value_lambda: f32) -> Estimate {
            let gamma = f64::from(gamma);
            let targets = self.discounted_deltas(gamma, gamma * f64::from(value_lambda));
            (
                self.discounted_deltas(gamma, gamma * f64::from(lambda)),
                targets
                    .iter()
                    .enumerate()
                    .map(|(i, target)| target + self.value(i))
                    .collect(),
            )
        }

        fn n_step(&self, gamma: f32, n: usize) -> Estimate {
            let targets: Vec<f64> = (0..self.rewards.len())
                .map(|i| {
                    let steps: Vec<_> =
                        (i..self.rewards.len()).step_by(self.envs).take(n).collect();
                    let (target, discount) =
                        steps.iter().fold((0.0, 1.0), |(target, discount), &j| {
                            (
                                target + discount * self.reward(j),
                                discount * self.mask(j) * f64::from(gamma),
                            )
                        });
                    target + discount * self.next_value(*steps.last().unwrap())
                })
                .collect();
            (self.minus_values(&targets), targets)
        }

        /// Returns UPGO advantages; value targets come from the wrapped estimator.
        fn upgo_advantages(&self, gamma: f32) -> Vec<f64> {
            let mut returns = vec![0.0; self.rewards.len()];
            let mut q_values = returns.clone();
            for i in (0..returns.len()).rev() {
                let bootstrap = f64::from(self.bootstrap[i % self.envs]);
                let next = |column: &[f64]| column.get(i + self.envs).copied().unwrap_or(bootstrap);
                let (next_return, next_q, next_value) =
                    (next(&returns), next(&q_values), self.next_value(i));
                let continuation = if next_q >= next_value {
                    next_return
                } else {
                    next_value
                };
                let discount = self.mask(i) * f64::from(gamma);
                returns[i] = self.reward(i) + discount * continuation;
                q_values[i] = self.reward(i) + discount * next_value;
            }
            self.minus_values(&returns)
        }

        fn minus_values(&self, column: &[f64]) -> Vec<f64> {
            column
                .iter()
                .enumerate()
                .map(|(i, item)| item - self.value(i))
                .collect()
        }
    }

    /// Flattened `(advantages, value_targets)`.
    type Estimate = (Vec<f64>, Vec<f64>);

    fn assert_close(actual: &[f64], expected: &[f64], dtype: DType) {
        let tolerance = if dtype == DType::F64 { 1e-9 } else { 1e-4 };
        assert_eq!(actual.len(), expected.len());
        for (i, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
                "[{i}]: {actual} != {expected}"
            );
        }
    }

    fn assert_estimates_close(actual: Estimate, expected: Estimate, dtype: DType) {
        assert_close(&actual.0, &expected.0, dtype);
        assert_close(&actual.1, &expected.1, dtype);
    }

    fn unit_interval() -> impl Strategy<Value = f32> {
        prop_oneof![Just(0.0f32), Just(0.5), Just(1.0), 0.0f32..=1.0]
    }

    /// Rollouts of 1..=12 steps over 1..=4 environments; episode ends are sparse so multi-step
    /// recurrences run across several steps before a mask cuts them.
    fn rollout() -> impl Strategy<Value = Rollout> {
        (1usize..=12, 1usize..=4).prop_flat_map(|(time_steps, envs)| {
            let len = time_steps * envs;
            let flags = || vec(prop::bool::weighted(0.2), len);
            (
                vec(-10.0f32..10.0, len),
                vec(-10.0f32..10.0, len),
                flags(),
                flags(),
                vec(-10.0f32..10.0, envs),
            )
                .prop_map(move |(rewards, values, dones, truncateds, bootstrap)| {
                    Rollout {
                        envs,
                        rewards,
                        values,
                        dones,
                        truncateds,
                        bootstrap,
                    }
                })
        })
    }

    fn float_dtype() -> impl Strategy<Value = DType> {
        prop_oneof![Just(DType::F32), Just(DType::F64)]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn gae_matches_scalar_reference(
            rollout in rollout(),
            gamma in unit_interval(),
            lambda in unit_interval(),
            value_lambda in unit_interval(),
            dtype in float_dtype(),
        ) {
            let estimator = GAE::new(lambda).unwrap().with_value_lambda(value_lambda).unwrap();
            let actual = rollout.estimate(&estimator, gamma, dtype);
            assert_estimates_close(actual, rollout.gae(gamma, lambda, value_lambda), dtype);
        }

        #[test]
        fn n_step_matches_scalar_reference(
            rollout in rollout(),
            gamma in unit_interval(),
            n in 1usize..=16,
            dtype in float_dtype(),
        ) {
            let actual = rollout.estimate(&NStep::new(n).unwrap(), gamma, dtype);
            assert_estimates_close(actual, rollout.n_step(gamma, n), dtype);
        }

        #[test]
        fn n_step_matches_gae_at_extremes(
            rollout in rollout(),
            gamma in unit_interval(),
            dtype in float_dtype(),
        ) {
            for (n, lambda) in [(1, 0.0), (rollout.rewards.len() / rollout.envs, 1.0)] {
                let n_step = rollout.estimate(&NStep::new(n).unwrap(), gamma, dtype);
                let gae = rollout.estimate(&GAE::new(lambda).unwrap(), gamma, dtype);
                assert_estimates_close(n_step, gae, dtype);
            }
        }

        // F64 only: UPGO branches on `q >= V`, and F32 rounding can flip near-ties.
        #[test]
        fn upgo_matches_scalar_reference(
            rollout in rollout(),
            gamma in unit_interval(),
            lambda in unit_interval(),
        ) {
            let actual = rollout.estimate(&UPGO::new(GAE::new(lambda).unwrap()), gamma, DType::F64);
            let expected = (rollout.upgo_advantages(gamma), rollout.gae(gamma, lambda, lambda).1);
            assert_estimates_close(actual, expected, DType::F64);
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
            assert_eq!(
                GAE::new(0.9).unwrap().with_value_lambda(lambda),
                Err(GAEError::InvalidLambda)
            );
        }
        assert_eq!(GAE::new(0.0).unwrap().lambda(), 0.0);
        assert_eq!(GAE::new(1.0).unwrap().lambda(), 1.0);
        assert_eq!(GAE::new(0.9).unwrap().value_lambda(), 0.9);
        assert_eq!(
            GAE::new(0.9)
                .unwrap()
                .with_value_lambda(1.0)
                .unwrap()
                .value_lambda(),
            1.0
        );
    }
}

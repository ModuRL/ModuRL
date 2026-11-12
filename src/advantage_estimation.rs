use burn::tensor::{Bool, Tensor};

/// Advantages and critic targets [rows, 1]; dense kernels use rank 3 internally.
#[derive(Debug, Clone)]
pub struct AdvantageEstimate<const D: usize = 2> {
    /// Per-row advantages, with the same leading dimensions as the input values.
    pub advantages: Tensor<D>,
    /// Critic targets, with the same leading dimensions as the input values.
    pub value_targets: Tensor<D>,
}

/// Computes advantages and critic value targets for an on-policy rollout.
trait DenseAdvantageEstimator {
    /// Returns advantages and value targets shaped `[time, env_count, 1]` from `rewards`,
    /// `values`, `next_dones`, and `next_truncateds` shaped `[time, env_count, 1]`, and
    /// `bootstrapped_values` shaped `[env_count, 1]` (critic values of the final next observations).
    ///
    /// Float inputs share the environment reward dtype and device. Both outputs must have shape
    /// `[time, env_count, 1]` and the reward dtype and device of `values`. Rewards already include
    /// `gamma * V(terminal)` for truncated transitions. Values are detached. The critic is trained
    /// toward `value_targets`.
    fn estimate_dense(
        &self,
        rewards: &Tensor<3>,
        values: &Tensor<3>,
        next_dones: &Tensor<3, Bool>,
        next_truncateds: &Tensor<3, Bool>,
        bootstrapped_values: &Tensor<2>,
        gamma: f32,
    ) -> AdvantageEstimate<3>;
}

/// Host temporal links for compact steps, preserving collection row order.
#[derive(Debug, Clone)]
pub struct RolloutLayout {
    step_offsets: Vec<usize>,
    successor: Vec<Option<usize>>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RolloutLayoutError {
    #[error("duplicate slot {slot} in step {step}")]
    DuplicateSlot { step: usize, slot: usize },
    #[error("rollout row count overflow")]
    TooManyRows,
    #[error("invalid compact rollout field: {0}")]
    InvalidBatch(&'static str),
}

impl RolloutLayout {
    /// Builds links from slots in each step's compact row order.
    pub fn from_steps(steps: &[&[usize]]) -> Result<Self, RolloutLayoutError> {
        for (step, slots) in steps.iter().enumerate() {
            crate::multi_agent::validate_slots(slots.len(), slots).map_err(
                |error| match error {
                    crate::multi_agent::SlotError::DuplicateSlot(slot) => {
                        RolloutLayoutError::DuplicateSlot { step, slot }
                    }
                    crate::multi_agent::SlotError::LengthMismatch { .. } => {
                        RolloutLayoutError::InvalidBatch("slots length")
                    }
                },
            )?;
        }
        Self::from_valid_steps(steps)
    }

    pub(crate) fn from_valid_steps(steps: &[&[usize]]) -> Result<Self, RolloutLayoutError> {
        let mut step_offsets = vec![0usize];
        for slots in steps {
            step_offsets.push(
                step_offsets
                    .last()
                    .copied()
                    .unwrap_or(0)
                    .checked_add(slots.len())
                    .ok_or(RolloutLayoutError::TooManyRows)?,
            );
        }
        let mut successor = vec![None; *step_offsets.last().unwrap_or(&0)];
        let mut latest = std::collections::HashMap::new();
        for (step, slots) in steps.iter().enumerate().rev() {
            for (row, &slot) in slots.iter().enumerate() {
                let index = step_offsets[step] + row;
                successor[index] = latest.insert(slot, index);
            }
        }
        Ok(Self {
            step_offsets,
            successor,
        })
    }

    pub fn step_offsets(&self) -> &[usize] {
        &self.step_offsets
    }
    pub fn successor(&self) -> &[Option<usize>] {
        &self.successor
    }
}

/// Validated flat rewards and values [rows, 1], with bootstrap values [tails, 1].
/// Rewards include gamma times terminal values for non-done truncations.
/// Bootstrap values follow flat row order for non-ended rows without a successor.
#[derive(Clone, Copy)]
pub struct RaggedRollout<'a> {
    rewards: &'a Tensor<2>,
    values: &'a Tensor<2>,
    dones: &'a [bool],
    truncateds: &'a [bool],
    layout: &'a RolloutLayout,
    bootstrap: &'a Tensor<2>,
}

impl<'a> RaggedRollout<'a> {
    pub fn rewards(&self) -> &Tensor<2> {
        self.rewards
    }
    pub fn values(&self) -> &Tensor<2> {
        self.values
    }
    pub fn dones(&self) -> &[bool] {
        self.dones
    }
    pub fn truncateds(&self) -> &[bool] {
        self.truncateds
    }
    pub fn layout(&self) -> &RolloutLayout {
        self.layout
    }
    pub fn bootstrap(&self) -> &Tensor<2> {
        self.bootstrap
    }

    /// Validates tensors [rows, 1], flags, layout and bootstrap values [tails, 1].
    pub fn new(
        rewards: &'a Tensor<2>,
        values: &'a Tensor<2>,
        dones: &'a [bool],
        truncateds: &'a [bool],
        layout: &'a RolloutLayout,
        bootstrap: &'a Tensor<2>,
    ) -> Result<Self, RolloutLayoutError> {
        let n = layout.successor.len();
        for (field, valid) in [
            ("rewards", rewards.dims() == [n, 1]),
            ("values", values.dims() == [n, 1]),
            ("dones", dones.len() == n),
            ("truncateds", truncateds.len() == n),
            (
                "dtype",
                rewards.dtype() == values.dtype() && bootstrap.dtype() == values.dtype(),
            ),
            (
                "device",
                rewards.device() == values.device() && bootstrap.device() == values.device(),
            ),
        ] {
            if !valid {
                return Err(RolloutLayoutError::InvalidBatch(field));
            }
        }
        let tails = (0..n)
            .filter(|&i| !dones[i] && !truncateds[i] && layout.successor[i].is_none())
            .count();
        if bootstrap.dims() != [tails, 1] {
            return Err(RolloutLayoutError::InvalidBatch("bootstrap"));
        }
        Ok(Self {
            rewards,
            values,
            dones,
            truncateds,
            layout,
            bootstrap,
        })
    }
}

/// Dense inputs use [time, env_count, 1] and final bootstrap values [env_count, 1],
/// without a slot layout. Both variants include truncation bootstraps in rewards.
#[derive(Clone, Copy)]
pub enum AdvantageInput<'a> {
    Dense {
        rewards: &'a Tensor<3>,
        values: &'a Tensor<3>,
        next_dones: &'a Tensor<3, Bool>,
        next_truncateds: &'a Tensor<3, Bool>,
        bootstrapped_values: &'a Tensor<2>,
    },
    Ragged(RaggedRollout<'a>),
}

impl AdvantageInput<'_> {
    /// Returns rewards and values [rows, 1] without padding.
    pub fn rewards_and_values(&self) -> (Tensor<2>, Tensor<2>) {
        match self {
            Self::Dense {
                rewards, values, ..
            } => (
                (*rewards).clone().flatten(0, 1),
                (*values).clone().flatten(0, 1),
            ),
            Self::Ragged(r) => (r.rewards.clone(), r.values.clone()),
        }
    }
    pub fn step_count(&self) -> usize {
        match self {
            Self::Dense { rewards, .. } => rewards.dims()[0],
            Self::Ragged(r) => r.layout.step_offsets.len() - 1,
        }
    }
}

/// Computes advantages and critic targets [rows, 1] in collection order.
/// Outputs must preserve the input dtype and device.
pub trait AdvantageEstimator {
    fn estimate(&self, rollout: AdvantageInput<'_>, gamma: f32) -> AdvantageEstimate;
}

impl AdvantageEstimator for Box<dyn AdvantageEstimator> {
    fn estimate(&self, rollout: AdvantageInput<'_>, gamma: f32) -> AdvantageEstimate {
        (**self).estimate(rollout, gamma)
    }
}

fn indices(rows: &[usize], device: &burn::tensor::Device) -> Tensor<1, burn::tensor::Int> {
    Tensor::from_data(
        burn::tensor::TensorData::new(
            rows.iter().map(|&i| i as i64).collect::<Vec<_>>(),
            [rows.len()],
        ),
        (device, burn::tensor::DType::I64),
    )
}

/// Builds a Float continuation column [rows, 1] matching the reference tensor.
fn flags(values: Vec<bool>, like: &Tensor<2>) -> Tensor<2> {
    let n = values.len();
    Tensor::<2, Bool>::from_data(
        burn::tensor::TensorData::new(values, [n, 1]),
        &like.device(),
    )
    .float()
    .cast(like.dtype())
}

impl RaggedRollout<'_> {
    fn masks(&self) -> Tensor<2> {
        flags(
            self.dones
                .iter()
                .zip(self.truncateds)
                .map(|(&done, &truncated)| !(done || truncated))
                .collect(),
            self.values,
        )
    }
    fn next_values(&self) -> Tensor<2> {
        let n = self.layout.successor.len();
        let mut tail = n;
        let rows: Vec<_> = (0..n)
            .map(|i| {
                if !self.dones[i] && !self.truncateds[i] && self.layout.successor[i].is_none() {
                    let index = tail;
                    tail += 1;
                    index
                } else {
                    self.layout.successor[i].unwrap_or(i)
                }
            })
            .collect();
        let source = if tail == n {
            self.values.clone().detach()
        } else {
            Tensor::cat(
                vec![
                    self.values.clone().detach(),
                    self.bootstrap.clone().detach(),
                ],
                0,
            )
        };
        let device = source.device();
        source.select(0, indices(&rows, &device))
    }
    fn empty(&self) -> AdvantageEstimate {
        AdvantageEstimate {
            advantages: self.values.clone().detach(),
            value_targets: self.values.clone().detach(),
        }
    }
    fn gae(&self, gamma: f32, lambda: f32, value_lambda: f32) -> AdvantageEstimate {
        let n = self.layout.successor.len();
        if n == 0 {
            return self.empty();
        }
        let masks = self.masks();
        let deltas = self.rewards.clone() + self.next_values() * masks.clone() * f64::from(gamma)
            - self.values.clone().detach();
        let mut accumulators =
            Tensor::<2>::zeros([n, 2], (&self.values.device(), self.values.dtype()));
        let discounts = Tensor::from_data(
            [[lambda, value_lambda]],
            (&self.values.device(), self.values.dtype()),
        ) * f64::from(gamma);
        for step in self.layout.step_offsets.windows(2).rev() {
            let (start, end) = (step[0], step[1]);
            if start == end {
                continue;
            }
            let rows: Vec<_> = (start..end)
                .map(|i| self.layout.successor[i].unwrap_or(i))
                .collect();
            let later = accumulators
                .clone()
                .select(0, indices(&rows, &self.values.device()));
            let current = deltas.clone().slice_dim(0, start..end).repeat_dim(1, 2)
                + later * masks.clone().slice_dim(0, start..end) * discounts.clone();
            accumulators = accumulators.slice_assign([start..end, 0..2], current);
        }
        AdvantageEstimate {
            advantages: accumulators.clone().slice_dim(1, 0..1),
            value_targets: self.values.clone().detach() + accumulators.slice_dim(1, 1..2),
        }
    }
    fn n_step(&self, gamma: f32, n_steps: usize) -> AdvantageEstimate {
        let n = self.layout.successor.len();
        if n == 0 {
            return self.empty();
        }
        let source = Tensor::cat(vec![self.rewards.clone(), self.next_values()], 1);
        let mut rows: Vec<_> = (0..n).collect();
        let mut live = vec![true; n];
        let mut targets = Tensor::<2>::zeros([n, 1], (&self.values.device(), self.values.dtype()));
        let mut discount = Tensor::<2>::ones([n, 1], (&self.values.device(), self.values.dtype()));
        for offset in 0..n_steps {
            let current = source
                .clone()
                .select(0, indices(&rows, &self.values.device()));
            targets = targets + discount.clone() * current.clone().slice_dim(1, 0..1);
            let continuing: Vec<_> = rows
                .iter()
                .enumerate()
                .map(|(i, &row)| live[i] && !self.dones[row] && !self.truncateds[row])
                .collect();
            discount = discount * flags(continuing.clone(), self.values) * f64::from(gamma);
            let next_live: Vec<_> = rows
                .iter()
                .enumerate()
                .map(|(i, &row)| {
                    continuing[i] && offset + 1 < n_steps && self.layout.successor[row].is_some()
                })
                .collect();
            targets = targets
                + discount.clone()
                    * current.slice_dim(1, 1..2)
                    * flags(next_live.iter().map(|&v| !v).collect(), self.values);
            discount = discount * flags(next_live.clone(), self.values);
            for (i, row) in rows.iter_mut().enumerate() {
                if next_live[i] {
                    *row = self.layout.successor[*row].unwrap_or(*row);
                }
            }
            live = next_live;
            if !live.contains(&true) {
                break;
            }
        }
        AdvantageEstimate {
            advantages: targets.clone() - self.values.clone().detach(),
            value_targets: targets,
        }
    }
    /// Computes compact upgoing advantages with critic targets [rows, 1].
    fn upgo(&self, gamma: f32, value_targets: Tensor<2>) -> AdvantageEstimate {
        let n = self.layout.successor.len();
        if n == 0 {
            return self.empty();
        }
        let next_values = self.next_values();
        let masks = self.masks();
        let mut accumulators =
            Tensor::<2>::zeros([n, 2], (&self.values.device(), self.values.dtype()));
        for step in self.layout.step_offsets.windows(2).rev() {
            let (start, end) = (step[0], step[1]);
            if start == end {
                continue;
            }
            let rows: Vec<_> = (start..end)
                .map(|i| self.layout.successor[i].unwrap_or(i))
                .collect();
            let next_value = next_values.clone().slice_dim(0, start..end);
            let tail = Tensor::<2, Bool>::from_data(
                burn::tensor::TensorData::new(
                    (start..end)
                        .map(|i| self.layout.successor[i].is_none())
                        .collect::<Vec<_>>(),
                    [end - start, 1],
                ),
                &self.values.device(),
            )
            .repeat_dim(1, 2);
            let later = accumulators
                .clone()
                .select(0, indices(&rows, &self.values.device()))
                .mask_where(tail, next_value.clone().repeat_dim(1, 2));
            let continuation = next_value.clone().mask_where(
                later
                    .clone()
                    .slice_dim(1, 0..1)
                    .greater_equal(next_value.clone()),
                later.slice_dim(1, 1..2),
            );
            let reward = self.rewards.clone().slice_dim(0, start..end);
            let mask = masks.clone().slice_dim(0, start..end);
            let current_return = reward.clone() + continuation * mask.clone() * f64::from(gamma);
            let q = reward + next_value * mask * f64::from(gamma);
            accumulators = accumulators
                .slice_assign([start..end, 0..2], Tensor::cat(vec![q, current_return], 1));
        }
        AdvantageEstimate {
            advantages: accumulators.slice_dim(1, 1..2) - self.values.clone().detach(),
            value_targets,
        }
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

impl DenseAdvantageEstimator for GAE {
    /// Computes GAE advantages shaped `[time, env_count, 1]`; value targets are the
    /// `value_lambda` λ-returns.
    fn estimate_dense(
        &self,
        rewards: &Tensor<3>,
        values: &Tensor<3>,
        next_dones: &Tensor<3, Bool>,
        next_truncateds: &Tensor<3, Bool>,
        bootstrapped_values: &Tensor<2>,
        gamma: f32,
    ) -> AdvantageEstimate<3> {
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

impl DenseAdvantageEstimator for NStep {
    /// Computes n-step advantages and value targets shaped `[time, env_count, 1]`.
    fn estimate_dense(
        &self,
        rewards: &Tensor<3>,
        values: &Tensor<3>,
        next_dones: &Tensor<3, Bool>,
        next_truncateds: &Tensor<3, Bool>,
        bootstrapped_values: &Tensor<2>,
        gamma: f32,
    ) -> AdvantageEstimate<3> {
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

impl<V: AdvantageEstimator> DenseAdvantageEstimator for UPGO<V> {
    /// Computes upgoing advantages shaped `[time, env_count, 1]`; value targets come from
    /// `value_estimator`.
    fn estimate_dense(
        &self,
        rewards: &Tensor<3>,
        values: &Tensor<3>,
        next_dones: &Tensor<3, Bool>,
        next_truncateds: &Tensor<3, Bool>,
        bootstrapped_values: &Tensor<2>,
        gamma: f32,
    ) -> AdvantageEstimate<3> {
        let value_targets = self
            .value_estimator
            .estimate(
                AdvantageInput::Dense {
                    rewards,
                    values,
                    next_dones,
                    next_truncateds,
                    bootstrapped_values,
                },
                gamma,
            )
            .value_targets
            .reshape(values.dims());
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

macro_rules! estimate_input {
    ($self:ident, $input:ident, $gamma:ident, $ragged:expr) => {
        match $input {
            AdvantageInput::Dense {
                rewards,
                values,
                next_dones,
                next_truncateds,
                bootstrapped_values,
            } => {
                let result = $self.estimate_dense(
                    rewards,
                    values,
                    next_dones,
                    next_truncateds,
                    bootstrapped_values,
                    $gamma,
                );
                AdvantageEstimate {
                    advantages: result.advantages.flatten(0, 1),
                    value_targets: result.value_targets.flatten(0, 1),
                }
            }
            AdvantageInput::Ragged(r) => ($ragged)(r),
        }
    };
}
impl AdvantageEstimator for GAE {
    fn estimate(&self, input: AdvantageInput<'_>, gamma: f32) -> AdvantageEstimate {
        estimate_input!(self, input, gamma, |r: RaggedRollout<'_>| r.gae(
            gamma,
            self.lambda,
            self.value_lambda
        ))
    }
}
impl AdvantageEstimator for NStep {
    fn estimate(&self, input: AdvantageInput<'_>, gamma: f32) -> AdvantageEstimate {
        estimate_input!(self, input, gamma, |r: RaggedRollout<'_>| r
            .n_step(gamma, self.n))
    }
}
impl<V: AdvantageEstimator> AdvantageEstimator for UPGO<V> {
    fn estimate(&self, input: AdvantageInput<'_>, gamma: f32) -> AdvantageEstimate {
        estimate_input!(self, input, gamma, |r: RaggedRollout<'_>| r.upgo(
            gamma,
            self.value_estimator
                .estimate(AdvantageInput::Ragged(r), gamma)
                .value_targets
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AdvantageEstimator, AdvantageInput, GAE, GAEError, NStep, NStepError, RaggedRollout,
        RolloutLayout, RolloutLayoutError, UPGO,
    };
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
                AdvantageInput::Dense {
                    rewards: &float(&self.rewards),
                    values: &float(&self.values),
                    next_dones: &flags(&self.dones),
                    next_truncateds: &flags(&self.truncateds),
                    bootstrapped_values: &Tensor::from_data(bootstrap, (&device, dtype)),
                },
                gamma,
            );
            let values = |tensor: Tensor<2>| {
                assert_eq!(
                    (tensor.dims(), tensor.dtype()),
                    ([self.rewards.len(), 1], dtype)
                );
                tensor
                    .into_data()
                    .convert::<f64>()
                    .try_to_vec::<f64>()
                    .unwrap()
            };
            (values(estimate.advantages), values(estimate.value_targets))
        }

        fn estimate_ragged(
            &self,
            estimator: &impl AdvantageEstimator,
            gamma: f32,
            dtype: DType,
        ) -> Estimate {
            let device = Device::flex().autodiff();
            let steps: Vec<_> = (0..self.rewards.len() / self.envs)
                .map(|_| (0..self.envs).collect::<Vec<_>>())
                .collect();
            let layout =
                RolloutLayout::from_steps(&steps.iter().map(Vec::as_slice).collect::<Vec<_>>())
                    .unwrap();
            let n = self.rewards.len();
            let float = |v: Vec<f32>| {
                let count = v.len();
                Tensor::from_data(TensorData::new(v, [count, 1]), (&device, dtype))
            };
            let rewards = float(self.rewards.clone());
            let values = float(self.values.clone());
            let bootstrap = float(
                (n - self.envs..n)
                    .filter(|&i| !self.dones[i] && !self.truncateds[i])
                    .map(|i| self.bootstrap[i % self.envs])
                    .collect(),
            );
            let result = estimator.estimate(
                AdvantageInput::Ragged(
                    RaggedRollout::new(
                        &rewards,
                        &values,
                        &self.dones,
                        &self.truncateds,
                        &layout,
                        &bootstrap,
                    )
                    .unwrap(),
                ),
                gamma,
            );
            let data = |t: Tensor<2>| {
                assert_eq!(t.dims(), [n, 1]);
                assert_eq!(t.dtype(), dtype);
                t.to_data().convert::<f64>().try_to_vec::<f64>().unwrap()
            };
            (data(result.advantages), data(result.value_targets))
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
            assert_estimates_close(rollout.estimate_ragged(&estimator, gamma, dtype), actual.clone(), dtype);
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
            assert_estimates_close(rollout.estimate_ragged(&NStep::new(n).unwrap(), gamma, dtype), actual.clone(), dtype);
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
            assert_estimates_close(rollout.estimate_ragged(&UPGO::new(GAE::new(lambda).unwrap()), gamma, DType::F64), actual.clone(), DType::F64);
            let expected = (rollout.upgo_advantages(gamma), rollout.gae(gamma, lambda, lambda).1);
            assert_estimates_close(actual, expected, DType::F64);
        }
    }

    #[test]
    fn irregular_interleaved_rollouts_match_independent_slot_references() {
        let steps = [
            vec![3, 7],
            vec![usize::MAX, 3],
            vec![],
            vec![7],
            vec![3, usize::MAX],
            vec![7, usize::MAX, 3],
        ];
        let layout =
            RolloutLayout::from_steps(&steps.iter().map(Vec::as_slice).collect::<Vec<_>>())
                .unwrap();
        assert_eq!(
            layout.successor(),
            &[
                Some(3),
                Some(4),
                Some(6),
                Some(5),
                Some(7),
                Some(9),
                Some(8),
                None,
                None,
                None
            ]
        );
        let slots: Vec<_> = steps.into_iter().flatten().collect();
        let gamma = 0.91f32;
        let mut rewards = vec![1.0, -2.0, 3.5, 2.0, -4.0, 7.0, 1.0, 2.5, 4.0, -1.0];
        rewards[3] += gamma * 5.5;
        rewards[8] += gamma * 3.75;
        let values = vec![0.2, 1.0, -2.5, 3.0, 0.7, -1.0, 5.0, 1.5, -0.4, 2.0];
        let dones = vec![
            false, false, true, false, true, false, false, false, false, false,
        ];
        let truncateds = vec![
            false, false, true, true, false, false, false, false, true, false,
        ];
        for dtype in [DType::F32, DType::F64] {
            let device = Device::flex().autodiff();
            let float = |v: Vec<f32>| {
                let n = v.len();
                Tensor::from_data(TensorData::new(v, [n, 1]), (&device, dtype))
            };
            let r = float(rewards.clone());
            let v = float(values.clone());
            let b = float(vec![11.0, 17.0]);
            let ragged = RaggedRollout::new(&r, &v, &dones, &truncateds, &layout, &b).unwrap();
            let estimators: Vec<Box<dyn AdvantageEstimator>> = vec![
                Box::new(GAE::new(0.73).unwrap().with_value_lambda(0.41).unwrap()),
                Box::new(NStep::new(1).unwrap()),
                Box::new(NStep::new(2).unwrap()),
                Box::new(NStep::new(6).unwrap()),
                Box::new(UPGO::new(
                    GAE::new(0.73).unwrap().with_value_lambda(0.41).unwrap(),
                )),
            ];
            for (kind, estimator) in estimators.iter().enumerate() {
                let estimate = estimator.estimate(AdvantageInput::Ragged(ragged), gamma);
                let mut expected = (vec![0.0; slots.len()], vec![0.0; slots.len()]);
                for slot in [3, 7, usize::MAX] {
                    let rows: Vec<_> = slots
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &s)| (s == slot).then_some(i))
                        .collect();
                    let sequence = Rollout {
                        envs: 1,
                        rewards: rows.iter().map(|&i| rewards[i]).collect(),
                        values: rows.iter().map(|&i| values[i]).collect(),
                        dones: rows.iter().map(|&i| dones[i]).collect(),
                        truncateds: rows.iter().map(|&i| truncateds[i]).collect(),
                        bootstrap: vec![if slot == 3 {
                            17.0
                        } else if slot == 7 {
                            11.0
                        } else {
                            0.0
                        }],
                    };
                    let reference = match kind {
                        0 => sequence.gae(gamma, 0.73, 0.41),
                        1..=3 => sequence.n_step(gamma, [1, 2, 6][kind - 1]),
                        _ => (
                            sequence.upgo_advantages(gamma),
                            sequence.gae(gamma, 0.73, 0.41).1,
                        ),
                    };
                    for (j, &row) in rows.iter().enumerate() {
                        expected.0[row] = reference.0[j];
                        expected.1[row] = reference.1[j];
                    }
                }
                let data = |t: Tensor<2>| t.to_data().convert::<f64>().try_to_vec::<f64>().unwrap();
                assert_estimates_close(
                    (data(estimate.advantages), data(estimate.value_targets)),
                    expected,
                    dtype,
                );
            }
        }
    }

    #[test]
    fn empty_and_invalid_compact_rollouts_are_checked() {
        assert!(matches!(
            RolloutLayout::from_steps(&[&[7, 7]]),
            Err(RolloutLayoutError::DuplicateSlot { slot: 7, step: 0 })
        ));
        let layout = RolloutLayout::from_steps(&[&[], &[]]).unwrap();
        let device = Device::flex();
        let empty = Tensor::zeros([0, 1], &device);
        let rollout = RaggedRollout::new(&empty, &empty, &[], &[], &layout, &empty).unwrap();
        for estimator in [
            Box::new(GAE::new(0.9).unwrap()) as Box<dyn AdvantageEstimator>,
            Box::new(NStep::new(2).unwrap()),
            Box::new(UPGO::new(GAE::new(0.9).unwrap())),
        ] {
            let result = estimator.estimate(AdvantageInput::Ragged(rollout), 0.99);
            assert_eq!(result.advantages.dims(), [0, 1]);
            assert_eq!(result.value_targets.dims(), [0, 1]);
        }
        assert!(RaggedRollout::new(&empty, &empty, &[true], &[], &layout, &empty).is_err());
        let one = Tensor::zeros([1, 1], &device);
        assert!(RaggedRollout::new(&one, &empty, &[], &[], &layout, &empty).is_err());
        assert!(RaggedRollout::new(&empty, &empty, &[], &[], &layout, &one).is_err());
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

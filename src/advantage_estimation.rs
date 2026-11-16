use burn::tensor::{Bool, Tensor};

/// Advantages and critic targets [rows, 1].
#[derive(Debug, Clone)]
pub struct AdvantageEstimate {
    /// Per-row advantages, with the same leading dimensions as the input values.
    pub advantages: Tensor<2>,
    /// Critic targets, with the same leading dimensions as the input values.
    pub value_targets: Tensor<2>,
}

/// Host temporal links for compact steps, preserving collection row order.
#[derive(Debug, Clone)]
pub struct RolloutLayout {
    step_offsets: Vec<usize>,
    successor: Vec<Option<usize>>,
    links: Vec<StepLinks>,
    successor_prefix: std::ops::Range<usize>,
}

#[derive(Debug, Clone)]
struct StepLinks {
    range: Option<std::ops::Range<usize>>,
    has_successors: bool,
    has_tails: bool,
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
    /// Builds links for identity-ordered slots in equally sized steps without slot maps.
    pub fn dense(time_steps: usize, env_count: usize) -> Result<Self, RolloutLayoutError> {
        let rows = time_steps
            .checked_mul(env_count)
            .ok_or(RolloutLayoutError::TooManyRows)?;
        let offsets = time_steps
            .checked_add(1)
            .ok_or(RolloutLayoutError::TooManyRows)?;
        let mut step_offsets = Vec::new();
        step_offsets
            .try_reserve_exact(offsets)
            .map_err(|_| RolloutLayoutError::TooManyRows)?;
        let mut successor = Vec::new();
        successor
            .try_reserve_exact(rows)
            .map_err(|_| RolloutLayoutError::TooManyRows)?;
        step_offsets.extend((0..offsets).map(|step| step * env_count));
        successor
            .extend((0..rows).map(|row| row.checked_add(env_count).filter(|&next| next < rows)));
        Self::with_links(step_offsets, successor)
    }

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
        Self::with_links(step_offsets, successor)
    }

    fn with_links(
        step_offsets: Vec<usize>,
        successor: Vec<Option<usize>>,
    ) -> Result<Self, RolloutLayoutError> {
        let mut links = Vec::new();
        links
            .try_reserve_exact(step_offsets.len() - 1)
            .map_err(|_| RolloutLayoutError::TooManyRows)?;
        for step in step_offsets.windows(2) {
            let (start, end) = (step[0], step[1]);
            let rows = &successor[start..end];
            let first = rows.first().copied().flatten();
            let range = first.filter(|&first| {
                rows.iter()
                    .enumerate()
                    .all(|(i, &row)| row == Some(first + i))
            });
            links.push(StepLinks {
                range: range.map(|first| first..first + end - start),
                has_successors: rows.iter().any(Option::is_some),
                has_tails: rows.iter().any(Option::is_none),
            });
        }
        let first = successor.first().copied().flatten().unwrap_or(0);
        let prefix = successor
            .iter()
            .enumerate()
            .take_while(|&(i, row)| *row == Some(first + i))
            .count();
        Ok(Self {
            step_offsets,
            successor,
            links,
            successor_prefix: first..first + prefix,
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
pub struct AdvantageInput<'a> {
    rewards: &'a Tensor<2>,
    values: &'a Tensor<2>,
    dones: &'a [bool],
    truncateds: &'a [bool],
    layout: &'a RolloutLayout,
    bootstrap: &'a Tensor<2>,
}

impl<'a> AdvantageInput<'a> {
    /// Returns flat rewards and values [rows, 1] without padding.
    pub fn rewards_and_values(&self) -> (Tensor<2>, Tensor<2>) {
        (self.rewards.clone(), self.values.clone())
    }

    pub fn step_count(&self) -> usize {
        self.layout.step_offsets.len() - 1
    }

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

impl AdvantageInput<'_> {
    fn successor_indices(&self) -> Option<Tensor<1, burn::tensor::Int>> {
        self.layout
            .links
            .iter()
            .any(|step| step.has_successors && step.range.is_none())
            .then(|| {
                Tensor::from_data(
                    burn::tensor::TensorData::new(
                        self.layout
                            .successor
                            .iter()
                            .enumerate()
                            .map(|(i, row)| row.unwrap_or(i) as i64)
                            .collect::<Vec<_>>(),
                        [self.layout.successor.len()],
                    ),
                    (&self.values.device(), burn::tensor::DType::I64),
                )
            })
    }

    /// Selects successor rows from values [rows, columns] using optional flat indices [rows].
    fn successors(
        &self,
        values: &Tensor<2>,
        step: usize,
        indices: &Option<Tensor<1, burn::tensor::Int>>,
    ) -> Tensor<2> {
        if let Some(range) = &self.layout.links[step].range {
            values.clone().slice_dim(0, range.clone())
        } else if let Some(indices) = indices {
            let start = self.layout.step_offsets[step];
            let end = self.layout.step_offsets[step + 1];
            values
                .clone()
                .select(0, indices.clone().slice([start..end]))
        } else {
            values.clone().slice_dim(
                0,
                self.layout.step_offsets[step]..self.layout.step_offsets[step + 1],
            )
        }
    }

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
        let prefix = &self.layout.successor_prefix;
        let rows: Vec<_> = (prefix.len()..n)
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
        if prefix.len() == n {
            source.slice_dim(0, prefix.clone())
        } else if !prefix.is_empty() {
            let device = source.device();
            Tensor::cat(
                vec![
                    source.clone().slice_dim(0, prefix.clone()),
                    source.select(0, indices(&rows, &device)),
                ],
                0,
            )
        } else {
            let device = source.device();
            source.select(0, indices(&rows, &device))
        }
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
        let columns = if lambda == value_lambda { 1 } else { 2 };
        let mut accumulators =
            Tensor::<2>::zeros([n, columns], (&self.values.device(), self.values.dtype()));
        let discounts = if columns == 1 {
            masks * (f64::from(gamma) * f64::from(lambda))
        } else {
            masks
                * (Tensor::from_data(
                    [[lambda, value_lambda]],
                    (&self.values.device(), self.values.dtype()),
                ) * f64::from(gamma))
        };
        let deltas = if columns == 1 {
            deltas
        } else {
            deltas.repeat_dim(1, columns)
        };
        let successor_indices = self.successor_indices();
        for step in (0..self.step_count()).rev() {
            let (start, end) = (
                self.layout.step_offsets[step],
                self.layout.step_offsets[step + 1],
            );
            if start == end {
                continue;
            }
            let current = if self.layout.links[step].has_successors {
                deltas.clone().slice_dim(0, start..end)
                    + self.successors(&accumulators, step, &successor_indices)
                        * discounts.clone().slice_dim(0, start..end)
            } else {
                deltas.clone().slice_dim(0, start..end)
            };
            accumulators = accumulators.slice_assign([start..end, 0..columns], current);
        }
        AdvantageEstimate {
            advantages: accumulators.clone().slice_dim(1, 0..1),
            value_targets: self.values.clone().detach()
                + accumulators.slice_dim(1, columns - 1..columns),
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
        let discounts = self.masks() * f64::from(gamma);
        let q = self.rewards.clone() + next_values.clone() * discounts.clone();
        let mut accumulators =
            Tensor::<2>::zeros([n, 1], (&self.values.device(), self.values.dtype()));
        let successor_indices = self.successor_indices();
        let tails = self
            .layout
            .links
            .iter()
            .any(|step| step.has_tails && step.has_successors)
            .then(|| {
                Tensor::<2, Bool>::from_data(
                    burn::tensor::TensorData::new(
                        self.layout
                            .successor
                            .iter()
                            .map(Option::is_none)
                            .collect::<Vec<_>>(),
                        [n, 1],
                    ),
                    &self.values.device(),
                )
            });
        for step in (0..self.step_count()).rev() {
            let (start, end) = (
                self.layout.step_offsets[step],
                self.layout.step_offsets[step + 1],
            );
            if start == end {
                continue;
            }
            let current_return = if self.layout.links[step].has_successors {
                let next_value = next_values.clone().slice_dim(0, start..end);
                let later = self.successors(&accumulators, step, &successor_indices);
                let later = if self.layout.links[step].has_tails {
                    if let Some(tails) = &tails {
                        later.mask_where(tails.clone().slice_dim(0, start..end), next_value.clone())
                    } else {
                        later
                    }
                } else {
                    later
                };
                let continuation = next_value.clone().mask_where(
                    self.successors(&q, step, &successor_indices)
                        .greater_equal(next_value),
                    later,
                );
                self.rewards.clone().slice_dim(0, start..end)
                    + continuation * discounts.clone().slice_dim(0, start..end)
            } else {
                q.clone().slice_dim(0, start..end)
            };
            accumulators = accumulators.slice_assign([start..end, 0..1], current_return);
        }
        AdvantageEstimate {
            advantages: accumulators - self.values.clone().detach(),
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

impl AdvantageEstimator for GAE {
    fn estimate(&self, input: AdvantageInput<'_>, gamma: f32) -> AdvantageEstimate {
        input.gae(gamma, self.lambda, self.value_lambda)
    }
}
impl AdvantageEstimator for NStep {
    fn estimate(&self, input: AdvantageInput<'_>, gamma: f32) -> AdvantageEstimate {
        input.n_step(gamma, self.n)
    }
}
impl<V: AdvantageEstimator> AdvantageEstimator for UPGO<V> {
    fn estimate(&self, input: AdvantageInput<'_>, gamma: f32) -> AdvantageEstimate {
        input.upgo(
            gamma,
            self.value_estimator.estimate(input, gamma).value_targets,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AdvantageEstimator, AdvantageInput, GAE, GAEError, NStep, NStepError, RolloutLayout,
        RolloutLayoutError, UPGO,
    };
    use burn::tensor::{DType, Device, Tensor, TensorData};
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
            let layout = RolloutLayout::dense(self.rewards.len() / self.envs, self.envs).unwrap();
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
                AdvantageInput::new(
                    &rewards,
                    &values,
                    &self.dones,
                    &self.truncateds,
                    &layout,
                    &bootstrap,
                )
                .unwrap(),
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
    fn dense_layout_links_match_identity_steps_and_check_overflow() {
        for time in 0..6 {
            for envs in 0..5 {
                let slots: Vec<_> = (0..envs).collect();
                let steps = vec![slots.as_slice(); time];
                let dense = RolloutLayout::dense(time, envs).unwrap();
                let compact = RolloutLayout::from_steps(&steps).unwrap();
                assert_eq!(dense.step_offsets(), compact.step_offsets());
                assert_eq!(dense.successor(), compact.successor());
                assert_eq!(dense.successor_prefix, compact.successor_prefix);
                for (a, b) in dense.links.iter().zip(&compact.links) {
                    assert_eq!(a.range, b.range);
                    assert_eq!(a.has_successors, b.has_successors);
                    assert_eq!(a.has_tails, b.has_tails);
                }
            }
        }
        for (time, envs) in [
            (usize::MAX, 0),
            (usize::MAX, 2),
            (1, usize::MAX),
            (usize::MAX - 1, 0),
            (usize::MAX / 2, 2),
        ] {
            assert!(matches!(
                RolloutLayout::dense(time, envs),
                Err(RolloutLayoutError::TooManyRows)
            ));
        }
    }

    #[test]
    fn contiguous_links_span_gaps_and_preserve_mixed_tails() {
        let layout = RolloutLayout::from_steps(&[&[3, 7], &[9], &[3, 7, 11], &[9, 11]]).unwrap();
        assert_eq!(layout.links[0].range, Some(3..5));
        assert_eq!(layout.successor_prefix, 3..5);
        assert_eq!(layout.links[1].range, Some(6..7));
        assert_eq!(layout.links[2].range, None);
        assert!(layout.links[2].has_successors);
        assert!(layout.links[2].has_tails);
        assert!(!layout.links[3].has_successors);
        let device = Device::flex();
        for dtype in [DType::F32, DType::F64] {
            let float = |v: Vec<f32>| {
                let count = v.len();
                Tensor::from_data(TensorData::new(v, [count, 1]), (&device, dtype))
            };
            let rewards = float(vec![1.0, 2.0, 3.0, 4.5, 5.0, 6.0, 7.0, 8.0]);
            let values = float(vec![0.0; 8]);
            let bootstrap = float(vec![3.0, 4.0]);
            let input = AdvantageInput::new(
                &rewards,
                &values,
                &[false, false, false, false, true, false, false, false],
                &[false, false, false, true, false, false, false, false],
                &layout,
                &bootstrap,
            )
            .unwrap();
            let returns = vec![3.25, 4.5, 7.25, 4.5, 5.0, 11.0, 8.5, 10.0];
            for estimator in [
                Box::new(GAE::new(1.0).unwrap()) as Box<dyn AdvantageEstimator>,
                Box::new(NStep::new(5).unwrap()),
                Box::new(UPGO::new(GAE::new(1.0).unwrap())),
            ] {
                let result = estimator.estimate(input, 0.5);
                let data = |t: Tensor<2>| t.to_data().convert::<f64>().try_to_vec::<f64>().unwrap();
                assert_estimates_close(
                    (data(result.advantages), data(result.value_targets)),
                    (returns.clone(), returns.clone()),
                    dtype,
                );
            }
        }
    }

    #[test]
    fn dense_layout_stops_at_episode_ends_and_bootstraps_live_tails() {
        let rollout = Rollout {
            envs: 2,
            rewards: vec![1.0, 10.0, 2.0, 23.5, 7.5, 30.0, 4.0, 40.0],
            values: vec![0.5, 5.0, 1.0, 10.0, 1.5, 15.0, 2.0, 20.0],
            dones: vec![false, false, true, false, false, false, false, true],
            truncateds: vec![false, false, false, true, true, false, false, false],
            bootstrap: vec![6.0, 1000.0],
        };
        let gamma = 0.5;
        let n_step = rollout.n_step(gamma, 5);
        assert_eq!(n_step.1, vec![2.0, 21.75, 2.0, 23.5, 7.5, 50.0, 7.0, 40.0]);
        for dtype in [DType::F32, DType::F64] {
            let gae = GAE::new(0.7).unwrap().with_value_lambda(0.4).unwrap();
            assert_estimates_close(
                rollout.estimate(&gae, gamma, dtype),
                rollout.gae(gamma, 0.7, 0.4),
                dtype,
            );
            assert_estimates_close(
                rollout.estimate(&NStep::new(5).unwrap(), gamma, dtype),
                n_step.clone(),
                dtype,
            );
            assert_estimates_close(
                rollout.estimate(&UPGO::new(gae), gamma, dtype),
                (
                    rollout.upgo_advantages(gamma),
                    rollout.gae(gamma, 0.7, 0.4).1,
                ),
                dtype,
            );
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
            let ragged = AdvantageInput::new(&r, &v, &dones, &truncateds, &layout, &b).unwrap();
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
                let estimate = estimator.estimate(ragged, gamma);
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
        let rollout = AdvantageInput::new(&empty, &empty, &[], &[], &layout, &empty).unwrap();
        for estimator in [
            Box::new(GAE::new(0.9).unwrap()) as Box<dyn AdvantageEstimator>,
            Box::new(NStep::new(2).unwrap()),
            Box::new(UPGO::new(GAE::new(0.9).unwrap())),
        ] {
            let result = estimator.estimate(rollout, 0.99);
            assert_eq!(result.advantages.dims(), [0, 1]);
            assert_eq!(result.value_targets.dims(), [0, 1]);
        }
        assert!(AdvantageInput::new(&empty, &empty, &[true], &[], &layout, &empty).is_err());
        let one = Tensor::zeros([1, 1], &device);
        assert!(AdvantageInput::new(&one, &empty, &[], &[], &layout, &empty).is_err());
        assert!(AdvantageInput::new(&empty, &empty, &[], &[], &layout, &one).is_err());
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

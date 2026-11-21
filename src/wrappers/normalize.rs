//! Running observation and reward normalization wrappers.

use crate::tensor_rank::{NextRank, PrevRank, TensorBatchTypes};
use burn::{
    module::{Module, Param},
    tensor::{DType, Device, Int, Tensor, TensorData, TensorReadError, kind::Basic},
};

use crate::{
    gym::{Gym, LegalActionsGym, PerfectStateGym, ResetInfo, StepInfo},
    spaces::{ActionSpace, ObservationSpace},
};

const RUNNING_STATS_EPSILON: f64 = 1e-4;
const NORMALIZATION_EPSILON: f64 = 1e-8;

/// Running per-component mean, variance, and sample count used by the normalization wrappers.
///
/// It is a Burn [`Module`], so it saves and loads like a model: export it from a trained wrapper
/// with `statistics()`, save it with `save_file`, and load it with `try_load_file` into a
/// template of the same length from [`new`](Self::new).
#[derive(Module, Debug)]
pub struct RunningMeanStd {
    mean: Param<Tensor<1>>,
    variance: Param<Tensor<1>>,
    count: Param<Tensor<1>>,
}

#[derive(Debug, thiserror::Error)]
pub enum RunningMeanStdError {
    #[error("statistics must have at least one component")]
    Empty,
    #[error("mean has {mean} components but variance has {variance}")]
    LengthMismatch { mean: usize, variance: usize },
    #[error("mean must be finite")]
    NonFiniteMean,
    #[error("variance must be finite and nonnegative")]
    InvalidVariance,
    #[error("count must be finite and positive")]
    InvalidCount,
    #[error("mean, variance, and count must be on the same device")]
    DeviceMismatch,
    #[error("mean, variance, and count must have the same dtype")]
    DTypeMismatch,
    #[error("failed to read statistics: {0}")]
    TensorRead(#[source] TensorReadError),
}

/// Wraps a `[size]` statistics tensor as a saved parameter that never takes gradients.
fn fixed(tensor: Tensor<1>) -> Param<Tensor<1>> {
    Param::from_tensor(tensor).set_require_grad(false)
}

impl RunningMeanStd {
    /// Starts `size` components at zero mean and unit variance on `device`. This is also the
    /// template `try_load_file` loads saved statistics into; a file of another length is rejected.
    pub fn new(size: usize, device: &Device) -> Self {
        let dtype = if device.supports_dtype(DType::F64) {
            DType::F64
        } else {
            device.settings().float_dtype.into()
        };
        Self {
            mean: fixed(Tensor::zeros([size], (device, dtype))),
            variance: fixed(Tensor::ones([size], (device, dtype))),
            count: fixed(Tensor::full([1], RUNNING_STATS_EPSILON, (device, dtype))),
        }
    }

    /// Builds statistics from `[size]` per-component means and variances and the number of samples they summarize.
    pub fn from_tensors(
        mean: Tensor<1>,
        variance: Tensor<1>,
        count: f64,
    ) -> Result<Self, RunningMeanStdError> {
        let count = Tensor::full([1], count, (&mean.device(), mean.dtype()));
        let statistics = Self {
            mean: fixed(mean),
            variance: fixed(variance),
            count: fixed(count),
        };
        statistics.validate()?;
        Ok(statistics)
    }

    /// Checks the invariants [`from_tensors`](Self::from_tensors) enforces, including that all
    /// `[size]` tensors share a device. `try_load_file` checks names and shapes but not values, and
    /// updates from non-finite samples can break these, so call it before saving and after loading.
    pub fn validate(&self) -> Result<(), RunningMeanStdError> {
        let device = self.mean.val().device();
        if self.variance.val().device() != device || self.count.val().device() != device {
            return Err(RunningMeanStdError::DeviceMismatch);
        }
        let dtype = self.mean.val().dtype();
        if self.variance.val().dtype() != dtype || self.count.val().dtype() != dtype {
            return Err(RunningMeanStdError::DTypeMismatch);
        }
        let read = |param: &Param<Tensor<1>>| {
            param
                .val()
                .try_to_vec_as::<f64>()
                .map_err(RunningMeanStdError::TensorRead)
        };
        let (mean, variance, count) =
            (read(&self.mean)?, read(&self.variance)?, read(&self.count)?);
        if mean.is_empty() {
            return Err(RunningMeanStdError::Empty);
        }
        if mean.len() != variance.len() {
            return Err(RunningMeanStdError::LengthMismatch {
                mean: mean.len(),
                variance: variance.len(),
            });
        }
        if !mean.iter().all(|value| value.is_finite()) {
            return Err(RunningMeanStdError::NonFiniteMean);
        }
        if !variance
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
        {
            return Err(RunningMeanStdError::InvalidVariance);
        }
        if !matches!(count[..], [count] if count.is_finite() && count > 0.0) {
            return Err(RunningMeanStdError::InvalidCount);
        }
        Ok(())
    }

    pub fn mean(&self) -> Tensor<1> {
        self.mean.val()
    }

    pub fn variance(&self) -> Tensor<1> {
        self.variance.val()
    }

    /// The number of samples summarized, as a one-element tensor.
    pub fn count(&self) -> Tensor<1> {
        self.count.val()
    }

    /// Number of normalized components.
    pub fn len(&self) -> usize {
        self.mean.val().dims()[0]
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Builds a one-dimensional tensor on the statistics' device with their dtype.
    fn tensor(&self, data: TensorData) -> Tensor<1> {
        let mean = self.mean.val();
        Tensor::from_data(data, (&mean.device(), mean.dtype()))
    }

    /// Folds one `[size]` sample into the running mean and variance.
    fn update(&mut self, values: Tensor<1>) {
        let count = self.count.val().expand([self.len()]);
        let total = count.clone().add_scalar(1.0);
        let delta = values - self.mean.val();
        let mean = self.mean.val() + delta.clone() / total.clone();
        let variance =
            (self.variance.val() * count.clone() + delta.square() * count / total.clone()) / total;
        self.mean = fixed(mean);
        self.variance = fixed(variance);
        self.count = fixed(self.count.val().add_scalar(1.0));
    }

    /// Centers and scales `[size]` values, then clips them to `±clip`.
    fn normalize(&self, values: Tensor<1>, clip: Option<f64>) -> Tensor<1> {
        let normalized = (values - self.mean.val())
            / self.variance.val().add_scalar(NORMALIZATION_EPSILON).sqrt();
        clip.map_or(normalized.clone(), |limit| normalized.clamp(-limit, limit))
    }

    /// Casts half-precision `[size]` statistics to F32, where the stabilizing epsilon and
    /// squared deviations stay representable; wider dtypes are unchanged.
    fn promoted(self) -> Self {
        if !matches!(self.mean.val().dtype(), DType::F16 | DType::BF16) {
            return self;
        }
        let cast = |param: Param<Tensor<1>>| fixed(param.val().cast(DType::F32));
        Self {
            mean: cast(self.mean),
            variance: cast(self.variance),
            count: cast(self.count),
        }
    }

    /// Single-component statistics holding `moments` on `device`, in `f64` where supported.
    fn from_moments(device: &Device, moments: RewardMoments) -> Self {
        Self::new(1, device).with_scalars(moments)
    }

    /// Reads the single-component mean, variance, and count to the host.
    fn scalars(&self) -> Result<RewardMoments, RunningMeanStdError> {
        let read = |param: &Param<Tensor<1>>| {
            param
                .val()
                .try_to_vec_as::<f64>()
                .map(|values| values[0])
                .map_err(RunningMeanStdError::TensorRead)
        };
        Ok(RewardMoments {
            mean: read(&self.mean)?,
            variance: read(&self.variance)?,
            count: read(&self.count)?,
        })
    }

    /// Single-component statistics holding `moments`, on this module's device and dtype.
    fn with_scalars(&self, moments: RewardMoments) -> Self {
        let tensor = |value: f64| fixed(self.tensor(TensorData::new(vec![value], [1])));
        Self {
            mean: tensor(moments.mean),
            variance: tensor(moments.variance),
            count: tensor(moments.count),
        }
    }
}

/// Host copy of single-component reward statistics, so stepping never reads from the device.
#[derive(Debug, Clone, Copy)]
struct RewardMoments {
    mean: f64,
    variance: f64,
    count: f64,
}

impl RewardMoments {
    fn update(&mut self, value: f64) {
        let total = self.count + 1.0;
        let delta = value - self.mean;
        self.mean += delta / total;
        self.variance = (self.variance * self.count + delta * delta * self.count / total) / total;
        self.count = total;
    }

    fn scale(&self, value: f64, clip: Option<f64>) -> f64 {
        let scaled = value / (self.variance + NORMALIZATION_EPSILON).sqrt();
        clip.map_or(scaled, |limit| scaled.clamp(-limit, limit))
    }
}

/// Opaque saved state for observation normalization.
#[derive(Clone)]
pub struct NormalizeObservationGymState<S> {
    inner_state: S,
    statistics: Option<RunningMeanStd>,
    update_statistics: bool,
}

impl<S> NormalizeObservationGymState<S> {
    /// The saved running statistics, or `None` before the first observation.
    pub fn statistics(&self) -> Option<&RunningMeanStd> {
        self.statistics.as_ref()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum NormalizeObservationGymError<E> {
    #[error("wrapped gym error: {0}")]
    GymError(#[source] E),
    #[error("failed to read an observation: {0}")]
    TensorRead(#[source] TensorReadError),
    #[error("statistics have {statistics} components but the observation has {observation}")]
    StatisticsSizeMismatch {
        statistics: usize,
        observation: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum NormalizeObservationGymConfigurationError {
    #[error("clip must be finite and positive")]
    InvalidClip,
}

/// Normalizes each observation component using its running mean and variance.
pub struct NormalizeObservationGym<G> {
    gym: G,
    statistics: Option<RunningMeanStd>,
    clip: Option<f64>,
    update_statistics: bool,
    statistics_shape_checked: bool,
}

impl<G> NormalizeObservationGym<G> {
    pub fn new(gym: G) -> Self {
        Self {
            gym,
            statistics: None,
            clip: None,
            update_statistics: true,
            statistics_shape_checked: true,
        }
    }

    /// Starts from `statistics`, for example ones saved after training, instead of zero mean and unit variance.
    /// Their length must equal the number of observation components; `reset` and `step` report a mismatch.
    pub fn with_statistics(mut self, statistics: RunningMeanStd) -> Self {
        self.statistics = Some(statistics.promoted());
        self.statistics_shape_checked = false;
        self
    }

    /// Sets whether `reset` and `step` update the running statistics. Disable updates to evaluate
    /// with fixed training statistics.
    pub fn with_update_statistics(mut self, update: bool) -> Self {
        self.update_statistics = update;
        self
    }

    /// The current running statistics, or `None` before the first observation.
    pub fn statistics(&self) -> Option<&RunningMeanStd> {
        self.statistics.as_ref()
    }

    /// Replaces the running statistics; `None` restarts from the first observation.
    pub fn set_statistics(&mut self, statistics: Option<RunningMeanStd>) {
        self.statistics = statistics.map(RunningMeanStd::promoted);
        self.statistics_shape_checked = false;
    }

    pub fn update_statistics(&self) -> bool {
        self.update_statistics
    }

    pub fn set_update_statistics(&mut self, update: bool) {
        self.update_statistics = update;
    }

    pub fn inner(&self) -> &G {
        &self.gym
    }

    pub fn inner_mut(&mut self) -> &mut G {
        &mut self.gym
    }

    pub fn into_inner(self) -> G {
        self.gym
    }

    /// Clips normalized observations to `[-limit, limit]`; rejects a non-finite or nonpositive `limit`.
    pub fn with_clip(
        mut self,
        limit: f64,
    ) -> Result<Self, NormalizeObservationGymConfigurationError> {
        if !limit.is_finite() || limit <= 0.0 {
            return Err(NormalizeObservationGymConfigurationError::InvalidClip);
        }
        self.clip = Some(limit);
        Ok(self)
    }

    /// Rejects installed or restored statistics that cannot match the declared observation shape
    /// before the inner gym changes. Runs once per install, since `observation_space()` may allocate.
    fn check_statistics_shape<E>(
        &mut self,
        shape: impl FnOnce(&G) -> Vec<usize>,
    ) -> Result<(), NormalizeObservationGymError<E>> {
        if self.statistics_shape_checked {
            return Ok(());
        }
        if let Some(statistics) = &self.statistics {
            let observation = shape(&self.gym).iter().product();
            if statistics.len() != observation {
                return Err(NormalizeObservationGymError::StatisticsSizeMismatch {
                    statistics: statistics.len(),
                    observation,
                });
            }
        }
        self.statistics_shape_checked = true;
        Ok(())
    }

    /// Normalizes an unbatched rank-`O` observation, preserving its shape, kind, dtype, and device.
    /// Reads components as F32 into the statistics' device and dtype and casts normalized values back to the input dtype.
    fn normalize<const O: usize, K: Basic, E>(
        &mut self,
        observation: &Tensor<O, K>,
    ) -> Result<Tensor<O, K>, NormalizeObservationGymError<E>> {
        let shape = observation.dims();
        let device = observation.device();
        let dtype = observation.dtype();
        let values = observation
            .try_to_vec_as::<f32>()
            .map_err(NormalizeObservationGymError::TensorRead)?;
        let size = values.len();
        let statistics = self
            .statistics
            .get_or_insert_with(|| RunningMeanStd::new(size, &device).promoted());
        if statistics.len() != size {
            return Err(NormalizeObservationGymError::StatisticsSizeMismatch {
                statistics: statistics.len(),
                observation: size,
            });
        }
        let values = statistics.tensor(TensorData::new(values, [size]));
        if self.update_statistics {
            statistics.update(values.clone());
        }
        let normalized = statistics
            .normalize(values, self.clip)
            .try_to_vec_as::<f32>()
            .map_err(NormalizeObservationGymError::TensorRead)?;
        Ok(Tensor::from_data(
            TensorData::new(normalized, shape),
            (&device, dtype),
        ))
    }
}

impl<G, I, const O: usize, const BO: usize, const BA: usize, K: Basic, AK: Basic> Gym<I, BO, BA>
    for NormalizeObservationGym<G>
where
    G: Gym<I, BO, BA>,
    G::ObservationSpace: ObservationSpace<BO, Kind = K, Item = Tensor<O, K>>,
    G::ActionSpace: ActionSpace<BA, Kind = AK>,
    Tensor<BO, K>: PrevRank<Prev = Tensor<O, K>>,
    Tensor<O, K>: NextRank<Next = Tensor<BO, K>>,
{
    type Error = NormalizeObservationGymError<G::Error>;
    type ActionSpace = G::ActionSpace;
    type ObservationSpace = G::ObservationSpace;

    /// Normalizes the reset observation without changing its unbatched rank-`O` shape, kind, dtype, or device.
    fn reset(&mut self) -> Result<ResetInfo<I, Tensor<O, K>>, Self::Error> {
        self.check_statistics_shape(|gym| gym.observation_space().shape())?;
        let mut reset = self
            .gym
            .reset()
            .map_err(NormalizeObservationGymError::GymError)?;
        reset.observation = self.normalize(&reset.observation)?;
        Ok(reset)
    }

    /// Forwards an unbatched rank-`BA - 1` action and returns an unbatched rank-`O` observation; scalars use `[1]`.
    /// Normalizes observation values while preserving tensor shapes, kinds, dtypes, and devices.
    fn step(
        &mut self,
        action: <Self::ActionSpace as TensorBatchTypes>::Item,
    ) -> Result<StepInfo<I, Tensor<O, K>>, Self::Error> {
        self.check_statistics_shape(|gym| gym.observation_space().shape())?;
        let mut step = self
            .gym
            .step(action)
            .map_err(NormalizeObservationGymError::GymError)?;
        step.observation = self.normalize(&step.observation)?;
        Ok(step)
    }

    fn action_space(&self) -> Self::ActionSpace {
        self.gym.action_space()
    }

    fn observation_space(&self) -> Self::ObservationSpace {
        self.gym.observation_space()
    }
}

impl<G, I, const O: usize, const BO: usize, const BA: usize, K: Basic, AK: Basic>
    PerfectStateGym<I, BO, BA> for NormalizeObservationGym<G>
where
    G: PerfectStateGym<I, BO, BA>,
    G::ObservationSpace: ObservationSpace<BO, Kind = K, Item = Tensor<O, K>>,
    G::ActionSpace: ActionSpace<BA, Kind = AK>,
    Tensor<BO, K>: PrevRank<Prev = Tensor<O, K>>,
    Tensor<O, K>: NextRank<Next = Tensor<BO, K>>,
{
    type State = NormalizeObservationGymState<G::State>;

    fn save_state(&self) -> Result<Self::State, Self::Error> {
        Ok(NormalizeObservationGymState {
            inner_state: self
                .gym
                .save_state()
                .map_err(NormalizeObservationGymError::GymError)?,
            statistics: self.statistics.clone(),
            update_statistics: self.update_statistics,
        })
    }

    fn set_state(&mut self, state: &Self::State) -> Result<(), Self::Error> {
        self.gym
            .set_state(&state.inner_state)
            .map_err(NormalizeObservationGymError::GymError)?;
        self.statistics.clone_from(&state.statistics);
        self.update_statistics = state.update_statistics;
        self.statistics_shape_checked = false;
        Ok(())
    }
}

impl<G, I, const O: usize, const BO: usize, K: Basic> LegalActionsGym<I, BO>
    for NormalizeObservationGym<G>
where
    G: LegalActionsGym<I, BO>,
    G::ObservationSpace: ObservationSpace<BO, Kind = K, Item = Tensor<O, K>>,
    G::ActionSpace: ActionSpace<2, Kind = Int>,
    Tensor<BO, K>: PrevRank<Prev = Tensor<O, K>>,
    Tensor<O, K>: NextRank<Next = Tensor<BO, K>>,
{
    fn legal_actions(&self) -> Result<Vec<usize>, Self::Error> {
        self.gym
            .legal_actions()
            .map_err(NormalizeObservationGymError::GymError)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum NormalizeRewardGymConfigurationError {
    #[error("gamma must be finite and in 0..=1")]
    InvalidGamma,
    #[error("clip must be finite and positive")]
    InvalidClip,
    #[error("reward statistics must have exactly one component, got {0}")]
    InvalidStatisticsSize(usize),
    #[error("reward statistics are invalid; `RunningMeanStd::validate` reports why")]
    InvalidStatistics,
}

/// Scales immediate rewards by the running standard deviation of discounted rewards.
pub struct NormalizeRewardGym<G> {
    gym: G,
    device: Device,
    moments: RewardMoments,
    discounted_reward: f64,
    gamma: f64,
    clip: Option<f64>,
    update_statistics: bool,
}

impl<G> NormalizeRewardGym<G> {
    /// Rejects a non-finite `gamma` or one outside `0..=1`.
    pub fn new(gym: G, gamma: f64) -> Result<Self, NormalizeRewardGymConfigurationError> {
        if !gamma.is_finite() || !(0.0..=1.0).contains(&gamma) {
            return Err(NormalizeRewardGymConfigurationError::InvalidGamma);
        }
        Ok(Self {
            gym,
            device: Device::default(),
            moments: RewardMoments {
                mean: 0.0,
                variance: 1.0,
                count: RUNNING_STATS_EPSILON,
            },
            discounted_reward: 0.0,
            gamma,
            clip: None,
            update_statistics: true,
        })
    }

    /// Starts from saved discounted-return statistics; rejects statistics without exactly one component.
    pub fn with_statistics(
        mut self,
        statistics: RunningMeanStd,
    ) -> Result<Self, NormalizeRewardGymConfigurationError> {
        self.set_statistics(statistics)?;
        Ok(self)
    }

    /// Sets whether `step` updates the running statistics. Disable updates to evaluate
    /// with fixed training statistics.
    pub fn with_update_statistics(mut self, update: bool) -> Self {
        self.update_statistics = update;
        self
    }

    /// The current running statistics of the discounted return, with one component, built from
    /// the wrapper's host `f64` copy on the device of the installed statistics (Burn's default
    /// device otherwise), in `f64` where that device supports it.
    pub fn statistics(&self) -> RunningMeanStd {
        RunningMeanStd::from_moments(&self.device, self.moments)
    }

    /// Replaces the running statistics; rejects statistics without exactly one component.
    pub fn set_statistics(
        &mut self,
        statistics: RunningMeanStd,
    ) -> Result<(), NormalizeRewardGymConfigurationError> {
        if statistics.len() != 1 {
            return Err(NormalizeRewardGymConfigurationError::InvalidStatisticsSize(
                statistics.len(),
            ));
        }
        statistics
            .validate()
            .map_err(|_| NormalizeRewardGymConfigurationError::InvalidStatistics)?;
        self.moments = statistics
            .scalars()
            .map_err(|_| NormalizeRewardGymConfigurationError::InvalidStatistics)?;
        self.device = statistics.mean.val().device();
        Ok(())
    }

    pub fn update_statistics(&self) -> bool {
        self.update_statistics
    }

    pub fn set_update_statistics(&mut self, update: bool) {
        self.update_statistics = update;
    }

    pub fn inner(&self) -> &G {
        &self.gym
    }

    pub fn inner_mut(&mut self) -> &mut G {
        &mut self.gym
    }

    pub fn into_inner(self) -> G {
        self.gym
    }

    /// Clips normalized rewards to `[-limit, limit]`; rejects a non-finite or nonpositive `limit`.
    pub fn with_clip(mut self, limit: f64) -> Result<Self, NormalizeRewardGymConfigurationError> {
        if !limit.is_finite() || limit <= 0.0 {
            return Err(NormalizeRewardGymConfigurationError::InvalidClip);
        }
        self.clip = Some(limit);
        Ok(self)
    }
}

/// Opaque saved state for reward normalization.
#[derive(Clone)]
pub struct NormalizeRewardGymState<S> {
    inner_state: S,
    device: Device,
    moments: RewardMoments,
    discounted_reward: f64,
    update_statistics: bool,
}

impl<S> NormalizeRewardGymState<S> {
    /// The saved running statistics of the discounted return, built like
    /// [`NormalizeRewardGym::statistics`].
    pub fn statistics(&self) -> RunningMeanStd {
        RunningMeanStd::from_moments(&self.device, self.moments)
    }
}

impl<G, I, const BO: usize, const BA: usize, K: Basic, AK: Basic> Gym<I, BO, BA>
    for NormalizeRewardGym<G>
where
    G: Gym<I, BO, BA>,
    G::ObservationSpace: ObservationSpace<BO, Kind = K>,
    G::ActionSpace: ActionSpace<BA, Kind = AK>,
{
    type Error = G::Error;
    type ActionSpace = G::ActionSpace;
    type ObservationSpace = G::ObservationSpace;

    /// Returns the inner gym's unbatched rank-`BO - 1` observation, preserving its kind, dtype, device, and item axes.
    fn reset(
        &mut self,
    ) -> Result<ResetInfo<I, <Self::ObservationSpace as TensorBatchTypes>::Item>, Self::Error> {
        self.gym.reset()
    }

    /// Forwards an unbatched rank-`BA - 1` action and returns an unbatched rank-`BO - 1` observation; scalars use `[1]`.
    /// Preserves tensor shapes, kinds, dtypes, and devices, subject to the inner gym's input requirements.
    fn step(
        &mut self,
        action: <Self::ActionSpace as TensorBatchTypes>::Item,
    ) -> Result<StepInfo<I, <Self::ObservationSpace as TensorBatchTypes>::Item>, Self::Error> {
        let mut step = self.gym.step(action)?;
        self.discounted_reward = self.discounted_reward * self.gamma + f64::from(step.reward);
        if self.update_statistics {
            self.moments.update(self.discounted_reward);
        }
        if step.terminated || step.truncated {
            self.discounted_reward = 0.0;
        }

        step.reward = self.moments.scale(f64::from(step.reward), self.clip) as f32;
        Ok(step)
    }

    fn action_space(&self) -> Self::ActionSpace {
        self.gym.action_space()
    }

    fn observation_space(&self) -> Self::ObservationSpace {
        self.gym.observation_space()
    }
}

impl<G, I, const BO: usize, const BA: usize, K: Basic, AK: Basic> PerfectStateGym<I, BO, BA>
    for NormalizeRewardGym<G>
where
    G: PerfectStateGym<I, BO, BA>,
    G::ObservationSpace: ObservationSpace<BO, Kind = K>,
    G::ActionSpace: ActionSpace<BA, Kind = AK>,
{
    type State = NormalizeRewardGymState<G::State>;

    fn save_state(&self) -> Result<Self::State, Self::Error> {
        Ok(NormalizeRewardGymState {
            inner_state: self.gym.save_state()?,
            device: self.device.clone(),
            moments: self.moments,
            discounted_reward: self.discounted_reward,
            update_statistics: self.update_statistics,
        })
    }

    fn set_state(&mut self, state: &Self::State) -> Result<(), Self::Error> {
        self.gym.set_state(&state.inner_state)?;
        self.device.clone_from(&state.device);
        self.moments = state.moments;
        self.discounted_reward = state.discounted_reward;
        self.update_statistics = state.update_statistics;
        Ok(())
    }
}

impl<G, I, const BO: usize, K: Basic> LegalActionsGym<I, BO> for NormalizeRewardGym<G>
where
    G: LegalActionsGym<I, BO>,
    G::ObservationSpace: ObservationSpace<BO, Kind = K>,
    G::ActionSpace: ActionSpace<2, Kind = Int>,
{
    fn legal_actions(&self) -> Result<Vec<usize>, Self::Error> {
        self.gym.legal_actions()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        gym::Gym,
        wrappers::{
            state_test_support::{StateTestGym, action as state_action, assert_state_round_trip},
            test_support::{TestGym, action, scalar},
        },
    };
    use burn::module::Module;
    use burn::tensor::{DType, Device, Tensor, TensorData};

    use super::{
        NormalizeObservationGym, NormalizeObservationGymConfigurationError,
        NormalizeObservationGymError, NormalizeRewardGym, NormalizeRewardGymConfigurationError,
        RunningMeanStd, RunningMeanStdError,
    };
    use crate::gym::PerfectStateGym;

    fn stats(
        mean: &[f64],
        variance: &[f64],
        count: f64,
    ) -> Result<RunningMeanStd, RunningMeanStdError> {
        let device = Device::flex();
        let tensor = |values: &[f64]| {
            Tensor::<1>::from_data(
                TensorData::new(values.to_vec(), [values.len()]),
                (&device, DType::F64),
            )
        };
        RunningMeanStd::from_tensors(tensor(mean), tensor(variance), count)
    }

    /// Reads a `[size]` statistics tensor back to the host.
    fn host(tensor: Tensor<1>) -> Vec<f64> {
        tensor.try_to_vec_as::<f64>().unwrap()
    }

    fn assert_same(left: &RunningMeanStd, right: &RunningMeanStd) {
        assert_eq!(host(left.mean()), host(right.mean()));
        assert_eq!(host(left.variance()), host(right.variance()));
        assert_eq!(host(left.count()), host(right.count()));
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("modurl-{name}-{}.bpk", std::process::id()))
    }

    #[test]
    fn rejects_invalid_normalization_settings() {
        for limit in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                NormalizeObservationGym::new(()).with_clip(limit),
                Err(NormalizeObservationGymConfigurationError::InvalidClip)
            ));
            assert!(matches!(
                NormalizeRewardGym::new((), 0.99).unwrap().with_clip(limit),
                Err(NormalizeRewardGymConfigurationError::InvalidClip)
            ));
        }
        for gamma in [-0.1, 1.1, f64::NAN] {
            assert!(matches!(
                NormalizeRewardGym::new((), gamma),
                Err(NormalizeRewardGymConfigurationError::InvalidGamma)
            ));
        }
    }

    #[test]
    fn observation_normalization_preserves_rank_dtype_and_device() {
        let device = Device::flex();
        let observations =
            Tensor::<2>::from_data([[1.0f64, 2.0], [3.0, 4.0]], (&device, DType::F64));
        let mut wrapper = NormalizeObservationGym::new(());
        let normalized = wrapper
            .normalize::<2, _, std::convert::Infallible>(&observations)
            .unwrap();
        assert_eq!(normalized.dims(), [2, 2]);
        assert_eq!(normalized.dtype(), DType::F64);
        assert_eq!(normalized.device(), device);
        assert!(
            normalized
                .into_data()
                .try_to_vec::<f64>()
                .unwrap()
                .iter()
                .all(|value| value.abs() < 0.02)
        );
    }

    #[test]
    fn normalizes_observations_and_preserves_metadata() {
        let gym = TestGym::new([TestGym::step(1.0, 0.0, false, false, 7)]);
        let mut wrapper = NormalizeObservationGym::new(gym).with_clip(10.0).unwrap();

        let reset = wrapper.reset().unwrap();
        let step = wrapper.step(action()).unwrap();

        assert!(scalar(&reset.observation).abs() < 0.02);
        assert!((scalar(&step.observation) + 1.0).abs() < 0.01);
        assert_eq!(step.info.sequence, 7);
    }

    #[test]
    fn normalizes_and_clips_rewards() {
        let gym = TestGym::new([
            TestGym::step(1.0, 1.0, false, false, 1),
            TestGym::step(2.0, 1.0, false, false, 2),
        ]);
        let mut wrapper = NormalizeRewardGym::new(gym, 0.99)
            .unwrap()
            .with_clip(10.0)
            .unwrap();

        let first = wrapper.step(action()).unwrap();
        assert_eq!(first.reward, 10.0);

        let second = wrapper.step(action()).unwrap();
        assert!(second.reward > 1.9 && second.reward < 2.1);
        assert_eq!(second.info.sequence, 2);
    }

    #[test]
    fn resets_discounted_reward_after_truncation() {
        let gym = TestGym::new([
            TestGym::step(1.0, 1.0, false, false, 1),
            TestGym::step(2.0, 2.0, false, true, 2),
            TestGym::step(3.0, 3.0, false, false, 3),
        ]);
        let mut wrapper = NormalizeRewardGym::new(gym, 0.99).unwrap();

        wrapper.step(action()).unwrap();
        wrapper.step(action()).unwrap();
        assert_eq!(wrapper.discounted_reward, 0.0);

        wrapper.step(action()).unwrap();
        assert_eq!(wrapper.discounted_reward, 3.0);
    }

    #[test]
    fn observation_normalization_restores_running_statistics_and_inner_state() {
        assert_state_round_trip(
            NormalizeObservationGym::new(StateTestGym::new(37)),
            state_action(),
            2,
            4,
            |observation| observation.into_data(),
            |_: &()| (),
        );
    }

    #[test]
    fn reward_normalization_restores_running_statistics_and_inner_state() {
        assert_state_round_trip(
            NormalizeRewardGym::new(StateTestGym::new(41), 0.9).unwrap(),
            state_action(),
            1,
            2,
            |observation| observation.into_data(),
            |_: &()| (),
        );
    }

    #[test]
    fn rejects_invalid_running_statistics() {
        assert!(matches!(
            stats(&[], &[], 1.0),
            Err(RunningMeanStdError::Empty)
        ));
        assert!(matches!(
            stats(&[0.0], &[1.0, 1.0], 1.0),
            Err(RunningMeanStdError::LengthMismatch {
                mean: 1,
                variance: 2
            })
        ));
        assert!(matches!(
            stats(&[f64::NAN], &[1.0], 1.0),
            Err(RunningMeanStdError::NonFiniteMean)
        ));
        for variance in [-1.0, f64::INFINITY] {
            assert!(matches!(
                stats(&[0.0], &[variance], 1.0),
                Err(RunningMeanStdError::InvalidVariance)
            ));
        }
        for count in [0.0, -1.0, f64::NAN] {
            assert!(matches!(
                stats(&[0.0], &[1.0], count),
                Err(RunningMeanStdError::InvalidCount)
            ));
        }
    }

    #[test]
    fn running_statistics_round_trip_through_burn_files() {
        let statistics = stats(&[1.5, -2.0], &[0.25, 4.0], 12.0).unwrap();
        let path = temp_path("running-mean-std");
        statistics.clone().save_file(&path).unwrap();

        let loaded = RunningMeanStd::new(2, &Device::flex()).try_load_file(&path);
        let wrong_length = RunningMeanStd::new(3, &Device::flex()).try_load_file(&path);
        let loaded = loaded.unwrap();
        assert_same(&loaded, &statistics);
        assert_eq!(loaded.mean().dtype(), DType::F64);
        assert!(wrong_length.is_err());
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn new_statistics_accumulate_in_f64_where_supported() {
        let statistics = RunningMeanStd::new(1, &Device::flex());
        assert_eq!(statistics.mean().dtype(), DType::F64);

        let mut statistics = stats(&[0.0], &[1.0], 16_777_216.0).unwrap();
        let sample = statistics.tensor(TensorData::new(vec![1.0f64], [1]));
        statistics.update(sample);
        assert_eq!(host(statistics.count()), vec![16_777_217.0]);
    }

    #[test]
    fn statistics_and_model_files_are_not_interchangeable() {
        use burn::nn::LinearConfig;

        let device = Device::flex();
        let model_path = temp_path("model");
        let statistics_path = temp_path("statistics");
        LinearConfig::new(2, 2)
            .init(&device)
            .save_file(&model_path)
            .unwrap();
        RunningMeanStd::new(2, &device)
            .save_file(&statistics_path)
            .unwrap();

        let statistics_from_model = RunningMeanStd::new(2, &device).try_load_file(&model_path);
        let model_from_statistics = LinearConfig::new(2, 2)
            .init(&device)
            .try_load_file(&statistics_path);
        std::fs::remove_file(&model_path).unwrap();
        std::fs::remove_file(&statistics_path).unwrap();
        assert!(statistics_from_model.is_err());
        assert!(model_from_statistics.is_err());
    }

    #[test]
    fn loaded_observation_statistics_normalize_without_updating() {
        let statistics = stats(&[2.0], &[4.0], 100.0).unwrap();
        let gym = TestGym::new([TestGym::step(4.0, 0.0, false, false, 1)]);
        let mut wrapper = NormalizeObservationGym::new(gym)
            .with_statistics(statistics.clone())
            .with_update_statistics(false);

        let reset = wrapper.reset().unwrap();
        let step = wrapper.step(action()).unwrap();

        assert!((scalar(&reset.observation) - 49.0).abs() < 1e-6);
        assert!((scalar(&step.observation) - 1.0).abs() < 1e-6);
        assert_same(wrapper.statistics().unwrap(), &statistics);
    }

    #[test]
    fn exported_observation_statistics_normalize_evaluation_observations() {
        let mut training = NormalizeObservationGym::new(TestGym::new([
            TestGym::step(1.0, 0.0, false, false, 1),
            TestGym::step(3.0, 0.0, false, false, 2),
        ]));
        training.reset().unwrap();
        training.step(action()).unwrap();
        training.step(action()).unwrap();
        let path = temp_path("exported-statistics");
        training
            .statistics()
            .unwrap()
            .clone()
            .save_file(&path)
            .unwrap();

        let loaded = RunningMeanStd::new(1, &Device::flex()).try_load_file(&path);
        let loaded = loaded.unwrap();
        let expected = (5.0 - host(loaded.mean())[0]) / (host(loaded.variance())[0] + 1e-8).sqrt();
        let mut evaluation =
            NormalizeObservationGym::new(TestGym::new([TestGym::step(5.0, 0.0, false, false, 1)]))
                .with_statistics(loaded)
                .with_update_statistics(false);
        let step = evaluation.step(action()).unwrap();

        assert!((f64::from(scalar(&step.observation)) - expected).abs() < 1e-5);
        assert_same(
            evaluation.statistics().unwrap(),
            training.statistics().unwrap(),
        );
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn reports_observation_statistics_size_mismatch_before_inner_gym_runs() {
        let statistics = stats(&[0.0, 0.0], &[1.0, 1.0], 1.0).unwrap();
        // An empty script panics if the inner gym is stepped.
        let mut wrapper =
            NormalizeObservationGym::new(TestGym::new([])).with_statistics(statistics);
        for result in [
            wrapper.reset().map(|_| ()),
            wrapper.step(action()).map(|_| ()),
        ] {
            assert!(matches!(
                result,
                Err(NormalizeObservationGymError::StatisticsSizeMismatch {
                    statistics: 2,
                    observation: 1
                })
            ));
        }
    }

    #[test]
    fn validate_reports_statistics_broken_by_non_finite_rewards() {
        let gym = TestGym::new([TestGym::step(0.0, f32::NAN, false, false, 1)]);
        let mut wrapper = NormalizeRewardGym::new(gym, 0.99).unwrap();
        wrapper.step(action()).unwrap();
        assert!(matches!(
            wrapper.statistics().validate(),
            Err(RunningMeanStdError::NonFiniteMean)
        ));
    }

    #[test]
    fn saved_states_restore_update_statistics() {
        let mut observation = NormalizeObservationGym::new(StateTestGym::new(3));
        let state = observation.save_state().unwrap();
        observation.set_update_statistics(false);
        observation.set_state(&state).unwrap();
        assert!(observation.update_statistics());

        let mut reward = NormalizeRewardGym::new(StateTestGym::new(5), 0.9)
            .unwrap()
            .with_update_statistics(false);
        let state = reward.save_state().unwrap();
        reward.set_update_statistics(true);
        reward.set_state(&state).unwrap();
        assert!(!reward.update_statistics());
    }

    #[test]
    fn loaded_reward_statistics_scale_without_updating() {
        let path = temp_path("reward_rms.bpk");
        stats(&[0.0], &[4.0], 100.0)
            .unwrap()
            .save_file(&path)
            .unwrap();
        let statistics = RunningMeanStd::new(1, &Device::flex())
            .try_load_file(&path)
            .unwrap();
        statistics.validate().unwrap();
        let gym = TestGym::new([
            TestGym::step(0.0, 2.0, false, false, 1),
            TestGym::step(0.0, 4.0, false, false, 2),
        ]);
        let mut wrapper = NormalizeRewardGym::new(gym, 0.99)
            .unwrap()
            .with_statistics(statistics.clone())
            .unwrap()
            .with_update_statistics(false);

        assert!((wrapper.step(action()).unwrap().reward - 1.0).abs() < 1e-6);
        assert!((wrapper.step(action()).unwrap().reward - 2.0).abs() < 1e-6);
        assert_same(&wrapper.statistics(), &statistics);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn frozen_f16_statistics_with_zero_variance_stay_finite() {
        let device = Device::flex();
        let f16 = |value: f64| {
            Tensor::from_data(TensorData::new(vec![value], [1]), (&device, DType::F16))
        };
        let statistics = RunningMeanStd::from_tensors(f16(2.0), f16(0.0), 100.0).unwrap();
        let gym = TestGym::new([TestGym::step(2.0, 0.0, false, false, 1)]);
        let mut wrapper = NormalizeObservationGym::new(gym)
            .with_statistics(statistics)
            .with_update_statistics(false);

        let observation = wrapper.step(action()).unwrap().observation;
        assert_eq!(
            observation.into_data().try_to_vec::<f32>().unwrap(),
            vec![0.0]
        );
    }

    #[test]
    fn rejects_multi_component_reward_statistics() {
        let statistics = stats(&[0.0, 0.0], &[1.0, 1.0], 1.0).unwrap();
        assert!(matches!(
            NormalizeRewardGym::new((), 0.99)
                .unwrap()
                .with_statistics(statistics),
            Err(NormalizeRewardGymConfigurationError::InvalidStatisticsSize(
                2
            ))
        ));
    }

    #[test]
    fn saved_states_expose_statistics() {
        let mut observation = NormalizeObservationGym::new(StateTestGym::new(3));
        assert!(observation.save_state().unwrap().statistics().is_none());
        observation.reset().unwrap();
        assert_same(
            observation.save_state().unwrap().statistics().unwrap(),
            observation.statistics().unwrap(),
        );

        let mut reward = NormalizeRewardGym::new(StateTestGym::new(5), 0.9).unwrap();
        reward.step(state_action()).unwrap();
        assert_same(
            &reward.save_state().unwrap().statistics(),
            &reward.statistics(),
        );
    }

    #[test]
    fn statistics_reject_mixed_dtypes() {
        let device = Device::flex();
        let tensor = |value: f64, dtype: DType| {
            Tensor::from_data(TensorData::new(vec![value], [1]), (&device, dtype))
        };
        assert!(matches!(
            RunningMeanStd::from_tensors(tensor(0.0, DType::F32), tensor(1.0, DType::F16), 1.0),
            Err(RunningMeanStdError::DTypeMismatch)
        ));
    }

    #[test]
    fn half_precision_observation_statistics_train_in_f32() {
        let device = Device::flex();
        let f16 = |value: f64| {
            Tensor::from_data(TensorData::new(vec![value], [1]), (&device, DType::F16))
        };
        let statistics = RunningMeanStd::from_tensors(f16(0.0), f16(1.0), 1.0).unwrap();
        let gym = TestGym::new([TestGym::step(300.0, 0.0, false, false, 1)]);
        let mut wrapper = NormalizeObservationGym::new(gym).with_statistics(statistics);

        wrapper.step(action()).unwrap();
        let statistics = wrapper.statistics().unwrap();
        assert_eq!(statistics.variance().dtype(), DType::F32);
        statistics.validate().unwrap();
    }

    #[test]
    fn reward_statistics_export_live_f64_moments() {
        let device = Device::flex();
        let f16 = |value: f64| {
            Tensor::from_data(TensorData::new(vec![value], [1]), (&device, DType::F16))
        };
        let statistics = RunningMeanStd::from_tensors(f16(0.0), f16(1.0), 1.0).unwrap();
        let gym = TestGym::new([TestGym::step(0.0, 300.0, false, false, 1)]);
        let mut wrapper = NormalizeRewardGym::new(gym, 0.99)
            .unwrap()
            .with_statistics(statistics)
            .unwrap();

        wrapper.step(action()).unwrap();
        let statistics = wrapper.statistics();
        assert_eq!(statistics.variance().dtype(), DType::F64);
        statistics.validate().unwrap();
        assert_eq!(host(statistics.count()), vec![2.0]);
    }
}

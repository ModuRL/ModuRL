//! Running observation and reward normalization wrappers.

use crate::tensor_rank::{NextRank, PrevRank, TensorBatchTypes};
use burn::tensor::{Int, Tensor, TensorData, TensorReadError, kind::Basic};

use crate::{
    gym::{Gym, LegalActionsGym, PerfectStateGym, ResetInfo, StepInfo},
    spaces::{ActionSpace, ObservationSpace},
};

const RUNNING_STATS_EPSILON: f64 = 1e-4;
const NORMALIZATION_EPSILON: f64 = 1e-8;

const STATISTICS_FORMAT_KEY: &str = "format";
const STATISTICS_FORMAT: &str = "modurl.running_mean_std";
const STATISTICS_VERSION_KEY: &str = "version";
const STATISTICS_VERSION: u64 = 1;
const STATISTICS_COUNT_KEY: &str = "count";
const STATISTICS_MEAN_KEY: &str = "mean";
const STATISTICS_VARIANCE_KEY: &str = "variance";

/// Running per-component mean, variance, and sample count used by the normalization wrappers.
///
/// Export it from a trained wrapper with `statistics()`, persist it with [`save_file`](Self::save_file)
/// or [`to_bytes`](Self::to_bytes), and load it into an evaluation wrapper with `with_statistics`.
#[derive(Debug, Clone, PartialEq)]
pub struct RunningMeanStd {
    mean: Vec<f64>,
    variance: Vec<f64>,
    count: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
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
}

#[derive(Debug, thiserror::Error)]
pub enum RunningMeanStdFileError {
    #[error("failed to read or write statistics: {0}")]
    Record(#[source] burn::store::RecordError),
    #[error("not a ModuRL normalization statistics file")]
    NotStatisticsFile,
    #[error("unsupported normalization statistics format version {0}")]
    UnsupportedVersion(u64),
    #[error("statistics entry `{0}` is missing or malformed")]
    InvalidEntry(&'static str),
    #[error("invalid statistics: {0}")]
    InvalidStatistics(#[source] RunningMeanStdError),
}

impl From<burn_pack::Error> for RunningMeanStdFileError {
    fn from(error: burn_pack::Error) -> Self {
        Self::Record(error.into())
    }
}

impl RunningMeanStd {
    /// Builds statistics from per-component means and variances and the number of samples they summarize.
    pub fn new(
        mean: Vec<f64>,
        variance: Vec<f64>,
        count: f64,
    ) -> Result<Self, RunningMeanStdError> {
        let statistics = Self {
            mean,
            variance,
            count,
        };
        statistics.validate()?;
        Ok(statistics)
    }

    /// Checks the invariants [`new`](Self::new) enforces; updates from non-finite samples can break them.
    pub fn validate(&self) -> Result<(), RunningMeanStdError> {
        let (mean, variance, count) = (&self.mean, &self.variance, self.count);
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
        if !count.is_finite() || count <= 0.0 {
            return Err(RunningMeanStdError::InvalidCount);
        }
        Ok(())
    }

    fn initial(size: usize) -> Self {
        Self {
            mean: vec![0.0; size],
            variance: vec![1.0; size],
            count: RUNNING_STATS_EPSILON,
        }
    }

    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    pub fn variance(&self) -> &[f64] {
        &self.variance
    }

    pub fn count(&self) -> f64 {
        self.count
    }

    /// Number of normalized components.
    pub fn len(&self) -> usize {
        self.mean.len()
    }

    /// Always `false`: statistics have at least one component.
    pub fn is_empty(&self) -> bool {
        self.mean.is_empty()
    }

    /// Encodes the statistics as an in-memory burnpack, the format Burn's `save_file` writes.
    /// Fails like [`validate`](Self::validate), so every encoded pack can be loaded again.
    pub fn to_bytes(&self) -> Result<Vec<u8>, RunningMeanStdFileError> {
        Ok(self.burnpack_writer()?.into_bytes()?.to_vec())
    }

    /// Decodes statistics written by [`to_bytes`](Self::to_bytes) and validates them like [`new`](Self::new).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, RunningMeanStdFileError> {
        Self::from_burnpack(burn_pack::Reader::from_bytes(
            burn_pack::Bytes::from_bytes_vec(bytes.to_vec()),
        )?)
    }

    /// Saves the statistics to a burnpack file with Burn's writer, the same file handling as
    /// Burn's `Module::save_file`: `.bpk` is appended when `path` has no extension, and the
    /// file is replaced atomically so a failed save leaves an existing file untouched.
    pub fn save_file<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), RunningMeanStdFileError> {
        Ok(self.burnpack_writer()?.write_to_file(path)?)
    }

    /// Loads statistics written by [`save_file`](Self::save_file).
    pub fn load_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, RunningMeanStdFileError> {
        Self::from_burnpack(burn_pack::Reader::from_file(path)?)
    }

    fn burnpack_writer(&self) -> Result<burn_pack::Writer, RunningMeanStdFileError> {
        self.validate()
            .map_err(RunningMeanStdFileError::InvalidStatistics)?;
        let tensor = |name: &str, values: &[f64]| {
            burn_pack::Tensor::new(
                name.to_string(),
                burn_pack::DType::F64,
                [values.len()],
                None,
                burn_pack::Bytes::from_bytes_vec(
                    values
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect(),
                ),
            )
        };
        Ok(burn_pack::Writer::new(vec![
            tensor(STATISTICS_MEAN_KEY, &self.mean),
            tensor(STATISTICS_VARIANCE_KEY, &self.variance),
        ])
        .with_metadata(STATISTICS_FORMAT_KEY, STATISTICS_FORMAT)
        .with_scalar(STATISTICS_VERSION_KEY, STATISTICS_VERSION.into())
        .with_scalar(STATISTICS_COUNT_KEY, self.count.into()))
    }

    fn from_burnpack(reader: burn_pack::Reader) -> Result<Self, RunningMeanStdFileError> {
        if reader
            .metadata()
            .get(STATISTICS_FORMAT_KEY)
            .map(String::as_str)
            != Some(STATISTICS_FORMAT)
        {
            return Err(RunningMeanStdFileError::NotStatisticsFile);
        }
        let scalar = |key: &'static str| {
            reader
                .scalars()
                .get(key)
                .cloned()
                .ok_or(RunningMeanStdFileError::InvalidEntry(key))
        };
        let version = u64::try_from(scalar(STATISTICS_VERSION_KEY)?)
            .map_err(|_| RunningMeanStdFileError::InvalidEntry(STATISTICS_VERSION_KEY))?;
        if version != STATISTICS_VERSION {
            return Err(RunningMeanStdFileError::UnsupportedVersion(version));
        }
        let count = f64::try_from(scalar(STATISTICS_COUNT_KEY)?)
            .map_err(|_| RunningMeanStdFileError::InvalidEntry(STATISTICS_COUNT_KEY))?;

        let tensors = reader.into_tensors()?;
        let values = |key: &'static str| -> Result<Vec<f64>, RunningMeanStdFileError> {
            let tensor = tensors
                .iter()
                .find(|tensor| tensor.name == key)
                .filter(|tensor| {
                    tensor.dtype == burn_pack::DType::F64 && tensor.shape.num_dims() == 1
                })
                .ok_or(RunningMeanStdFileError::InvalidEntry(key))?;
            let bytes = tensor.to_bytes()?;
            if bytes.len() != 8 * tensor.shape[0] {
                return Err(RunningMeanStdFileError::InvalidEntry(key));
            }
            Ok(bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|value| f64::from_le_bytes(*value))
                .collect())
        };
        Self::new(
            values(STATISTICS_MEAN_KEY)?,
            values(STATISTICS_VARIANCE_KEY)?,
            count,
        )
        .map_err(RunningMeanStdFileError::InvalidStatistics)
    }

    fn update(&mut self, values: &[f64]) {
        assert_eq!(values.len(), self.mean.len());
        let total_count = self.count + 1.0;
        for ((mean, variance), value) in self.mean.iter_mut().zip(&mut self.variance).zip(values) {
            let delta = value - *mean;
            let new_mean = *mean + delta / total_count;
            let m2 = *variance * self.count + delta * delta * self.count / total_count;
            *mean = new_mean;
            *variance = m2 / total_count;
        }
        self.count = total_count;
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
        self.statistics = Some(statistics);
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
        self.statistics = statistics;
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
    /// Reads components as F32 for host statistics and casts normalized F32 values back to the input dtype.
    fn normalize<const O: usize, K: Basic, E>(
        &mut self,
        observation: &Tensor<O, K>,
    ) -> Result<Tensor<O, K>, NormalizeObservationGymError<E>> {
        let shape = observation.dims();
        let device = observation.device();
        let dtype = observation.dtype();
        let values = observation
            .try_to_vec_as::<f32>()
            .map_err(NormalizeObservationGymError::TensorRead)?
            .into_iter()
            .map(f64::from)
            .collect::<Vec<_>>();
        let statistics = self
            .statistics
            .get_or_insert_with(|| RunningMeanStd::initial(values.len()));
        if statistics.len() != values.len() {
            return Err(NormalizeObservationGymError::StatisticsSizeMismatch {
                statistics: statistics.len(),
                observation: values.len(),
            });
        }
        if self.update_statistics {
            statistics.update(&values);
        }

        let normalized = values
            .iter()
            .zip(&statistics.mean)
            .zip(&statistics.variance)
            .map(|((value, mean), variance)| {
                let value = (value - mean) / (variance + NORMALIZATION_EPSILON).sqrt();
                self.clip.map_or(value, |limit| value.clamp(-limit, limit)) as f32
            })
            .collect::<Vec<_>>();
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
}

/// Scales immediate rewards by the running standard deviation of discounted rewards.
pub struct NormalizeRewardGym<G> {
    gym: G,
    statistics: RunningMeanStd,
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
            statistics: RunningMeanStd::initial(1),
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

    /// The current running statistics of the discounted return, with one component.
    pub fn statistics(&self) -> &RunningMeanStd {
        &self.statistics
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
        self.statistics = statistics;
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
    statistics: RunningMeanStd,
    discounted_reward: f64,
    update_statistics: bool,
}

impl<S> NormalizeRewardGymState<S> {
    /// The saved running statistics of the discounted return.
    pub fn statistics(&self) -> &RunningMeanStd {
        &self.statistics
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
            self.statistics.update(&[self.discounted_reward]);
        }
        if step.terminated || step.truncated {
            self.discounted_reward = 0.0;
        }

        let normalized =
            f64::from(step.reward) / (self.statistics.variance[0] + NORMALIZATION_EPSILON).sqrt();
        step.reward =
            self.clip
                .map_or(normalized, |limit| normalized.clamp(-limit, limit)) as f32;
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
            statistics: self.statistics.clone(),
            discounted_reward: self.discounted_reward,
            update_statistics: self.update_statistics,
        })
    }

    fn set_state(&mut self, state: &Self::State) -> Result<(), Self::Error> {
        self.gym.set_state(&state.inner_state)?;
        self.statistics.clone_from(&state.statistics);
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
    use burn::tensor::{DType, Device, Tensor};

    use super::{
        NormalizeObservationGym, NormalizeObservationGymConfigurationError,
        NormalizeObservationGymError, NormalizeRewardGym, NormalizeRewardGymConfigurationError,
        RunningMeanStd, RunningMeanStdError, RunningMeanStdFileError,
    };
    use crate::gym::PerfectStateGym;

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
        assert_eq!(
            RunningMeanStd::new(vec![], vec![], 1.0),
            Err(RunningMeanStdError::Empty)
        );
        assert_eq!(
            RunningMeanStd::new(vec![0.0], vec![1.0, 1.0], 1.0),
            Err(RunningMeanStdError::LengthMismatch {
                mean: 1,
                variance: 2
            })
        );
        assert_eq!(
            RunningMeanStd::new(vec![f64::NAN], vec![1.0], 1.0),
            Err(RunningMeanStdError::NonFiniteMean)
        );
        for variance in [-1.0, f64::INFINITY] {
            assert_eq!(
                RunningMeanStd::new(vec![0.0], vec![variance], 1.0),
                Err(RunningMeanStdError::InvalidVariance)
            );
        }
        for count in [0.0, -1.0, f64::NAN] {
            assert_eq!(
                RunningMeanStd::new(vec![0.0], vec![1.0], count),
                Err(RunningMeanStdError::InvalidCount)
            );
        }
    }

    #[test]
    fn running_statistics_round_trip_through_bytes_and_files() {
        let statistics = RunningMeanStd::new(vec![1.5, -2.0], vec![0.25, 4.0], 12.0).unwrap();
        assert_eq!(
            RunningMeanStd::from_bytes(&statistics.to_bytes().unwrap()).unwrap(),
            statistics
        );

        let path = std::env::temp_dir().join(format!(
            "modurl-running-mean-std-{}.bpk",
            std::process::id()
        ));
        statistics.save_file(&path).unwrap();
        let loaded = RunningMeanStd::load_file(&path);
        std::fs::remove_file(&path).unwrap();
        assert_eq!(loaded.unwrap(), statistics);
    }

    fn statistics_pack(
        format: &str,
        version: u64,
        count: f64,
        mean_dtype: burn_pack::DType,
    ) -> Vec<u8> {
        let tensor = |name: &str, dtype| {
            let bytes = match dtype {
                burn_pack::DType::F32 => 1.0f32.to_le_bytes().to_vec(),
                _ => 1.0f64.to_le_bytes().to_vec(),
            };
            burn_pack::Tensor::new(
                name.to_string(),
                dtype,
                [1],
                None,
                burn_pack::Bytes::from_bytes_vec(bytes),
            )
        };
        burn_pack::Writer::new(vec![
            tensor("mean", mean_dtype),
            tensor("variance", burn_pack::DType::F64),
        ])
        .with_metadata("format", format)
        .with_scalar("version", version.into())
        .with_scalar("count", count.into())
        .into_bytes()
        .unwrap()
        .to_vec()
    }

    #[test]
    fn rejects_malformed_statistics_bytes() {
        let valid = statistics_pack("modurl.running_mean_std", 1, 2.0, burn_pack::DType::F64);
        assert!(RunningMeanStd::from_bytes(&valid).is_ok());
        assert!(matches!(
            RunningMeanStd::from_bytes(b"not statistics"),
            Err(RunningMeanStdFileError::Record(_))
        ));
        assert!(matches!(
            RunningMeanStd::from_bytes(&valid[..valid.len() - 1]),
            Err(RunningMeanStdFileError::Record(_))
        ));
        assert!(matches!(
            RunningMeanStd::from_bytes(&statistics_pack("other", 1, 2.0, burn_pack::DType::F64)),
            Err(RunningMeanStdFileError::NotStatisticsFile)
        ));
        assert!(matches!(
            RunningMeanStd::from_bytes(&statistics_pack(
                "modurl.running_mean_std",
                2,
                2.0,
                burn_pack::DType::F64
            )),
            Err(RunningMeanStdFileError::UnsupportedVersion(2))
        ));
        assert!(matches!(
            RunningMeanStd::from_bytes(&statistics_pack(
                "modurl.running_mean_std",
                1,
                2.0,
                burn_pack::DType::F32
            )),
            Err(RunningMeanStdFileError::InvalidEntry("mean"))
        ));
        assert!(matches!(
            RunningMeanStd::from_bytes(&statistics_pack(
                "modurl.running_mean_std",
                1,
                0.0,
                burn_pack::DType::F64
            )),
            Err(RunningMeanStdFileError::InvalidStatistics(
                RunningMeanStdError::InvalidCount
            ))
        ));
    }

    #[test]
    fn loaded_observation_statistics_normalize_without_updating() {
        let statistics = RunningMeanStd::new(vec![2.0], vec![4.0], 100.0).unwrap();
        let gym = TestGym::new([TestGym::step(4.0, 0.0, false, false, 1)]);
        let mut wrapper = NormalizeObservationGym::new(gym)
            .with_statistics(statistics.clone())
            .with_update_statistics(false);

        let reset = wrapper.reset().unwrap();
        let step = wrapper.step(action()).unwrap();

        assert!((scalar(&reset.observation) - 49.0).abs() < 1e-6);
        assert!((scalar(&step.observation) - 1.0).abs() < 1e-6);
        assert_eq!(wrapper.statistics(), Some(&statistics));
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
        let bytes = training.statistics().unwrap().to_bytes().unwrap();

        let loaded = RunningMeanStd::from_bytes(&bytes).unwrap();
        let expected = (5.0 - loaded.mean()[0]) / (loaded.variance()[0] + 1e-8).sqrt();
        let mut evaluation =
            NormalizeObservationGym::new(TestGym::new([TestGym::step(5.0, 0.0, false, false, 1)]))
                .with_statistics(loaded)
                .with_update_statistics(false);
        let step = evaluation.step(action()).unwrap();

        assert!((f64::from(scalar(&step.observation)) - expected).abs() < 1e-5);
        assert_eq!(evaluation.statistics(), training.statistics());
    }

    #[test]
    fn reports_observation_statistics_size_mismatch_before_inner_gym_runs() {
        let statistics = RunningMeanStd::new(vec![0.0, 0.0], vec![1.0, 1.0], 1.0).unwrap();
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
    fn saving_invalid_statistics_fails_before_writing() {
        let gym = TestGym::new([TestGym::step(0.0, f32::NAN, false, false, 1)]);
        let mut wrapper = NormalizeRewardGym::new(gym, 0.99).unwrap();
        wrapper.step(action()).unwrap();
        assert!(matches!(
            wrapper.statistics().to_bytes(),
            Err(RunningMeanStdFileError::InvalidStatistics(
                RunningMeanStdError::NonFiniteMean
            ))
        ));

        let path = std::env::temp_dir().join(format!(
            "modurl-invalid-running-mean-std-{}.bpk",
            std::process::id()
        ));
        assert!(matches!(
            wrapper.statistics().save_file(&path),
            Err(RunningMeanStdFileError::InvalidStatistics(
                RunningMeanStdError::NonFiniteMean
            ))
        ));
        assert!(!path.exists());
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
        let statistics = RunningMeanStd::new(vec![0.0], vec![4.0], 100.0).unwrap();
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
        assert_eq!(wrapper.statistics(), &statistics);
    }

    #[test]
    fn rejects_multi_component_reward_statistics() {
        let statistics = RunningMeanStd::new(vec![0.0, 0.0], vec![1.0, 1.0], 1.0).unwrap();
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
        assert_eq!(
            observation.save_state().unwrap().statistics(),
            observation.statistics()
        );

        let mut reward = NormalizeRewardGym::new(StateTestGym::new(5), 0.9).unwrap();
        reward.step(state_action()).unwrap();
        assert_eq!(
            reward.save_state().unwrap().statistics(),
            reward.statistics()
        );
    }
}

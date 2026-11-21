//! Running observation and reward normalization wrappers.

use crate::tensor_rank::{NextRank, PrevRank, TensorBatchTypes};
use burn::tensor::{Int, Slice, Tensor, TensorData, TensorReadError, kind::Basic};

use crate::{
    gym::{Gym, LegalActionsGym, PerfectStateGym, ResetInfo, StepInfo},
    spaces::{ActionSpace, ObservationSpace},
};

const RUNNING_STATS_EPSILON: f64 = 1e-4;
const NORMALIZATION_EPSILON: f64 = 1e-8;

#[derive(Clone)]
struct RunningMeanStd {
    mean: Vec<f64>,
    variance: Vec<f64>,
    count: f64,
}

impl RunningMeanStd {
    fn new(size: usize) -> Self {
        Self {
            mean: vec![0.0; size],
            variance: vec![1.0; size],
            count: RUNNING_STATS_EPSILON,
        }
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
}

#[derive(Debug, thiserror::Error)]
pub enum NormalizeObservationGymError<E> {
    /// A passthrough column exceeds the flattened unbatched observation size.
    #[error("passthrough column {column} is outside the {features} observation features")]
    InvalidPassthroughColumn { column: usize, features: usize },
    #[error("wrapped gym error: {0}")]
    GymError(#[source] E),
    #[error("failed to read an observation: {0}")]
    TensorRead(#[source] TensorReadError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum NormalizeObservationGymConfigurationError {
    #[error("clip must be finite and positive")]
    InvalidClip,
}

/// Normalizes each observation component using its running mean and variance.
/// Use [`Self::with_passthrough_columns`] to preserve discrete packed columns.
pub struct NormalizeObservationGym<G> {
    gym: G,
    statistics: Option<RunningMeanStd>,
    clip: Option<f64>,
    passthrough_columns: Vec<usize>,
}

impl<G> NormalizeObservationGym<G> {
    pub fn new(gym: G) -> Self {
        Self {
            gym,
            statistics: None,
            clip: None,
            passthrough_columns: Vec::new(),
        }
    }

    /// Skips normalization and clipping for selected components, preserving exact values.
    /// Indices address a row-major flattened unbatched observation; for flat
    /// observations they are feature columns. Duplicate indices are ignored.
    /// Replaces any previous selection. The default is empty (normalize everything).
    /// Out-of-range indices return an error on reset/step, before updating statistics.
    pub fn with_passthrough_columns(mut self, columns: impl IntoIterator<Item = usize>) -> Self {
        self.passthrough_columns = columns.into_iter().collect();
        self.passthrough_columns.sort_unstable();
        self.passthrough_columns.dedup();
        self
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

    /// Normalizes an unbatched rank-`O` observation, preserving its shape, kind, dtype, and device.
    /// Reads components as F32 for host statistics and casts normalized F32 values back to the input dtype.
    fn normalize<const O: usize, K: Basic, E>(
        &mut self,
        observation: &Tensor<O, K>,
    ) -> Result<Tensor<O, K>, NormalizeObservationGymError<E>> {
        let shape = observation.dims();
        let features = shape.iter().product::<usize>();
        if let Some(&column) = self
            .passthrough_columns
            .iter()
            .find(|&&column| column >= features)
        {
            return Err(NormalizeObservationGymError::InvalidPassthroughColumn {
                column,
                features,
            });
        }
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
            .get_or_insert_with(|| RunningMeanStd::new(values.len()));
        statistics.update(&values);

        let normalized = values
            .iter()
            .zip(&statistics.mean)
            .zip(&statistics.variance)
            .map(|((value, mean), variance)| {
                let value = (value - mean) / (variance + NORMALIZATION_EPSILON).sqrt();
                self.clip.map_or(value, |limit| value.clamp(-limit, limit)) as f32
            })
            .collect::<Vec<_>>();
        let normalized = Tensor::from_data(TensorData::new(normalized, shape), (&device, dtype));
        if self.passthrough_columns.is_empty() {
            return Ok(normalized);
        }
        let original = observation.clone().reshape([features]);
        let mut normalized = normalized.reshape([features]);
        for &column in &self.passthrough_columns {
            normalized = normalized.slice_assign(
                [Slice::from(column..column + 1)],
                original.clone().slice_dim(0, column..column + 1),
            );
        }
        Ok(normalized.reshape(shape))
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
        let mut reset = self
            .gym
            .reset()
            .map_err(NormalizeObservationGymError::GymError)?;
        reset.observation = self.normalize::<O, K, G::Error>(&reset.observation)?;
        Ok(reset)
    }

    /// Forwards an unbatched rank-`BA - 1` action and returns an unbatched rank-`O` observation; scalars use `[1]`.
    /// Normalizes observation values while preserving tensor shapes, kinds, dtypes, and devices.
    fn step(
        &mut self,
        action: <Self::ActionSpace as TensorBatchTypes>::Item,
    ) -> Result<StepInfo<I, Tensor<O, K>>, Self::Error> {
        let mut step = self
            .gym
            .step(action)
            .map_err(NormalizeObservationGymError::GymError)?;
        step.observation = self.normalize::<O, K, G::Error>(&step.observation)?;
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
        })
    }

    fn set_state(&mut self, state: &Self::State) -> Result<(), Self::Error> {
        self.gym
            .set_state(&state.inner_state)
            .map_err(NormalizeObservationGymError::GymError)?;
        self.statistics.clone_from(&state.statistics);
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
}

/// Scales immediate rewards by the running standard deviation of discounted rewards.
pub struct NormalizeRewardGym<G> {
    gym: G,
    statistics: RunningMeanStd,
    discounted_reward: f64,
    gamma: f64,
    clip: Option<f64>,
}

impl<G> NormalizeRewardGym<G> {
    /// Rejects a non-finite `gamma` or one outside `0..=1`.
    pub fn new(gym: G, gamma: f64) -> Result<Self, NormalizeRewardGymConfigurationError> {
        if !gamma.is_finite() || !(0.0..=1.0).contains(&gamma) {
            return Err(NormalizeRewardGymConfigurationError::InvalidGamma);
        }
        Ok(Self {
            gym,
            statistics: RunningMeanStd::new(1),
            discounted_reward: 0.0,
            gamma,
            clip: None,
        })
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
        self.statistics.update(&[self.discounted_reward]);
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
        })
    }

    fn set_state(&mut self, state: &Self::State) -> Result<(), Self::Error> {
        self.gym.set_state(&state.inner_state)?;
        self.statistics.clone_from(&state.statistics);
        self.discounted_reward = state.discounted_reward;
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
        gym::{Gym, ResetInfo, StepInfo},
        spaces::{BoxSpace, Discrete, DiscretePart, MultiPartSpace, SpaceError},
        wrappers::{
            state_test_support::{StateTestGym, action as state_action, assert_state_round_trip},
            test_support::{TestGym, action, scalar},
        },
    };
    use burn::tensor::{DType, Device, Int, IntDType, Tensor};

    use super::{
        NormalizeObservationGym, NormalizeObservationGymConfigurationError, NormalizeRewardGym,
        NormalizeRewardGymConfigurationError,
    };

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
    fn passthrough_components_keep_exact_values_at_any_rank() {
        let device = Device::flex();
        let observations = Tensor::<2>::from_data(
            [[1.0f64, 2.000000000000001], [3.0, 4.000000000000001]],
            (&device, DType::F64),
        );
        let mut wrapper = NormalizeObservationGym::new(())
            .with_passthrough_columns([3, 1, 3])
            .with_clip(0.01)
            .unwrap();
        let normalized = wrapper
            .normalize::<2, _, std::convert::Infallible>(&observations)
            .unwrap();
        assert_eq!(normalized.dims(), [2, 2]);
        assert_eq!(normalized.dtype(), DType::F64);
        let values = normalized.into_data().try_to_vec::<f64>().unwrap();
        assert_eq!(values[1], 2.000000000000001);
        assert_eq!(values[3], 4.000000000000001);
        assert!(values[0].abs() <= 0.01 && values[2].abs() <= 0.01);
        let mut wrapper = NormalizeObservationGym::new(()).with_passthrough_columns([]);
        let normalized = wrapper
            .normalize::<2, _, std::convert::Infallible>(&observations)
            .unwrap();
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
    fn invalid_passthrough_columns_do_not_update_statistics() {
        let mut wrapper = NormalizeObservationGym::new(()).with_passthrough_columns([4]);
        let observations = Tensor::<2>::zeros([2, 2], &Device::flex());
        assert!(matches!(
            wrapper.normalize::<2, _, std::convert::Infallible>(&observations),
            Err(
                super::NormalizeObservationGymError::InvalidPassthroughColumn {
                    column: 4,
                    features: 4
                }
            )
        ));
        assert!(wrapper.statistics.is_none());
        let mut wrapper =
            NormalizeObservationGym::new(TestGym::new([TestGym::step(1.0, 0.0, false, false, 1)]))
                .with_passthrough_columns([1]);
        assert!(matches!(
            wrapper.reset(),
            Err(
                super::NormalizeObservationGymError::InvalidPassthroughColumn {
                    column: 1,
                    features: 1
                }
            )
        ));
        assert!(matches!(
            wrapper.step(action()),
            Err(
                super::NormalizeObservationGymError::InvalidPassthroughColumn {
                    column: 1,
                    features: 1
                }
            )
        ));
        assert!(wrapper.statistics.is_none());
    }

    #[test]
    fn multipart_reset_and_step_preserve_reported_categories() {
        type Space = MultiPartSpace<(BoxSpace<2>, DiscretePart)>;
        struct Env {
            space: Space,
            mode: i64,
        }
        impl Env {
            fn observation(&self) -> Result<Tensor<1>, SpaceError> {
                let device = Device::flex();
                self.space
                    .pack((
                        Tensor::from_data([[self.mode as f64 * 10.0, 20.0]], (&device, DType::F64)),
                        Tensor::from_data([[self.mode]], (&device, DType::I64)),
                    ))
                    .map(|batch| batch.reshape([3]))
            }
        }
        impl Gym<i64> for Env {
            type Error = SpaceError;
            type ActionSpace = Discrete;
            type ObservationSpace = Space;
            fn reset(&mut self) -> Result<ResetInfo<i64>, Self::Error> {
                self.mode = 2;
                Ok(ResetInfo {
                    observation: self.observation()?,
                    info: self.mode,
                })
            }
            /// Consumes an unbatched action `[1]` and returns a packed observation `[3]`.
            fn step(&mut self, _action: Tensor<1, Int>) -> Result<StepInfo<i64>, Self::Error> {
                self.mode = 3;
                Ok(StepInfo {
                    observation: self.observation()?,
                    reward: 1.0,
                    terminated: false,
                    truncated: false,
                    info: self.mode,
                })
            }
            fn action_space(&self) -> Discrete {
                Discrete::new(2).unwrap()
            }
            fn observation_space(&self) -> Space {
                self.space.clone()
            }
        }
        let device = Device::flex();
        let space = MultiPartSpace::new((
            BoxSpace::new_unbounded([1, 2], &device),
            DiscretePart::new(Discrete::new(4).unwrap(), IntDType::I64),
        ));
        let mut wrapper = NormalizeObservationGym::new(Env {
            space: space.clone(),
            mode: 0,
        })
        .with_passthrough_columns(space.discrete_columns())
        .with_clip(0.125)
        .unwrap();
        let reset = wrapper.reset().unwrap();
        let fields = space.unpack(reset.observation.reshape([1, 3])).unwrap();
        assert_eq!(reset.info, 2);
        assert_eq!(fields.1.dtype(), DType::I64);
        assert_eq!(fields.1.into_scalar::<i64>(), reset.info);
        assert!(
            fields
                .0
                .into_data()
                .try_to_vec::<f64>()
                .unwrap()
                .iter()
                .all(|value| value.abs() <= 0.125)
        );
        let step = wrapper.step(Tensor::from_data([0i32], &device)).unwrap();
        let fields = space.unpack(step.observation.reshape([1, 3])).unwrap();
        assert_eq!(step.info, 3);
        assert_eq!(fields.1.into_scalar::<i64>(), step.info);
        assert!(
            fields
                .0
                .into_data()
                .try_to_vec::<f64>()
                .unwrap()
                .iter()
                .all(|value| value.abs() <= 0.125)
        );
        let mut blanket = NormalizeObservationGym::new(Env {
            space: space.clone(),
            mode: 0,
        });
        assert!(matches!(
            space.unpack(blanket.reset().unwrap().observation.reshape([1, 3])),
            Err(SpaceError::InvalidDiscretePartValue { part: 1 })
        ));
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
}

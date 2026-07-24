//! Running observation and reward normalization wrappers.

use candle_core::{DType, Device, Tensor};

use crate::gym::{Gym, ResetInfo, StepInfo};

const RUNNING_STATS_EPSILON: f64 = 1e-4;
const NORMALIZATION_EPSILON: f64 = 1e-8;

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

#[derive(Debug)]
pub enum NormalizeObservationGymError<E> {
    GymError(E),
    CandleError(candle_core::Error),
}

/// Normalizes each observation component using its running mean and variance.
pub struct NormalizeObservationGym<G> {
    gym: G,
    statistics: Option<RunningMeanStd>,
    clip: Option<f64>,
}

impl<G> NormalizeObservationGym<G> {
    pub fn new(gym: G) -> Self {
        Self {
            gym,
            statistics: None,
            clip: None,
        }
    }

    /// Clips normalized observations to `[-limit, limit]`.
    pub fn with_clip(mut self, limit: f64) -> Self {
        assert!(limit.is_finite() && limit > 0.0);
        self.clip = Some(limit);
        self
    }

    /// Normalizes one observation of arbitrary `observation_shape` and returns
    /// a tensor with that same shape.
    fn normalize(&mut self, observation: &Tensor) -> candle_core::Result<Tensor> {
        let shape = observation.shape().clone();
        let device = observation.device().clone();
        let values = observation
            .to_dtype(DType::F32)
            .and_then(|tensor| tensor.to_device(&Device::Cpu))
            .and_then(|tensor| tensor.flatten_all())
            .and_then(|tensor| tensor.to_vec1::<f32>())?
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
        Tensor::from_vec(normalized, shape, &Device::Cpu)
            .and_then(|tensor| tensor.to_device(&device))
    }
}

impl<G, I> Gym<I> for NormalizeObservationGym<G>
where
    G: Gym<I>,
{
    type Error = NormalizeObservationGymError<<G as Gym<I>>::Error>;
    type SpaceError = <G as Gym<I>>::SpaceError;

    fn reset(&mut self) -> Result<ResetInfo<I>, Self::Error> {
        let mut reset = self
            .gym
            .reset()
            .map_err(NormalizeObservationGymError::GymError)?;
        reset.state = self
            .normalize(&reset.state)
            .map_err(NormalizeObservationGymError::CandleError)?;
        Ok(reset)
    }

    /// Forwards one unbatched environment action shaped `action_shape`.
    fn step(&mut self, action: Tensor) -> Result<StepInfo<I>, Self::Error> {
        let mut step = self
            .gym
            .step(action)
            .map_err(NormalizeObservationGymError::GymError)?;
        step.state = self
            .normalize(&step.state)
            .map_err(NormalizeObservationGymError::CandleError)?;
        Ok(step)
    }

    fn action_space(&self) -> Box<dyn crate::spaces::Space<Error = Self::SpaceError>> {
        self.gym.action_space()
    }

    fn observation_space(&self) -> Box<dyn crate::spaces::Space<Error = Self::SpaceError>> {
        self.gym.observation_space()
    }
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
    pub fn new(gym: G, gamma: f64) -> Self {
        assert!((0.0..=1.0).contains(&gamma));
        Self {
            gym,
            statistics: RunningMeanStd::new(1),
            discounted_reward: 0.0,
            gamma,
            clip: None,
        }
    }

    /// Clips normalized rewards to `[-limit, limit]`.
    pub fn with_clip(mut self, limit: f64) -> Self {
        assert!(limit.is_finite() && limit > 0.0);
        self.clip = Some(limit);
        self
    }
}

impl<G, I> Gym<I> for NormalizeRewardGym<G>
where
    G: Gym<I>,
{
    type Error = <G as Gym<I>>::Error;
    type SpaceError = <G as Gym<I>>::SpaceError;

    fn reset(&mut self) -> Result<ResetInfo<I>, Self::Error> {
        self.gym.reset()
    }

    /// Forwards one unbatched environment action shaped `action_shape`.
    fn step(&mut self, action: Tensor) -> Result<StepInfo<I>, Self::Error> {
        let mut step = self.gym.step(action)?;
        self.discounted_reward =
            self.discounted_reward * self.gamma * if step.done { 0.0 } else { 1.0 }
                + f64::from(step.reward);
        self.statistics.update(&[self.discounted_reward]);

        let normalized =
            f64::from(step.reward) / (self.statistics.variance[0] + NORMALIZATION_EPSILON).sqrt();
        step.reward =
            self.clip
                .map_or(normalized, |limit| normalized.clamp(-limit, limit)) as f32;
        Ok(step)
    }

    fn action_space(&self) -> Box<dyn crate::spaces::Space<Error = Self::SpaceError>> {
        self.gym.action_space()
    }

    fn observation_space(&self) -> Box<dyn crate::spaces::Space<Error = Self::SpaceError>> {
        self.gym.observation_space()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        gym::Gym,
        wrappers::test_support::{TestGym, action, scalar},
    };

    use super::{NormalizeObservationGym, NormalizeRewardGym};

    #[test]
    fn normalizes_observations_and_preserves_metadata() {
        let gym = TestGym::new([TestGym::step(1.0, 0.0, false, false, 7)]);
        let mut wrapper = NormalizeObservationGym::new(gym).with_clip(10.0);

        let reset = wrapper.reset().unwrap();
        let step = wrapper.step(action()).unwrap();

        assert!(scalar(&reset.state).abs() < 0.02);
        assert!((scalar(&step.state) + 1.0).abs() < 0.01);
        assert_eq!(step.info.sequence, 7);
    }

    #[test]
    fn normalizes_and_clips_rewards() {
        let gym = TestGym::new([
            TestGym::step(1.0, 1.0, false, false, 1),
            TestGym::step(2.0, 1.0, false, false, 2),
        ]);
        let mut wrapper = NormalizeRewardGym::new(gym, 0.99).with_clip(10.0);

        let first = wrapper.step(action()).unwrap();
        assert_eq!(first.reward, 10.0);

        let second = wrapper.step(action()).unwrap();
        assert!(second.reward > 1.9 && second.reward < 2.1);
        assert_eq!(second.info.sequence, 2);
    }
}

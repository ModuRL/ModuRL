use crate::AtariInfo;
use burn::tensor::{
    DType, Device, Int, Tensor, TensorData, TensorReadError,
    kind::{Basic, Kind},
};
use modurl::{
    gym::{Gym, ResetInfo, StepInfo},
    sampling::{SamplingError, sample_u32_inclusive},
    spaces::{Discrete, ObservationSpace},
    tensor_rank::{NextRank, PrevRank, TensorBatchTypes},
    wrappers::EpisodeStatisticsInfo,
};
use std::sync::OnceLock;

/// Takes a random number of no-op actions after reset.
///
/// This randomizes Atari starting states so a policy does not overfit to one
/// deterministic opening sequence.
pub struct NoopResetGym<G> {
    gym: G,
    noop_max: u32,
}

impl<G> NoopResetGym<G> {
    pub fn new(gym: G) -> Self {
        Self { gym, noop_max: 30 }
    }

    pub fn new_with_noop_max(gym: G, noop_max: u32) -> Self {
        assert!(noop_max > 0, "noop_max must be at least 1");
        Self { gym, noop_max }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum NoopResetGymError<E> {
    #[error("wrapped gym error: {0}")]
    GymError(#[source] E),
    #[error("no-op sampling failed: {0}")]
    Sampling(#[source] SamplingError),
}

impl<G, I, const O: usize> Gym<I, O, 2> for NoopResetGym<G>
where
    G: Gym<I, O, 2, ActionSpace = Discrete>,
{
    type Error = NoopResetGymError<G::Error>;
    type ObservationSpace = G::ObservationSpace;
    type ActionSpace = Discrete;

    fn reset(
        &mut self,
    ) -> Result<ResetInfo<I, <G::ObservationSpace as TensorBatchTypes>::Item>, Self::Error> {
        let mut reset = self.gym.reset().map_err(NoopResetGymError::GymError)?;
        for _ in 0..sample_u32_inclusive(1, self.noop_max, &Device::flex())
            .map_err(NoopResetGymError::Sampling)?
        {
            let noop_action = Tensor::from_data([0i32], &Device::flex());
            let step = self
                .gym
                .step(noop_action)
                .map_err(NoopResetGymError::GymError)?;
            reset = if step.done || step.truncated {
                self.gym.reset().map_err(NoopResetGymError::GymError)?
            } else {
                ResetInfo {
                    observation: step.observation,
                    info: step.info,
                }
            };
        }
        Ok(reset)
    }

    /// Forwards Int Atari actions [1], preserving the wrapped observation item shape, kind, dtype, and device.
    fn step(
        &mut self,
        action: Tensor<1, Int>,
    ) -> Result<StepInfo<I, <G::ObservationSpace as TensorBatchTypes>::Item>, Self::Error> {
        self.gym.step(action).map_err(NoopResetGymError::GymError)
    }

    fn action_space(&self) -> Discrete {
        self.gym.action_space()
    }

    fn observation_space(&self) -> Self::ObservationSpace {
        self.gym.observation_space()
    }
}

/// Metadata that exposes the number of lives remaining in an Atari game.
pub trait AtariLives {
    /// Returns the current number of lives reported by ALE.
    fn lives(&self) -> u32;
}

impl AtariLives for AtariInfo {
    fn lives(&self) -> u32 {
        self.lives
    }
}

impl<I> AtariLives for EpisodeStatisticsInfo<I>
where
    I: AtariLives,
{
    fn lives(&self) -> u32 {
        self.inner.lives()
    }
}

/// Treats a lost life as an episode end while preserving the current game.
///
/// This supplies denser reset boundaries for Atari training without discarding
/// progress from the remaining lives.
pub struct EpisodicLifeGym<G> {
    gym: G,
    last_lives: u32,
    was_real_done: bool,
    device: Device,
}

impl<G> EpisodicLifeGym<G> {
    pub fn new(gym: G, device: &Device) -> Self {
        Self {
            gym,
            last_lives: 0,
            was_real_done: true,
            device: device.clone(),
        }
    }
}

impl<G, I, const O: usize> Gym<I, O, 2> for EpisodicLifeGym<G>
where
    G: Gym<I, O, 2, ActionSpace = Discrete>,
    I: AtariLives,
{
    type Error = G::Error;
    type ObservationSpace = G::ObservationSpace;
    type ActionSpace = Discrete;

    fn reset(
        &mut self,
    ) -> Result<ResetInfo<I, <G::ObservationSpace as TensorBatchTypes>::Item>, Self::Error> {
        let reset = if self.was_real_done {
            self.gym.reset()?
        } else {
            let noop_action = Tensor::from_data([0i32], &self.device);
            let step = self.gym.step(noop_action)?;
            if step.done || step.truncated {
                self.gym.reset()?
            } else {
                ResetInfo {
                    observation: step.observation,
                    info: step.info,
                }
            }
        };

        self.last_lives = reset.info.lives();
        Ok(reset)
    }

    /// Forwards Int Atari actions [1], preserving the wrapped observation item shape, kind, dtype, and device.
    fn step(
        &mut self,
        action: Tensor<1, Int>,
    ) -> Result<StepInfo<I, <G::ObservationSpace as TensorBatchTypes>::Item>, Self::Error> {
        let mut step = self.gym.step(action)?;
        let current_lives = step.info.lives();

        self.was_real_done = step.done || step.truncated;

        if current_lives < self.last_lives {
            step.done = true;
        }

        self.last_lives = current_lives;

        Ok(step)
    }

    fn action_space(&self) -> Discrete {
        self.gym.action_space()
    }

    fn observation_space(&self) -> Self::ObservationSpace {
        self.gym.observation_space()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum FireResetGymError<E> {
    #[error("wrapped gym error: {0}")]
    GymError(#[source] E),
}

/// Sends the standard FIRE startup actions after reset.
///
/// Games such as Breakout require FIRE before play begins, so this makes each
/// reset immediately reach an interactive state.
pub struct FireResetGym<G> {
    gym: G,
}

impl<G> FireResetGym<G> {
    pub fn new(gym: G) -> Self {
        Self { gym }
    }
}

impl<G, I, const O: usize> Gym<I, O, 2> for FireResetGym<G>
where
    G: Gym<I, O, 2, ActionSpace = Discrete>,
{
    type Error = FireResetGymError<G::Error>;
    type ObservationSpace = G::ObservationSpace;
    type ActionSpace = Discrete;

    fn reset(
        &mut self,
    ) -> Result<ResetInfo<I, <G::ObservationSpace as TensorBatchTypes>::Item>, Self::Error> {
        self.gym.reset().map_err(FireResetGymError::GymError)?;
        let first = self
            .gym
            .step(Tensor::from_data([1i32], &Device::flex()))
            .map_err(FireResetGymError::GymError)?;
        if first.done || first.truncated {
            self.gym.reset().map_err(FireResetGymError::GymError)?;
        }
        let second = self
            .gym
            .step(Tensor::from_data([2i32], &Device::flex()))
            .map_err(FireResetGymError::GymError)?;
        if second.done || second.truncated {
            self.gym.reset().map_err(FireResetGymError::GymError)
        } else {
            Ok(ResetInfo {
                observation: second.observation,
                info: second.info,
            })
        }
    }

    /// Forwards Int Atari actions [1], preserving the wrapped observation item shape, kind, dtype, and device.
    fn step(
        &mut self,
        action: Tensor<1, Int>,
    ) -> Result<StepInfo<I, <G::ObservationSpace as TensorBatchTypes>::Item>, Self::Error> {
        self.gym.step(action).map_err(FireResetGymError::GymError)
    }

    fn action_space(&self) -> Discrete {
        self.gym.action_space()
    }

    fn observation_space(&self) -> Self::ObservationSpace {
        self.gym.observation_space()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum WarpGymError<E> {
    #[error("wrapped gym error: {0}")]
    GymError(#[source] E),
    #[error("reading Atari observation failed: {0}")]
    Tensor(#[from] TensorReadError),
    #[error(
        "warp requires nonempty grayscale [height, width] or RGB [height, width, 3], got {0:?}"
    )]
    Shape(Vec<usize>),
    #[error("warp requires U8 Int or F32 Float observations, got {0:?}")]
    DType(DType),
    #[error("warp observations must use CPU Flex")]
    Device,
}

/// Converts CPU RGB or grayscale observations into grayscale [84, 84].
/// Preserves U8 Int or F32 Float representation and uses the reference area-resize calculation.
pub struct WarpGym<G, const O: usize = 4> {
    gym: G,
}

/// Warped CPU observations [batch_size, 84, 84] with the input tensor kind.
pub struct WarpObservationSpace<K: Basic> {
    kind: std::marker::PhantomData<K>,
}

impl<K: Basic> TensorBatchTypes for WarpObservationSpace<K> {
    type Kind = K;
    type Item = Tensor<2, K>;
    type Batch = Tensor<3, K>;
}

impl<K: Basic> ObservationSpace<3> for WarpObservationSpace<K> {
    /// Checks observations [batch_size, 84, 84] on CPU Flex without changing kind or axes.
    fn contains(&self, observations: &Tensor<3, K>) -> bool {
        let dtype_matches = if const { matches!(K::KIND, Kind::Int) } {
            observations.dtype() == DType::U8
        } else if const { matches!(K::KIND, Kind::Float) } {
            observations.dtype() == DType::F32
        } else {
            false
        };
        observations.dims()[1..] == [84, 84]
            && observations.device() == Device::flex()
            && dtype_matches
    }

    fn shape(&self) -> Vec<usize> {
        vec![84, 84]
    }
}

#[derive(Clone, Copy)]
struct AreaSpan {
    start: usize,
    len: usize,
    weights: [f32; 3],
}

fn area_spans(source_len: usize, target_len: usize) -> Vec<AreaSpan> {
    let scale = source_len as f32 / target_len as f32;
    (0..target_len)
        .map(|output| {
            let start_f = output as f32 * scale;
            let end_f = (output + 1) as f32 * scale;
            let start = start_f.floor() as usize;
            let end = end_f.ceil().min(source_len as f32) as usize;
            let len = end - start;
            debug_assert!(len <= 3);
            let mut weights = [0.0; 3];
            for (offset, weight) in weights.iter_mut().enumerate().take(len) {
                let source = start + offset;
                *weight =
                    (end_f.min((source + 1) as f32) - start_f.max(source as f32)).max(0.0) / scale;
            }
            AreaSpan {
                start,
                len,
                weights,
            }
        })
        .collect()
}

static ATARI_HORIZONTAL_SPANS: OnceLock<Vec<AreaSpan>> = OnceLock::new();
static ATARI_VERTICAL_SPANS: OnceLock<Vec<AreaSpan>> = OnceLock::new();

impl<G, const O: usize> WarpGym<G, O> {
    pub fn new(gym: G) -> Self {
        Self { gym }
    }

    fn resize_area_data(
        input: &[f32],
        input_h: usize,
        input_w: usize,
        target_h: usize,
        target_w: usize,
    ) -> Vec<f32> {
        let scale_h = input_h as f32 / target_h as f32;
        let scale_w = input_w as f32 / target_w as f32;

        // INTER_AREA is a separable box filter. The two one-dimensional
        // passes avoid recomputing every horizontal contribution for each
        // vertically overlapping output pixel.
        let mut horizontal = vec![0.0f32; input_h * target_w];
        for y in 0..input_h {
            for output_x in 0..target_w {
                let x_start = output_x as f32 * scale_w;
                let x_end = (output_x + 1) as f32 * scale_w;
                let mut sum = 0.0f32;
                let mut x = x_start.floor() as usize;
                while (x as f32) < x_end && x < input_w {
                    let weight = (x_end.min((x + 1) as f32) - x_start.max(x as f32)).max(0.0);
                    sum += input[y * input_w + x] * weight;
                    x += 1;
                }
                horizontal[y * target_w + output_x] = sum / scale_w;
            }
        }

        let mut output = vec![0.0f32; target_h * target_w];
        for output_y in 0..target_h {
            let y_start = output_y as f32 * scale_h;
            let y_end = (output_y + 1) as f32 * scale_h;
            for x in 0..target_w {
                let mut sum = 0.0f32;
                let mut y = y_start.floor() as usize;
                while (y as f32) < y_end && y < input_h {
                    let weight = (y_end.min((y + 1) as f32) - y_start.max(y as f32)).max(0.0);
                    sum += horizontal[y * target_w + x] * weight;
                    y += 1;
                }
                output[output_y * target_w + x] = sum / scale_h;
            }
        }
        output
    }

    fn resize_atari_rgb_u8(rgb: &[u8], height: usize, width: usize) -> Vec<u8> {
        debug_assert_eq!((height, width), (210, 160));
        let horizontal_spans = ATARI_HORIZONTAL_SPANS.get_or_init(|| area_spans(160, 84));
        let vertical_spans = ATARI_VERTICAL_SPANS.get_or_init(|| area_spans(210, 84));

        // Fuse RGB-to-luminance with the horizontal area pass. This avoids a
        // 210x160 f32 grayscale allocation and a second traversal of it.
        let mut horizontal = vec![0.0f32; height * 84];
        for y in 0..height {
            for (output_x, span) in horizontal_spans.iter().enumerate() {
                let mut sum = 0.0;
                for offset in 0..span.len {
                    let pixel = (y * width + span.start + offset) * 3;
                    let luminance = rgb[pixel] as f32 * 0.299
                        + rgb[pixel + 1] as f32 * 0.587
                        + rgb[pixel + 2] as f32 * 0.114;
                    sum += luminance * span.weights[offset];
                }
                horizontal[y * 84 + output_x] = sum;
            }
        }

        let mut output = vec![0u8; 84 * 84];
        for (output_y, span) in vertical_spans.iter().enumerate() {
            for x in 0..84 {
                let mut sum = 0.0;
                for offset in 0..span.len {
                    sum += horizontal[(span.start + offset) * 84 + x] * span.weights[offset];
                }
                output[output_y * 84 + x] = sum.round().clamp(0.0, 255.0) as u8;
            }
        }
        output
    }

    /// Converts CPU rank-D RGB [height, width, 3] or grayscale [height, width] into [84, 84].
    /// Preserves U8 Int or F32 Float data; reads input to the host and returns a CPU tensor with no batch axis.
    fn preprocess_observation<K: Basic, E, const D: usize>(
        &self,
        observation: Tensor<D, K>,
    ) -> Result<Tensor<2, K>, WarpGymError<E>> {
        const {
            assert!(
                D == 2 || D == 3,
                "warp accepts grayscale rank 2 or RGB rank 3"
            );
        }
        let dims = observation.dims();
        if dims[0] == 0
            || dims[1] == 0
            || (if const { D == 3 } {
                dims[2] != 3
            } else {
                false
            })
        {
            return Err(WarpGymError::Shape(dims.to_vec()));
        }
        if observation.device() != Device::flex() {
            return Err(WarpGymError::Device);
        }
        let (height, width) = (dims[0], dims[1]);
        let dtype = observation.dtype();
        let data = observation.try_into_data().map_err(TensorReadError::from)?;
        if dtype == DType::U8 {
            let bytes = data.as_slice::<u8>().map_err(TensorReadError::from)?;
            if const { D == 3 } && (height, width) == (210, 160) {
                return Ok(Tensor::from_data(
                    TensorData::new(Self::resize_atari_rgb_u8(bytes, height, width), [84, 84]),
                    (&Device::flex(), DType::U8),
                ));
            }
            let luminance: Vec<f32> = if const { D == 2 } {
                bytes.iter().copied().map(f32::from).collect()
            } else {
                bytes
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|pixel| {
                        pixel[0] as f32 * 0.299 + pixel[1] as f32 * 0.587 + pixel[2] as f32 * 0.114
                    })
                    .collect()
            };
            let resized = Self::resize_area_data(&luminance, height, width, 84, 84)
                .into_iter()
                .map(|value| value.round().clamp(0.0, 255.0) as u8)
                .collect::<Vec<_>>();
            return Ok(Tensor::from_data(
                TensorData::new(resized, [84, 84]),
                (&Device::flex(), DType::U8),
            ));
        }
        if dtype != DType::F32 {
            return Err(WarpGymError::DType(dtype));
        }
        let values = data.try_to_vec::<f32>().map_err(TensorReadError::from)?;
        let luminance = if const { D == 2 } {
            values
        } else {
            values
                .as_chunks::<3>()
                .0
                .iter()
                .map(|pixel| pixel[0] * 0.299 + pixel[1] * 0.587 + pixel[2] * 0.114)
                .collect()
        };
        Ok(Tensor::from_data(
            TensorData::new(
                Self::resize_area_data(&luminance, height, width, 84, 84),
                [84, 84],
            ),
            &Device::flex(),
        ))
    }
}

impl<G, I, K: Basic, const O: usize, const D: usize> Gym<I, 3, 2> for WarpGym<G, O>
where
    G: Gym<I, O, 2, ActionSpace = Discrete>,
    G::ObservationSpace: ObservationSpace<O, Kind = K, Item = Tensor<D, K>>,
    Tensor<D, K>: NextRank<Next = Tensor<O, K>>,
    Tensor<O, K>: PrevRank<Prev = Tensor<D, K>>,
{
    type Error = WarpGymError<G::Error>;
    type ObservationSpace = WarpObservationSpace<K>;
    type ActionSpace = Discrete;

    /// Warps rank-D CPU RGB/grayscale reset observations to rank 2 [84, 84], preserving native dtype and metadata.
    fn reset(&mut self) -> Result<ResetInfo<I, Tensor<2, K>>, Self::Error> {
        let reset = self.gym.reset().map_err(WarpGymError::GymError)?;
        Ok(ResetInfo {
            observation: self.preprocess_observation(reset.observation)?,
            info: reset.info,
        })
    }

    /// Forwards Int actions [1] and warps rank-D CPU observations to rank 2 [84, 84]; preserves reward and end flags.
    fn step(&mut self, action: Tensor<1, Int>) -> Result<StepInfo<I, Tensor<2, K>>, Self::Error> {
        let step = self.gym.step(action).map_err(WarpGymError::GymError)?;
        Ok(StepInfo {
            observation: self.preprocess_observation(step.observation)?,
            reward: step.reward,
            done: step.done,
            truncated: step.truncated,
            info: step.info,
        })
    }

    fn action_space(&self) -> Discrete {
        self.gym.action_space()
    }

    fn observation_space(&self) -> Self::ObservationSpace {
        WarpObservationSpace {
            kind: std::marker::PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use modurl::{
        spaces::{BoxSpace, Discrete},
        wrappers::{EpisodeStatistics, RecordEpisodeStatisticsGym},
    };

    #[derive(Clone)]
    struct ScriptStep {
        observation: f32,
        reward: f32,
        done: bool,
        truncated: bool,
        lives: u32,
    }

    struct ScriptGym {
        device: Device,
        reset_count: u32,
        actions: Vec<u32>,
        lives: u32,
        steps: Vec<ScriptStep>,
        step_index: usize,
    }

    impl ScriptGym {
        fn new(steps: Vec<ScriptStep>) -> Self {
            Self {
                device: Device::flex(),
                reset_count: 0,
                actions: vec![],
                lives: 3,
                steps,
                step_index: 0,
            }
        }

        fn step(
            observation: f32,
            reward: f32,
            done: bool,
            truncated: bool,
            lives: u32,
        ) -> ScriptStep {
            ScriptStep {
                observation,
                reward,
                done,
                truncated,
                lives,
            }
        }

        fn tensor(&self, value: f32) -> Tensor<1> {
            Tensor::from_data([value], &self.device)
        }
    }

    impl Gym<AtariInfo> for ScriptGym {
        type Error = TensorReadError;
        type ObservationSpace = BoxSpace<2>;
        type ActionSpace = Discrete;

        /// Steps with one scalar discrete action shaped `[1]`.
        fn step(&mut self, action: Tensor<1, Int>) -> Result<StepInfo<AtariInfo>, Self::Error> {
            self.actions.push(action.try_into_scalar::<u32>()?);
            let step = self.steps.get(self.step_index).cloned().unwrap_or_else(|| {
                Self::step(
                    10_000.0 + self.step_index as f32,
                    0.0,
                    false,
                    false,
                    self.lives,
                )
            });
            self.step_index += 1;
            self.lives = step.lives;
            Ok(StepInfo {
                observation: self.tensor(step.observation),
                reward: step.reward,
                done: step.done,
                truncated: step.truncated,
                info: AtariInfo {
                    lives: self.lives,
                    frame_number: self.step_index as u32,
                    episode_frame_number: self.step_index as u32,
                },
            })
        }

        fn reset(&mut self) -> Result<ResetInfo<AtariInfo>, Self::Error> {
            self.reset_count += 1;
            self.lives = 3;
            Ok(ResetInfo {
                observation: self.tensor(100.0 * self.reset_count as f32),
                info: AtariInfo {
                    lives: self.lives,
                    frame_number: 0,
                    episode_frame_number: 0,
                },
            })
        }

        fn observation_space(&self) -> Self::ObservationSpace {
            BoxSpace::new_with_universal_bounds([1, 1], 0.0, 10_000.0, &self.device)
        }

        fn action_space(&self) -> Discrete {
            Discrete::new(4).unwrap()
        }
    }

    /// Reads one scalar tensor shaped `[1]`.
    fn scalar(tensor: &Tensor<1>) -> f32 {
        tensor.clone().try_into_scalar::<f32>().unwrap()
    }

    struct ImageGym;

    impl Gym<AtariInfo, 4, 2> for ImageGym {
        type Error = std::convert::Infallible;
        type ObservationSpace = crate::AtariObservationSpace<4>;
        type ActionSpace = Discrete;

        /// Returns CPU U8 RGB observations [210, 160, 3] without a batch axis.
        fn reset(&mut self) -> Result<ResetInfo<AtariInfo, Tensor<3, Int>>, Self::Error> {
            Ok(ResetInfo {
                observation: Tensor::zeros([210, 160, 3], (&Device::flex(), DType::U8)),
                info: AtariInfo {
                    lives: 3,
                    frame_number: 0,
                    episode_frame_number: 0,
                },
            })
        }

        /// Accepts CPU Int actions [1] and returns terminal RGB observations [210, 160, 3].
        fn step(
            &mut self,
            action: Tensor<1, Int>,
        ) -> Result<StepInfo<AtariInfo, Tensor<3, Int>>, Self::Error> {
            assert_eq!(action.dims(), [1]);
            Ok(StepInfo {
                observation: Tensor::full([210, 160, 3], 37, (&Device::flex(), DType::U8)),
                reward: 2.0,
                done: true,
                truncated: false,
                info: AtariInfo {
                    lives: 0,
                    frame_number: 1,
                    episode_frame_number: 1,
                },
            })
        }

        fn action_space(&self) -> Discrete {
            Discrete::new(4).unwrap()
        }

        fn observation_space(&self) -> Self::ObservationSpace {
            crate::AtariObservationSpace::new(vec![210, 160, 3])
        }
    }

    #[test]
    fn warped_vectorized_observations_keep_terminal_frames_and_native_kind() {
        use modurl::gym::{MultiGym, VectorizedGymWrapper};
        let gym = WarpGym::new(FireResetGym::new(ImageGym));
        let mut env = VectorizedGymWrapper::from(gym);
        let reset = env.reset().unwrap();
        assert_eq!(reset.dims(), [1, 84, 84]);
        assert_eq!(reset.dtype(), DType::U8);
        let step = env
            .step(Tensor::from_data([[0i32]], &Device::flex()))
            .unwrap();
        assert_eq!(step.dones, [true]);
        assert_eq!(step.truncateds, [false]);
        assert_eq!(step.observations.dtype(), DType::U8);
        let terminal = step.transition_next_observations();
        assert_eq!(terminal.dims(), [1, 84, 84]);
        assert_eq!(
            terminal.into_data().try_to_vec::<u8>().unwrap(),
            vec![37; 84 * 84]
        );
        assert_eq!(
            step.observations.into_data().try_to_vec::<u8>().unwrap(),
            vec![0; 84 * 84]
        );
    }

    #[test]
    fn warp_preserves_u8_through_grayscale_and_resize() {
        let wrapper = WarpGym::<_, 4>::new(ScriptGym::new(vec![]));
        let rgb = Tensor::<3, Int>::zeros([210, 160, 3], (&Device::flex(), DType::U8));

        let warped = wrapper
            .preprocess_observation::<_, std::convert::Infallible, 3>(rgb)
            .unwrap();

        assert_eq!(warped.dtype(), DType::U8);
        assert_eq!(warped.dims(), [84, 84]);
    }

    #[test]
    fn warp_preserves_grayscale_bytes_and_float_values() {
        let wrapper = WarpGym::<_, 3>::new(ScriptGym::new(vec![]));
        let bytes = Tensor::<2, Int>::full([84, 84], 37, (&Device::flex(), DType::U8));
        let output = wrapper
            .preprocess_observation::<_, std::convert::Infallible, 2>(bytes)
            .unwrap();
        assert_eq!(output.dtype(), DType::U8);
        assert_eq!(
            output.into_data().try_to_vec::<u8>().unwrap(),
            vec![37; 84 * 84]
        );
        let values = Tensor::<2>::full([84, 84], 0.5, &Device::flex());
        let output = wrapper
            .preprocess_observation::<_, std::convert::Infallible, 2>(values)
            .unwrap();
        assert_eq!(output.dtype(), DType::F32);
        assert_eq!(
            output.into_data().try_to_vec::<f32>().unwrap(),
            vec![0.5; 84 * 84]
        );
        let invalid = Tensor::<3, Int>::zeros([84, 84, 2], (&Device::flex(), DType::U8));
        assert!(matches!(
            wrapper.preprocess_observation::<_, std::convert::Infallible, 3>(invalid),
            Err(WarpGymError::Shape(_))
        ));
        let invalid = Tensor::<2>::zeros([84, 84], (&Device::flex(), DType::F64));
        assert!(matches!(
            wrapper.preprocess_observation::<_, std::convert::Infallible, 2>(invalid),
            Err(WarpGymError::DType(DType::F64))
        ));
    }

    #[test]
    fn fused_u8_warp_matches_separable_reference() {
        let rgb = (0..210 * 160 * 3)
            .map(|index| ((index * 37 + 11) % 256) as u8)
            .collect::<Vec<_>>();
        let luminance = rgb
            .as_chunks::<3>()
            .0
            .iter()
            .map(|pixel| {
                pixel[0] as f32 * 0.299 + pixel[1] as f32 * 0.587 + pixel[2] as f32 * 0.114
            })
            .collect::<Vec<_>>();
        let expected = WarpGym::<ScriptGym>::resize_area_data(&luminance, 210, 160, 84, 84)
            .into_iter()
            .map(|value| value.round().clamp(0.0, 255.0) as u8)
            .collect::<Vec<_>>();

        let actual = WarpGym::<ScriptGym>::resize_atari_rgb_u8(&rgb, 210, 160);

        assert_eq!(actual, expected);
    }

    #[test]
    fn u8_warp_differs_from_old_float_pipeline_only_by_final_rounding() {
        let rgb = (0..210 * 160 * 3)
            .map(|index| ((index * 37 + 11) % 256) as u8)
            .collect::<Vec<_>>();
        let normalized_luminance = rgb
            .as_chunks::<3>()
            .0
            .iter()
            .map(|pixel| {
                (pixel[0] as f32 * 0.299 + pixel[1] as f32 * 0.587 + pixel[2] as f32 * 0.114)
                    / 255.0
            })
            .collect::<Vec<_>>();
        let old_float =
            WarpGym::<ScriptGym>::resize_area_data(&normalized_luminance, 210, 160, 84, 84);
        let new_float = WarpGym::<ScriptGym>::resize_atari_rgb_u8(&rgb, 210, 160)
            .into_iter()
            .map(|value| value as f32 / 255.0)
            .collect::<Vec<_>>();

        let maximum_error = old_float
            .iter()
            .zip(new_float)
            .map(|(old, new)| (old - new).abs())
            .fold(0.0f32, f32::max);

        assert!(maximum_error <= 0.5 / 255.0 + 1e-6, "{maximum_error}");
    }

    #[test]
    fn noop_reset_uses_noop_action_and_resets_if_noop_ends_episode() {
        let gym = ScriptGym::new(vec![ScriptGym::step(1.0, 0.0, true, false, 0)]);
        let mut wrapper = NoopResetGym::new_with_noop_max(gym, 1);

        let obs = wrapper.reset().unwrap();

        assert_eq!(wrapper.gym.actions, vec![0]);
        assert_eq!(wrapper.gym.reset_count, 2);
        assert_eq!(scalar(&obs.observation), 200.0);
    }

    #[test]
    fn fire_reset_takes_fire_then_action_two() {
        let gym = ScriptGym::new(vec![
            ScriptGym::step(11.0, 0.0, false, false, 3),
            ScriptGym::step(12.0, 0.0, false, false, 3),
        ]);
        let mut wrapper = FireResetGym::new(gym);

        let obs = wrapper.reset().unwrap();

        assert_eq!(wrapper.gym.actions, vec![1, 2]);
        assert_eq!(wrapper.gym.reset_count, 1);
        assert_eq!(scalar(&obs.observation), 12.0);
    }

    #[test]
    fn fire_reset_resets_after_first_reset_action_done() {
        let gym = ScriptGym::new(vec![
            ScriptGym::step(11.0, 0.0, true, false, 0),
            ScriptGym::step(22.0, 0.0, false, false, 3),
        ]);
        let mut wrapper = FireResetGym::new(gym);

        let obs = wrapper.reset().unwrap();

        assert_eq!(wrapper.gym.actions, vec![1, 2]);
        assert_eq!(wrapper.gym.reset_count, 2);
        assert_eq!(scalar(&obs.observation), 22.0);
    }

    #[test]
    fn fire_reset_returns_the_new_reset_observation_after_second_action_done() {
        let gym = ScriptGym::new(vec![
            ScriptGym::step(11.0, 0.0, false, false, 3),
            ScriptGym::step(12.0, 0.0, true, false, 0),
        ]);
        let mut wrapper = FireResetGym::new(gym);

        let obs = wrapper.reset().unwrap();

        assert_eq!(wrapper.gym.actions, vec![1, 2]);
        assert_eq!(wrapper.gym.reset_count, 2);
        assert_eq!(scalar(&obs.observation), 200.0);
    }

    #[test]
    fn episodic_life_marks_life_loss_done_without_real_reset() {
        let gym = ScriptGym::new(vec![
            ScriptGym::step(11.0, 0.0, false, false, 2),
            ScriptGym::step(12.0, 0.0, false, false, 2),
        ]);
        let mut wrapper = EpisodicLifeGym::new(gym, &Device::flex());

        let _ = wrapper.reset().unwrap();
        let step = wrapper
            .step(Tensor::from_data([1i32], &Device::flex()))
            .unwrap();
        let reset_obs = wrapper.reset().unwrap();

        assert!(step.done);
        assert_eq!(wrapper.gym.actions, vec![1, 0]);
        assert_eq!(wrapper.gym.reset_count, 1);
        assert_eq!(scalar(&reset_obs.observation), 12.0);
    }

    #[test]
    fn episodic_life_preserves_complete_episode_statistics() {
        let gym = ScriptGym::new(vec![
            ScriptGym::step(11.0, 1.0, false, false, 2),
            ScriptGym::step(12.0, 0.0, false, false, 2),
            ScriptGym::step(13.0, 2.0, true, false, 0),
        ]);
        let gym = RecordEpisodeStatisticsGym::new(gym);
        let mut wrapper = EpisodicLifeGym::new(gym, &Device::flex());

        wrapper.reset().unwrap();
        let life_loss = wrapper
            .step(Tensor::from_data([1i32], &Device::flex()))
            .unwrap();
        assert!(life_loss.done);
        assert!(life_loss.info.completed_episode.is_none());

        wrapper.reset().unwrap();
        let game_over = wrapper
            .step(Tensor::from_data([1i32], &Device::flex()))
            .unwrap();
        assert!(game_over.done);
        assert_eq!(
            game_over.info.completed_episode,
            Some(EpisodeStatistics {
                episode_return: 3.0,
                episode_length: 3,
            })
        );
    }

    #[test]
    fn episodic_life_real_resets_if_noop_continuation_terminates() {
        let gym = ScriptGym::new(vec![
            ScriptGym::step(11.0, 0.0, false, false, 2),
            ScriptGym::step(12.0, 0.0, true, false, 0),
        ]);
        let mut wrapper = EpisodicLifeGym::new(gym, &Device::flex());

        let _ = wrapper.reset().unwrap();
        let step = wrapper
            .step(Tensor::from_data([1i32], &Device::flex()))
            .unwrap();
        let reset_obs = wrapper.reset().unwrap();

        assert!(step.done);
        assert_eq!(wrapper.gym.actions, vec![1, 0]);
        assert_eq!(wrapper.gym.reset_count, 2);
        assert_eq!(scalar(&reset_obs.observation), 200.0);
    }

    #[test]
    fn episodic_life_treats_truncation_as_real_done_for_reset() {
        let gym = ScriptGym::new(vec![
            ScriptGym::step(11.0, 0.0, false, true, 3),
            ScriptGym::step(12.0, 0.0, false, false, 3),
        ]);
        let mut wrapper = EpisodicLifeGym::new(gym, &Device::flex());

        let _ = wrapper.reset().unwrap();
        let step = wrapper
            .step(Tensor::from_data([1i32], &Device::flex()))
            .unwrap();
        let reset_obs = wrapper.reset().unwrap();

        assert!(!step.done);
        assert!(step.truncated);
        assert_eq!(wrapper.gym.actions, vec![1]);
        assert_eq!(wrapper.gym.reset_count, 2);
        assert_eq!(scalar(&reset_obs.observation), 200.0);
    }
}

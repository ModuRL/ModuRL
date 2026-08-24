use crate::AtariInfo;
use candle_core::{DType, Device, Tensor};
use modurl::{
    gym::{Gym, ResetInfo, StepInfo},
    sampling::sample_u32_inclusive,
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

#[derive(Debug)]
pub enum NoopResetGymError<E> {
    GymError(E),
    CandleError(candle_core::Error),
}

impl<G, I> Gym<I> for NoopResetGym<G>
where
    G: Gym<I>,
{
    type Error = NoopResetGymError<<G as Gym<I>>::Error>;
    type SpaceError = <G as Gym<I>>::SpaceError;

    fn reset(&mut self) -> Result<ResetInfo<I>, Self::Error> {
        let mut reset = self.gym.reset().map_err(NoopResetGymError::GymError)?;
        for _ in 0..sample_u32_inclusive(1, self.noop_max, reset.state.device()).unwrap() {
            let noop_action =
                Tensor::new(0u32, reset.state.device()).map_err(NoopResetGymError::CandleError)?;
            let step = self
                .gym
                .step(noop_action)
                .map_err(NoopResetGymError::GymError)?;
            reset = if step.done || step.truncated {
                self.gym.reset().map_err(NoopResetGymError::GymError)?
            } else {
                ResetInfo {
                    state: step.state,
                    info: step.info,
                }
            };
        }
        Ok(reset)
    }

    /// Forwards one scalar Atari action shaped `[]`.
    fn step(&mut self, action: candle_core::Tensor) -> Result<StepInfo<I>, Self::Error> {
        self.gym.step(action).map_err(NoopResetGymError::GymError)
    }

    fn action_space(&self) -> Box<dyn modurl::spaces::Space<Error = Self::SpaceError>> {
        self.gym.action_space()
    }

    fn observation_space(&self) -> Box<dyn modurl::spaces::Space<Error = Self::SpaceError>> {
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
    device: candle_core::Device,
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

impl<G, I> Gym<I> for EpisodicLifeGym<G>
where
    G: Gym<I>,
    I: AtariLives,
{
    type Error = G::Error;
    type SpaceError = G::SpaceError;

    fn reset(&mut self) -> Result<ResetInfo<I>, Self::Error> {
        let reset = if self.was_real_done {
            self.gym.reset()?
        } else {
            let noop_action = Tensor::new(0u32, &self.device).unwrap();
            let step = self.gym.step(noop_action)?;
            if step.done || step.truncated {
                self.gym.reset()?
            } else {
                ResetInfo {
                    state: step.state,
                    info: step.info,
                }
            }
        };

        self.last_lives = reset.info.lives();
        Ok(reset)
    }

    /// Forwards one scalar Atari action shaped `[]`.
    fn step(&mut self, action: candle_core::Tensor) -> Result<StepInfo<I>, Self::Error> {
        let mut step = self.gym.step(action)?;
        let current_lives = step.info.lives();

        self.was_real_done = step.done || step.truncated;

        if current_lives < self.last_lives {
            step.done = true;
        }

        self.last_lives = current_lives;

        Ok(step)
    }

    fn action_space(&self) -> Box<dyn modurl::spaces::Space<Error = Self::SpaceError>> {
        self.gym.action_space()
    }

    fn observation_space(&self) -> Box<dyn modurl::spaces::Space<Error = Self::SpaceError>> {
        self.gym.observation_space()
    }
}

#[derive(Debug)]
pub enum FireResetGymError<E> {
    GymError(E),
    CandleError(candle_core::Error),
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

impl<G, I> Gym<I> for FireResetGym<G>
where
    G: Gym<I>,
{
    type Error = FireResetGymError<<G as Gym<I>>::Error>;
    type SpaceError = <G as Gym<I>>::SpaceError;

    fn reset(&mut self) -> Result<ResetInfo<I>, Self::Error> {
        let mut reset = self.gym.reset().map_err(FireResetGymError::GymError)?;
        let fire_action =
            Tensor::new(1u32, reset.state.device()).map_err(FireResetGymError::CandleError)?;
        let step = self
            .gym
            .step(fire_action)
            .map_err(FireResetGymError::GymError)?;
        reset = if step.done || step.truncated {
            self.gym.reset().map_err(FireResetGymError::GymError)?
        } else {
            ResetInfo {
                state: step.state,
                info: step.info,
            }
        };

        let second_action =
            Tensor::new(2u32, reset.state.device()).map_err(FireResetGymError::CandleError)?;
        let step = self
            .gym
            .step(second_action)
            .map_err(FireResetGymError::GymError)?;
        reset = if step.done || step.truncated {
            self.gym.reset().map_err(FireResetGymError::GymError)?
        } else {
            ResetInfo {
                state: step.state,
                info: step.info,
            }
        };

        Ok(reset)
    }

    /// Forwards one scalar Atari action shaped `[]`.
    fn step(&mut self, action: candle_core::Tensor) -> Result<StepInfo<I>, Self::Error> {
        self.gym.step(action).map_err(FireResetGymError::GymError)
    }

    fn action_space(&self) -> Box<dyn modurl::spaces::Space<Error = Self::SpaceError>> {
        self.gym.action_space()
    }

    fn observation_space(&self) -> Box<dyn modurl::spaces::Space<Error = Self::SpaceError>> {
        self.gym.observation_space()
    }
}

#[derive(Debug)]
pub enum WarpGymError<E> {
    GymError(E),
    CandleError(candle_core::Error),
}

/// Converts RGB observations to luminance and resizes them to 84×84.
///
/// This produces the compact grayscale input shape used by the standard Atari
/// convolutional policy.
pub struct WarpGym<G> {
    gym: G,
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

impl<G> WarpGym<G> {
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

    /// Converts an Atari observation `[210, 160, 3]` or `[210, 160]` to the
    /// warped grayscale shape `[84, 84]`.
    fn preprocess_observation(&self, obs: &Tensor) -> Result<Tensor, candle_core::Error> {
        let device = obs.device().clone();
        let dims = obs.dims();
        let (height, width) = match dims {
            [height, width] | [height, width, 3] => (*height, *width),
            _ => candle_core::bail!("expected an Atari RGB or grayscale observation, got {dims:?}"),
        };

        if obs.dtype() == DType::U8 {
            let bytes = obs.flatten_all()?.to_vec1::<u8>()?;
            if matches!(dims, [_, _, 3]) && (height, width) == (210, 160) {
                return Tensor::from_vec(
                    Self::resize_atari_rgb_u8(&bytes, height, width),
                    (84, 84),
                    &device,
                );
            }
            let luminance: Vec<f32> = match dims {
                [_, _] => bytes.into_iter().map(f32::from).collect(),
                [_, _, 3] => bytes
                    .chunks_exact(3)
                    .map(|pixel| {
                        pixel[0] as f32 * 0.299 + pixel[1] as f32 * 0.587 + pixel[2] as f32 * 0.114
                    })
                    .collect(),
                _ => unreachable!(),
            };
            let resized = Self::resize_area_data(&luminance, height, width, 84, 84)
                .into_iter()
                .map(|value| value.round().clamp(0.0, 255.0) as u8)
                .collect::<Vec<_>>();
            return Tensor::from_vec(resized, (84, 84), &device);
        }

        let luminance = match dims {
            [height, width] => {
                let _ = (height, width);
                obs.flatten_all()?.to_vec1::<f32>()?
            }
            [_, _, 3] => obs
                .flatten_all()?
                .to_vec1::<f32>()?
                .chunks_exact(3)
                .map(|pixel| pixel[0] * 0.299 + pixel[1] * 0.587 + pixel[2] * 0.114)
                .collect(),
            _ => unreachable!(),
        };
        let resized = Self::resize_area_data(&luminance, height, width, 84, 84);
        Tensor::from_vec(resized, (84, 84), &device)
    }
}

impl<G, I> Gym<I> for WarpGym<G>
where
    G: Gym<I>,
{
    type Error = WarpGymError<<G as Gym<I>>::Error>;
    type SpaceError = <G as Gym<I>>::SpaceError;

    fn reset(&mut self) -> Result<ResetInfo<I>, Self::Error> {
        let mut reset = self.gym.reset().map_err(WarpGymError::GymError)?;
        reset.state = self
            .preprocess_observation(&reset.state)
            .map_err(WarpGymError::CandleError)?;
        Ok(reset)
    }

    /// Forwards one scalar Atari action shaped `[]`.
    fn step(&mut self, action: candle_core::Tensor) -> Result<StepInfo<I>, Self::Error> {
        let mut info = self.gym.step(action).map_err(WarpGymError::GymError)?;
        info.state = self
            .preprocess_observation(&info.state)
            .map_err(WarpGymError::CandleError)?;
        Ok(info)
    }

    fn action_space(&self) -> Box<dyn modurl::spaces::Space<Error = Self::SpaceError>> {
        self.gym.action_space()
    }

    fn observation_space(&self) -> Box<dyn modurl::spaces::Space<Error = Self::SpaceError>> {
        self.gym.observation_space()
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
        state: f32,
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
                device: Device::Cpu,
                reset_count: 0,
                actions: vec![],
                lives: 3,
                steps,
                step_index: 0,
            }
        }

        fn step(state: f32, reward: f32, done: bool, truncated: bool, lives: u32) -> ScriptStep {
            ScriptStep {
                state,
                reward,
                done,
                truncated,
                lives,
            }
        }

        fn tensor(&self, value: f32) -> Tensor {
            Tensor::new(value, &self.device).unwrap()
        }
    }

    impl Gym<AtariInfo> for ScriptGym {
        type Error = candle_core::Error;
        type SpaceError = candle_core::Error;

        /// Steps with one scalar discrete action shaped `[]`.
        fn step(&mut self, action: Tensor) -> Result<StepInfo<AtariInfo>, Self::Error> {
            self.actions.push(action.to_scalar::<u32>()?);
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
                state: self.tensor(step.state),
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
                state: self.tensor(100.0 * self.reset_count as f32),
                info: AtariInfo {
                    lives: self.lives,
                    frame_number: 0,
                    episode_frame_number: 0,
                },
            })
        }

        fn observation_space(&self) -> Box<dyn modurl::spaces::Space<Error = Self::SpaceError>> {
            Box::new(BoxSpace::new(
                Tensor::full(0.0, &[], &self.device).unwrap(),
                Tensor::full(10_000.0, &[], &self.device).unwrap(),
            ))
        }

        fn action_space(&self) -> Box<dyn modurl::spaces::Space<Error = Self::SpaceError>> {
            Box::new(Discrete::new(4))
        }
    }

    /// Reads one scalar tensor shaped `[]`.
    fn scalar(tensor: &Tensor) -> f32 {
        tensor.to_scalar::<f32>().unwrap()
    }

    #[test]
    fn warp_preserves_u8_through_grayscale_and_resize() {
        let wrapper = WarpGym::new(ScriptGym::new(vec![]));
        let rgb = Tensor::zeros((210, 160, 3), DType::U8, &Device::Cpu).unwrap();

        let warped = wrapper.preprocess_observation(&rgb).unwrap();

        assert_eq!(warped.dtype(), DType::U8);
        assert_eq!(warped.dims(), &[84, 84]);
    }

    #[test]
    fn fused_u8_warp_matches_separable_reference() {
        let rgb = (0..210 * 160 * 3)
            .map(|index| ((index * 37 + 11) % 256) as u8)
            .collect::<Vec<_>>();
        let luminance = rgb
            .chunks_exact(3)
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
            .chunks_exact(3)
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
        assert_eq!(scalar(&obs.state), 200.0);
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
        assert_eq!(scalar(&obs.state), 12.0);
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
        assert_eq!(scalar(&obs.state), 22.0);
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
        assert_eq!(scalar(&obs.state), 200.0);
    }

    #[test]
    fn episodic_life_marks_life_loss_done_without_real_reset() {
        let gym = ScriptGym::new(vec![
            ScriptGym::step(11.0, 0.0, false, false, 2),
            ScriptGym::step(12.0, 0.0, false, false, 2),
        ]);
        let mut wrapper = EpisodicLifeGym::new(gym, &Device::Cpu);

        let _ = wrapper.reset().unwrap();
        let step = wrapper
            .step(Tensor::new(1u32, &Device::Cpu).unwrap())
            .unwrap();
        let reset_obs = wrapper.reset().unwrap();

        assert!(step.done);
        assert_eq!(wrapper.gym.actions, vec![1, 0]);
        assert_eq!(wrapper.gym.reset_count, 1);
        assert_eq!(scalar(&reset_obs.state), 12.0);
    }

    #[test]
    fn episodic_life_preserves_complete_episode_statistics() {
        let gym = ScriptGym::new(vec![
            ScriptGym::step(11.0, 1.0, false, false, 2),
            ScriptGym::step(12.0, 0.0, false, false, 2),
            ScriptGym::step(13.0, 2.0, true, false, 0),
        ]);
        let gym = RecordEpisodeStatisticsGym::new(gym);
        let mut wrapper = EpisodicLifeGym::new(gym, &Device::Cpu);

        wrapper.reset().unwrap();
        let life_loss = wrapper
            .step(Tensor::new(1u32, &Device::Cpu).unwrap())
            .unwrap();
        assert!(life_loss.done);
        assert!(life_loss.info.completed_episode.is_none());

        wrapper.reset().unwrap();
        let game_over = wrapper
            .step(Tensor::new(1u32, &Device::Cpu).unwrap())
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
        let mut wrapper = EpisodicLifeGym::new(gym, &Device::Cpu);

        let _ = wrapper.reset().unwrap();
        let step = wrapper
            .step(Tensor::new(1u32, &Device::Cpu).unwrap())
            .unwrap();
        let reset_obs = wrapper.reset().unwrap();

        assert!(step.done);
        assert_eq!(wrapper.gym.actions, vec![1, 0]);
        assert_eq!(wrapper.gym.reset_count, 2);
        assert_eq!(scalar(&reset_obs.state), 200.0);
    }

    #[test]
    fn episodic_life_treats_truncation_as_real_done_for_reset() {
        let gym = ScriptGym::new(vec![
            ScriptGym::step(11.0, 0.0, false, true, 3),
            ScriptGym::step(12.0, 0.0, false, false, 3),
        ]);
        let mut wrapper = EpisodicLifeGym::new(gym, &Device::Cpu);

        let _ = wrapper.reset().unwrap();
        let step = wrapper
            .step(Tensor::new(1u32, &Device::Cpu).unwrap())
            .unwrap();
        let reset_obs = wrapper.reset().unwrap();

        assert!(!step.done);
        assert!(step.truncated);
        assert_eq!(wrapper.gym.actions, vec![1]);
        assert_eq!(wrapper.gym.reset_count, 2);
        assert_eq!(scalar(&reset_obs.state), 200.0);
    }
}

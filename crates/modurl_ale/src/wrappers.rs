use crate::AtariInfo;
use candle_core::{Device, IndexOp, Tensor};
use modurl::{
    gym::{Gym, ResetInfo, StepInfo},
    tensor_operations::gen_range_int_tensor,
};

/// Takes a random number of no-op actions after reset.
///
/// This randomizes Atari starting states so a policy does not overfit to one
/// deterministic opening sequence.
pub struct NoopResetGym<G> {
    gym: G,
    noop_max: u32,
}

impl<G> NoopResetGym<G> {
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
        for _ in 0..gen_range_int_tensor(1, self.noop_max, reset.state.device()).unwrap() {
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

/// Treats a lost life as an episode end while preserving the current game.
///
/// This supplies denser reset boundaries for Atari training without discarding
/// progress from the remaining lives.
pub struct EpisodicLifeGym<G> {
    gym: G,
    last_lives: u8,
    was_real_done: bool,
    device: candle_core::Device,
}

impl<G> EpisodicLifeGym<G>
where
    G: Gym<AtariInfo>,
    <G as Gym<AtariInfo>>::Error: std::fmt::Debug,
{
    pub fn new(gym: G, device: &Device) -> Self {
        Self {
            gym,
            last_lives: 0,
            was_real_done: true, // Start as true so first reset is a real reset
            device: device.clone(),
        }
    }
}

impl<G> Gym<AtariInfo> for EpisodicLifeGym<G>
where
    G: Gym<AtariInfo>,
{
    type Error = <G as Gym<AtariInfo>>::Error;
    type SpaceError = <G as Gym<AtariInfo>>::SpaceError;

    fn reset(&mut self) -> Result<ResetInfo<AtariInfo>, Self::Error> {
        let reset = if self.was_real_done {
            // Real game over, do a full reset
            self.gym.reset()?
        } else {
            // Life lost but game not over, just take a no-op to continue
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

        self.last_lives = reset.info.lives as u8;
        Ok(reset)
    }

    fn step(&mut self, action: candle_core::Tensor) -> Result<StepInfo<AtariInfo>, Self::Error> {
        let mut info = self.gym.step(action)?;
        let current_lives = info.info.lives;

        self.was_real_done = info.done || info.truncated;

        // If lives decreased, mark episode as done (but game may continue)
        if (current_lives as u8) < self.last_lives {
            info.done = true;
        }

        self.last_lives = current_lives as u8;

        Ok(info)
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
    #[allow(dead_code)]
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
            let reset = ResetInfo {
                state: step.state,
                info: step.info,
            };
            self.gym.reset().map_err(FireResetGymError::GymError)?;
            reset
        } else {
            ResetInfo {
                state: step.state,
                info: step.info,
            }
        };

        Ok(reset)
    }

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
    #[allow(dead_code)]
    CandleError(candle_core::Error),
}

/// Converts RGB observations to luminance and resizes them to 84×84.
///
/// This produces the compact grayscale input shape used by the standard Atari
/// convolutional policy.
pub struct WarpGym<G> {
    gym: G,
}

impl<G> WarpGym<G> {
    pub fn new(gym: G) -> Self {
        Self { gym }
    }

    fn resize_observation(&self, obs: Tensor) -> Result<Tensor, candle_core::Error> {
        // Resize from [210, 160] to [84, 84] using area-based interpolation
        // This mimics OpenCV's INTER_AREA which is standard for Atari preprocessing
        self.resize_area_interp(obs, 84, 84)
    }

    fn resize_area_interp(
        &self,
        input: Tensor,
        target_h: usize,
        target_w: usize,
    ) -> Result<Tensor, candle_core::Error> {
        // Get input dimensions
        let shape = input.shape();
        let input_h = shape.dims()[0];
        let input_w = shape.dims()[1];

        // Calculate scaling factors
        let scale_h = input_h as f32 / target_h as f32;
        let scale_w = input_w as f32 / target_w as f32;

        let mut output_data = vec![0.0f32; target_h * target_w];
        let input_data = input.to_vec2::<f32>()?;

        // True area average: source pixels only partially covered by the
        // destination pixel's box are weighted by their covered fraction,
        // matching cv2 INTER_AREA. Equal-weight averaging distorts pixels on
        // feature boundaries by up to ~13% (33/255 measured on Breakout).
        for i in 0..target_h {
            let y_start = i as f32 * scale_h;
            let y_end = (i as f32 + 1.0) * scale_h;
            for j in 0..target_w {
                let x_start = j as f32 * scale_w;
                let x_end = (j as f32 + 1.0) * scale_w;

                let mut sum = 0.0f32;
                let mut weight_total = 0.0f32;

                let mut y = y_start.floor() as usize;
                while (y as f32) < y_end && y < input_h {
                    let wy = (y_end.min((y + 1) as f32) - y_start.max(y as f32)).max(0.0);
                    let mut x = x_start.floor() as usize;
                    while (x as f32) < x_end && x < input_w {
                        let wx = (x_end.min((x + 1) as f32) - x_start.max(x as f32)).max(0.0);
                        let w = wy * wx;
                        sum += input_data[y][x] * w;
                        weight_total += w;
                        x += 1;
                    }
                    y += 1;
                }

                output_data[i * target_w + j] = if weight_total > 0.0 {
                    sum / weight_total
                } else {
                    0.0
                };
            }
        }

        Tensor::from_vec(output_data, (target_h, target_w), input.device())
    }

    fn extract_luminance(&self, obs: &Tensor) -> Result<Tensor, candle_core::Error> {
        if obs.dims().len() == 2 {
            return Ok(obs.clone());
        }
        // Assuming obs is [210, 160, 3] in RGB format
        let r = (obs.i((.., .., 0))? * 0.299)?;
        let g = (obs.i((.., .., 1))? * 0.587)?;
        let b = (obs.i((.., .., 2))? * 0.114)?;
        r.add(&g)?.add(&b)
    }

    fn preprocess_observation(&self, obs: &Tensor) -> Result<Tensor, candle_core::Error> {
        let luminance = self.extract_luminance(obs)?;
        let resized = self.resize_observation(luminance)?;
        Ok(resized)
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
    use modurl::spaces::{BoxSpace, Discrete};

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

    fn scalar(tensor: &Tensor) -> f32 {
        tensor.to_scalar::<f32>().unwrap()
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
    fn fire_reset_resets_after_second_reset_action_done_but_returns_step_obs() {
        let gym = ScriptGym::new(vec![
            ScriptGym::step(11.0, 0.0, false, false, 3),
            ScriptGym::step(12.0, 0.0, true, false, 0),
        ]);
        let mut wrapper = FireResetGym::new(gym);

        let obs = wrapper.reset().unwrap();

        assert_eq!(wrapper.gym.actions, vec![1, 2]);
        assert_eq!(wrapper.gym.reset_count, 2);
        assert_eq!(scalar(&obs.state), 12.0);
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

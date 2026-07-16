use bon::bon;
use candle_core::{Device, Tensor};
use modurl::{
    gym::{Gym, ResetInfo, StepInfo},
    spaces::Space,
};

use crate::{
    MujocoError,
    core::{MujocoCore, validate_noise_scale},
};

const MODEL: &str = include_str!("../assets/half_cheetah.xml");

/// Gymnasium-compatible `HalfCheetah-v5` with the default configuration.
pub struct HalfCheetahV5 {
    core: MujocoCore,
    forward_reward_weight: f64,
    control_cost_weight: f64,
    reset_noise_scale: f64,
    exclude_x: bool,
}

#[bon]
impl HalfCheetahV5 {
    /// Creates an environment. All parameters default to Gymnasium v5 values.
    #[builder]
    pub fn new(
        #[builder(default = &Device::Cpu)] device: &Device,
        #[builder(default = 5)] frame_skip: usize,
        #[builder(default = 1.0)] forward_reward_weight: f64,
        #[builder(default = 0.1)] ctrl_cost_weight: f64,
        #[builder(default = 0.1)] reset_noise_scale: f64,
        #[builder(default = true)] exclude_current_positions_from_observation: bool,
        #[cfg(feature = "rendering")]
        #[builder(default = false)]
        render: bool,
    ) -> Result<Self, MujocoError> {
        #[cfg(not(feature = "rendering"))]
        let render = false;

        validate_noise_scale(reset_noise_scale)?;
        Ok(Self {
            core: MujocoCore::new(MODEL, frame_skip, device, render)?,
            forward_reward_weight,
            control_cost_weight: ctrl_cost_weight,
            reset_noise_scale,
            exclude_x: exclude_current_positions_from_observation,
        })
    }

    /// Re-seeds the reset-noise generator.
    pub fn seed(&mut self, seed: u64) {
        self.core.seed(seed);
    }

    /// Sets an exact MuJoCo state, useful for reproducible evaluation and parity tests.
    pub fn set_state(&mut self, qpos: &[f64], qvel: &[f64]) -> Result<Tensor, MujocoError> {
        self.core.set_state(qpos, qvel)?;
        self.core.render()?;
        self.observation()
    }

    fn observation(&self) -> Result<Tensor, MujocoError> {
        self.core
            .tensor(&self.core.observation(self.exclude_x, false))
    }
}

impl Gym for HalfCheetahV5 {
    type Error = MujocoError;
    type SpaceError = candle_core::Error;

    fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
        self.core.reset_half_cheetah(self.reset_noise_scale)?;
        self.core.render()?;
        Ok(ResetInfo {
            state: self.observation()?,
            info: (),
        })
    }

    fn step(&mut self, action: Tensor) -> Result<StepInfo, Self::Error> {
        let x_before = self.core.qpos()[0];
        let action = self.core.step(&action)?;
        let x_velocity = (self.core.qpos()[0] - x_before) / self.core.dt();
        let control_cost = self.control_cost_weight * action.iter().map(|x| x * x).sum::<f64>();
        let reward = self.forward_reward_weight * x_velocity - control_cost;
        self.core.render()?;
        Ok(StepInfo {
            state: self.observation()?,
            reward: reward as f32,
            done: false,
            truncated: false,
            info: (),
        })
    }

    fn observation_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
        self.core.observation_space(self.exclude_x)
    }

    fn action_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
        self.core.action_space()
    }
}

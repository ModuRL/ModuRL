use bon::bon;
use candle_core::{Device, Tensor};
use modurl::{
    gym::{Gym, ResetInfo, StepInfo},
    spaces::Space,
};

use crate::{
    MujocoError,
    core::{MujocoCore, validate_noise_scale, validate_range},
};

const MODEL: &str = include_str!("../assets/walker2d.xml");

/// Gymnasium-compatible `Walker2d-v5` with the default configuration.
pub struct Walker2dV5 {
    core: MujocoCore,
    forward_reward_weight: f64,
    control_cost_weight: f64,
    healthy_reward: f64,
    terminate_when_unhealthy: bool,
    healthy_z_range: (f64, f64),
    healthy_angle_range: (f64, f64),
    reset_noise_scale: f64,
    exclude_x: bool,
}

#[bon]
impl Walker2dV5 {
    /// Creates an environment. All parameters default to Gymnasium v5 values.
    #[builder]
    pub fn new(
        #[builder(default = &Device::Cpu)] device: &Device,
        #[builder(default = 4)] frame_skip: usize,
        #[builder(default = 1.0)] forward_reward_weight: f64,
        #[builder(default = 1e-3)] ctrl_cost_weight: f64,
        #[builder(default = 1.0)] healthy_reward: f64,
        #[builder(default = true)] terminate_when_unhealthy: bool,
        #[builder(default = (0.8, 2.0))] healthy_z_range: (f64, f64),
        #[builder(default = (-1.0, 1.0))] healthy_angle_range: (f64, f64),
        #[builder(default = 0.005)] reset_noise_scale: f64,
        #[builder(default = true)] exclude_current_positions_from_observation: bool,
        #[cfg(feature = "rendering")]
        #[builder(default = false)]
        render: bool,
    ) -> Result<Self, MujocoError> {
        #[cfg(not(feature = "rendering"))]
        let render = false;

        validate_noise_scale(reset_noise_scale)?;
        validate_range("healthy_z_range", healthy_z_range)?;
        validate_range("healthy_angle_range", healthy_angle_range)?;
        Ok(Self {
            core: MujocoCore::new(MODEL, frame_skip, device, render)?,
            forward_reward_weight,
            control_cost_weight: ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            healthy_angle_range,
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

    fn is_healthy(&self) -> bool {
        let qpos = self.core.qpos();
        self.healthy_z_range.0 < qpos[1]
            && qpos[1] < self.healthy_z_range.1
            && self.healthy_angle_range.0 < qpos[2]
            && qpos[2] < self.healthy_angle_range.1
    }

    fn observation(&self) -> Result<Tensor, MujocoError> {
        self.core
            .tensor(&self.core.observation(self.exclude_x, true))
    }
}

impl Gym for Walker2dV5 {
    type Error = MujocoError;
    type SpaceError = candle_core::Error;

    fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
        self.core.reset_uniform(self.reset_noise_scale)?;
        self.core.render()?;
        Ok(ResetInfo {
            state: self.observation()?,
            info: (),
        })
    }

    /// Steps with one continuous actuator vector shaped `[action_size]`.
    fn step(&mut self, action: Tensor) -> Result<StepInfo, Self::Error> {
        let x_before = self.core.qpos()[0];
        let action = self.core.step(&action)?;
        let x_velocity = (self.core.qpos()[0] - x_before) / self.core.dt();
        let healthy = self.is_healthy();
        let control_cost = self.control_cost_weight * action.iter().map(|x| x * x).sum::<f64>();
        let reward = self.forward_reward_weight * x_velocity
            + if healthy { self.healthy_reward } else { 0.0 }
            - control_cost;
        self.core.render()?;
        Ok(StepInfo {
            state: self.observation()?,
            reward: reward as f32,
            done: !healthy && self.terminate_when_unhealthy,
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

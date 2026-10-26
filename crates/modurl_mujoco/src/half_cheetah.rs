use bon::bon;
use burn::tensor::Tensor;
use modurl::{
    gym::{Gym, ResetInfo, StepInfo},
    spaces::BoxSpace,
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
            core: MujocoCore::new(MODEL, frame_skip, render)?,
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
    /// Accepts host positions `[nq]` and velocities `[nv]`; returns unbatched F32 observations `[observation_size]` on CPU Flex.
    pub fn set_state(&mut self, qpos: &[f64], qvel: &[f64]) -> Result<Tensor<1>, MujocoError> {
        self.core.set_state(qpos, qvel)?;
        self.core.render()?;
        self.observation()
    }

    /// Returns unbatched F32 observations `[observation_size]` on CPU Flex, preserving the configured observation component order.
    fn observation(&self) -> Result<Tensor<1>, MujocoError> {
        Ok(self
            .core
            .tensor(&self.core.observation(self.exclude_x, false)))
    }
}

impl Gym for HalfCheetahV5 {
    type Error = MujocoError;
    type ObservationSpace = BoxSpace<2>;
    type ActionSpace = BoxSpace<2>;

    /// Returns unbatched F32 observations `[observation_size]` on CPU Flex with reset metadata.
    fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
        self.core
            .reset_uniform_positions_normal_velocities(self.reset_noise_scale)?;
        self.core.render()?;
        Ok(ResetInfo {
            observation: self.observation()?,
            info: (),
        })
    }

    /// Steps with one continuous actuator vector shaped `[action_size]`.
    /// Accepts Float actions on CPU Flex; returns unbatched F32 observations `[observation_size]` on CPU Flex and an F32 reward.
    fn step(&mut self, action: Tensor<1>) -> Result<StepInfo, Self::Error> {
        let x_before = self.core.qpos()[0];
        let action = self.core.step(&action)?;
        let x_velocity = (self.core.qpos()[0] - x_before) / self.core.dt();
        let control_cost = self.control_cost_weight * action.iter().map(|x| x * x).sum::<f64>();
        let reward = self.forward_reward_weight * x_velocity - control_cost;
        self.core.render()?;
        Ok(StepInfo {
            observation: self.observation()?,
            reward: reward as f32,
            done: false,
            truncated: false,
            info: (),
        })
    }

    fn observation_space(&self) -> BoxSpace<2> {
        self.core.observation_space(self.exclude_x)
    }

    fn action_space(&self) -> BoxSpace<2> {
        self.core.action_space()
    }
}

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

const MODEL: &str = include_str!("../assets/ant.xml");

/// Gymnasium-compatible `Ant-v5` with the default model.
pub struct AntV5 {
    core: MujocoCore,
    forward_reward_weight: f64,
    control_cost_weight: f64,
    contact_cost_weight: f64,
    healthy_reward: f64,
    terminate_when_unhealthy: bool,
    healthy_z_range: (f64, f64),
    contact_force_range: (f64, f64),
    reset_noise_scale: f64,
    exclude_xy: bool,
    include_contact_forces: bool,
}

#[bon]
impl AntV5 {
    /// Creates an environment. All parameters default to Gymnasium v5 values.
    #[builder]
    pub fn new(
        #[builder(default = &Device::Cpu)] device: &Device,
        #[builder(default = 5)] frame_skip: usize,
        #[builder(default = 1.0)] forward_reward_weight: f64,
        #[builder(default = 0.5)] ctrl_cost_weight: f64,
        #[builder(default = 5e-4)] contact_cost_weight: f64,
        #[builder(default = 1.0)] healthy_reward: f64,
        #[builder(default = true)] terminate_when_unhealthy: bool,
        #[builder(default = (0.2, 1.0))] healthy_z_range: (f64, f64),
        #[builder(default = (-1.0, 1.0))] contact_force_range: (f64, f64),
        #[builder(default = 0.1)] reset_noise_scale: f64,
        #[builder(default = true)] exclude_current_positions_from_observation: bool,
        #[builder(default = true)] include_cfrc_ext_in_observation: bool,
        #[cfg(feature = "rendering")]
        #[builder(default = false)]
        render: bool,
    ) -> Result<Self, MujocoError> {
        #[cfg(not(feature = "rendering"))]
        let render = false;

        validate_noise_scale(reset_noise_scale)?;
        validate_range("healthy_z_range", healthy_z_range)?;
        validate_range("contact_force_range", contact_force_range)?;
        Ok(Self {
            core: MujocoCore::new(MODEL, frame_skip, device, render)?,
            forward_reward_weight,
            control_cost_weight: ctrl_cost_weight,
            contact_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            contact_force_range,
            reset_noise_scale,
            exclude_xy: exclude_current_positions_from_observation,
            include_contact_forces: include_cfrc_ext_in_observation,
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
        let z = self.core.qpos()[2];
        self.core
            .qpos()
            .iter()
            .chain(self.core.qvel())
            .all(|value| value.is_finite())
            && self.healthy_z_range.0 <= z
            && z <= self.healthy_z_range.1
    }

    fn contact_cost(&self) -> f64 {
        let (minimum, maximum) = self.contact_force_range;
        self.contact_cost_weight
            * self
                .core
                .cfrc_ext()
                .iter()
                .flatten()
                .map(|force| force.clamp(minimum, maximum).powi(2))
                .sum::<f64>()
    }

    fn observation_size(&self) -> usize {
        let positions = self.core.nq() - if self.exclude_xy { 2 } else { 0 };
        let contacts = if self.include_contact_forces {
            (self.core.nbody() - 1) * 6
        } else {
            0
        };
        positions + self.core.nv() + contacts
    }

    fn observation(&self) -> Result<Tensor, MujocoError> {
        let mut observation = Vec::with_capacity(self.observation_size());
        observation.extend_from_slice(&self.core.qpos()[if self.exclude_xy { 2 } else { 0 }..]);
        observation.extend_from_slice(self.core.qvel());
        if self.include_contact_forces {
            let (minimum, maximum) = self.contact_force_range;
            observation.extend(
                self.core.cfrc_ext()[1..]
                    .iter()
                    .flatten()
                    .map(|force| force.clamp(minimum, maximum)),
            );
        }
        self.core.tensor(&observation)
    }
}

impl Gym for AntV5 {
    type Error = MujocoError;
    type SpaceError = candle_core::Error;

    fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
        self.core
            .reset_uniform_positions_normal_velocities(self.reset_noise_scale)?;
        self.core.render()?;
        Ok(ResetInfo {
            state: self.observation()?,
            info: (),
        })
    }

    /// Steps with one continuous actuator vector shaped `[8]`.
    fn step(&mut self, action: Tensor) -> Result<StepInfo, Self::Error> {
        let x_before = self.core.body_position(1)[0];
        let action = self.core.step(&action)?;
        let x_velocity = (self.core.body_position(1)[0] - x_before) / self.core.dt();
        let healthy = self.is_healthy();
        let control_cost = self.control_cost_weight * action.iter().map(|x| x * x).sum::<f64>();
        let reward = self.forward_reward_weight * x_velocity
            + if healthy { self.healthy_reward } else { 0.0 }
            - control_cost
            - self.contact_cost();
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
        self.core
            .unbounded_observation_space(self.observation_size())
    }

    fn action_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
        self.core.action_space()
    }
}

//! Load an MJCF file and provide a task-specific observation and reward.

use std::path::Path;

use candle_core::{Device, Tensor};
use modurl::{
    gym::{Gym, ResetInfo, StepInfo},
    spaces::Space,
};

use crate::{MujocoError, core::MujocoCore};

/// Simulator state available to task callbacks. Body positions use MuJoCo body indices.
#[derive(Debug, Clone, Default)]
pub struct MujocoState {
    pub contact_substeps: Vec<Vec<[usize; 2]>>,
    pub physics_timestep: f64,
    pub subtree_angular_momenta: Vec<[f64; 3]>,
    pub joint_position_limits: Vec<[f64; 2]>,
    pub qpos: Vec<f64>,
    pub qvel: Vec<f64>,
    pub controls: Vec<f64>,
    pub actuator_forces: Vec<f64>,
    pub sensor_data: Vec<f64>,
    pub body_positions: Vec<[f64; 3]>,
    /// World-frame force from contacts with the world body, indexed by body.
    pub world_contact_forces: Vec<[f64; 3]>,
    pub contact_body_pairs: Vec<[usize; 2]>,
    pub body_parent_ids: Vec<usize>,
    pub site_positions: Vec<[f64; 3]>,
    pub site_linear_velocities: Vec<[f64; 3]>,
    pub site_body_ids: Vec<usize>,
}

impl MujocoState {
    fn capture(core: &MujocoCore) -> Self {
        Self {
            contact_substeps: core.contact_substeps(),
            physics_timestep: core.physics_timestep(),
            subtree_angular_momenta: core.subtree_angular_momenta(),
            joint_position_limits: core.joint_position_limits(),
            qpos: core.qpos().to_vec(),
            qvel: core.qvel().to_vec(),
            controls: core.controls().to_vec(),
            actuator_forces: core.actuator_forces().to_vec(),
            sensor_data: core.sensor_data().to_vec(),
            body_positions: (0..core.nbody())
                .map(|index| core.body_position(index))
                .collect(),
            world_contact_forces: core.world_contact_forces(),
            contact_body_pairs: core.contact_body_pairs(),
            body_parent_ids: core.body_parent_ids(),
            site_positions: core.site_positions(),
            site_linear_velocities: core.site_linear_velocities(),
            site_body_ids: core.site_body_ids(),
        }
    }
}

/// Task output after one MuJoCo control step.
#[derive(Debug, Clone, Copy)]
pub struct TaskStep {
    pub reward: f32,
    pub done: bool,
    pub truncated: bool,
}

/// Define observations, rewards, and episode boundaries for an arbitrary MJCF model.
/// A task may hold episode state such as a sampled goal.
pub trait MujocoTask {
    /// Optional policy space and control mapping. Rewards receive original actions.
    fn action_dim(&self) -> Option<usize> { None }
    fn physical_control_targets(&self) -> bool { false }
    fn map_action(&self, action: &Tensor) -> Result<Tensor, candle_core::Error> { Ok(action.clone()) }

    /// Optionally edits generalized position or velocity after the simulator's
    /// default reset and before the task observes the new episode.
    fn reset_state(&mut self, _state: &mut MujocoState) {}
    fn reset(&mut self, _state: &MujocoState) {}
    fn randomization_step(&self) -> usize { 0 }
    /// Optionally edits generalized position or velocity immediately before a
    /// control step. This supports task disturbances such as push impulses.
    fn before_step(&mut self, _state: &mut MujocoState) {}
    fn observation(&self, state: &MujocoState) -> Vec<f64>;
    fn transition(
        &mut self,
        previous: &MujocoState,
        next: &MujocoState,
        action: &[f64],
    ) -> TaskStep;
}

/// A `Gym` environment backed by a user-supplied MJCF XML and task.
pub struct CustomMujoco<T: MujocoTask> {
    core: MujocoCore,
    task: T,
    observation_dim: usize,
    action_dim: usize,
}

impl<T: MujocoTask> CustomMujoco<T> {
    /// Loads XML from disk, including assets resolved relative to that XML.
    pub fn from_xml_path(
        path: impl AsRef<Path>,
        frame_skip: usize,
        device: &Device,
        task: T,
        observation_dim: usize,
    ) -> Result<Self, MujocoError> {
        if observation_dim == 0 {
            return Err(MujocoError::InvalidInput(
                "observation_dim must be nonzero".into(),
            ));
        }
        Self::from_xml_path_with_rendering(path, frame_skip, device, task, observation_dim, false)
    }

    /// Loads XML with Microduck-compatible per-environment physics randomization.
    pub fn from_xml_path_with_randomization(
        path: impl AsRef<Path>,
        frame_skip: usize,
        device: &Device,
        task: T,
        observation_dim: usize,
    ) -> Result<Self, MujocoError> {
        let core = MujocoCore::from_xml_path_with_randomization(path.as_ref(), frame_skip, device)?;
        let action_dim = task.action_dim().unwrap_or(core.nu());
        let env = Self { core, task, observation_dim, action_dim };
        env.make_observation()?;
        Ok(env)
    }

    /// Loads XML and opens an interactive viewer when `render` is true.
    pub fn from_xml_path_with_rendering(
        path: impl AsRef<Path>,
        frame_skip: usize,
        device: &Device,
        task: T,
        observation_dim: usize,
        render: bool,
    ) -> Result<Self, MujocoError> {
        if observation_dim == 0 {
            return Err(MujocoError::InvalidInput(
                "observation_dim must be nonzero".into(),
            ));
        }
        let core = MujocoCore::from_xml_path(path.as_ref(), frame_skip, device, render)?;
        let env = Self {
            action_dim: task.action_dim().unwrap_or(core.nu()),
            core,
            task,
            observation_dim,
        };
        env.make_observation()?;
        Ok(env)
    }

    /// Returns false after the interactive viewer window closes.
    pub fn viewer_running(&self) -> bool {
        self.core.viewer_running()
    }

    pub fn seed(&mut self, seed: u64) {
        self.core.seed(seed);
    }

    pub fn actuator_count(&self) -> usize {
        self.core.nu()
    }

    fn make_observation(&self) -> Result<Tensor, MujocoError> {
        let values = self.task.observation(&MujocoState::capture(&self.core));
        if values.len() != self.observation_dim || !values.iter().all(|value| value.is_finite()) {
            return Err(MujocoError::InvalidInput(format!(
                "task observation must contain {} finite values, got {}",
                self.observation_dim,
                values.len()
            )));
        }
        self.core.tensor(&values)
    }
}

impl<T: MujocoTask> Gym for CustomMujoco<T> {
    type Error = MujocoError;
    type SpaceError = candle_core::Error;

    fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
        self.core.randomize_nano_owl_physics(self.task.randomization_step());
        self.core.reset_uniform(0.0)?;
        let mut state = MujocoState::capture(&self.core);
        let original_qpos = state.qpos.clone();
        let original_qvel = state.qvel.clone();
        self.task.reset_state(&mut state);
        if state.qpos != original_qpos || state.qvel != original_qvel {
            self.core.set_task_state(&state.qpos, &state.qvel)?;
            state = MujocoState::capture(&self.core);
        }
        self.task.reset(&state);
        self.core.render()?;
        Ok(ResetInfo {
            state: self.make_observation()?,
            info: (),
        })
    }

    fn step(&mut self, action: Tensor) -> Result<StepInfo, Self::Error> {
        let mut previous = MujocoState::capture(&self.core);
        let original_qpos = previous.qpos.clone();
        let original_qvel = previous.qvel.clone();
        self.task.before_step(&mut previous);
        if previous.qpos != original_qpos || previous.qvel != original_qvel {
            self.core.set_task_state(&previous.qpos, &previous.qvel)?;
            previous = MujocoState::capture(&self.core);
        }
        let action_values = action.to_dtype(candle_core::DType::F64)?.to_vec1::<f64>()?;
        if action_values.len() != self.action_dim || !action_values.iter().all(|a| a.is_finite()) {
            return Err(MujocoError::InvalidInput("invalid policy action shape or nonfinite action".into()));
        }
        let mapped = self.task.map_action(&action)?;
        if self.task.physical_control_targets() {
            self.core.step_physical(&mapped)?;
        } else {
            self.core.step_normalized(&mapped)?;
        }
        let next = MujocoState::capture(&self.core);
        let outcome = self.task.transition(&previous, &next, &action_values);
        self.core.render()?;
        if !outcome.reward.is_finite() {
            return Err(MujocoError::InvalidInput(
                "task reward must be finite".into(),
            ));
        }
        Ok(StepInfo {
            state: self.make_observation()?,
            reward: outcome.reward,
            done: outcome.done,
            truncated: outcome.truncated,
            info: (),
        })
    }

    fn observation_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
        self.core.unbounded_observation_space(self.observation_dim)
    }

    fn action_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
        if self.task.physical_control_targets() {
            self.core.unbounded_action_space_for_dim(self.action_dim)
        } else {
            self.core.action_space_for_dim(self.action_dim)
        }
    }
}

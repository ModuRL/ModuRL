//! Load an MJCF file and provide a task-specific observation and reward.

use std::path::Path;

use bon::bon;
use candle_core::{Device, Tensor};
use modurl::{
    gym::{Gym, ResetInfo, StepInfo},
    spaces::Space,
};

use crate::{MjModel, MujocoError, core::MujocoCore};

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
    fn action_dim(&self) -> Option<usize> {
        None
    }
    fn physical_control_targets(&self) -> bool {
        false
    }
    /// Maps a policy action of shape `[action_dim]` to actuator controls of shape
    /// `[nu]`. By default, `action_dim` is the model actuator count `nu`.
    fn map_action(&self, action: &Tensor) -> Result<Tensor, candle_core::Error> {
        Ok(action.clone())
    }

    /// Optionally edits generalized position or velocity after the simulator's
    /// default reset and before the task observes the new episode.
    fn reset_state(&mut self, _state: &mut MujocoState) {}
    fn reset(&mut self, _state: &MujocoState) {}
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

#[bon]
impl<T: MujocoTask> CustomMujoco<T> {
    /// Loads XML and relative assets, sharing an immutable compiled model with
    /// other environments loaded from the same path. Each owns its simulation state.
    #[builder]
    pub fn new(
        path: impl AsRef<Path>,
        task: T,
        observation_dim: usize,
        #[builder(default = 1)] frame_skip: usize,
        #[builder(default = &Device::Cpu)] device: &Device,
        #[cfg(feature = "rendering")]
        #[builder(default = false)]
        render: bool,
    ) -> Result<Self, MujocoError> {
        #[cfg(not(feature = "rendering"))]
        let render = false;

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

    /// Read the current MuJoCo model, including name lookup and physics settings.
    pub fn model(&self) -> &MjModel {
        self.core.model()
    }

    /// Edit the current model between steps, without resetting the episode.
    ///
    /// Edits run on a private candidate and persist across resets. An error or
    /// panic in the closure leaves the live model unchanged. Model sharing is
    /// retained until the first successful edit. Each edit clones the model.
    /// Derived constants and current-state quantities are recomputed on success;
    /// time, positions, velocities, controls, and actuator activation are preserved.
    /// Edits to `qpos0` become the default pose for subsequent resets. An open
    /// viewer reloads the edited model without replacing its window; a closed
    /// viewer stays closed.
    ///
    /// Use MuJoCo's runtime-editable numerical parameters. Structural changes
    /// (a different compiled signature) are rejected. The caller is responsible
    /// for valid MuJoCo parameter values and combinations; this API does not
    /// validate every field. Randomness, scheduling, and baselines belong to the
    /// caller. Rebuild the environment to change model structure or render assets.
    pub fn edit_model(
        &mut self,
        edit: impl FnOnce(&mut MjModel) -> Result<(), MujocoError>,
    ) -> Result<(), MujocoError> {
        self.core.edit_model(edit)
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

    /// Steps with a floating policy action of shape `[action_dim]`; the task
    /// maps it to the model's `[nu]` actuator controls.
    fn step(&mut self, action: Tensor) -> Result<StepInfo, Self::Error> {
        if action.rank() != 1 || !action.dtype().is_float() {
            return Err(MujocoError::InvalidInput(
                "invalid policy action shape or dtype".into(),
            ));
        }
        let action_values = action.to_dtype(candle_core::DType::F64)?.to_vec1::<f64>()?;
        if action_values.len() != self.action_dim || !action_values.iter().all(|a| a.is_finite()) {
            return Err(MujocoError::InvalidInput(
                "invalid policy action shape or nonfinite action".into(),
            ));
        }
        let mapped = self.task.map_action(&action)?;
        let physical = self.task.physical_control_targets();
        if mapped.rank() != 1 || mapped.dims()[0] != self.core.nu() || !mapped.dtype().is_float() {
            return Err(MujocoError::InvalidInput(
                "invalid mapped action shape or dtype".into(),
            ));
        }
        let mapped_values = mapped.to_dtype(candle_core::DType::F64)?.to_vec1::<f64>()?;
        if !mapped_values.iter().all(|value| {
            value.is_finite() && (physical || (-1.0 - 1e-6..=1.0 + 1e-6).contains(value))
        }) {
            return Err(MujocoError::InvalidInput(
                "invalid mapped action value".into(),
            ));
        }
        let mut previous = MujocoState::capture(&self.core);
        let original_qpos = previous.qpos.clone();
        let original_qvel = previous.qvel.clone();
        self.task.before_step(&mut previous);
        if previous.qpos != original_qpos || previous.qvel != original_qvel {
            self.core.set_task_state(&previous.qpos, &previous.qvel)?;
            previous = MujocoState::capture(&self.core);
        }
        if physical {
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

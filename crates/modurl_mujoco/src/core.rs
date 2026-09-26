use candle_core::{DType, Device, Tensor};
use modurl::spaces::{BoxSpace, Space};
use mujoco_rs::prelude::{MjData, MjModel, MjtObj};
#[cfg(feature = "rendering")]
use mujoco_rs::viewer::MjViewer;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::StandardNormal;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock, Weak},
};

use crate::MujocoError;

// Weak entries let a model disappear when the last environment using it is dropped.
// Hold the mutex during compilation so simultaneous constructors compile only once.
fn shared_xml_model(xml: &str) -> Result<Arc<MjModel>, MujocoError> {
    static MODELS: OnceLock<Mutex<HashMap<String, Weak<MjModel>>>> = OnceLock::new();
    let mut models = MODELS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap();
    if let Some(model) = models.get(xml).and_then(Weak::upgrade) {
        return Ok(model);
    }
    let model = Arc::new(MjModel::from_xml_string(xml)?);
    models.insert(xml.to_owned(), Arc::downgrade(&model));
    Ok(model)
}

fn shared_path_model(path: &Path) -> Result<Arc<MjModel>, MujocoError> {
    static MODELS: OnceLock<Mutex<HashMap<PathBuf, Weak<MjModel>>>> = OnceLock::new();
    // Preserve MuJoCo's original error for paths that cannot be canonicalized.
    let Ok(key) = path.canonicalize() else {
        return Ok(Arc::new(MjModel::from_xml(path)?));
    };
    let mut models = MODELS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap();
    if let Some(model) = models.get(&key).and_then(Weak::upgrade) {
        return Ok(model);
    }
    let model = Arc::new(MjModel::from_xml(&key)?);
    models.insert(key, Arc::downgrade(&model));
    Ok(model)
}

pub(crate) struct MujocoCore {
    data: MjData<Arc<MjModel>>,
    contact_substeps: Vec<Vec<[usize; 2]>>,
    initial_qpos: Vec<f64>,
    initial_qvel: Vec<f64>,
    frame_skip: usize,
    device: Device,
    rng: StdRng,
    #[cfg(feature = "rendering")]
    viewer: Option<MjViewer>,
}

impl MujocoCore {
    pub(crate) fn new(
        xml: &str,
        frame_skip: usize,
        device: &Device,
        render: bool,
    ) -> Result<Self, MujocoError> {
        Self::from_model(shared_xml_model(xml)?, frame_skip, device, render)
    }

    pub(crate) fn from_xml_path(
        path: &Path,
        frame_skip: usize,
        device: &Device,
        render: bool,
    ) -> Result<Self, MujocoError> {
        Self::from_model(shared_path_model(path)?, frame_skip, device, render)
    }

    fn from_model(
        model: Arc<MjModel>,
        frame_skip: usize,
        device: &Device,
        render: bool,
    ) -> Result<Self, MujocoError> {
        if frame_skip == 0 {
            return Err(MujocoError::InvalidInput(
                "frame_skip must be greater than zero".into(),
            ));
        }
        let data = MjData::new(model);
        let initial_qpos = data.qpos().to_vec();
        let initial_qvel = data.qvel().to_vec();
        #[cfg(feature = "rendering")]
        let viewer = if render {
            Some(
                MjViewer::builder()
                    .max_user_geoms(0)
                    .build_passive(data.model())?,
            )
        } else {
            None
        };
        #[cfg(not(feature = "rendering"))]
        let _ = render;
        Ok(Self {
            data,
            contact_substeps: Vec::new(),
            initial_qpos,
            initial_qvel,
            frame_skip,
            device: device.clone(),
            rng: rand::make_rng(),
            #[cfg(feature = "rendering")]
            viewer,
        })
    }

    pub(crate) fn seed(&mut self, seed: u64) {
        self.rng = StdRng::seed_from_u64(seed);
    }

    pub(crate) fn qpos(&self) -> &[f64] {
        self.data.qpos()
    }

    pub(crate) fn qvel(&self) -> &[f64] {
        self.data.qvel()
    }

    pub(crate) fn set_task_state(&mut self, qpos: &[f64], qvel: &[f64]) -> Result<(), MujocoError> {
        if qpos.len() != self.nq() || qvel.len() != self.nv() {
            return Err(MujocoError::InvalidInput(format!(
                "state shape mismatch: expected qpos ({},) and qvel ({},), got ({},) and ({},)",
                self.nq(),
                self.nv(),
                qpos.len(),
                qvel.len()
            )));
        }
        if !qpos.iter().chain(qvel).all(|value| value.is_finite()) {
            return Err(MujocoError::InvalidInput(
                "qpos and qvel must contain only finite values".into(),
            ));
        }
        // Preserve time, actuator activation, controls, and history during an
        // in-episode disturbance; only recompute quantities derived from qpos/qvel.
        self.data.qpos_mut().copy_from_slice(qpos);
        self.data.qvel_mut().copy_from_slice(qvel);
        self.contact_substeps.clear();
        self.data.forward();
        self.data.subtree_vel();
        Ok(())
    }

    pub(crate) fn nq(&self) -> usize {
        self.data.qpos().len()
    }

    pub(crate) fn nv(&self) -> usize {
        self.data.qvel().len()
    }

    pub(crate) fn sensor_data(&self) -> &[f64] {
        self.data.sensordata()
    }

    pub(crate) fn controls(&self) -> &[f64] {
        self.data.ctrl()
    }

    pub(crate) fn actuator_forces(&self) -> &[f64] {
        self.data.actuator_force()
    }

    pub(crate) fn nu(&self) -> usize {
        self.data.ctrl().len()
    }

    pub(crate) fn nbody(&self) -> usize {
        self.data.cfrc_ext().len()
    }

    pub(crate) fn body_position(&self, body_index: usize) -> [f64; 3] {
        self.data.xpos()[body_index]
    }

    /// Body IDs for each active MuJoCo contact, including ground contacts.
    pub(crate) fn contact_body_pairs(&self) -> Vec<[usize; 2]> {
        let geom_bodies = self.data.model().geom_bodyid();
        self.data
            .contact()
            .iter()
            .filter_map(|contact| {
                let geom1 = usize::try_from(contact.geom1).ok()?;
                let geom2 = usize::try_from(contact.geom2).ok()?;
                Some([
                    usize::try_from(*geom_bodies.get(geom1)?).ok()?,
                    usize::try_from(*geom_bodies.get(geom2)?).ok()?,
                ])
            })
            .collect()
    }

    /// Sum world-frame contact forces exerted by the world body on each body.
    /// MuJoCo reports each contact's force in its contact frame, acting on geom2.
    pub(crate) fn world_contact_forces(&self) -> Vec<[f64; 3]> {
        let geom_bodies = self.data.model().geom_bodyid();
        let mut forces = vec![[0.0; 3]; self.nbody()];
        for (index, contact) in self.data.contact().iter().enumerate() {
            let Ok(geom1) = usize::try_from(contact.geom1) else {
                continue;
            };
            let Ok(geom2) = usize::try_from(contact.geom2) else {
                continue;
            };
            let Some(&body1) = geom_bodies.get(geom1) else {
                continue;
            };
            let Some(&body2) = geom_bodies.get(geom2) else {
                continue;
            };
            let (body, sign) = match (body1, body2) {
                (0, other) if other > 0 => (other as usize, 1.0),
                (other, 0) if other > 0 => (other as usize, -1.0),
                _ => continue,
            };
            let force = self.data.contact_force(index);
            for axis in 0..3 {
                forces[body][axis] += sign
                    * (0..3)
                        .map(|component| force[component] * contact.frame[component * 3 + axis])
                        .sum::<f64>();
            }
        }
        forces
    }

    pub(crate) fn body_parent_ids(&self) -> Vec<usize> {
        self.data
            .model()
            .body_parentid()
            .iter()
            .map(|&id| usize::try_from(id).unwrap_or(0))
            .collect()
    }

    pub(crate) fn contact_substeps(&self) -> Vec<Vec<[usize; 2]>> {
        self.contact_substeps.clone()
    }
    pub(crate) fn physics_timestep(&self) -> f64 {
        self.data.model().opt().timestep
    }

    pub(crate) fn subtree_angular_momenta(&self) -> Vec<[f64; 3]> {
        self.data.subtree_angmom().to_vec()
    }

    pub(crate) fn joint_position_limits(&self) -> Vec<[f64; 2]> {
        let model = self.data.model();
        model
            .jnt_range()
            .iter()
            .zip(model.jnt_type())
            .filter_map(|(range, kind)| {
                // Hinge/slide only; exclude free and ball joints.
                if *kind as i32 >= 2 {
                    Some(*range)
                } else {
                    None
                }
            })
            .collect()
    }

    pub(crate) fn site_positions(&self) -> Vec<[f64; 3]> {
        self.data.site_xpos().to_vec()
    }

    pub(crate) fn site_linear_velocities(&self) -> Vec<[f64; 3]> {
        (0..self.data.site_xpos().len())
            .map(|id| {
                let velocity = self.data.object_velocity(MjtObj::mjOBJ_SITE, id, false);
                [velocity[3], velocity[4], velocity[5]]
            })
            .collect()
    }

    pub(crate) fn site_body_ids(&self) -> Vec<usize> {
        self.data
            .model()
            .site_bodyid()
            .iter()
            .map(|&id| usize::try_from(id).unwrap_or(0))
            .collect()
    }

    pub(crate) fn tendon_lengths(&self) -> &[f64] {
        self.data.ten_length()
    }

    pub(crate) fn tendon_velocities(&self) -> &[f64] {
        self.data.ten_velocity()
    }

    pub(crate) fn dt(&self) -> f64 {
        self.data.model_opt().timestep * self.frame_skip as f64
    }

    pub(crate) fn set_state(&mut self, qpos: &[f64], qvel: &[f64]) -> Result<(), MujocoError> {
        if qpos.len() != self.nq() || qvel.len() != self.nv() {
            return Err(MujocoError::InvalidInput(format!(
                "state shape mismatch: expected qpos ({},) and qvel ({},), got ({},) and ({},)",
                self.nq(),
                self.nv(),
                qpos.len(),
                qvel.len()
            )));
        }
        if !qpos.iter().chain(qvel).all(|value| value.is_finite()) {
            return Err(MujocoError::InvalidInput(
                "qpos and qvel must contain only finite values".into(),
            ));
        }
        // qpos/qvel are not MuJoCo's entire state. Clear warm-start and other
        // solver state so an explicit state always defines a reproducible
        // transition, independent of the environment's previous episode.
        self.data.reset();
        self.data.qpos_mut().copy_from_slice(qpos);
        self.data.qvel_mut().copy_from_slice(qvel);
        self.contact_substeps.clear();
        self.data.forward();
        self.data.subtree_vel();
        Ok(())
    }

    pub(crate) fn reset_uniform(&mut self, noise_scale: f64) -> Result<(), MujocoError> {
        if noise_scale == 0.0 {
            let qpos = self.initial_qpos.clone();
            let qvel = self.initial_qvel.clone();
            return self.set_state(&qpos, &qvel);
        }
        let qpos = self
            .initial_qpos
            .iter()
            .map(|value| value + self.rng.random_range(-noise_scale..noise_scale))
            .collect::<Vec<_>>();
        let qvel = self
            .initial_qvel
            .iter()
            .map(|value| value + self.rng.random_range(-noise_scale..noise_scale))
            .collect::<Vec<_>>();
        self.set_state(&qpos, &qvel)
    }

    pub(crate) fn reset_uniform_positions_normal_velocities(
        &mut self,
        noise_scale: f64,
    ) -> Result<(), MujocoError> {
        if noise_scale == 0.0 {
            let qpos = self.initial_qpos.clone();
            let qvel = self.initial_qvel.clone();
            return self.set_state(&qpos, &qvel);
        }
        let qpos = self
            .initial_qpos
            .iter()
            .map(|value| value + self.rng.random_range(-noise_scale..noise_scale))
            .collect::<Vec<_>>();
        let qvel = self
            .initial_qvel
            .iter()
            .map(|value| {
                let noise: f64 = self.rng.sample(StandardNormal);
                value + noise_scale * noise
            })
            .collect::<Vec<_>>();
        self.set_state(&qpos, &qvel)
    }

    pub(crate) fn mass_center_xy(&self) -> [f64; 2] {
        let masses = self.data.model().body_mass();
        let positions = self.data.xipos();
        let total_mass = masses.iter().sum::<f64>();
        let mut center = [0.0; 2];
        for (mass, position) in masses.iter().zip(positions) {
            center[0] += mass * position[0];
            center[1] += mass * position[1];
        }
        center[0] /= total_mass;
        center[1] /= total_mass;
        center
    }

    pub(crate) fn cinert(&self) -> &[[f64; 10]] {
        self.data.cinert()
    }

    pub(crate) fn cvel(&self) -> &[[f64; 6]] {
        self.data.cvel()
    }

    pub(crate) fn qfrc_actuator(&self) -> &[f64] {
        self.data.qfrc_actuator()
    }

    pub(crate) fn cfrc_ext(&self) -> &[[f64; 6]] {
        self.data.cfrc_ext()
    }

    /// Advances MuJoCo with one actuator vector `action` shaped `[nu]`.
    pub(crate) fn step(&mut self, action: &Tensor) -> Result<Vec<f64>, MujocoError> {
        self.step_bounded(action, -1.0, 1.0)
    }

    /// Advances MuJoCo with one bounded actuator vector `action` shaped `[nu]`.
    pub(crate) fn step_bounded(
        &mut self,
        action: &Tensor,
        minimum: f64,
        maximum: f64,
    ) -> Result<Vec<f64>, MujocoError> {
        if action.rank() != 1 || action.dims()[0] != self.nu() {
            return Err(MujocoError::InvalidInput(format!(
                "action shape mismatch: expected ({},), got {:?}",
                self.nu(),
                action.dims()
            )));
        }
        if !action.dtype().is_float() {
            return Err(MujocoError::InvalidInput(format!(
                "action dtype mismatch: expected a floating-point dtype, got {:?}",
                action.dtype()
            )));
        }
        let values = action.to_dtype(DType::F64)?.to_vec1::<f64>()?;
        if !values
            .iter()
            .all(|value| value.is_finite() && (minimum..=maximum).contains(value))
        {
            return Err(MujocoError::InvalidInput(format!(
                "actions must be finite and within [{minimum}, {maximum}]"
            )));
        }
        self.data
            .ctrl_mut()
            .iter_mut()
            .zip(&values)
            .for_each(|(control, value)| *control = *value);
        for _ in 0..self.frame_skip {
            self.data.step();
        }
        // MuJoCo does not populate center-of-mass external forces during a
        // normal step unless a sensor requests them. Gymnasium explicitly
        // runs this pass after simulation, so do the same before environments
        // read `cfrc_ext` for observations and contact costs.
        self.data.rne_post_constraint();
        self.data.subtree_vel();
        Ok(values)
    }

    /// Sends physical actuator targets directly, as required by unbounded
    /// joint-position action tasks. The MJCF actuator defines the control units.
    pub(crate) fn step_physical(&mut self, action: &Tensor) -> Result<(), MujocoError> {
        if action.rank() != 1 || action.dims()[0] != self.nu() || !action.dtype().is_float() {
            return Err(MujocoError::InvalidInput(format!(
                "physical action must be a floating tensor of shape ({},)",
                self.nu()
            )));
        }
        let targets = action.to_dtype(DType::F64)?.to_vec1::<f64>()?;
        if !targets.iter().all(|value| value.is_finite()) {
            return Err(MujocoError::InvalidInput(
                "physical actions must be finite".into(),
            ));
        }
        self.data.ctrl_mut().copy_from_slice(&targets);
        self.contact_substeps.clear();
        for _ in 0..self.frame_skip {
            self.data.step();
            self.contact_substeps.push(self.contact_body_pairs());
        }
        self.data.rne_post_constraint();
        self.data.subtree_vel();
        Ok(())
    }

    /// Maps normalized policy actions `[-1, 1]` into each limited actuator's
    /// XML control range before stepping. Unbounded actuators receive the raw value.
    pub(crate) fn step_normalized(&mut self, action: &Tensor) -> Result<Vec<f64>, MujocoError> {
        if action.rank() != 1 || action.dims()[0] != self.nu() || !action.dtype().is_float() {
            return Err(MujocoError::InvalidInput(format!(
                "normalized action must be a floating tensor of shape ({},)",
                self.nu()
            )));
        }
        let mut values = action.to_dtype(DType::F64)?.to_vec1::<f64>()?;
        if !values
            .iter()
            .all(|value| value.is_finite() && (-1.0 - 1e-6..=1.0 + 1e-6).contains(value))
        {
            return Err(MujocoError::InvalidInput(format!(
                "normalized actions must be finite and within [-1, 1]; got {:?}",
                values
            )));
        }
        values
            .iter_mut()
            .for_each(|value| *value = value.clamp(-1.0, 1.0));
        let model = self.data.model();
        let physical = values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                if model.actuator_ctrllimited()[index] {
                    let [minimum, maximum] = model.actuator_ctrlrange()[index];
                    minimum + (value + 1.0) * 0.5 * (maximum - minimum)
                } else {
                    *value
                }
            })
            .collect::<Vec<_>>();
        self.data.ctrl_mut().copy_from_slice(&physical);
        self.contact_substeps.clear();
        for _ in 0..self.frame_skip {
            self.data.step();
            self.contact_substeps.push(self.contact_body_pairs());
        }
        self.data.rne_post_constraint();
        self.data.subtree_vel();
        Ok(values)
    }

    pub(crate) fn observation(&self, exclude_x: bool, clip_velocity: bool) -> Vec<f64> {
        let mut observation = Vec::with_capacity(self.nq() + self.nv() - usize::from(exclude_x));
        observation.extend_from_slice(&self.qpos()[usize::from(exclude_x)..]);
        if clip_velocity {
            observation.extend(self.qvel().iter().map(|value| value.clamp(-10.0, 10.0)));
        } else {
            observation.extend_from_slice(self.qvel());
        }
        observation
    }

    pub(crate) fn tensor(&self, values: &[f64]) -> Result<Tensor, MujocoError> {
        let values = values.iter().map(|value| *value as f32).collect::<Vec<_>>();
        let len = values.len();
        Ok(Tensor::from_vec(values, len, &self.device)?)
    }

    pub(crate) fn viewer_running(&self) -> bool {
        #[cfg(feature = "rendering")]
        {
            return self.viewer.as_ref().is_some_and(MjViewer::running);
        }
        #[cfg(not(feature = "rendering"))]
        {
            false
        }
    }

    pub(crate) fn render(&mut self) -> Result<(), MujocoError> {
        #[cfg(feature = "rendering")]
        if let Some(viewer) = &mut self.viewer
            && viewer.running()
        {
            viewer.sync_data(&mut self.data);
            viewer.render()?;
        }
        Ok(())
    }

    pub(crate) fn action_space(&self) -> Box<dyn Space<Error = candle_core::Error>> {
        self.action_space_bounded(-1.0, 1.0)
    }

    pub(crate) fn unbounded_action_space_for_dim(
        &self,
        dim: usize,
    ) -> Box<dyn Space<Error = candle_core::Error>> {
        Box::new(BoxSpace::new_unbounded(vec![dim], &self.device))
    }

    pub(crate) fn action_space_for_dim(
        &self,
        dim: usize,
    ) -> Box<dyn Space<Error = candle_core::Error>> {
        Box::new(BoxSpace::new_with_universal_bounds(
            vec![dim],
            -1.0,
            1.0,
            &self.device,
        ))
    }

    pub(crate) fn action_space_bounded(
        &self,
        minimum: f64,
        maximum: f64,
    ) -> Box<dyn Space<Error = candle_core::Error>> {
        Box::new(BoxSpace::new_with_universal_bounds(
            vec![self.nu()],
            minimum as f32,
            maximum as f32,
            &self.device,
        ))
    }

    pub(crate) fn observation_space(
        &self,
        exclude_x: bool,
    ) -> Box<dyn Space<Error = candle_core::Error>> {
        Box::new(BoxSpace::new_unbounded(
            vec![self.nq() + self.nv() - usize::from(exclude_x)],
            &self.device,
        ))
    }

    pub(crate) fn unbounded_observation_space(
        &self,
        size: usize,
    ) -> Box<dyn Space<Error = candle_core::Error>> {
        Box::new(BoxSpace::new_unbounded(vec![size], &self.device))
    }
}

pub(crate) fn validate_noise_scale(noise_scale: f64) -> Result<(), MujocoError> {
    if noise_scale.is_finite() && noise_scale >= 0.0 {
        Ok(())
    } else {
        Err(MujocoError::InvalidInput(
            "reset_noise_scale must be finite and non-negative".into(),
        ))
    }
}

pub(crate) fn validate_range(
    name: &str,
    (minimum, maximum): (f64, f64),
) -> Result<(), MujocoError> {
    if !minimum.is_nan() && !maximum.is_nan() && minimum <= maximum {
        Ok(())
    } else {
        Err(MujocoError::InvalidInput(format!(
            "{name} must have a non-NaN minimum no greater than its maximum"
        )))
    }
}

#[cfg(test)]
mod shared_model_tests {
    use super::*;

    #[test]
    fn task_state_edit_preserves_simulation_time() {
        let xml = "<mujoco><worldbody><body><freejoint/><geom size=\"0.1\" mass=\"1\"/></body></worldbody></mujoco>";
        let mut core = MujocoCore::new(xml, 1, &Device::Cpu, false).unwrap();
        core.data.step();
        let time = core.data.ffi().time;
        let mut qpos = core.qpos().to_vec();
        let qvel = core.qvel().to_vec();
        qpos[0] += 0.1;
        core.set_task_state(&qpos, &qvel).unwrap();
        assert_eq!(core.data.ffi().time, time);
    }

    #[test]
    fn path_instances_share_model_across_threads_and_release_it() {
        let path = Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/assets/slide.xml"
        ));
        let model = shared_path_model(path).unwrap();
        let weak = Arc::downgrade(&model);
        let first = MujocoCore::from_xml_path(path, 1, &Device::Cpu, false).unwrap();
        assert!(std::ptr::eq(model.as_ref(), first.data.model()));
        let other = std::thread::spawn(move || {
            MujocoCore::from_xml_path(path, 1, &Device::Cpu, false).unwrap()
        })
        .join()
        .unwrap();
        assert!(std::ptr::eq(first.data.model(), other.data.model()));
        drop(model);
        drop(first);
        assert!(weak.upgrade().is_some());
        drop(other);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn same_xml_shares_model_but_not_data() {
        let xml = "<mujoco><worldbody><body><freejoint/><geom size=\"0.1\" mass=\"1\"/></body></worldbody></mujoco>";
        let mut first = MujocoCore::new(xml, 1, &Device::Cpu, false).unwrap();
        let second = MujocoCore::new(xml, 1, &Device::Cpu, false).unwrap();
        assert!(std::ptr::eq(first.data.model(), second.data.model()));
        first.data.qpos_mut()[0] = 2.0;
        assert_ne!(first.data.qpos()[0], second.data.qpos()[0]);
    }
}

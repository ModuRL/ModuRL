use candle_core::{DType, Device, Tensor};
use modurl::spaces::{BoxSpace, Space};
use mujoco_rs::prelude::{MjData, MjModel};
#[cfg(feature = "rendering")]
use mujoco_rs::viewer::MjViewer;
use rand::{Rng, SeedableRng, rngs::StdRng};
use rand_distr::StandardNormal;

use crate::MujocoError;

pub(crate) struct MujocoCore {
    data: MjData<Box<MjModel>>,
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
        if frame_skip == 0 {
            return Err(MujocoError::InvalidInput(
                "frame_skip must be greater than zero".into(),
            ));
        }
        let model = Box::new(MjModel::from_xml_string(xml)?);
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
            initial_qpos,
            initial_qvel,
            frame_skip,
            device: device.clone(),
            rng: StdRng::from_os_rng(),
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

    pub(crate) fn nq(&self) -> usize {
        self.data.qpos().len()
    }

    pub(crate) fn nv(&self) -> usize {
        self.data.qvel().len()
    }

    pub(crate) fn nu(&self) -> usize {
        self.data.ctrl().len()
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
        self.data.forward();
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

    pub(crate) fn reset_half_cheetah(&mut self, noise_scale: f64) -> Result<(), MujocoError> {
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

    pub(crate) fn step(&mut self, action: &Tensor) -> Result<Vec<f64>, MujocoError> {
        if action.rank() != 1 || action.dims()[0] != self.nu() {
            return Err(MujocoError::InvalidInput(format!(
                "action shape mismatch: expected ({},), got {:?}",
                self.nu(),
                action.dims()
            )));
        }
        if action.dtype() != DType::F32 {
            return Err(MujocoError::InvalidInput(format!(
                "action dtype mismatch: expected F32, got {:?}",
                action.dtype()
            )));
        }
        let values = action.to_vec1::<f32>()?;
        if !values
            .iter()
            .all(|value| value.is_finite() && (-1.0..=1.0).contains(value))
        {
            return Err(MujocoError::InvalidInput(
                "actions must be finite and within [-1, 1]".into(),
            ));
        }
        self.data
            .ctrl_mut()
            .iter_mut()
            .zip(&values)
            .for_each(|(control, value)| *control = f64::from(*value));
        for _ in 0..self.frame_skip {
            self.data.step();
        }
        Ok(values.into_iter().map(f64::from).collect())
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
        Box::new(BoxSpace::new_with_universal_bounds(
            vec![self.nu()],
            -1.0,
            1.0,
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
    if !minimum.is_nan() && !maximum.is_nan() && minimum < maximum {
        Ok(())
    } else {
        Err(MujocoError::InvalidInput(format!(
            "{name} must have a non-NaN minimum smaller than its maximum"
        )))
    }
}

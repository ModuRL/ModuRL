use burn::tensor::{Device, Tensor};
use modurl::gym::Gym;
use modurl_mujoco::{CustomMujoco, MjModel, MujocoError, MujocoState, MujocoTask, TaskStep};
use mujoco_rs::mujoco_c::{mjtCamLight, mjtEq, mjtGeom, mjtProjection};
use std::{
    fmt::Debug,
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

const SLIDE_XML: &str = r#"
<mujoco model="slide">
  <option timestep="0.01" gravity="0 0 0"/>
  <worldbody>
    <body name="slider">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.05" mass="1"/>
      <site name="marker" size="0.01"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slide" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"#;

struct TempModel {
    directory: PathBuf,
    path: PathBuf,
}

impl TempModel {
    fn new(xml: &str) -> Self {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
        let directory = std::env::temp_dir().join(format!(
            "modurl-mujoco-edit-{}-{}",
            std::process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("model.xml");
        fs::write(&path, xml).unwrap();
        Self { directory, path }
    }
}

impl Drop for TempModel {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

struct StateTask;

impl MujocoTask for StateTask {
    fn observation(&self, state: &MujocoState) -> Vec<f64> {
        vec![state.qpos.first().copied().unwrap_or_default()]
    }

    fn transition(&mut self, _: &MujocoState, _: &MujocoState, _: &[f64]) -> TaskStep {
        TaskStep {
            reward: 0.0,
            done: false,
            truncated: false,
        }
    }
}

fn build(path: &Path) -> CustomMujoco<StateTask> {
    CustomMujoco::builder()
        .path(path)
        .task(StateTask)
        .observation_dim(1)
        .build()
        .unwrap()
}

fn step(env: &mut CustomMujoco<StateTask>) {
    env.step(Tensor::from_data([0.0_f32], &Device::flex()))
        .unwrap();
}

fn assert_rejected_and_unchanged<T: Debug + PartialEq>(
    env: &mut CustomMujoco<StateTask>,
    field: &str,
    current: impl Fn(&MjModel) -> T,
    edit: impl FnOnce(&mut MjModel) -> Result<(), MujocoError>,
) {
    let original = current(env.model());
    let error = env.edit_model(edit).unwrap_err().to_string();
    assert!(error.contains(field), "expected {field} in {error}");
    assert_eq!(current(env.model()), original);
    step(env);
}

#[test]
fn hazardous_edits_are_rejected_without_changing_the_live_model() {
    let model = TempModel::new(SLIDE_XML);
    let mut env = build(&model.path);
    env.reset().unwrap();

    let sdf_initpoints = env.model().opt().sdf_initpoints + 1;
    assert_rejected_and_unchanged(
        &mut env,
        "opt.sdf_initpoints",
        |model| model.opt().sdf_initpoints,
        move |model| {
            model.opt_mut().sdf_initpoints = sdf_initpoints;
            Ok(())
        },
    );
    assert_rejected_and_unchanged(
        &mut env,
        "dof_simplenum",
        |model| model.dof_simplenum().to_vec(),
        |model| {
            model.dof_simplenum_mut()[0] = 17;
            Ok(())
        },
    );
    assert_rejected_and_unchanged(
        &mut env,
        "body_simple",
        |model| model.body_simple().to_vec(),
        |model| {
            let value = if model.body_simple()[1] == 0 { 1 } else { 0 };
            model.body_simple_mut()[1] = value;
            Ok(())
        },
    );
    let solver = if env.model().opt().solver == 0 { 1 } else { 0 };
    assert_rejected_and_unchanged(
        &mut env,
        "opt.solver",
        |model| model.opt().solver,
        move |model| {
            model.opt_mut().solver = solver;
            Ok(())
        },
    );
    assert_rejected_and_unchanged(
        &mut env,
        "opt.iterations",
        |model| model.opt().iterations,
        |model| {
            model.opt_mut().iterations = -1;
            Ok(())
        },
    );
}

#[test]
fn contact_sensor_dataspec_cannot_be_changed() {
    let model = TempModel::new(
        r#"
        <mujoco model="contact">
          <option timestep="0.01" gravity="0 0 0"/>
          <worldbody>
            <body name="slider">
              <joint name="slide" type="slide" axis="1 0 0"/>
              <geom name="moving" type="sphere" size="0.05" mass="1"/>
            </body>
            <geom name="fixed" type="sphere" size="0.05" pos="0.1 0 0"/>
          </worldbody>
          <actuator>
            <motor joint="slide" ctrllimited="true" ctrlrange="-1 1"/>
          </actuator>
          <sensor>
            <contact name="touch" geom1="moving" geom2="fixed"/>
          </sensor>
        </mujoco>
        "#,
    );
    let mut env = build(&model.path);
    env.reset().unwrap();
    assert_ne!(env.model().sensor_intprm()[0][0], 0);
    assert_rejected_and_unchanged(
        &mut env,
        "sensor_intprm",
        |model| model.sensor_intprm().to_vec(),
        |model| {
            model.sensor_intprm_mut()[0][0] = 0;
            Ok(())
        },
    );
}

#[test]
fn only_finite_physics_edits_and_allowlisted_flags_are_accepted() {
    let model = TempModel::new(SLIDE_XML);
    let mut env = build(&model.path);
    let disableflags = env.model().opt().disableflags ^ 1;
    env.edit_model(|model| {
        model.geom_contype_mut()[0] = 7;
        model.actuator_group_mut()[0] = 2;
        model.opt_mut().iterations = 17;
        model.opt_mut().gravity = [0.0, 0.0, -1.0];
        model.geom_friction_mut()[0][0] = 0.25;
        model.opt_mut().disableflags = disableflags;
        Ok(())
    })
    .unwrap();

    assert_eq!(env.model().geom_contype()[0], 7);
    assert_eq!(env.model().actuator_group()[0], 2);
    assert_eq!(env.model().opt().iterations, 17);
    assert_eq!(env.model().opt().gravity, [0.0, 0.0, -1.0]);
    assert_eq!(env.model().geom_friction()[0][0], 0.25);
    assert_eq!(env.model().opt().disableflags, disableflags);
    env.reset().unwrap();
    step(&mut env);
}

#[test]
fn camera_and_light_render_fields_are_editable() {
    let model = TempModel::new(
        r#"
        <mujoco model="camera-light">
          <worldbody>
            <body name="target">
              <joint name="slide" type="slide" axis="1 0 0"/>
              <geom type="sphere" size="0.05" mass="1"/>
            </body>
            <light name="runtime-light" mode="fixed" target="target" pos="0 0 1"/>
            <camera name="runtime-camera" mode="fixed" target="target" pos="0 -1 1"/>
          </worldbody>
          <actuator>
            <motor joint="slide" ctrllimited="true" ctrlrange="-1 1"/>
          </actuator>
        </mujoco>
        "#,
    );
    let mut env = build(&model.path);
    env.reset().unwrap();

    let light_active = env.model().light_active()[0];
    let light_castshadow = env.model().light_castshadow()[0];
    env.edit_model(|model| {
        model.light_active_mut()[0] = !light_active;
        model.light_castshadow_mut()[0] = !light_castshadow;
        model.cam_projection_mut()[0] = mjtProjection::mjPROJ_ORTHOGRAPHIC;
        model.cam_mode_mut()[0] = mjtCamLight::mjCAMLIGHT_TRACK;
        model.light_mode_mut()[0] = mjtCamLight::mjCAMLIGHT_TRACK;
        Ok(())
    })
    .unwrap();

    assert_eq!(env.model().light_active()[0], !light_active);
    assert_eq!(env.model().light_castshadow()[0], !light_castshadow);
    assert_eq!(
        env.model().cam_projection()[0],
        mjtProjection::mjPROJ_ORTHOGRAPHIC
    );
    assert_eq!(env.model().cam_mode()[0], mjtCamLight::mjCAMLIGHT_TRACK);
    assert_eq!(env.model().light_mode()[0], mjtCamLight::mjCAMLIGHT_TRACK);
    env.reset().unwrap();
    assert_eq!(env.model().light_active()[0], !light_active);
    assert_eq!(env.model().light_castshadow()[0], !light_castshadow);
    assert_eq!(
        env.model().cam_projection()[0],
        mjtProjection::mjPROJ_ORTHOGRAPHIC
    );
    assert_eq!(env.model().cam_mode()[0], mjtCamLight::mjCAMLIGHT_TRACK);
    assert_eq!(env.model().light_mode()[0], mjtCamLight::mjCAMLIGHT_TRACK);
    step(&mut env);

    env.edit_model(|model| {
        model.light_active_mut()[0] = light_active;
        Ok(())
    })
    .unwrap();
    assert_eq!(env.model().light_active()[0], light_active);
    step(&mut env);
}

#[test]
fn site_type_cannot_be_changed() {
    let model = TempModel::new(SLIDE_XML);
    let mut env = build(&model.path);
    env.reset().unwrap();
    assert_eq!(env.model().site_type().len(), 1);

    assert_rejected_and_unchanged(
        &mut env,
        "site_type",
        |model| model.site_type().to_vec(),
        |model| {
            model.site_type_mut()[0] = mjtGeom::mjGEOM_FLEX;
            Ok(())
        },
    );
}

#[test]
fn flexstrain_eq_data_is_frozen_but_weld_eq_data_is_editable() {
    let flex_model = TempModel::new(
        r#"
        <mujoco model="flexstrain">
          <worldbody>
            <body name="slider">
              <joint name="slide" type="slide" axis="1 0 0"/>
              <geom type="sphere" size="0.05" mass="1"/>
            </body>
            <body name="soft">
              <freejoint/>
              <geom type="sphere" size="0.05" mass="1"/>
              <flexcomp name="soft_flex" type="grid" count="2 2 2"
                        spacing="0.1 0.1 0.1" dim="3" mass="1" radius="0.01">
                <edge equality="strain"/>
              </flexcomp>
            </body>
          </worldbody>
          <actuator>
            <motor joint="slide" ctrllimited="true" ctrlrange="-1 1"/>
          </actuator>
        </mujoco>
        "#,
    );
    let mut flex_env = build(&flex_model.path);
    let flexstrain_index = flex_env
        .model()
        .eq_type()
        .iter()
        .position(|eq_type| *eq_type == mjtEq::mjEQ_FLEXSTRAIN)
        .expect("flexcomp should compile a flexstrain equality");
    flex_env.reset().unwrap();
    let changed_value = flex_env.model().eq_data()[flexstrain_index][0] + 1.0;
    assert_rejected_and_unchanged(
        &mut flex_env,
        "eq_data",
        |model| model.eq_data().to_vec(),
        move |model| {
            model.eq_data_mut()[flexstrain_index][0] = changed_value;
            Ok(())
        },
    );

    let weld_model = TempModel::new(
        r#"
        <mujoco model="weld">
          <option timestep="0.01" gravity="0 0 0"/>
          <worldbody>
            <body name="slider">
              <joint name="slide" type="slide" axis="1 0 0"/>
              <geom type="sphere" size="0.05" mass="1"/>
            </body>
            <body name="welded">
              <freejoint/>
              <geom type="sphere" size="0.05" mass="1"/>
            </body>
          </worldbody>
          <actuator>
            <motor joint="slide" ctrllimited="true" ctrlrange="-1 1"/>
          </actuator>
          <equality>
            <weld body1="welded"/>
          </equality>
        </mujoco>
        "#,
    );
    let mut weld_env = build(&weld_model.path);
    weld_env.reset().unwrap();
    let changed_value = weld_env.model().eq_data()[0][0] + 0.01;
    weld_env
        .edit_model(move |model| {
            model.eq_data_mut()[0][0] = changed_value;
            Ok(())
        })
        .unwrap();
    assert_eq!(weld_env.model().eq_data()[0][0], changed_value);
    step(&mut weld_env);
}

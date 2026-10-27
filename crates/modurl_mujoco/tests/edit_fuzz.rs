use burn::tensor::{Device, Tensor, TensorData};
use modurl::gym::Gym;
use modurl_mujoco::{CustomMujoco, MjModel, MujocoState, MujocoTask, TaskStep};
use mujoco_rs::mujoco_c::{mjtCamLight, mjtColorSpace, mjtLightType, mjtObj, mjtProjection};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::{
    env, fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

const DEFAULT_SEED: u64 = 0x4d4f_4455_524c;
const DEFAULT_ITERS: usize = 32;

const MODELS: &[(&str, &str)] = &[
    (
        "heightfield-rangefinder",
        r#"
        <mujoco model="hfield-ray">
          <option timestep="0.01" gravity="0 0 0"/>
          <asset>
            <hfield name="terrain" nrow="3" ncol="3" size="1 1 1 0.1"
                    elevation="0 0 0 0 0.5 0 0 0 0"/>
          </asset>
          <worldbody>
            <geom name="ground" type="hfield" hfield="terrain"/>
            <body name="probe" pos="-2 0 0.5">
              <joint name="slide" type="slide" axis="0 0 1"/>
              <geom type="sphere" size="0.01" mass="1" contype="0" conaffinity="0"/>
              <site name="rf" zaxis="1 0 0"/>
            </body>
          </worldbody>
          <actuator><motor joint="slide"/></actuator>
          <sensor><rangefinder name="range" site="rf"/></sensor>
        </mujoco>
        "#,
    ),
    (
        "mesh-pair-render-sensor",
        r#"
        <mujoco model="mesh-pair">
          <option timestep="0.01" gravity="0 0 -9.81"/>
          <asset>
            <mesh name="tetra"
                  vertex="0 0 0 0.1 0 0 0 0.1 0 0 0 0.1"
                  face="0 1 2 0 3 1 0 2 3 1 3 2"/>
            <texture name="checker" type="2d" builtin="checker" width="4" height="4"
                     rgb1="0.1 0.2 0.3" rgb2="0.8 0.7 0.6"/>
            <material name="mat" texture="checker" texuniform="true"/>
          </asset>
          <worldbody>
            <light name="key" pos="0 0 3" dir="0 0 -1"/>
            <geom name="obstacle" type="sphere" pos="0.1 0 0" size="0.1"/>
            <body name="slider">
              <joint name="slide" type="slide" axis="1 0 0"/>
              <geom name="mesh_geom" type="mesh" mesh="tetra" mass="1" material="mat"/>
              <site name="touch_site" type="sphere" size="0.1"/>
              <camera name="camera" pos="0 -2 1"/>
            </body>
          </worldbody>
          <contact><pair geom1="mesh_geom" geom2="obstacle"/></contact>
          <actuator><motor joint="slide" gear="1" ctrllimited="true" ctrlrange="-1 1"/></actuator>
          <sensor><touch name="touch" site="touch_site"/></sensor>
          <custom>
            <numeric name="fuzz-values" data="0.5 -0.5"/>
            <tuple name="fuzz-objects">
              <element objtype="body" objname="slider"/>
              <element objtype="geom" objname="mesh_geom"/>
            </tuple>
          </custom>
        </mujoco>
        "#,
    ),
    (
        "flex",
        r#"
        <mujoco model="flex-fuzz">
          <option timestep="0.01" gravity="0 0 0"/>
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
          <actuator><motor joint="slide"/></actuator>
        </mujoco>
        "#,
    ),
    (
        "tendon-equality-actuator",
        r#"
        <mujoco model="tendon-fuzz">
          <option timestep="0.01" gravity="0 0 0"/>
          <worldbody>
            <site name="world_site" pos="0 1 0"/>
            <body name="slider">
              <joint name="slide" type="slide" axis="1 0 0"/>
              <geom type="sphere" size="0.05" mass="1"/>
              <site name="slider_site" pos="0 0.1 0"/>
            </body>
            <body name="second">
              <joint name="slide2" type="slide" axis="0 1 0"/>
              <geom type="sphere" size="0.05" mass="1"/>
            </body>
          </worldbody>
          <tendon>
            <spatial name="rope" width="0.01">
              <site site="slider_site"/>
              <site site="world_site"/>
            </spatial>
          </tendon>
          <equality>
            <joint name="joint-equality" joint1="slide" joint2="slide2"
                   polycoef="0 1 0 0 0" active="false"/>
          </equality>
          <actuator>
            <motor name="joint-motor" joint="slide" gear="2"
                   ctrllimited="true" ctrlrange="-1 1"
                   forcelimited="true" forcerange="-2 2"/>
            <motor name="tendon-motor" tendon="rope" gear="0.5"
                   ctrllimited="true" ctrlrange="-1 1"/>
          </actuator>
        </mujoco>
        "#,
    ),
];

struct Fixtures {
    directory: PathBuf,
    paths: Vec<(&'static str, PathBuf)>,
}

impl Fixtures {
    fn new() -> Self {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
        let directory = std::env::temp_dir().join(format!(
            "modurl-mujoco-edit-fuzz-{}-{}",
            std::process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&directory).expect("create model directory");
        let paths = MODELS
            .iter()
            .enumerate()
            .map(|(index, (name, xml))| {
                let path = directory.join(format!("model-{index}.xml"));
                fs::write(&path, xml).expect("write MJCF");
                (*name, path)
            })
            .collect();
        Self { directory, paths }
    }
}

impl Drop for Fixtures {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

struct FuzzTask;

impl MujocoTask for FuzzTask {
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

#[test]
fn deterministic_runtime_model_edit_fuzz() {
    let seed = env::var("MODURL_FUZZ_SEED")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_SEED);
    let iterations = env::var("MODURL_FUZZ_ITERS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_ITERS);
    let trace = env::var_os("MODURL_FUZZ_TRACE").is_some();
    let mut rng = StdRng::seed_from_u64(seed);
    let fixtures = Fixtures::new();
    let editors = all_editors();

    for (name, path) in &fixtures.paths {
        CustomMujoco::builder()
            .path(path)
            .task(FuzzTask)
            .observation_dim(1)
            .build()
            .unwrap_or_else(|error| panic!("build {name}: {error}"));
    }

    for iteration in 0..iterations {
        let (name, path) = &fixtures.paths[rng.random_range(0..fixtures.paths.len())];
        if trace {
            eprintln!("mujoco seed={seed} iteration={iteration} model={name}");
        }
        let mut env = CustomMujoco::builder()
            .path(path)
            .task(FuzzTask)
            .observation_dim(1)
            .build()
            .unwrap_or_else(|error| panic!("build {name}: {error}"));
        let original = env.model().clone();
        let edit_count = rng.random_range(1..=4);
        let edit_result = env.edit_model(|candidate| {
            for edit_index in 0..edit_count {
                let description = edit_one(&mut rng, &editors, &original, candidate);
                if trace {
                    eprintln!(
                        "mujoco seed={seed} iteration={iteration} edit={edit_index} {description}"
                    );
                }
            }
            Ok(())
        });
        if edit_result.is_err() || env.reset().is_err() {
            continue;
        }

        let action_count = env.actuator_count();
        for _ in 0..5 {
            let action: Vec<f32> = (0..action_count)
                .map(|_| rng.random_range(-1.0_f32..=1.0))
                .collect();
            if env
                .step(Tensor::from_data(
                    TensorData::new(action.clone(), [action_count]),
                    &Device::flex(),
                ))
                .is_err()
            {
                break;
            }
        }
    }
}

trait FloatValues {
    fn scalar_count(&self) -> usize;
    fn scalar_at(&self, index: usize) -> f64;
    fn set_scalar(&mut self, index: usize, value: f64);
}

impl FloatValues for f64 {
    fn scalar_count(&self) -> usize {
        1
    }

    fn scalar_at(&self, _: usize) -> f64 {
        *self
    }

    fn set_scalar(&mut self, _: usize, value: f64) {
        *self = value;
    }
}

impl FloatValues for f32 {
    fn scalar_count(&self) -> usize {
        1
    }

    fn scalar_at(&self, _: usize) -> f64 {
        f64::from(*self)
    }

    fn set_scalar(&mut self, _: usize, value: f64) {
        *self = value as f32;
    }
}

impl<T: FloatValues> FloatValues for [T] {
    fn scalar_count(&self) -> usize {
        self.iter().map(FloatValues::scalar_count).sum()
    }

    fn scalar_at(&self, mut index: usize) -> f64 {
        for value in self {
            let count = value.scalar_count();
            if index < count {
                return value.scalar_at(index);
            }
            index -= count;
        }
        unreachable!("float scalar index is in bounds")
    }

    fn set_scalar(&mut self, mut index: usize, scalar: f64) {
        for value in self {
            let count = value.scalar_count();
            if index < count {
                value.set_scalar(index, scalar);
                return;
            }
            index -= count;
        }
        unreachable!("float scalar index is in bounds")
    }
}

impl<T: FloatValues, const N: usize> FloatValues for [T; N] {
    fn scalar_count(&self) -> usize {
        self.as_slice().scalar_count()
    }

    fn scalar_at(&self, index: usize) -> f64 {
        self.as_slice().scalar_at(index)
    }

    fn set_scalar(&mut self, index: usize, value: f64) {
        self.as_mut_slice().set_scalar(index, value);
    }
}

type FieldEditor = fn(&mut StdRng, &MjModel, &mut MjModel) -> Option<String>;

fn edit_float_field<T: FloatValues + ?Sized>(
    rng: &mut StdRng,
    field: &str,
    original: &T,
    candidate: &mut T,
) -> Option<String> {
    let count = candidate.scalar_count();
    if count == 0 || count != original.scalar_count() {
        return None;
    }
    let index = rng.random_range(0..count);
    let old = original.scalar_at(index);
    let value = random_float(rng, old);
    candidate.set_scalar(index, value);
    Some(format!("{field}[{index}]={value:?}"))
}

fn random_float(rng: &mut StdRng, original: f64) -> f64 {
    match rng.random_range(0..11) {
        0 => 0.0,
        1 => -0.0,
        2 => -1.0,
        3 => 1e-300,
        4 => 1e300,
        5 => -1e300,
        6 => f64::INFINITY,
        7 => f64::NEG_INFINITY,
        8 => f64::NAN,
        9 => rng.random_range(-1.0..=1.0),
        _ => original * rng.random_range(-10.0..=10.0),
    }
}

#[derive(Clone, Copy)]
enum IntKind {
    ContactMask,
    GroupPriority,
    Bounded,
    CameraOutput,
}

fn edit_i32_field(
    rng: &mut StdRng,
    field: &str,
    candidate: &mut [i32],
    kind: IntKind,
) -> Option<String> {
    if candidate.is_empty() {
        return None;
    }
    let index = rng.random_range(0..candidate.len());
    let value = match kind {
        IntKind::ContactMask => rng.random(),
        IntKind::GroupPriority => match rng.random_range(0..8) {
            0 => i32::MIN,
            1 => -1_000_000,
            2 => -1,
            3 => 0,
            4 => 1,
            5 => 1_000_000,
            6 => i32::MAX,
            _ => rng.random(),
        },
        IntKind::Bounded => rng.random_range(0..=8),
        IntKind::CameraOutput => rng.random_range(0..=31),
    };
    candidate[index] = value;
    Some(format!("{field}[{index}]={value}"))
}

fn edit_choice<T: Copy + std::fmt::Debug>(
    rng: &mut StdRng,
    field: &str,
    candidate: &mut [T],
    choices: &[T],
) -> Option<String> {
    if candidate.is_empty() || choices.is_empty() {
        return None;
    }
    let index = rng.random_range(0..candidate.len());
    let value = choices[rng.random_range(0..choices.len())];
    candidate[index] = value;
    Some(format!("{field}[{index}]={value:?}"))
}

macro_rules! float_field_editors {
    ($($field:ident => $mut_field:ident),* $(,)?) => {
        const FLOAT_FIELD_EDITORS: &[FieldEditor] = &[
            $(|rng, original, candidate| {
                edit_float_field(
                    rng,
                    stringify!($field),
                    original.$field(),
                    candidate.$mut_field(),
                )
            },)*
        ];
    };
}

float_field_editors! {
    qpos0 => qpos0_mut, qpos_spring => qpos_spring_mut, body_pos => body_pos_mut,
    body_quat => body_quat_mut, body_ipos => body_ipos_mut, body_iquat => body_iquat_mut,
    body_mass => body_mass_mut, body_subtreemass => body_subtreemass_mut,
    body_inertia => body_inertia_mut, body_invweight0 => body_invweight0_mut,
    body_gravcomp => body_gravcomp_mut, body_margin => body_margin_mut,
    bvh_aabb => bvh_aabb_mut, oct_aabb => oct_aabb_mut, oct_coeff => oct_coeff_mut,
    jnt_solref => jnt_solref_mut, jnt_solimp => jnt_solimp_mut, jnt_pos => jnt_pos_mut,
    jnt_axis => jnt_axis_mut, jnt_stiffness => jnt_stiffness_mut,
    jnt_stiffnesspoly => jnt_stiffnesspoly_mut, jnt_range => jnt_range_mut,
    jnt_actfrcrange => jnt_actfrcrange_mut, jnt_margin => jnt_margin_mut,
    dof_solref => dof_solref_mut, dof_solimp => dof_solimp_mut,
    dof_frictionloss => dof_frictionloss_mut, dof_armature => dof_armature_mut,
    dof_damping => dof_damping_mut, dof_dampingpoly => dof_dampingpoly_mut,
    dof_invweight0 => dof_invweight0_mut, dof_length => dof_length_mut,
    geom_solmix => geom_solmix_mut, geom_solref => geom_solref_mut,
    geom_solimp => geom_solimp_mut, geom_size => geom_size_mut, geom_aabb => geom_aabb_mut,
    geom_rbound => geom_rbound_mut, geom_pos => geom_pos_mut, geom_quat => geom_quat_mut,
    geom_friction => geom_friction_mut, geom_margin => geom_margin_mut, geom_gap => geom_gap_mut,
    geom_fluid => geom_fluid_mut, geom_rgba => geom_rgba_mut, site_size => site_size_mut,
    site_pos => site_pos_mut, site_quat => site_quat_mut, site_rgba => site_rgba_mut,
    cam_pos => cam_pos_mut, cam_quat => cam_quat_mut, cam_poscom0 => cam_poscom0_mut,
    cam_pos0 => cam_pos0_mut, cam_mat0 => cam_mat0_mut, cam_fovy => cam_fovy_mut,
    cam_ipd => cam_ipd_mut, cam_sensorsize => cam_sensorsize_mut,
    cam_intrinsic => cam_intrinsic_mut, light_bulbradius => light_bulbradius_mut,
    light_intensity => light_intensity_mut, light_range => light_range_mut,
    light_pos => light_pos_mut, light_dir => light_dir_mut, light_poscom0 => light_poscom0_mut,
    light_pos0 => light_pos0_mut, light_dir0 => light_dir0_mut,
    light_attenuation => light_attenuation_mut, light_cutoff => light_cutoff_mut,
    light_exponent => light_exponent_mut, light_ambient => light_ambient_mut,
    light_diffuse => light_diffuse_mut, light_specular => light_specular_mut,
    flex_solmix => flex_solmix_mut, flex_solref => flex_solref_mut,
    flex_solimp => flex_solimp_mut, flex_friction => flex_friction_mut,
    flex_margin => flex_margin_mut, flex_gap => flex_gap_mut, flex_vert => flex_vert_mut,
    flex_vert0 => flex_vert0_mut, flex_vertmetric => flex_vertmetric_mut,
    flex_node => flex_node_mut, flex_node0 => flex_node0_mut,
    flexedge_length0 => flexedge_length0_mut, flexedge_invweight0 => flexedge_invweight0_mut,
    flex_radius => flex_radius_mut, flex_size => flex_size_mut,
    flex_stiffness => flex_stiffness_mut, flex_bending => flex_bending_mut,
    flex_damping => flex_damping_mut, flex_edgestiffness => flex_edgestiffness_mut,
    flex_edgedamping => flex_edgedamping_mut, flex_rgba => flex_rgba_mut,
    flex_texcoord => flex_texcoord_mut, mesh_vert => mesh_vert_mut,
    mesh_normal => mesh_normal_mut, mesh_texcoord => mesh_texcoord_mut,
    mesh_scale => mesh_scale_mut, mesh_pos => mesh_pos_mut, mesh_quat => mesh_quat_mut,
    mesh_polynormal => mesh_polynormal_mut, skin_rgba => skin_rgba_mut,
    skin_inflate => skin_inflate_mut, skin_vert => skin_vert_mut,
    skin_texcoord => skin_texcoord_mut, skin_bonebindpos => skin_bonebindpos_mut,
    skin_bonebindquat => skin_bonebindquat_mut, skin_bonevertweight => skin_bonevertweight_mut,
    hfield_size => hfield_size_mut, hfield_data => hfield_data_mut,
    mat_texrepeat => mat_texrepeat_mut, mat_emission => mat_emission_mut,
    mat_specular => mat_specular_mut, mat_shininess => mat_shininess_mut,
    mat_reflectance => mat_reflectance_mut, mat_metallic => mat_metallic_mut,
    mat_roughness => mat_roughness_mut, mat_rgba => mat_rgba_mut,
    pair_solref => pair_solref_mut, pair_solreffriction => pair_solreffriction_mut,
    pair_solimp => pair_solimp_mut, pair_margin => pair_margin_mut,
    pair_gap => pair_gap_mut, pair_friction => pair_friction_mut,
    eq_solref => eq_solref_mut, eq_solimp => eq_solimp_mut, eq_data => eq_data_mut,
    tendon_width => tendon_width_mut, tendon_solref_lim => tendon_solref_lim_mut,
    tendon_solimp_lim => tendon_solimp_lim_mut, tendon_solref_fri => tendon_solref_fri_mut,
    tendon_solimp_fri => tendon_solimp_fri_mut, tendon_range => tendon_range_mut,
    tendon_actfrcrange => tendon_actfrcrange_mut, tendon_margin => tendon_margin_mut,
    tendon_stiffness => tendon_stiffness_mut, tendon_stiffnesspoly => tendon_stiffnesspoly_mut,
    tendon_damping => tendon_damping_mut, tendon_dampingpoly => tendon_dampingpoly_mut,
    tendon_armature => tendon_armature_mut, tendon_frictionloss => tendon_frictionloss_mut,
    tendon_lengthspring => tendon_lengthspring_mut, tendon_length0 => tendon_length0_mut,
    tendon_invweight0 => tendon_invweight0_mut, tendon_rgba => tendon_rgba_mut,
    actuator_delay => actuator_delay_mut, actuator_dynprm => actuator_dynprm_mut,
    actuator_gainprm => actuator_gainprm_mut, actuator_biasprm => actuator_biasprm_mut,
    actuator_ctrlrange => actuator_ctrlrange_mut, actuator_forcerange => actuator_forcerange_mut,
    actuator_actrange => actuator_actrange_mut, actuator_damping => actuator_damping_mut,
    actuator_dampingpoly => actuator_dampingpoly_mut, actuator_armature => actuator_armature_mut,
    actuator_gear => actuator_gear_mut, actuator_cranklength => actuator_cranklength_mut,
    actuator_acc0 => actuator_acc0_mut, actuator_length0 => actuator_length0_mut,
    actuator_lengthrange => actuator_lengthrange_mut, sensor_cutoff => sensor_cutoff_mut,
    sensor_noise => sensor_noise_mut, sensor_delay => sensor_delay_mut,
    sensor_interval => sensor_interval_mut, numeric_data => numeric_data_mut,
    tuple_objprm => tuple_objprm_mut, key_time => key_time_mut, key_qpos => key_qpos_mut,
    key_qvel => key_qvel_mut, key_act => key_act_mut, key_mpos => key_mpos_mut,
    key_mquat => key_mquat_mut, key_ctrl => key_ctrl_mut, sensor_user => sensor_user_mut,
    actuator_user => actuator_user_mut, tendon_user => tendon_user_mut,
    cam_user => cam_user_mut, site_user => site_user_mut, geom_user => geom_user_mut,
    jnt_user => jnt_user_mut, body_user => body_user_mut,
}

macro_rules! option_float_field_editors {
    ($($field:ident),* $(,)?) => {
        const OPTION_FLOAT_FIELD_EDITORS: &[FieldEditor] = &[
            $(|rng, original, candidate| {
                edit_float_field(
                    rng,
                    concat!("opt.", stringify!($field)),
                    &original.opt().$field,
                    &mut candidate.opt_mut().$field,
                )
            },)*
        ];
    };
}

option_float_field_editors! {
    timestep, impratio, tolerance, ls_tolerance, noslip_tolerance, ccd_tolerance,
    sleep_tolerance, gravity, wind, magnetic, density, viscosity, o_margin, o_solref,
    o_solimp, o_friction,
}

macro_rules! bool_field_editors {
    ($($field:ident => $mut_field:ident),* $(,)?) => {
        const BOOL_FIELD_EDITORS: &[FieldEditor] = &[
            $(|rng, _, candidate| edit_choice(
                rng,
                stringify!($field),
                candidate.$mut_field(),
                &[false, true],
            ),)*
        ];
    };
}

bool_field_editors! {
    jnt_limited => jnt_limited_mut, jnt_actfrclimited => jnt_actfrclimited_mut,
    jnt_actgravcomp => jnt_actgravcomp_mut, light_castshadow => light_castshadow_mut,
    light_active => light_active_mut, flex_flatskin => flex_flatskin_mut,
    mat_texuniform => mat_texuniform_mut, eq_active0 => eq_active0_mut,
    tendon_limited => tendon_limited_mut, tendon_actfrclimited => tendon_actfrclimited_mut,
    actuator_ctrllimited => actuator_ctrllimited_mut,
    actuator_forcelimited => actuator_forcelimited_mut,
    actuator_actlimited => actuator_actlimited_mut, actuator_actearly => actuator_actearly_mut,
}

macro_rules! make_int_field_editors {
    ($name:ident; $kind:ident; $($field:ident => $mut_field:ident),* $(,)?) => {
        const $name: &[FieldEditor] = &[
            $(|rng, _, candidate| edit_i32_field(
                rng,
                stringify!($field),
                candidate.$mut_field(),
                IntKind::$kind,
            ),)*
        ];
    };
}

make_int_field_editors!(CONTACT_MASK_EDITORS; ContactMask;
    body_contype => body_contype_mut, body_conaffinity => body_conaffinity_mut,
    geom_contype => geom_contype_mut, geom_conaffinity => geom_conaffinity_mut,
    flex_contype => flex_contype_mut, flex_conaffinity => flex_conaffinity_mut,
);
make_int_field_editors!(GROUP_PRIORITY_EDITORS; GroupPriority;
    jnt_group => jnt_group_mut, geom_group => geom_group_mut,
    geom_priority => geom_priority_mut, site_group => site_group_mut,
    flex_priority => flex_priority_mut, flex_group => flex_group_mut,
    skin_group => skin_group_mut, tendon_group => tendon_group_mut,
    actuator_group => actuator_group_mut,
);
make_int_field_editors!(BOUNDED_INT_EDITORS; Bounded;
    bvh_depth => bvh_depth_mut, oct_depth => oct_depth_mut,
);
make_int_field_editors!(CAMERA_OUTPUT_EDITORS; CameraOutput; cam_output => cam_output_mut);

const BYTE_FIELD_EDITORS: &[FieldEditor] = &[|rng, _, candidate| {
    edit_choice(
        rng,
        "tex_data",
        candidate.tex_data_mut(),
        &[0_u8, 1, 127, 255],
    )
}];

const ENUM_FIELD_EDITORS: &[FieldEditor] = &[
    |rng, _, candidate| {
        edit_choice(
            rng,
            "cam_mode",
            candidate.cam_mode_mut(),
            &[
                mjtCamLight::mjCAMLIGHT_FIXED,
                mjtCamLight::mjCAMLIGHT_TRACK,
                mjtCamLight::mjCAMLIGHT_TRACKCOM,
                mjtCamLight::mjCAMLIGHT_TARGETBODY,
                mjtCamLight::mjCAMLIGHT_TARGETBODYCOM,
            ],
        )
    },
    |rng, _, candidate| {
        edit_choice(
            rng,
            "cam_projection",
            candidate.cam_projection_mut(),
            &[
                mjtProjection::mjPROJ_PERSPECTIVE,
                mjtProjection::mjPROJ_ORTHOGRAPHIC,
            ],
        )
    },
    |rng, _, candidate| {
        edit_choice(
            rng,
            "light_mode",
            candidate.light_mode_mut(),
            &[
                mjtCamLight::mjCAMLIGHT_FIXED,
                mjtCamLight::mjCAMLIGHT_TRACK,
                mjtCamLight::mjCAMLIGHT_TRACKCOM,
                mjtCamLight::mjCAMLIGHT_TARGETBODY,
                mjtCamLight::mjCAMLIGHT_TARGETBODYCOM,
            ],
        )
    },
    |rng, _, candidate| {
        edit_choice(
            rng,
            "light_type",
            candidate.light_type_mut(),
            &[
                mjtLightType::mjLIGHT_SPOT,
                mjtLightType::mjLIGHT_DIRECTIONAL,
                mjtLightType::mjLIGHT_POINT,
                mjtLightType::mjLIGHT_IMAGE,
            ],
        )
    },
    |rng, _, candidate| {
        edit_choice(
            rng,
            "tex_colorspace",
            candidate.tex_colorspace_mut(),
            &[
                mjtColorSpace::mjCOLORSPACE_AUTO,
                mjtColorSpace::mjCOLORSPACE_LINEAR,
                mjtColorSpace::mjCOLORSPACE_SRGB,
            ],
        )
    },
    |rng, _, candidate| {
        edit_choice(
            rng,
            "tuple_objtype",
            candidate.tuple_objtype_mut(),
            &[
                mjtObj::mjOBJ_BODY,
                mjtObj::mjOBJ_JOINT,
                mjtObj::mjOBJ_GEOM,
                mjtObj::mjOBJ_SITE,
                mjtObj::mjOBJ_CAMERA,
                mjtObj::mjOBJ_LIGHT,
                mjtObj::mjOBJ_FLEX,
                mjtObj::mjOBJ_MESH,
                mjtObj::mjOBJ_SKIN,
                mjtObj::mjOBJ_HFIELD,
                mjtObj::mjOBJ_TEXTURE,
                mjtObj::mjOBJ_MATERIAL,
                mjtObj::mjOBJ_PAIR,
                mjtObj::mjOBJ_EXCLUDE,
                mjtObj::mjOBJ_EQUALITY,
                mjtObj::mjOBJ_TENDON,
                mjtObj::mjOBJ_ACTUATOR,
                mjtObj::mjOBJ_SENSOR,
                mjtObj::mjOBJ_NUMERIC,
                mjtObj::mjOBJ_TEXT,
                mjtObj::mjOBJ_TUPLE,
                mjtObj::mjOBJ_KEY,
                mjtObj::mjOBJ_PLUGIN,
            ],
        )
    },
];

const OPTION_INT_EDITORS: &[FieldEditor] = &[
    |rng, _, candidate| {
        let value = rng.random_range(0..=1000);
        candidate.opt_mut().iterations = value;
        Some(format!("opt.iterations={value}"))
    },
    |rng, _, candidate| {
        let value = rng.random_range(0..=1000);
        candidate.opt_mut().ls_iterations = value;
        Some(format!("opt.ls_iterations={value}"))
    },
    |rng, _, candidate| {
        let value = rng.random_range(0..=1000);
        candidate.opt_mut().noslip_iterations = value;
        Some(format!("opt.noslip_iterations={value}"))
    },
    |rng, _, candidate| {
        let value = rng.random();
        candidate.opt_mut().disableflags = value;
        Some(format!("opt.disableflags={value}"))
    },
    |rng, _, candidate| {
        let value = rng.random();
        candidate.opt_mut().enableflags = value;
        Some(format!("opt.enableflags={value}"))
    },
    |rng, _, candidate| {
        let value = rng.random();
        candidate.opt_mut().disableactuator = value;
        Some(format!("opt.disableactuator={value}"))
    },
];

fn all_editors() -> Vec<FieldEditor> {
    FLOAT_FIELD_EDITORS
        .iter()
        .chain(OPTION_FLOAT_FIELD_EDITORS)
        .chain(BOOL_FIELD_EDITORS)
        .chain(CONTACT_MASK_EDITORS)
        .chain(GROUP_PRIORITY_EDITORS)
        .chain(BOUNDED_INT_EDITORS)
        .chain(CAMERA_OUTPUT_EDITORS)
        .chain(BYTE_FIELD_EDITORS)
        .chain(ENUM_FIELD_EDITORS)
        .chain(OPTION_INT_EDITORS)
        .copied()
        .collect()
}

fn edit_one(
    rng: &mut StdRng,
    editors: &[FieldEditor],
    original: &MjModel,
    candidate: &mut MjModel,
) -> String {
    loop {
        let editor = editors[rng.random_range(0..editors.len())];
        if let Some(description) = editor(rng, original, candidate) {
            return description;
        }
    }
}

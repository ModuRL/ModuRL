use candle_core::{DType, Device, Tensor};
use modurl::gym::Gym;
use modurl_mujoco::{AntV5, HalfCheetahV5, HopperV5, HumanoidV5, Walker2dV5};

#[test]
fn default_shapes_and_metadata_match_gymnasium() {
    let mut ant = AntV5::builder().build().unwrap();
    let ant_reset = ant.reset().unwrap();
    assert_eq!(ant_reset.state.dims(), &[105]);
    assert_eq!(ant_reset.state.dtype(), DType::F32);
    assert_eq!(ant.action_space().shape(), vec![8]);
    assert!(ant_reset.info.x_position.is_finite());
    assert!(ant_reset.info.x_velocity.is_none());

    let mut half_cheetah = HalfCheetahV5::builder().build().unwrap();
    let half_cheetah_reset = half_cheetah.reset().unwrap();
    assert_eq!(half_cheetah_reset.state.dims(), &[17]);
    assert_eq!(half_cheetah.action_space().shape(), vec![6]);

    let mut hopper = HopperV5::builder().build().unwrap();
    let hopper_reset = hopper.reset().unwrap();
    assert_eq!(hopper_reset.state.dims(), &[11]);
    assert_eq!(hopper.action_space().shape(), vec![3]);

    let mut humanoid = HumanoidV5::builder().build().unwrap();
    let humanoid_reset = humanoid.reset().unwrap();
    assert_eq!(humanoid_reset.state.dims(), &[348]);
    assert_eq!(humanoid.action_space().shape(), vec![17]);
    assert_eq!(humanoid_reset.info.tendon_length.len(), 2);
    assert_eq!(humanoid_reset.info.tendon_velocity.len(), 2);
    assert!(humanoid_reset.info.x_velocity.is_none());

    let mut walker = Walker2dV5::builder().build().unwrap();
    let walker_reset = walker.reset().unwrap();
    assert_eq!(walker_reset.state.dims(), &[17]);
    assert_eq!(walker.action_space().shape(), vec![6]);
}

#[test]
fn sampled_actions_are_f32_and_can_step_every_environment() {
    let mut ant = AntV5::builder().build().unwrap();
    ant.reset().unwrap();
    let action = ant.action_space().sample(&Device::Cpu).unwrap();
    let step = ant.step(action).unwrap();
    assert_eq!(step.state.dtype(), DType::F32);
    assert!(step.info.x_velocity.is_some());
    assert!(step.info.reward_forward.is_some());

    let mut half_cheetah = HalfCheetahV5::builder().build().unwrap();
    half_cheetah.reset().unwrap();
    let action = half_cheetah.action_space().sample(&Device::Cpu).unwrap();
    assert_eq!(half_cheetah.step(action).unwrap().state.dtype(), DType::F32);

    let mut hopper = HopperV5::builder().build().unwrap();
    hopper.reset().unwrap();
    let action = hopper.action_space().sample(&Device::Cpu).unwrap();
    assert_eq!(hopper.step(action).unwrap().state.dtype(), DType::F32);

    let mut humanoid = HumanoidV5::builder().build().unwrap();
    humanoid.reset().unwrap();
    let action = humanoid.action_space().sample(&Device::Cpu).unwrap();
    let step = humanoid.step(action).unwrap();
    assert_eq!(step.state.dtype(), DType::F32);
    assert!(step.info.y_velocity.is_some());
    assert!(step.info.reward_contact.is_some());

    let mut walker = Walker2dV5::builder().build().unwrap();
    walker.reset().unwrap();
    let action = walker.action_space().sample(&Device::Cpu).unwrap();
    assert_eq!(walker.step(action).unwrap().state.dtype(), DType::F32);
}

#[test]
fn f64_policy_actions_are_converted_to_mujoco_precision() {
    let mut environment = HalfCheetahV5::builder().build().unwrap();
    environment.reset().unwrap();
    let action = Tensor::zeros(6, DType::F64, &Device::Cpu).unwrap();
    assert_eq!(environment.step(action).unwrap().state.dtype(), DType::F32);
}

#[test]
fn invalid_action_is_reported_without_mutating() {
    let mut environment = HopperV5::builder().build().unwrap();
    environment.reset().unwrap();
    let invalid = Tensor::from_vec(vec![0.0_f32, 0.0], 2, &Device::Cpu).unwrap();
    assert!(environment.step(invalid).is_err());
}

#[test]
fn builders_expose_gymnasium_configuration() {
    let mut ant = AntV5::builder()
        .reset_noise_scale(0.0)
        .exclude_current_positions_from_observation(false)
        .include_cfrc_ext_in_observation(false)
        .build()
        .unwrap();
    assert_eq!(ant.reset().unwrap().state.dims(), &[29]);

    let mut humanoid = HumanoidV5::builder()
        .reset_noise_scale(0.0)
        .include_cinert_in_observation(false)
        .include_cvel_in_observation(false)
        .include_qfrc_actuator_in_observation(false)
        .include_cfrc_ext_in_observation(false)
        .build()
        .unwrap();
    assert_eq!(humanoid.reset().unwrap().state.dims(), &[45]);

    assert!(
        AntV5::builder()
            .contact_force_range((1.0, 1.0))
            .build()
            .is_ok()
    );
    assert!(
        HumanoidV5::builder()
            .contact_cost_range((10.0, 10.0))
            .build()
            .is_ok()
    );
}

#[test]
fn builders_reject_invalid_configuration() {
    assert!(
        HalfCheetahV5::builder()
            .reset_noise_scale(-0.1)
            .build()
            .is_err()
    );
    assert!(HopperV5::builder().frame_skip(0).build().is_err());
    assert!(
        AntV5::builder()
            .contact_force_range((1.0, -1.0))
            .build()
            .is_err()
    );
    assert!(
        HumanoidV5::builder()
            .healthy_z_range((2.0, 1.0))
            .build()
            .is_err()
    );
    assert!(
        Walker2dV5::builder()
            .healthy_z_range((2.0, 1.0))
            .build()
            .is_err()
    );
}

#[test]
fn humanoid_matches_gymnasium_for_out_of_space_actions() {
    let mut humanoid = HumanoidV5::builder().build().unwrap();
    humanoid.reset().unwrap();
    let action = Tensor::full(0.5_f32, 17, &Device::Cpu).unwrap();
    assert!(humanoid.step(action).is_ok());
}

#[cfg(feature = "rendering")]
#[test]
fn rendering_can_be_configured_on_every_builder() {
    AntV5::builder().render(false).build().unwrap();
    HalfCheetahV5::builder().render(false).build().unwrap();
    HopperV5::builder().render(false).build().unwrap();
    HumanoidV5::builder().render(false).build().unwrap();
    Walker2dV5::builder().render(false).build().unwrap();
}

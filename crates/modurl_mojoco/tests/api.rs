use candle_core::{DType, Device, Tensor};
use modurl::gym::Gym;
use modurl_mojoco::{HalfCheetahV5, HopperV5, Walker2dV5};

#[test]
fn default_shapes_match_gymnasium() {
    let mut half_cheetah = HalfCheetahV5::builder().build().unwrap();
    let half_cheetah_observation = half_cheetah.reset().unwrap();
    assert_eq!(half_cheetah_observation.state.dims(), &[17]);
    assert_eq!(half_cheetah_observation.state.dtype(), DType::F32);
    assert_eq!(half_cheetah.action_space().shape(), vec![6]);

    let mut hopper = HopperV5::builder().build().unwrap();
    let hopper_observation = hopper.reset().unwrap();
    assert_eq!(hopper_observation.state.dims(), &[11]);
    assert_eq!(hopper_observation.state.dtype(), DType::F32);
    assert_eq!(hopper.action_space().shape(), vec![3]);

    let mut walker = Walker2dV5::builder().build().unwrap();
    let walker_observation = walker.reset().unwrap();
    assert_eq!(walker_observation.state.dims(), &[17]);
    assert_eq!(walker_observation.state.dtype(), DType::F32);
    assert_eq!(walker.action_space().shape(), vec![6]);
}

#[test]
fn sampled_actions_are_f32_and_can_step_every_environment() {
    let mut half_cheetah = HalfCheetahV5::builder().build().unwrap();
    half_cheetah.reset().unwrap();
    let action = half_cheetah.action_space().sample(&Device::Cpu).unwrap();
    assert_eq!(action.dtype(), DType::F32);
    assert_eq!(half_cheetah.step(action).unwrap().state.dtype(), DType::F32);

    let mut hopper = HopperV5::builder().build().unwrap();
    hopper.reset().unwrap();
    let action = hopper.action_space().sample(&Device::Cpu).unwrap();
    assert_eq!(action.dtype(), DType::F32);
    assert_eq!(hopper.step(action).unwrap().state.dtype(), DType::F32);

    let mut walker = Walker2dV5::builder().build().unwrap();
    walker.reset().unwrap();
    let action = walker.action_space().sample(&Device::Cpu).unwrap();
    assert_eq!(action.dtype(), DType::F32);
    assert_eq!(walker.step(action).unwrap().state.dtype(), DType::F32);
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
    let mut half_cheetah = HalfCheetahV5::builder()
        .reset_noise_scale(0.0)
        .exclude_current_positions_from_observation(false)
        .build()
        .unwrap();
    assert_eq!(half_cheetah.reset().unwrap().state.dims(), &[18]);
    assert_eq!(half_cheetah.observation_space().shape(), vec![18]);

    let mut hopper = HopperV5::builder()
        .reset_noise_scale(0.0)
        .terminate_when_unhealthy(false)
        .healthy_z_range((2.0, 3.0))
        .build()
        .unwrap();
    hopper
        .set_state(&[0.0, 1.25, 0.0, 0.0, 0.0, 0.0], &[0.0; 6])
        .unwrap();
    let action = Tensor::zeros(3, candle_core::DType::F32, &Device::Cpu).unwrap();
    assert!(!hopper.step(action).unwrap().done);
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
        Walker2dV5::builder()
            .healthy_z_range((2.0, 1.0))
            .build()
            .is_err()
    );
}

#[cfg(feature = "rendering")]
#[test]
fn rendering_can_be_configured_on_every_builder() {
    HalfCheetahV5::builder().render(false).build().unwrap();
    HopperV5::builder().render(false).build().unwrap();
    Walker2dV5::builder().render(false).build().unwrap();
}

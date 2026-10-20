use burn::tensor::{DType, Device, Tensor};
use modurl::{gym::Gym, spaces::ActionSpace};
use modurl_mujoco::{AntV5, HalfCheetahV5, HopperV5, HumanoidV5, Walker2dV5};

#[test]
fn default_shapes_and_metadata_match_gymnasium() {
    let mut ant = AntV5::builder().build().unwrap();
    let ant_reset = ant.reset().unwrap();
    assert!(ant_reset.observation.device() == Device::flex());
    assert_eq!(ant_reset.observation.dims(), [105]);
    assert_eq!(ant_reset.observation.dtype(), DType::F32);
    assert_eq!(ant.action_space().shape(), vec![8]);
    assert!(ant_reset.info.x_position.is_finite());
    assert!(ant_reset.info.x_velocity.is_none());

    let mut half_cheetah = HalfCheetahV5::builder().build().unwrap();
    let half_cheetah_reset = half_cheetah.reset().unwrap();
    assert!(half_cheetah_reset.observation.device() == Device::flex());
    assert_eq!(half_cheetah_reset.observation.dims(), [17]);
    assert_eq!(half_cheetah.action_space().shape(), vec![6]);

    let mut hopper = HopperV5::builder().build().unwrap();
    let hopper_reset = hopper.reset().unwrap();
    assert!(hopper_reset.observation.device() == Device::flex());
    assert_eq!(hopper_reset.observation.dims(), [11]);
    assert_eq!(hopper.action_space().shape(), vec![3]);

    let mut humanoid = HumanoidV5::builder().build().unwrap();
    let humanoid_reset = humanoid.reset().unwrap();
    assert!(humanoid_reset.observation.device() == Device::flex());
    assert_eq!(humanoid_reset.observation.dims(), [348]);
    assert_eq!(humanoid.action_space().shape(), vec![17]);
    assert_eq!(humanoid_reset.info.tendon_length.len(), 2);
    assert_eq!(humanoid_reset.info.tendon_velocity.len(), 2);
    assert!(humanoid_reset.info.x_velocity.is_none());

    let mut walker = Walker2dV5::builder().build().unwrap();
    let walker_reset = walker.reset().unwrap();
    assert!(walker_reset.observation.device() == Device::flex());
    assert_eq!(walker_reset.observation.dims(), [17]);
    assert_eq!(walker.action_space().shape(), vec![6]);
}

#[test]
fn sampled_actions_are_f32_and_can_step_every_environment() {
    let mut ant = AntV5::builder().build().unwrap();
    ant.reset().unwrap();
    let action = ant
        .action_space()
        .sample_batch(1, &Device::flex())
        .unwrap()
        .squeeze_dim::<1>(0);
    let step = ant.step(action).unwrap();
    assert_eq!(step.observation.dtype(), DType::F32);
    assert!(step.observation.device() == Device::flex());
    assert!(step.info.x_velocity.is_some());
    assert!(step.info.reward_forward.is_some());

    let mut half_cheetah = HalfCheetahV5::builder().build().unwrap();
    half_cheetah.reset().unwrap();
    let action = half_cheetah
        .action_space()
        .sample_batch(1, &Device::flex())
        .unwrap()
        .squeeze_dim::<1>(0);
    assert_eq!(
        half_cheetah.step(action).unwrap().observation.dtype(),
        DType::F32
    );

    let mut hopper = HopperV5::builder().build().unwrap();
    hopper.reset().unwrap();
    let action = hopper
        .action_space()
        .sample_batch(1, &Device::flex())
        .unwrap()
        .squeeze_dim::<1>(0);
    assert_eq!(hopper.step(action).unwrap().observation.dtype(), DType::F32);

    let mut humanoid = HumanoidV5::builder().build().unwrap();
    humanoid.reset().unwrap();
    let action = humanoid
        .action_space()
        .sample_batch(1, &Device::flex())
        .unwrap()
        .squeeze_dim::<1>(0);
    let step = humanoid.step(action).unwrap();
    assert_eq!(step.observation.dtype(), DType::F32);
    assert!(step.observation.device() == Device::flex());
    assert!(step.info.y_velocity.is_some());
    assert!(step.info.reward_contact.is_some());

    let mut walker = Walker2dV5::builder().build().unwrap();
    walker.reset().unwrap();
    let action = walker
        .action_space()
        .sample_batch(1, &Device::flex())
        .unwrap()
        .squeeze_dim::<1>(0);
    assert_eq!(walker.step(action).unwrap().observation.dtype(), DType::F32);
}

#[test]
fn f64_policy_actions_are_converted_to_mujoco_precision() {
    let mut environment = HalfCheetahV5::builder().build().unwrap();
    environment.reset().unwrap();
    let action = Tensor::zeros([6], (&Device::flex(), DType::F64));
    assert_eq!(
        environment.step(action).unwrap().observation.dtype(),
        DType::F32
    );
}

#[test]
fn invalid_action_is_reported_without_mutating() {
    let mut environment = HopperV5::builder().build().unwrap();
    environment.reset().unwrap();
    let invalid = Tensor::from_data([0.0_f32, 0.0], &Device::flex());
    assert!(environment.step(invalid).is_err());
}

#[test]
fn invalid_actions_preserve_the_next_physics_transition() {
    let mut environment = HalfCheetahV5::builder()
        .reset_noise_scale(0.0)
        .build()
        .unwrap();
    let mut reference = HalfCheetahV5::builder()
        .reset_noise_scale(0.0)
        .build()
        .unwrap();
    environment.reset().unwrap();
    reference.reset().unwrap();
    let device = Device::flex();
    for invalid in [
        Tensor::zeros([5], &device),
        Tensor::full([6], f32::NAN, &device),
        Tensor::full([6], f32::INFINITY, &device),
        Tensor::full([6], 2.0, &device),
    ] {
        assert!(environment.step(invalid).is_err());
        let actual = environment.step(Tensor::zeros([6], &device)).unwrap();
        let expected = reference.step(Tensor::zeros([6], &device)).unwrap();
        assert_eq!(actual.reward, expected.reward);
        assert_eq!(
            actual.observation.into_data(),
            expected.observation.into_data()
        );
    }
}

#[test]
fn seeded_resets_preserve_rng_advancement_and_observation_dtype() {
    let mut first = HalfCheetahV5::builder().build().unwrap();
    let mut second = HalfCheetahV5::builder().build().unwrap();
    first.seed(42);
    second.seed(42);
    let initial = first.reset().unwrap().observation.into_data();
    assert_eq!(initial, second.reset().unwrap().observation.into_data());
    let next = first.reset().unwrap().observation.into_data();
    assert_ne!(initial, next);
    assert_eq!(next, second.reset().unwrap().observation.into_data());
    first.seed(42);
    assert_eq!(initial, first.reset().unwrap().observation.into_data());
}

#[test]
fn builders_expose_gymnasium_configuration() {
    let mut ant = AntV5::builder()
        .reset_noise_scale(0.0)
        .exclude_current_positions_from_observation(false)
        .include_cfrc_ext_in_observation(false)
        .build()
        .unwrap();
    assert_eq!(ant.reset().unwrap().observation.dims(), [29]);

    let mut humanoid = HumanoidV5::builder()
        .reset_noise_scale(0.0)
        .include_cinert_in_observation(false)
        .include_cvel_in_observation(false)
        .include_qfrc_actuator_in_observation(false)
        .include_cfrc_ext_in_observation(false)
        .build()
        .unwrap();
    assert_eq!(humanoid.reset().unwrap().observation.dims(), [45]);

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
    let action = Tensor::full([17], 0.5_f32, (&Device::flex(), DType::F32));
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

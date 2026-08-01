use candle_core::{DType, Device, Tensor};
use modurl::gym::{Gym, MultiGym, StackedMultiGym};
use modurl_mojoco::{HalfCheetahV5, HopperV5, SumoAnts, Walker2dV5};

const SUMO_INITIAL_QPOS: [f64; 30] = [
    -1.0, 0.0, 0.65, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.65,
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0,
];

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
fn f64_policy_actions_are_converted_to_mujoco_precision() {
    let mut environment = HalfCheetahV5::builder().build().unwrap();
    environment.reset().unwrap();
    let action = Tensor::zeros(6, DType::F64, &Device::Cpu).unwrap();

    let step = environment.step(action).unwrap();

    assert_eq!(step.state.dtype(), DType::F32);
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
    assert!(SumoAnts::builder().max_episode_steps(0).build().is_err());
    assert!(SumoAnts::builder().ring_radius(0.0).build().is_err());
    assert!(SumoAnts::builder().player_count(1).build().is_err());
    assert!(SumoAnts::builder().player_count(9).build().is_err());
}

#[test]
fn sumo_ants_batches_two_players_from_one_shared_game() {
    let mut environment = SumoAnts::builder().reset_noise_scale(0.0).build().unwrap();

    let observations = environment.reset().unwrap();
    assert_eq!(environment.num_envs(), 2);
    assert_eq!(observations.dims(), &[2, 58]);
    assert_eq!(observations.dtype(), DType::F32);
    assert_eq!(environment.observation_space().shape(), vec![58]);
    assert_eq!(environment.action_space().shape(), vec![8]);

    let actions = Tensor::zeros((2, 8), DType::F32, &Device::Cpu).unwrap();
    let transition = environment.step(actions).unwrap();
    assert_eq!(transition.states.dims(), &[2, 58]);
    assert_eq!(transition.rewards.dims(), &[2]);
    assert_eq!(transition.infos.len(), 2);
    assert_eq!(transition.infos[0].player_index, 0);
    assert_eq!(transition.infos[1].player_index, 1);
    assert_eq!(transition.dones, vec![false, false]);
    assert_eq!(transition.truncateds, vec![false, false]);
    assert!(transition.terminal_states.iter().all(Option::is_none));
}

#[test]
fn sumo_ants_player_count_controls_the_shared_batch() {
    let mut environment = SumoAnts::builder()
        .player_count(3)
        .reset_noise_scale(0.0)
        .build()
        .unwrap();

    assert_eq!(environment.num_envs(), 3);
    assert_eq!(environment.reset().unwrap().dims(), &[3, 87]);
    assert_eq!(environment.observation_space().shape(), vec![87]);
    let actions = Tensor::zeros((3, 8), DType::F32, &Device::Cpu).unwrap();
    let transition = environment.step(actions).unwrap();
    assert_eq!(transition.states.dims(), &[3, 87]);
    assert_eq!(transition.rewards.dims(), &[3]);
    assert_eq!(transition.infos.len(), 3);
}

#[test]
fn stacked_sumo_ants_flattens_games_and_players_into_one_batch() {
    let games = vec![
        SumoAnts::builder().reset_noise_scale(0.0).build().unwrap(),
        SumoAnts::builder().reset_noise_scale(0.0).build().unwrap(),
    ];
    let mut environment = StackedMultiGym::try_new(games).unwrap();

    assert_eq!(environment.num_groups(), 2);
    assert_eq!(environment.num_envs(), 4);
    assert_eq!(environment.group_offsets(), &[0, 2, 4]);
    assert_eq!(environment.reset().unwrap().dims(), &[4, 58]);

    let actions = Tensor::zeros((4, 8), DType::F32, &Device::Cpu).unwrap();
    let transition = environment.step(actions).unwrap();
    assert_eq!(transition.states.dims(), &[4, 58]);
    assert_eq!(transition.rewards.dims(), &[4]);
    assert_eq!(transition.infos.len(), 4);
    assert_eq!(
        transition
            .infos
            .iter()
            .map(|info| info.player_index)
            .collect::<Vec<_>>(),
        vec![0, 1, 0, 1]
    );
}

#[test]
fn sumo_ants_rejects_actions_without_player_rows() {
    let mut environment = SumoAnts::builder().build().unwrap();
    environment.reset().unwrap();
    let action = Tensor::zeros(16, DType::F32, &Device::Cpu).unwrap();

    assert!(environment.step(action).is_err());
}

#[test]
fn sumo_ants_terminates_and_resets_both_players_together() {
    let mut environment = SumoAnts::builder().reset_noise_scale(0.0).build().unwrap();
    let mut qpos = SUMO_INITIAL_QPOS;
    qpos[0] = 3.5;
    environment.set_state(&qpos, &[0.0; 28]).unwrap();

    let actions = Tensor::zeros((2, 8), DType::F32, &Device::Cpu).unwrap();
    let transition = environment.step(actions).unwrap();

    assert_eq!(transition.dones, vec![true, true]);
    assert_eq!(transition.truncateds, vec![false, false]);
    assert!(transition.terminal_states.iter().all(Option::is_some));
    assert!(transition.infos[0].lost);
    assert!(transition.infos[1].won);
    let rewards = transition.rewards.to_vec1::<f32>().unwrap();
    assert!(rewards[1] > rewards[0]);
    assert_eq!(
        transition.transition_next_states().unwrap().dims(),
        &[2, 58]
    );
    let reset_observations = transition.states.to_vec2::<f32>().unwrap();
    assert_eq!(reset_observations[0][0], -1.5);
    assert_eq!(reset_observations[1][0], 1.5);
}

#[test]
fn sumo_ants_truncates_at_its_shared_time_limit() {
    let mut environment = SumoAnts::builder()
        .reset_noise_scale(0.0)
        .max_episode_steps(1)
        .build()
        .unwrap();
    environment.reset().unwrap();

    let actions = Tensor::zeros((2, 8), DType::F32, &Device::Cpu).unwrap();
    let transition = environment.step(actions).unwrap();

    assert_eq!(transition.dones, vec![false, false]);
    assert_eq!(transition.truncateds, vec![true, true]);
    assert!(transition.terminal_states.iter().all(Option::is_some));
}

#[cfg(feature = "rendering")]
#[test]
fn rendering_can_be_configured_on_every_builder() {
    HalfCheetahV5::builder().render(false).build().unwrap();
    HopperV5::builder().render(false).build().unwrap();
    Walker2dV5::builder().render(false).build().unwrap();
    SumoAnts::builder().render(false).build().unwrap();
}

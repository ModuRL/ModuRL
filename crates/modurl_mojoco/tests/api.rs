use candle_core::{DType, Device, Tensor};
use modurl::gym::{Gym, MultiGym, StackedMultiGym};
use modurl_mojoco::{AntV5, HalfCheetahV5, HopperV5, HumanoidV5, SumoAnts, SumoHumans, Walker2dV5};

const SUMO_INITIAL_QPOS: [f64; 30] = [
    -1.0, 0.0, 2.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.5, 0.0,
    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
];

const SUMO_HUMANS_INITIAL_QPOS: [f64; 48] = [
    -2.0, 0.0, 2.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
];

#[test]
fn default_shapes_match_gymnasium() {
    let mut ant = AntV5::builder().build().unwrap();
    let ant_observation = ant.reset().unwrap();
    assert_eq!(ant_observation.state.dims(), &[105]);
    assert_eq!(ant_observation.state.dtype(), DType::F32);
    assert_eq!(ant.action_space().shape(), vec![8]);

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

    let mut humanoid = HumanoidV5::builder().build().unwrap();
    let humanoid_observation = humanoid.reset().unwrap();
    assert_eq!(humanoid_observation.state.dims(), &[348]);
    assert_eq!(humanoid_observation.state.dtype(), DType::F32);
    assert_eq!(humanoid.action_space().shape(), vec![17]);

    let mut walker = Walker2dV5::builder().build().unwrap();
    let walker_observation = walker.reset().unwrap();
    assert_eq!(walker_observation.state.dims(), &[17]);
    assert_eq!(walker_observation.state.dtype(), DType::F32);
    assert_eq!(walker.action_space().shape(), vec![6]);
}

#[test]
fn sampled_actions_are_f32_and_can_step_every_environment() {
    let mut ant = AntV5::builder().build().unwrap();
    ant.reset().unwrap();
    let action = ant.action_space().sample(&Device::Cpu).unwrap();
    assert_eq!(action.dtype(), DType::F32);
    assert_eq!(ant.step(action).unwrap().state.dtype(), DType::F32);

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

    let mut humanoid = HumanoidV5::builder().build().unwrap();
    humanoid.reset().unwrap();
    let action = humanoid.action_space().sample(&Device::Cpu).unwrap();
    assert_eq!(action.dtype(), DType::F32);
    assert_eq!(humanoid.step(action).unwrap().state.dtype(), DType::F32);

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
    let mut ant = AntV5::builder()
        .reset_noise_scale(0.0)
        .exclude_current_positions_from_observation(false)
        .include_cfrc_ext_in_observation(false)
        .build()
        .unwrap();
    assert_eq!(ant.reset().unwrap().state.dims(), &[29]);
    assert_eq!(ant.observation_space().shape(), vec![29]);

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

    let mut humanoid = HumanoidV5::builder()
        .reset_noise_scale(0.0)
        .include_cinert_in_observation(false)
        .include_cvel_in_observation(false)
        .include_qfrc_actuator_in_observation(false)
        .include_cfrc_ext_in_observation(false)
        .build()
        .unwrap();
    assert_eq!(humanoid.reset().unwrap().state.dims(), &[45]);
    assert_eq!(humanoid.observation_space().shape(), vec![45]);
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
    assert!(SumoAnts::builder().max_episode_steps(0).build().is_err());
    assert!(SumoAnts::builder().min_radius(0.0).build().is_err());
    assert!(
        SumoAnts::builder()
            .min_radius(4.5)
            .max_radius(2.5)
            .build()
            .is_err()
    );
    assert!(SumoHumans::builder().max_episode_steps(0).build().is_err());
    assert!(SumoHumans::builder().min_radius(0.0).build().is_err());
    assert!(
        SumoHumans::builder()
            .min_radius(3.5)
            .max_radius(1.5)
            .build()
            .is_err()
    );
}

#[test]
fn humanoid_rejects_actions_outside_its_model_bounds() {
    let mut humanoid = HumanoidV5::builder().build().unwrap();
    humanoid.reset().unwrap();
    let action = Tensor::full(0.5_f32, 17, &Device::Cpu).unwrap();

    assert!(humanoid.step(action).is_err());
}

#[test]
fn sumo_ants_batches_two_players_from_one_shared_game() {
    let mut environment = SumoAnts::builder().reset_noise_scale(0.0).build().unwrap();

    let observations = environment.reset().unwrap();
    assert_eq!(environment.num_envs(), 2);
    assert_eq!(observations.dims(), &[2, 137]);
    assert_eq!(observations.dtype(), DType::F32);
    assert_eq!(environment.observation_space().shape(), vec![137]);
    assert_eq!(environment.action_space().shape(), vec![8]);

    let actions = Tensor::zeros((2, 8), DType::F32, &Device::Cpu).unwrap();
    let transition = environment.step(actions).unwrap();
    assert_eq!(transition.states.dims(), &[2, 137]);
    assert_eq!(transition.rewards.dims(), &[2]);
    assert_eq!(transition.infos.len(), 2);
    assert_eq!(transition.infos[0].player_index, 0);
    assert_eq!(transition.infos[1].player_index, 1);
    assert_eq!(transition.dones, vec![false, false]);
    assert_eq!(transition.truncateds, vec![false, false]);
    assert!(transition.terminal_states.iter().all(Option::is_none));
}

#[test]
fn sumo_humans_batches_two_players_from_one_shared_game() {
    let mut environment = SumoHumans::builder()
        .reset_noise_scale(0.0)
        .build()
        .unwrap();

    let observations = environment.reset().unwrap();
    assert_eq!(environment.num_envs(), 2);
    assert_eq!(observations.dims(), &[2, 395]);
    assert_eq!(observations.dtype(), DType::F32);
    assert_eq!(environment.observation_space().shape(), vec![395]);
    assert_eq!(environment.action_space().shape(), vec![17]);

    let actions = Tensor::zeros((2, 17), DType::F32, &Device::Cpu).unwrap();
    let transition = environment.step(actions).unwrap();
    assert_eq!(transition.states.dims(), &[2, 395]);
    assert_eq!(transition.rewards.dims(), &[2]);
    assert_eq!(transition.infos.len(), 2);
    assert_eq!(transition.infos[0].player_index, 0);
    assert_eq!(transition.infos[1].player_index, 1);
    assert_eq!(transition.dones, vec![false, false]);
    assert_eq!(transition.truncateds, vec![false, false]);
    assert!(transition.terminal_states.iter().all(Option::is_none));
}

#[test]
fn sumo_humans_observation_and_reward_layout_match_the_original() {
    let mut environment = SumoHumans::builder()
        .min_radius(3.5)
        .max_radius(3.5)
        .reset_noise_scale(0.0)
        .build()
        .unwrap();
    environment.seed(7);

    let observations = environment.reset().unwrap().to_vec2::<f32>().unwrap();
    assert_eq!(observations[0][2], 2.5);
    assert_eq!(observations[1][2], 2.5);
    assert_eq!(observations[0][391], 3.5);
    assert_eq!(observations[1][391], 3.5);
    assert_eq!(observations[0][394], 500.0);
    assert_eq!(observations[1][394], 500.0);
    assert_eq!(&observations[0][356..380], &observations[1][0..24]);
    assert_eq!(&observations[1][356..380], &observations[0][0..24]);
    assert!((observations[0][380] - (observations[1][0] - observations[0][0])).abs() < 1e-6);
    assert!((observations[0][381] - (observations[1][1] - observations[0][1])).abs() < 1e-6);

    let actions = Tensor::zeros((2, 17), DType::F32, &Device::Cpu).unwrap();
    let transition = environment.step(actions).unwrap();
    let rewards = transition.rewards.to_vec1::<f32>().unwrap();
    for (reward, info) in rewards.iter().zip(&transition.infos) {
        assert!((reward - (info.reward_move + info.reward_remaining)).abs() < 1e-4);
        assert!(
            (info.reward_move
                - (info.reward_center - info.reward_ctrl - info.reward_contact
                    + info.reward_survive))
                .abs()
                < 1e-4
        );
        assert_eq!(info.reward_ctrl, 0.0);
    }
    assert_eq!(transition.states.to_vec2::<f32>().unwrap()[0][394], 499.0);
}

#[test]
fn sumo_humans_rejects_out_of_bounds_and_malformed_actions() {
    let mut environment = SumoHumans::builder().build().unwrap();
    environment.reset().unwrap();

    let malformed = Tensor::zeros(34, DType::F32, &Device::Cpu).unwrap();
    assert!(environment.step(malformed).is_err());
    let out_of_bounds = Tensor::full(0.5_f32, (2, 17), &Device::Cpu).unwrap();
    assert!(environment.step(out_of_bounds).is_err());
}

#[test]
fn sumo_humans_terminates_and_resets_both_players_together() {
    let mut environment = SumoHumans::builder()
        .reset_noise_scale(0.0)
        .build()
        .unwrap();
    let mut qpos = SUMO_HUMANS_INITIAL_QPOS;
    qpos[0] = 4.0;
    environment.set_state(&qpos, &[0.0; 46]).unwrap();

    let actions = Tensor::zeros((2, 17), DType::F32, &Device::Cpu).unwrap();
    let transition = environment.step(actions).unwrap();

    assert_eq!(transition.dones, vec![true, true]);
    assert_eq!(transition.truncateds, vec![false, false]);
    assert!(transition.terminal_states.iter().all(Option::is_some));
    assert!(transition.infos[0].lost);
    assert!(!transition.infos[1].won);
    assert_eq!(transition.infos[0].reward_remaining, -1_000.0);
    assert_eq!(transition.infos[1].reward_remaining, 0.0);
    assert_eq!(transition.states.dims(), &[2, 395]);
}

#[test]
fn sumo_humans_terminates_at_its_original_shared_time_limit() {
    let mut environment = SumoHumans::builder()
        .reset_noise_scale(0.0)
        .max_episode_steps(1)
        .build()
        .unwrap();
    environment.reset().unwrap();

    let actions = Tensor::zeros((2, 17), DType::F32, &Device::Cpu).unwrap();
    let transition = environment.step(actions).unwrap();

    assert_eq!(transition.dones, vec![true, true]);
    assert_eq!(transition.truncateds, vec![false, false]);
    assert!(
        transition
            .infos
            .iter()
            .all(|info| info.reward_remaining == -1_000.0)
    );
}

#[test]
fn sumo_ants_observation_and_reward_layout_match_the_original() {
    let mut environment = SumoAnts::builder()
        .min_radius(4.5)
        .max_radius(4.5)
        .reset_noise_scale(0.0)
        .build()
        .unwrap();
    environment.seed(7);

    let observations = environment.reset().unwrap().to_vec2::<f32>().unwrap();
    assert_eq!(observations[0][2], 2.5);
    assert_eq!(observations[1][2], 2.5);
    assert_eq!(observations[0][133], 4.5);
    assert_eq!(observations[1][133], 4.5);
    assert_eq!(observations[0][136], 500.0);
    assert_eq!(observations[1][136], 500.0);
    assert!((-2.95..-0.3).contains(&observations[0][0]));
    assert!((0.3..2.95).contains(&observations[1][0]));
    assert_eq!(&observations[0][107..122], &observations[1][0..15]);
    assert_eq!(&observations[1][107..122], &observations[0][0..15]);
    assert!((observations[0][122] - (observations[1][0] - observations[0][0])).abs() < 1e-6);
    assert!((observations[0][123] - (observations[1][1] - observations[0][1])).abs() < 1e-6);

    let actions = Tensor::zeros((2, 8), DType::F32, &Device::Cpu).unwrap();
    let transition = environment.step(actions).unwrap();
    let rewards = transition.rewards.to_vec1::<f32>().unwrap();
    for (reward, info) in rewards.iter().zip(&transition.infos) {
        let expected = info.reward_move + info.reward_remaining;
        assert!((reward - expected).abs() < 1e-4);
        assert!(
            (info.reward_move
                - (info.reward_center - info.reward_ctrl - info.reward_contact
                    + info.reward_survive))
                .abs()
                < 1e-4
        );
        assert_eq!(info.reward_ctrl, 0.0);
    }
    assert_eq!(transition.states.to_vec2::<f32>().unwrap()[0][136], 499.0);
}

#[test]
fn sumo_ants_supports_the_original_radius_curriculum() {
    let mut environment = SumoAnts::builder().reset_noise_scale(0.0).build().unwrap();
    environment.seed(11);

    let observations = environment
        .reset_with_version(0.0)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    assert!((2.5..2.6).contains(&observations[0][133]));
    assert_eq!(observations[0][133], observations[1][133]);
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
    assert_eq!(environment.reset().unwrap().dims(), &[4, 137]);

    let actions = Tensor::zeros((4, 8), DType::F32, &Device::Cpu).unwrap();
    let transition = environment.step(actions).unwrap();
    assert_eq!(transition.states.dims(), &[4, 137]);
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
    qpos[0] = 5.0;
    environment.set_state(&qpos, &[0.0; 28]).unwrap();

    let actions = Tensor::zeros((2, 8), DType::F32, &Device::Cpu).unwrap();
    let transition = environment.step(actions).unwrap();

    assert_eq!(transition.dones, vec![true, true]);
    assert_eq!(transition.truncateds, vec![false, false]);
    assert!(transition.terminal_states.iter().all(Option::is_some));
    assert!(transition.infos[0].lost);
    assert!(!transition.infos[1].won);
    assert_eq!(transition.infos[0].reward_remaining, -1_000.0);
    assert_eq!(transition.infos[1].reward_remaining, 0.0);
    let rewards = transition.rewards.to_vec1::<f32>().unwrap();
    assert!(rewards[1] > rewards[0]);
    assert_eq!(
        transition.transition_next_states().unwrap().dims(),
        &[2, 137]
    );
    assert_eq!(transition.states.dims(), &[2, 137]);
}

#[test]
fn sumo_ants_ends_when_an_ant_is_below_the_original_fall_height() {
    let mut environment = SumoAnts::builder().reset_noise_scale(0.0).build().unwrap();
    let mut qpos = SUMO_INITIAL_QPOS;
    qpos[2] = 0.79;
    environment.set_state(&qpos, &[0.0; 28]).unwrap();

    let actions = Tensor::zeros((2, 8), DType::F32, &Device::Cpu).unwrap();
    let transition = environment.step(actions).unwrap();

    assert_eq!(transition.dones, vec![true, true]);
    assert!(transition.infos[0].lost);
    assert_eq!(transition.infos[0].reward_remaining, -1_000.0);
    assert!(!transition.infos[1].won);
}

#[test]
fn sumo_ants_terminates_at_its_original_shared_time_limit() {
    let mut environment = SumoAnts::builder()
        .reset_noise_scale(0.0)
        .max_episode_steps(1)
        .build()
        .unwrap();
    environment.reset().unwrap();

    let actions = Tensor::zeros((2, 8), DType::F32, &Device::Cpu).unwrap();
    let transition = environment.step(actions).unwrap();

    assert_eq!(transition.dones, vec![true, true]);
    assert_eq!(transition.truncateds, vec![false, false]);
    assert!(transition.terminal_states.iter().all(Option::is_some));
    assert!(
        transition
            .infos
            .iter()
            .all(|info| info.reward_remaining == -1_000.0)
    );
}

#[cfg(feature = "rendering")]
#[test]
fn rendering_can_be_configured_on_every_builder() {
    AntV5::builder().render(false).build().unwrap();
    HalfCheetahV5::builder().render(false).build().unwrap();
    HopperV5::builder().render(false).build().unwrap();
    HumanoidV5::builder().render(false).build().unwrap();
    Walker2dV5::builder().render(false).build().unwrap();
    SumoAnts::builder().render(false).build().unwrap();
    SumoHumans::builder().render(false).build().unwrap();
}

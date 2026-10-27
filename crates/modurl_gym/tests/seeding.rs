use burn::tensor::{DType, Device, Distribution, Tensor};
use modurl::gym::{Gym, MultiGym, MultithreadedVectorizedGymWrapper};
use modurl_gym::{
    box_2d::{bipedal_walker::BipedalWalkerV3, lunar_lander::LunarLanderV3},
    classic_control::{
        acrobot::AcrobotV1, cartpole::CartPoleV1, mountain_car::MountainCarV0, pendulum::PendulumV1,
    },
};
use std::sync::Mutex;

static RNG_LOCK: Mutex<()> = Mutex::new(());

fn threaded_cartpole_rollout() -> Vec<Vec<f32>> {
    let device = Device::flex();

    let env = CartPoleV1::builder().seed(42).build().unwrap();
    let constructors = (0..8)
        .map(|index| move || CartPoleV1::builder().seed(42 + index).build().unwrap())
        .collect();
    let mut gym: MultithreadedVectorizedGymWrapper<CartPoleV1> =
        MultithreadedVectorizedGymWrapper::new(
            constructors,
            env.observation_space().clone(),
            env.action_space().clone(),
        )
        .unwrap();

    let mut observations = vec![
        gym.reset()
            .unwrap()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap(),
    ];
    for _ in 0..20 {
        let step = gym.step(Tensor::from_data([[1i32]; 8], &device)).unwrap();
        observations.push(step.observations.into_data().try_to_vec::<f32>().unwrap());
    }
    observations
}

#[test]
fn threaded_envs_are_reproducible() {
    let first = threaded_cartpole_rollout();
    for _ in 0..10 {
        assert_eq!(threaded_cartpole_rollout(), first);
    }
}

#[cfg(feature = "rendering")]
fn lunar_lander_rollout(render: bool) -> Vec<Vec<f32>> {
    let device = Device::flex();
    let mut env = LunarLanderV3::builder()
        .seed(42)
        .render(render)
        .build()
        .unwrap();
    let mut observations = vec![
        env.reset()
            .unwrap()
            .observation
            .into_data()
            .try_to_vec::<f32>()
            .unwrap(),
    ];
    for _ in 0..20 {
        observations.push(
            env.step(Tensor::from_data([2i32], &device))
                .unwrap()
                .observation
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
        );
    }
    observations
}

#[test]
fn lunar_lander_seeded_trajectory_matches_golden() {
    let device = Device::flex();
    let mut env = LunarLanderV3::builder().seed(42).build().unwrap();
    let reset_observation = env
        .reset()
        .unwrap()
        .observation
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    let mut total_reward = 0.0;
    let mut final_observation = Vec::new();
    for index in 0..20 {
        let step = env.step(Tensor::from_data([index % 4], &device)).unwrap();
        total_reward += step.reward;
        final_observation = step.observation.into_data().try_to_vec::<f32>().unwrap();
    }
    let actual = (reset_observation, final_observation, total_reward);
    assert_eq!(
        actual,
        (
            vec![
                -0.0024300576,
                1.416066,
                -0.2461533,
                0.22869912,
                0.0028189588,
                0.055681963,
                0.0,
                0.0,
            ],
            vec![
                -0.052530956,
                1.4581959,
                -0.26300734,
                -0.038651336,
                0.10441409,
                0.100221366,
                0.0,
                0.0,
            ],
            -9.250286,
        )
    );
}

#[cfg(feature = "rendering")]
#[test]
#[ignore = "needs a display"]
fn lunar_lander_rendering_does_not_change_physics() {
    assert_eq!(lunar_lander_rollout(true), lunar_lander_rollout(false));
}

macro_rules! check_seeded_environment {
    ($builder:expr, $device:ident, $action:expr) => {{
        let action = $action;
        let rollout = |seed| {
            let mut env = ($builder).seed(seed).build().unwrap();
            let mut observations = vec![
                env.reset()
                    .unwrap()
                    .observation
                    .into_data()
                    .try_to_vec::<f32>()
                    .unwrap(),
            ];
            for _ in 0..3 {
                observations.push(
                    env.step(action.clone())
                        .unwrap()
                        .observation
                        .into_data()
                        .try_to_vec::<f32>()
                        .unwrap(),
                );
            }
            observations
        };
        let first = rollout(7);
        assert_eq!(rollout(7), first);
        assert_ne!(rollout(8)[0], first[0]);

        $device.seed(1);
        let expected =
            Tensor::<1>::random([7], Distribution::Uniform(0.0, 1.0), (&$device, DType::F32))
                .into_data();
        $device.seed(1);
        let mut env = ($builder).seed(7).build().unwrap();
        env.reset().unwrap();
        for _ in 0..3 {
            env.step(action.clone()).unwrap();
        }
        let actual =
            Tensor::<1>::random([7], Distribution::Uniform(0.0, 1.0), (&$device, DType::F32))
                .into_data();
        assert_eq!(actual, expected);
    }};
}

#[test]
fn all_environments_use_reproducible_host_rngs() {
    let _guard = RNG_LOCK.lock().unwrap();
    let device = Device::flex();
    check_seeded_environment!(
        CartPoleV1::builder(),
        device,
        Tensor::from_data([1i32], &device)
    );
    check_seeded_environment!(
        MountainCarV0::builder(),
        device,
        Tensor::from_data([1i32], &device)
    );
    check_seeded_environment!(
        AcrobotV1::builder().torque_noise_max(0.1),
        device,
        Tensor::from_data([1i32], &device)
    );
    check_seeded_environment!(
        PendulumV1::builder(),
        device,
        Tensor::from_data([0.0f32], &device)
    );
    check_seeded_environment!(
        LunarLanderV3::builder(),
        device,
        Tensor::from_data([2i32], &device)
    );
    check_seeded_environment!(
        BipedalWalkerV3::builder(),
        device,
        Tensor::zeros([4], (&device, DType::F32))
    );
}

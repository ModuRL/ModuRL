use burn::tensor::{DType, Device, Distribution, Tensor, kind::Basic};
use modurl::{
    gym::{Gym, MultiGym, VectorizedGymWrapper},
    spaces::{ActionSpace, BoxSpace, Discrete},
    wrappers::DeviceMultiGymWrapper,
};
use modurl_gym::{
    EnvironmentError,
    box_2d::{bipedal_walker::BipedalWalkerV3, lunar_lander::LunarLanderV3},
    classic_control::{
        acrobot::AcrobotV1, cartpole::CartPoleV1, mountain_car::MountainCarV0, pendulum::PendulumV1,
    },
};
use std::sync::Mutex;

static RNG_LOCK: Mutex<()> = Mutex::new(());

/// Checks CPU observations `[features]` using one native action `[action_components]`.
fn check_environment<G, K: Basic>(device: &Device, factory: impl Fn() -> G, action: Tensor<1, K>)
where
    G: Gym<Error = EnvironmentError, ObservationSpace = BoxSpace<2>>,
    G::ActionSpace: ActionSpace<2, Kind = K, Item = Tensor<1, K>>,
{
    let mut env = factory();
    assert_eq!(env.reset().unwrap().observation.device(), Device::flex());
    for _ in 0..3 {
        let step = env.step(action.clone()).unwrap();
        assert_eq!(step.observation.device(), Device::flex());
    }

    device.seed(42);
    let expected =
        Tensor::<1>::random([7], Distribution::Uniform(0.0, 1.0), (device, DType::F32)).into_data();
    device.seed(42);
    let mut env = factory();
    env.reset().unwrap();
    for _ in 0..3 {
        env.step(action.clone()).unwrap();
    }
    let actual =
        Tensor::<1>::random([7], Distribution::Uniform(0.0, 1.0), (device, DType::F32)).into_data();
    assert_eq!(
        actual, expected,
        "environment draws must not consume device RNG"
    );
}

#[test]
fn observations_stay_on_cpu_and_environment_draws_leave_device_rng_untouched() {
    let _guard = RNG_LOCK.lock().unwrap();
    let device = Device::flex();
    let discrete = || Tensor::from_data([1i32], &device);
    check_environment(
        &device,
        || CartPoleV1::builder().seed(42).build().unwrap(),
        discrete(),
    );
    check_environment(
        &device,
        || MountainCarV0::builder().seed(42).build().unwrap(),
        discrete(),
    );
    check_environment(
        &device,
        || {
            AcrobotV1::builder()
                .seed(42)
                .torque_noise_max(0.1)
                .build()
                .unwrap()
        },
        discrete(),
    );
    check_environment(
        &device,
        || PendulumV1::builder().seed(42).build().unwrap(),
        Tensor::from_data([0f32], &device),
    );
    check_environment(
        &device,
        || LunarLanderV3::builder().seed(42).build().unwrap(),
        discrete(),
    );
    check_environment(
        &device,
        || BipedalWalkerV3::builder().seed(42).build().unwrap(),
        Tensor::from_data([0f32; 4], &device),
    );
}

#[test]
fn batch_wrapper_preserves_observation_and_action_axes() {
    let _guard = RNG_LOCK.lock().unwrap();
    let device = Device::flex();
    let envs = (0..4)
        .map(|index| CartPoleV1::builder().seed(42 + index).build().unwrap())
        .collect::<Vec<_>>();
    let envs: VectorizedGymWrapper<CartPoleV1> = envs.try_into().unwrap();
    let mut gym = DeviceMultiGymWrapper::new(envs, Device::flex(), device.clone());
    let observations = gym.reset().unwrap();
    assert_eq!(observations.dims(), [4, 4]);
    assert_eq!(observations.device(), device);
    let step = gym.step(Tensor::from_data([[1i32]; 4], &device)).unwrap();
    assert_eq!(step.observations.dims(), [4, 4]);
    assert_eq!(step.rewards.dims(), [4, 1]);
    assert_eq!(step.observations.device(), device);
    assert_eq!(step.rewards.device(), device);
}

/// Checks discrete Int actions `[1]` and rejects incorrect item lengths or indices before changing observations.
fn check_discrete_actions<G>(mut env: G)
where
    G: Gym<Error = EnvironmentError, ObservationSpace = BoxSpace<2>, ActionSpace = Discrete>,
{
    env.reset().unwrap();
    let device = Device::flex();
    let action_count = env.action_space().get_possible_values() as i32;
    for action in [
        Tensor::from_data([0i32, 0], &device),
        Tensor::from_data([-1i32], &device),
        Tensor::from_data([action_count], &device),
    ] {
        assert!(matches!(
            env.step(action),
            Err(EnvironmentError::InvalidAction(_))
        ));
    }
    env.step(Tensor::from_data([0i32], &device)).unwrap();
    let action = env
        .action_space()
        .sample_batch(1, &device)
        .unwrap()
        .squeeze_dims::<1>(&[0]);
    assert_eq!(action.dims(), [1]);
    env.step(action).unwrap();
}

#[test]
fn discrete_actions_require_one_in_range_index() {
    let _guard = RNG_LOCK.lock().unwrap();
    check_discrete_actions(CartPoleV1::builder().build().unwrap());
    check_discrete_actions(MountainCarV0::builder().build().unwrap());
    check_discrete_actions(AcrobotV1::builder().build().unwrap());
    check_discrete_actions(LunarLanderV3::builder().build().unwrap());
}

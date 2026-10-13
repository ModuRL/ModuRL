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
    let rollout = || {
        let mut env = factory();
        let reset = env.reset().unwrap();
        assert_eq!(reset.observation.device(), Device::flex());
        let mut observations = vec![reset.observation.into_data().try_to_vec::<f32>().unwrap()];
        for _ in 0..3 {
            let step = env.step(action.clone()).unwrap();
            assert_eq!(step.observation.device(), Device::flex());
            observations.push(step.observation.into_data().try_to_vec::<f32>().unwrap());
        }
        let next = Tensor::<1>::random([7], Distribution::Uniform(0.0, 1.0), (device, DType::F32))
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        (observations, next)
    };
    device.seed(42);
    let first = rollout();
    device.seed(42);
    assert_eq!(
        first,
        rollout(),
        "environment must retain the selected device RNG"
    );
}

#[test]
fn observations_stay_on_cpu_and_rng_is_seeded() {
    let _guard = RNG_LOCK.lock().unwrap();
    let device = Device::flex();
    let discrete = || Tensor::from_data([1i32], &device);
    check_environment(
        &device,
        || CartPoleV1::builder().rng_device(&device).build().unwrap(),
        discrete(),
    );
    check_environment(
        &device,
        || {
            MountainCarV0::builder()
                .rng_device(&device)
                .build()
                .unwrap()
        },
        discrete(),
    );
    check_environment(
        &device,
        || {
            AcrobotV1::builder()
                .rng_device(&device)
                .torque_noise_max(0.1)
                .build()
                .unwrap()
        },
        discrete(),
    );
    check_environment(
        &device,
        || PendulumV1::builder().rng_device(&device).build().unwrap(),
        Tensor::from_data([0f32], &device),
    );
    check_environment(
        &device,
        || {
            LunarLanderV3::builder()
                .rng_device(device.clone())
                .build()
                .unwrap()
        },
        discrete(),
    );
    check_environment(
        &device,
        || {
            BipedalWalkerV3::builder()
                .rng_device(&device)
                .build()
                .unwrap()
        },
        Tensor::from_data([0f32; 4], &device),
    );
}

fn check_reset_rng<G>(device: &Device, mut env: G, size: usize, low: f64, high: f64, dtype: DType)
where
    G: Gym<Error = EnvironmentError, ObservationSpace = BoxSpace<2>>,
{
    device.seed(123);
    let _ =
        Tensor::<1>::random([size], Distribution::Uniform(low, high), (device, dtype)).into_data();
    let expected =
        Tensor::<1>::random([7], Distribution::Uniform(0.0, 1.0), (device, DType::F32)).into_data();
    device.seed(123);
    assert_eq!(env.reset().unwrap().observation.device(), Device::flex());
    let actual =
        Tensor::<1>::random([7], Distribution::Uniform(0.0, 1.0), (device, DType::F32)).into_data();
    assert_eq!(actual, expected);
}

#[test]
fn classic_control_reset_preserves_rng_advancement() {
    let _guard = RNG_LOCK.lock().unwrap();
    let device = Device::flex();
    check_reset_rng(
        &device,
        CartPoleV1::builder().rng_device(&device).build().unwrap(),
        4,
        -0.05,
        0.05,
        DType::F64,
    );
    check_reset_rng(
        &device,
        MountainCarV0::builder()
            .rng_device(&device)
            .build()
            .unwrap(),
        1,
        -0.6,
        -0.4,
        DType::F64,
    );
    check_reset_rng(
        &device,
        AcrobotV1::builder().rng_device(&device).build().unwrap(),
        4,
        -0.1,
        0.1,
        DType::F32,
    );
    check_reset_rng(
        &device,
        PendulumV1::builder().rng_device(&device).build().unwrap(),
        2,
        0.0,
        1.0,
        DType::F32,
    );
}

#[test]
fn batch_wrapper_preserves_observation_and_action_axes() {
    let _guard = RNG_LOCK.lock().unwrap();
    let device = Device::flex();
    let envs = (0..4)
        .map(|_| CartPoleV1::builder().rng_device(&device).build().unwrap())
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

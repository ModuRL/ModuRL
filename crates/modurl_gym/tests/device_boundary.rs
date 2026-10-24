use candle_core::{Device, Tensor};
use modurl::gym::{Gym, MultiGym, VectorizedGymWrapper};
use modurl::wrappers::DeviceMultiGymWrapper;
use modurl_gym::{
    EnvironmentError,
    box_2d::{bipedal_walker::BipedalWalkerV3, lunar_lander::LunarLanderV3},
    classic_control::{
        acrobot::AcrobotV1, cartpole::CartPoleV1, mountain_car::MountainCarV0, pendulum::PendulumV1,
    },
};

/// Checks an environment using an unbatched action shaped `action_space().shape()`.
fn check_environment<G: Gym<Error = EnvironmentError>>(
    device: &Device,
    factory: impl Fn() -> G,
    action: Tensor,
) {
    let rollout = || {
        let mut env = factory();
        let reset = env.reset().unwrap();
        assert!(reset.state.device().is_cpu());
        let mut observations = vec![reset.state.to_vec1::<f32>().unwrap()];
        for _ in 0..3 {
            let step = env.step(action.clone()).unwrap();
            assert!(step.state.device().is_cpu());
            observations.push(step.state.to_vec1::<f32>().unwrap());
        }
        let next = Tensor::rand(0f32, 1f32, 7, device)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        (observations, next)
    };
    if device.is_cpu() {
        rollout();
    } else {
        device.set_seed(42).unwrap();
        let first = rollout();
        device.set_seed(42).unwrap();
        assert_eq!(
            first,
            rollout(),
            "environment must retain the selected device RNG"
        );
    }
}

#[test]
fn observations_stay_on_cpu_with_cpu_or_cuda_rng() {
    for device in [Device::Cpu, Device::cuda_if_available(0).unwrap()] {
        let discrete = || Tensor::new(1u32, &Device::Cpu).unwrap();
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
            Tensor::new(&[0f32], &Device::Cpu).unwrap(),
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
            Tensor::new(&[0f32; 4], &Device::Cpu).unwrap(),
        );
    }
}

#[test]
fn classic_control_reset_preserves_cuda_rng_advancement() {
    let device = Device::cuda_if_available(0).unwrap();
    if device.is_cpu() {
        return;
    }
    let mut envs: Vec<Box<dyn Gym<Error = EnvironmentError, SpaceError = candle_core::Error>>> = vec![
        Box::new(CartPoleV1::builder().rng_device(&device).build().unwrap()),
        Box::new(
            MountainCarV0::builder()
                .rng_device(&device)
                .build()
                .unwrap(),
        ),
        Box::new(AcrobotV1::builder().rng_device(&device).build().unwrap()),
        Box::new(PendulumV1::builder().rng_device(&device).build().unwrap()),
    ];
    for (index, env) in envs.iter_mut().enumerate() {
        device.set_seed(123).unwrap();
        // Original reset draw shapes, dtypes and ranges, before any CPU transfer.
        match index {
            0 => {
                Tensor::rand(-0.05f64, 0.05, vec![4], &device).unwrap();
            }
            1 => {
                Tensor::rand(-0.6f64, -0.4, vec![1], &device).unwrap();
            }
            2 => {
                Tensor::rand(-0.1f32, 0.1, 4, &device).unwrap();
            }
            _ => {
                Tensor::rand(0f32, 1f32, 2, &device).unwrap();
            }
        }
        let expected = Tensor::rand(0f32, 1f32, 7, &device)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        device.set_seed(123).unwrap();
        assert!(env.reset().unwrap().state.device().is_cpu());
        let actual = Tensor::rand(0f32, 1f32, 7, &device)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(actual, expected);
    }
}

#[test]
fn batch_wrapper_moves_cpu_observations_to_agent_device() {
    let device = Device::cuda_if_available(0).unwrap();
    let envs = (0..4)
        .map(|_| CartPoleV1::builder().rng_device(&device).build().unwrap())
        .collect::<Vec<_>>();
    let mut gym = DeviceMultiGymWrapper::new(
        VectorizedGymWrapper::from(envs),
        Device::Cpu,
        device.clone(),
    );
    let states = gym.reset().unwrap();
    assert_eq!(states.dims(), &[4, 4]);
    assert!(states.device().same_device(&device));
    let step = gym.step(Tensor::new(&[1u32; 4], &device).unwrap()).unwrap();
    assert!(step.states.device().same_device(&device));
    assert!(step.rewards.device().same_device(&device));
}

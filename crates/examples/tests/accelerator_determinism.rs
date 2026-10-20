#![cfg(any(feature = "cuda", feature = "metal"))]

use burn::{
    nn::Tanh,
    optim::AdamWConfig,
    tensor::{DType, Device, Tensor},
};
use modurl::{prelude::*, sampling::shuffle_with_device_rng};
use modurl_gym::classic_control::cartpole::CartPoleV1;
use std::sync::Mutex;

static RNG_LOCK: Mutex<()> = Mutex::new(());

fn accelerator_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        Device::cuda(0).autodiff()
    }
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        Device::metal(burn::tensor::DeviceKind::DefaultDevice).autodiff()
    }
}

#[test]
fn mlp_initialization_is_device_seeded() {
    let _guard = RNG_LOCK.lock().unwrap();
    let device = accelerator_device();
    let input = Tensor::<2>::ones([1, 4], &device);
    let mut last_output = None;
    for _ in 0..10 {
        device.seed(42);
        let network = MLP::builder()
            .input_size(4)
            .output_size(2)
            .hidden_layer_sizes(vec![8, 8])
            .options(&device)
            .build()
            .unwrap();
        let output = network.forward(input.clone()).unwrap().into_data();
        if let Some(previous) = &last_output {
            assert_eq!(previous, &output);
        }
        last_output = Some(output);
    }
}

#[test]
fn shuffle_is_device_seeded() {
    let _guard = RNG_LOCK.lock().unwrap();
    let device = accelerator_device();
    let mut first: Vec<u32> = (0..100).collect();
    let mut second = first.clone();
    device.seed(42);
    shuffle_with_device_rng(&mut first, &device).unwrap();
    device.seed(42);
    shuffle_with_device_rng(&mut second, &device).unwrap();
    assert_eq!(first, second);
}

#[test]
fn cartpole_tensors_are_device_deterministic() {
    let _guard = RNG_LOCK.lock().unwrap();
    let device = accelerator_device();
    let cpu = Device::flex();
    let mut previous = None;
    for _ in 0..10 {
        cpu.seed(42);
        device.seed(42);
        let env =
            VectorizedGymWrapper::from(CartPoleV1::builder().rng_device(&cpu).build().unwrap());
        let mut env = DeviceMultiGymWrapper::new(env, cpu.clone(), device.clone());
        let mut observations = env.reset().unwrap();
        for _ in 0..10 {
            let actions = env.action_space().sample_batch(1, &device).unwrap();
            observations = env.step(actions).unwrap().observations;
        }
        let data = observations.into_data();
        if let Some(previous) = &previous {
            assert_eq!(previous, &data);
        }
        previous = Some(data);
    }
}

#[test]
fn ppo_training_is_device_deterministic() {
    let _guard = RNG_LOCK.lock().unwrap();
    let device = accelerator_device();
    let cpu = Device::flex();
    let mut previous = None;
    for _ in 0..3 {
        cpu.seed(42);
        device.seed(42);
        let env = VectorizedGymWrapper::new(
            (0..2)
                .map(|_| CartPoleV1::builder().rng_device(&cpu).build().unwrap())
                .collect(),
        )
        .unwrap();
        let mut env = DeviceMultiGymWrapper::new(env, cpu.clone(), device.clone());
        let model = |output| {
            MLP::builder()
                .input_size(4)
                .output_size(output)
                .hidden_layer_sizes(vec![8, 8])
                .activation(Tanh)
                .options((&device, DType::F32))
                .build()
                .unwrap()
        };
        let networks = SeparatePPONetwork::builder()
            .actor_network(ProbabilisticPolicyModel::new(
                model(2),
                CategoricalDistribution,
            ))
            .critic_network(model(1))
            .actor_optimizer(AdamWConfig::new().init())
            .critic_optimizer(AdamWConfig::new().init())
            .actor_learning_rate(3e-4)
            .critic_learning_rate(3e-4)
            .build()
            .unwrap();
        let mut agent = PPOAgent::builder()
            .network_info(networks)
            .observation_space(env.observation_space())
            .action_space(env.action_space())
            .batch_size(16)
            .mini_batch_size(4)
            .num_epochs(2)
            .training_horizon(32)
            .device(device.clone())
            .build()
            .unwrap();
        agent.learn(&mut env, 32).unwrap();
        let mut observations = env.reset().unwrap();
        let mut actions = Vec::new();
        for _ in 0..25 {
            let action = agent.act(&observations).unwrap();
            actions.push(action.clone().into_data());
            observations = env.step(action).unwrap().observations;
        }
        if let Some(previous) = &previous {
            assert_eq!(previous, &actions);
        }
        previous = Some(actions);
    }
}

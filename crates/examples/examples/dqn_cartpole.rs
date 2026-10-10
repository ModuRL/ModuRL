use burn::{
    nn::activation::Activation,
    optim::AdamWConfig,
    tensor::{DType, Device},
};
use modurl::prelude::*;
use modurl_gym::classic_control::cartpole::CartPoleV1;

const DTYPE: DType = DType::F32;

mod support;
use support::graphers::DQNGrapher;

fn main() {
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let device = Device::flex();
    #[cfg(feature = "cuda")]
    let device = Device::cuda(0);
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let device = Device::metal(burn::tensor::DeviceKind::DefaultDevice);
    let device = device.autodiff();
    println!("Using device: {device:?}");

    let environment_device = Device::flex();
    let envs = vec![
        CartPoleV1::builder()
            .rng_device(&environment_device)
            .build()
            .unwrap(),
    ];
    let env = VectorizedGymWrapper::from(envs);
    let mut env = DeviceMultiGymWrapper::new(env, environment_device, device.clone());
    let observation_space = env.observation_space();
    let online_q_network = MLP::builder()
        .input_size(observation_space.shape()[0])
        .output_size(env.action_space().get_possible_values())
        .options((&device, DTYPE))
        .activation(Activation::Tanh)
        .hidden_layer_sizes(vec![64, 64])
        .build()
        .expect("failed to build the online Q-network");
    let optimizer = AdamWConfig::new().init();

    let mut grapher = DQNGrapher::new();
    let mut agent = DQNAgent::builder()
        .dtype(DTYPE)
        .action_space(env.action_space())
        .observation_space(observation_space)
        .online_q_network(online_q_network)
        .optimizer(optimizer)
        .learning_rate(2.5e-4)
        .replay_capacity(10_000)
        .batch_size(128)
        .training_start(10_000)
        .update_frequency(10)
        .target_update_interval(500)
        .training_horizon(500_000)
        .epsilon_schedule(|progress: f64| {
            let exploration_progress = (progress / 0.5).min(1.0);
            1.0 + (0.05 - 1.0) * exploration_progress
        })
        .logger(&mut grapher)
        .replay_storage_config(ReplayStorageConfig::new(ReplayDeviceStrategy::OneDevice(
            device,
        )))
        .build()
        .expect("DQN configuration should be valid");

    agent.learn(&mut env, 500_000).expect("DQN learning failed");
    grapher.display();
}

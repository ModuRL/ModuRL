use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use modurl::prelude::*;
use modurl_gym::classic_control::cartpole::CartPoleV1;

#[path = "support/graphers.rs"]
mod graphers;
use graphers::DQNGrapher;

fn main() {
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let device = Device::Cpu;
    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0).unwrap();
    #[cfg(feature = "metal")]
    let device = Device::new_metal(0).unwrap();

    println!("Using device: {device:?}");

    let envs = vec![CartPoleV1::builder().device(&device).build()];
    let mut env = VectorizedGymWrapper::from(envs);
    let observation_space = env.observation_space();

    let online_var_map = VarMap::new();
    let online_q_network = MLP::builder()
        .input_size(observation_space.shape()[0])
        .output_size(2)
        .vb(VarBuilder::from_varmap(
            &online_var_map,
            DType::F32,
            &device,
        ))
        .activation(Box::new(Tensor::tanh))
        .hidden_layer_sizes(vec![64, 64])
        .build()
        .expect("failed to build the online Q-network");

    let mut target_var_map = VarMap::new();
    let target_q_network = MLP::builder()
        .input_size(observation_space.shape()[0])
        .output_size(2)
        .vb(VarBuilder::from_varmap(
            &target_var_map,
            DType::F32,
            &device,
        ))
        .activation(Box::new(Tensor::tanh))
        .hidden_layer_sizes(vec![64, 64])
        .build()
        .expect("failed to build the target Q-network");

    let optimizer = AdamW::new(
        online_var_map.all_vars(),
        ParamsAdamW {
            lr: 2.5e-4,
            ..Default::default()
        },
    )
    .expect("failed to build the optimizer");

    let mut grapher = DQNGrapher::new();
    let mut agent = DQNAgent::builder()
        .action_space(Discrete::new(2))
        .observation_space(observation_space)
        .online_q_network(Box::new(online_q_network))
        .target_q_network(Box::new(target_q_network))
        .online_vars(&online_var_map)
        .target_vars(&mut target_var_map)
        .optimizer(optimizer)
        .replay_capacity(10_000)
        .batch_size(128)
        .training_start(10_000)
        .update_frequency(10)
        .target_update_interval(500)
        .training_horizon(500_000)
        .epsilon_schedule(Box::new(|progress: f64| {
            let exploration_progress = (progress / 0.5).min(1.0);
            1.0 + (0.05 - 1.0) * exploration_progress
        }))
        .logger(&mut grapher)
        .device_strategy(QLearningDeviceStrategy::OneDevice(device))
        .build()
        .expect("DQN configuration should be valid");

    agent.learn(&mut env, 500_000).expect("DQN learning failed");
    drop(agent);
    grapher.display();
}

use burn::{
    nn::Tanh,
    optim::AdamWConfig,
    tensor::{DType, Device},
};
use modurl::prelude::*;
use modurl_gym::classic_control::cartpole::CartPoleV1;

#[test]
fn getting_started_program_trains_one_rollout() {
    let device = Device::flex().autodiff();

    let envs = (0..4)
        .map(|_| {
            CartPoleV1::builder()
                .rng_device(&Device::flex())
                .build()
                .unwrap()
        })
        .collect::<Vec<_>>();
    let mut env = DeviceMultiGymWrapper::new(
        VectorizedGymWrapper::new(envs).unwrap(),
        Device::flex(),
        device.clone(),
    );

    let observation_space = env.observation_space();
    let action_space = env.action_space();

    let actor_network = MLP::builder()
        .input_size(observation_space.shape()[0])
        .output_size(action_space.get_possible_values())
        .options((&device, DType::F32))
        .activation(Tanh)
        .hidden_layer_sizes(vec![64, 64])
        .build()
        .expect("failed to build actor network");

    let critic_network = MLP::builder()
        .input_size(observation_space.shape()[0])
        .output_size(1)
        .options((&device, DType::F32))
        .activation(Tanh)
        .hidden_layer_sizes(vec![64, 64])
        .build()
        .expect("failed to build critic network");

    let optimizer_config = AdamWConfig::new();
    let actor_optimizer = optimizer_config.init();
    let critic_optimizer = optimizer_config.init();

    let policy = ProbabilisticPolicyModel::new(actor_network, CategoricalDistribution);

    let network_info = SeparatePPONetwork::builder()
        .actor_network(policy)
        .critic_network(critic_network)
        .actor_optimizer(actor_optimizer)
        .critic_optimizer(critic_optimizer)
        .actor_learning_rate(3e-4)
        .critic_learning_rate(3e-4)
        .build()
        .unwrap();

    let mut agent = PPOAgent::builder()
        .observation_space(observation_space)
        .action_space(action_space)
        .network_info(network_info)
        .batch_size(1024)
        .mini_batch_size(64)
        .clip_range(ConstantSchedule::new(0.2))
        .training_horizon(1024)
        .device(device)
        .build()
        .unwrap();

    agent.learn(&mut env, 1024).expect("PPO learning failed");
    println!("Training complete.");
}

use burn::{
    nn::Initializer,
    optim::AdamWConfig,
    tensor::{DType, Device},
};
use modurl::prelude::*;
use modurl_gym::classic_control::cartpole::CartPoleV1;

const DTYPE: DType = DType::F32;

fn main() {
    ppo_cartpole();
}

fn ppo_cartpole() {
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let device = Device::flex();
    #[cfg(feature = "cuda")]
    let device = Device::cuda(0);
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let device = Device::metal(burn::tensor::DeviceKind::DefaultDevice);

    let device = device.autodiff();
    println!("Using device: {:?}", device);

    let mut envs = vec![];
    for index in 0..8 {
        let env = CartPoleV1::builder().seed(42 + index).build().unwrap();
        envs.push(env);
    }
    let vec_env: VectorizedGymWrapper<CartPoleV1> = VectorizedGymWrapper::new(envs).unwrap();
    let mut vec_env = DeviceMultiGymWrapper::new(vec_env, Device::flex(), device.clone());

    let observation_space = vec_env.observation_space();
    let action_space = vec_env.action_space();

    // Actor network: 2x64, tanh activation
    let actor_network = MLP::builder()
        .input_size(observation_space.shape()[0])
        .output_size(action_space.get_possible_values())
        .options((&device, DTYPE))
        .activation(burn::nn::Tanh)
        .hidden_layer_sizes(vec![64, 64])
        .hidden_initializer(Initializer::Orthogonal {
            gain: 2.0f64.sqrt(),
        })
        .output_initializer(Initializer::Orthogonal { gain: 0.01 })
        .build()
        .unwrap();
    // Optimizers: both with lr=3e-4
    let config = AdamWConfig::new().with_weight_decay(0.01);
    let actor_optimizer = config.init();

    // Critic network: 2x64, tanh activation
    let critic_network = MLP::builder()
        .input_size(observation_space.shape()[0])
        .output_size(1)
        .options((&device, DTYPE))
        .activation(burn::nn::Tanh)
        .hidden_layer_sizes(vec![64, 64])
        .hidden_initializer(Initializer::Orthogonal {
            gain: 2.0f64.sqrt(),
        })
        .output_initializer(Initializer::Orthogonal { gain: 1.0 })
        .build()
        .unwrap();

    let critic_optimizer = config.init();

    let ppo_network_info = SeparatePPONetwork::builder()
        .actor_network(ProbabilisticPolicyModel::new(
            actor_network,
            CategoricalDistribution,
        ))
        .critic_network(critic_network)
        .actor_optimizer(actor_optimizer)
        .critic_optimizer(critic_optimizer)
        .actor_learning_rate(3e-4)
        .critic_learning_rate(3e-4)
        .build()
        .unwrap();

    // PPO config
    // Stable baselines3 config:
    let mut agent = PPOAgent::builder()
        .observation_space(observation_space)
        .action_space(action_space)
        .network_info(ppo_network_info)
        .batch_size(2048)
        .mini_batch_size(64)
        .normalize_advantage(true)
        .ent_coef(0.005)
        .gamma(0.99)
        .vf_coef(0.5)
        .clip_range(ConstantSchedule::new(0.2))
        .clipped(true)
        .gae_lambda(0.95)
        .num_epochs(10)
        .training_horizon(1_000_000)
        .device(device)
        .build()
        .unwrap();

    let start = std::time::Instant::now();
    for _ in 0..10 {
        agent.learn(&mut vec_env, 100_000).unwrap();
    }
    let duration = start.elapsed();
    println!("Training took: {:?} per 100,000 steps", duration / 10);
}

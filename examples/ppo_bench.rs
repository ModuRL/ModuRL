use candle_core::Device;
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use modurl::prelude::*;
use modurl_gym::classic_control::cartpole::CartPoleV1;

fn main() {
    ppo_cartpole();
}

fn ppo_cartpole() {
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let device = Device::Cpu;
    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0).unwrap();
    #[cfg(feature = "metal")]
    let device = Device::new_metal(0).unwrap();

    println!("Using device: {:?}", device);

    let mut envs = vec![];
    for _ in 0..8 {
        let env = CartPoleV1::builder().device(&device).build();
        envs.push(env);
    }
    let mut vec_env: VectorizedGymWrapper<CartPoleV1> = envs.into();

    let observation_space = vec_env.observation_space();
    let action_space = vec_env.action_space();
    let var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &device);

    // Actor network: 2x64, tanh activation
    let actor_network = MLP::builder()
        .input_size(observation_space.shape()[0])
        .output_size(action_space.shape()[0])
        .vb(vb.clone())
        .activation(Box::new(tanh))
        .hidden_layer_sizes(vec![64, 64])
        .initializer(Box::new(OrthogonalMLPInitializer {
            hidden_gain: 2.0f64.sqrt(),
            output_gain: 0.01,
        }))
        .name("actor_network".to_string())
        .build()
        .unwrap();
    // Optimizers: both with lr=3e-4
    let config = ParamsAdamW {
        lr: 3e-4,
        ..Default::default()
    };
    let actor_optimizer =
        AdamW::new(var_map.all_vars(), config.clone()).expect("Failed to create AdamW");

    let critic_var_map = VarMap::new();
    let critic_vb = VarBuilder::from_varmap(&critic_var_map, candle_core::DType::F32, &device);

    // Critic network: 2x64, tanh activation
    let critic_network = MLP::builder()
        .input_size(observation_space.shape()[0])
        .output_size(1)
        .vb(critic_vb)
        .activation(Box::new(tanh))
        .hidden_layer_sizes(vec![64, 64])
        .initializer(Box::new(OrthogonalMLPInitializer {
            hidden_gain: 2.0f64.sqrt(),
            output_gain: 1.0,
        }))
        .name("critic_network".to_string())
        .build()
        .unwrap();

    let critic_optimizer =
        AdamW::new(critic_var_map.all_vars(), config.clone()).expect("Failed to create AdamW");

    let ppo_network_info = PPONetworkInfo::Separate(
        SeparatePPONetwork::builder()
            .actor_network(Box::new(
                ProbabilisticPolicyModel::<CategoricalDistribution>::new(Box::new(actor_network)),
            ))
            .critic_network(Box::new(critic_network))
            .actor_optimizer(actor_optimizer)
            .critic_optimizer(critic_optimizer)
            .build(),
    );

    // PPO config
    // Stable baselines3 config:
    let mut agent = PPOAgent::builder()
        .action_space(action_space)
        .network_info(ppo_network_info)
        .batch_size(2048)
        .mini_batch_size(64)
        .normalize_advantage(true)
        .ent_coef(0.005)
        .gamma(0.99)
        .vf_coef(0.5)
        .clip_range(Box::new(ConstantSchedule::new(0.2)))
        .clipped(true)
        .gae_lambda(0.95)
        .num_epochs(10)
        .device(device)
        .build();

    let start = std::time::Instant::now();
    for _ in 0..10 {
        agent.learn(&mut vec_env, 100_000).unwrap();
    }
    let duration = start.elapsed();
    println!("Training took: {:?} per 100,000 steps", duration / 10);
}

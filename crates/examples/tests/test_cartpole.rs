use burn::{
    nn::{Initializer, Tanh},
    optim::AdamWConfig,
    tensor::Device,
};
use modurl::prelude::*;
use modurl_gym::classic_control::cartpole::CartPoleV1;
use std::sync::Mutex;

static RNG_LOCK: Mutex<()> = Mutex::new(());

fn device() -> Device {
    #[cfg(feature = "cuda")]
    {
        Device::cuda(0).autodiff()
    }
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        Device::metal(burn::tensor::DeviceKind::DefaultDevice).autodiff()
    }
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    {
        Device::flex().autodiff()
    }
}

fn model(input: usize, output: usize, hidden: Vec<usize>, gain: f64, device: &Device) -> MLP {
    MLP::builder()
        .input_size(input)
        .output_size(output)
        .hidden_layer_sizes(hidden)
        .activation(Tanh)
        .hidden_initializer(Initializer::Orthogonal {
            gain: 2.0f64.sqrt(),
        })
        .output_initializer(Initializer::Orthogonal { gain })
        .options(device)
        .build()
        .unwrap()
}

/// Evaluates Float observations [1, 4] with Int actions [1, 1] over 100 completed CartPole episodes.
fn get_average_steps<A: Agent<ObservationSpace = BoxSpace<2>, ActionSpace = Discrete>>(
    agent: &mut A,
    device: &Device,
) -> f32
where
    A::Error: std::fmt::Debug,
{
    let env = VectorizedGymWrapper::from(CartPoleV1::builder().build().unwrap());
    let mut env = DeviceMultiGymWrapper::new(env, Device::flex(), device.clone());
    let mut observations = env.reset().unwrap();
    let mut total_steps = 0;
    let mut episodes = 0;
    while episodes < 100 {
        let step = env.step(agent.act(&observations).unwrap()).unwrap();
        episodes += usize::from(step.dones[0] || step.truncateds[0]);
        total_steps += 1;
        observations = step.observations;
    }
    total_steps as f32 / 100.0
}

#[test]
#[ignore = "slow solve test; run manually"]
fn ppo_cartpole() {
    let _guard = RNG_LOCK.lock().unwrap();
    let device = device();
    device.seed(42);
    let env = VectorizedGymWrapper::from(CartPoleV1::builder().build().unwrap());
    let mut env = DeviceMultiGymWrapper::new(env, Device::flex(), device.clone());
    let networks = SeparatePPONetwork::builder()
        .actor_network(ProbabilisticPolicyModel::new(
            model(4, 2, vec![64, 64], 0.01, &device),
            CategoricalDistribution,
        ))
        .critic_network(model(4, 1, vec![64, 64], 1.0, &device))
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
        .training_horizon(120_000)
        .device(device.clone())
        .build()
        .unwrap();
    for _ in 0..6 {
        agent.learn(&mut env, 20_000).unwrap();
        if get_average_steps(&mut agent, &device) >= 195.0 {
            return;
        }
    }
    panic!("PPO failed to solve CartPole-v1 within 120000 timesteps");
}

fn shared_network(
    device: &Device,
) -> SharedPPONetwork<
    MLP,
    impl burn::module::Module
    + ProbabilisticPolicy<2, 2, ObservationKind = burn::tensor::Float, Error: std::fmt::Debug>,
    MLP,
> {
    let trunk = MLP::builder()
        .input_size(4)
        .output_size(64)
        .hidden_layer_sizes(vec![64])
        .activation(Tanh)
        .output_activation(Tanh)
        .hidden_initializer(Initializer::Orthogonal {
            gain: 2.0f64.sqrt(),
        })
        .output_initializer(Initializer::Orthogonal {
            gain: 2.0f64.sqrt(),
        })
        .options(device)
        .build()
        .unwrap();
    SharedPPONetwork::builder()
        .shared_network(trunk)
        .actor_head(ProbabilisticPolicyModel::new(
            model(64, 2, vec![], 0.01, device),
            CategoricalDistribution,
        ))
        .critic_head(model(64, 1, vec![], 1.0, device))
        .optimizer(AdamWConfig::new().init())
        .learning_rate(3e-4)
        .build()
        .unwrap()
}

#[test]
#[ignore = "slow solve test; run manually"]
fn ppo_cartpole_shared() {
    let _guard = RNG_LOCK.lock().unwrap();
    let device = device();
    device.seed(42);
    let env = VectorizedGymWrapper::new(
        (0..8)
            .map(|_| CartPoleV1::builder().build().unwrap())
            .collect(),
    )
    .unwrap();
    let mut env = DeviceMultiGymWrapper::new(env, Device::flex(), device.clone());
    let vf_coef = std::env::var("TEST_VF_COEF")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0.5);
    let gradient_clip = std::env::var("TEST_GRAD_CLIP")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0.5);
    let mut agent = PPOAgent::builder()
        .network_info(shared_network(&device))
        .observation_space(env.observation_space())
        .action_space(env.action_space())
        .batch_size(2048)
        .mini_batch_size(64)
        .normalize_advantage(true)
        .ent_coef(0.005)
        .gamma(0.99)
        .vf_coef(vf_coef)
        .clip_range(ConstantSchedule::new(0.2))
        .clipped(true)
        .gae_lambda(0.95)
        .num_epochs(10)
        .gradient_clip(gradient_clip)
        .training_horizon(120_000)
        .device(device.clone())
        .build()
        .unwrap();
    for _ in 0..6 {
        agent.learn(&mut env, 20_000).unwrap();
        if get_average_steps(&mut agent, &device) >= 195.0 {
            return;
        }
    }
    panic!("Shared PPO failed to solve CartPole-v1 within 120000 timesteps");
}

#[test]
#[ignore = "slow solve test; run manually"]
#[cfg(feature = "multithreading")]
fn ppo_cartpole_shared_multithreaded() {
    let _guard = RNG_LOCK.lock().unwrap();
    let device = device();
    device.seed(42);
    let constructors: Vec<_> = (0..8)
        .map(|_| || CartPoleV1::builder().build().unwrap())
        .collect();
    let space = CartPoleV1::builder().build().unwrap().observation_space();
    let env =
        MultithreadedVectorizedGymWrapper::new(constructors, space, Discrete::new(2)).unwrap();
    let mut env = DeviceMultiGymWrapper::new(env, Device::flex(), device.clone());
    let mut agent = PPOAgent::builder()
        .network_info(shared_network(&device))
        .observation_space(env.observation_space())
        .action_space(env.action_space())
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
        .training_horizon(120_000)
        .device(device.clone())
        .build()
        .unwrap();
    for _ in 0..6 {
        agent.learn(&mut env, 20_000).unwrap();
        if get_average_steps(&mut agent, &device) >= 195.0 {
            return;
        }
    }
    panic!("Multithreaded shared PPO failed to solve CartPole-v1 within 120000 timesteps");
}

#[test]
#[ignore = "slow solve test; run manually for review prep"]
fn dqn_cartpole() {
    let _guard = RNG_LOCK.lock().unwrap();
    let environment_device = burn::tensor::Device::flex();
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let device = environment_device.clone().autodiff();
    #[cfg(feature = "cuda")]
    let device = burn::tensor::Device::cuda(0).autodiff();
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let device = burn::tensor::Device::metal(burn::tensor::DeviceKind::DefaultDevice).autodiff();
    device.seed(42);
    let env = VectorizedGymWrapper::from(
        CartPoleV1::builder()
            .rng_device(&environment_device)
            .build()
            .unwrap(),
    );
    let mut env = DeviceMultiGymWrapper::new(env, environment_device, device.clone());
    let online_q_network = MLP::builder()
        .input_size(4)
        .output_size(2)
        .options(&device)
        .hidden_layer_sizes(vec![64, 64])
        .build()
        .unwrap();
    let mut agent = DQNAgent::builder()
        .action_space(env.action_space())
        .observation_space(env.observation_space())
        .online_q_network(online_q_network)
        .optimizer(burn::optim::AdamWConfig::new().init())
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
        .replay_storage_config(ReplayStorageConfig::new(ReplayDeviceStrategy::OneDevice(
            device,
        )))
        .build()
        .unwrap();
    const TRAINING_TIMESTEPS: usize = 500_000;
    agent.learn(&mut env, TRAINING_TIMESTEPS).unwrap();
    let mut observations = env.reset().unwrap();
    let mut episode_count = 0;
    let mut total_steps = 0;
    while episode_count < 100 {
        let actions = agent.act(&observations).unwrap();
        let step = env.step(actions).unwrap();
        total_steps += 1;
        episode_count += usize::from(step.dones[0] || step.truncateds[0]);
        observations = step.observations;
    }
    let average_steps = total_steps as f64 / 100.0;
    assert!(
        average_steps >= 195.0,
        "DQN averaged {average_steps} steps after {TRAINING_TIMESTEPS} transitions"
    );
}

#[test]
#[ignore = "slow solve test; run manually for review prep"]
fn ddqn_cartpole() {
    let _guard = RNG_LOCK.lock().unwrap();
    let environment_device = burn::tensor::Device::flex();
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let device = environment_device.clone().autodiff();
    #[cfg(feature = "cuda")]
    let device = burn::tensor::Device::cuda(0).autodiff();
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let device = burn::tensor::Device::metal(burn::tensor::DeviceKind::DefaultDevice).autodiff();
    device.seed(42);
    let env = VectorizedGymWrapper::from(
        CartPoleV1::builder()
            .rng_device(&environment_device)
            .build()
            .unwrap(),
    );
    let mut env = DeviceMultiGymWrapper::new(env, environment_device, device.clone());
    let online_q_network = MLP::builder()
        .input_size(4)
        .output_size(2)
        .options(&device)
        .hidden_layer_sizes(vec![64, 64])
        .build()
        .unwrap();
    let mut agent = DDQNAgent::builder()
        .action_space(env.action_space())
        .observation_space(env.observation_space())
        .online_q_network(online_q_network)
        .optimizer(burn::optim::AdamWConfig::new().init())
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
        .replay_storage_config(ReplayStorageConfig::new(ReplayDeviceStrategy::OneDevice(
            device,
        )))
        .build()
        .unwrap();
    const TRAINING_TIMESTEPS: usize = 500_000;
    agent.learn(&mut env, TRAINING_TIMESTEPS).unwrap();
    let mut observations = env.reset().unwrap();
    let mut episode_count = 0;
    let mut total_steps = 0;
    while episode_count < 100 {
        let actions = agent.act(&observations).unwrap();
        let step = env.step(actions).unwrap();
        total_steps += 1;
        episode_count += usize::from(step.dones[0] || step.truncateds[0]);
        observations = step.observations;
    }
    let average_steps = total_steps as f64 / 100.0;
    assert!(
        average_steps >= 195.0,
        "DDQN averaged {average_steps} steps after {TRAINING_TIMESTEPS} transitions"
    );
}

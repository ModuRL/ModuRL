use burn::{
    nn::{Relu, Tanh},
    optim::AdamWConfig,
    tensor::{DType, Device},
};
use modurl::agents::action_value_critic::ScalarStateActionCritic;
use modurl::prelude::*;

mod support;
use support::{
    graphers::DeterministicActorCriticGrapher,
    mujoco::{self, ENVIRONMENT_NAME},
};

const TOTAL_TIMESTEPS: usize = 1_000_000;
const DTYPE: DType = DType::F32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let device = Device::flex();
    #[cfg(feature = "cuda")]
    let device = Device::cuda(0);
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let device = Device::metal(burn::tensor::DeviceKind::DefaultDevice);
    let device = device.autodiff();
    println!("Environment: {ENVIRONMENT_NAME}");
    println!("Using device: {device:?}");
    let env = VectorizedGymWrapper::from(TimeLimitGym::new(mujoco::build_environment(), 1_000));
    let mut env = DeviceMultiGymWrapper::new(env, Device::flex(), device.clone());
    let observation_space = env.observation_space();
    let action_space = env.action_space();
    let observation_size = observation_space.shape().iter().product();
    let action_size = action_space.shape().iter().product();
    let online_actor = MLP::builder()
        .input_size(observation_size)
        .output_size(action_size)
        .hidden_layer_sizes(vec![64, 64])
        .activation(Relu)
        .output_activation(Tanh)
        .options((&device, DTYPE))
        .build()?;
    let mut critics = Vec::new();
    for _ in 0..2 {
        let online_critic = MLP::builder()
            .input_size(observation_size + action_size)
            .output_size(1)
            .hidden_layer_sizes(vec![64, 64])
            .activation(Relu)
            .options((&device, DTYPE))
            .build()?;
        critics.push(
            ActionValueCritic::builder()
                .online_network(ScalarStateActionCritic::new(online_critic))
                .optimizer(AdamWConfig::new().with_weight_decay(0.0).init())
                .learning_rate(1e-3)
                .build()?,
        );
    }
    let mut grapher = DeterministicActorCriticGrapher::new();
    let mut agent = TD3Agent::builder()
        .dtype(DTYPE)
        .online_actor(online_actor)
        .actor_optimizer(AdamWConfig::new().with_weight_decay(0.0).init())
        .actor_learning_rate(1e-3)
        .critics(critics)
        .action_space(action_space)
        .observation_space(observation_space)
        .gamma(0.99)
        .tau(0.005)
        .exploration_noise(0.1)
        .target_policy_noise(0.2)
        .target_noise_clip(0.5)
        .actor_update_interval(2)
        .replay_capacity(1_000_000)
        .batch_size(256)
        .training_start(10_000)
        .training_horizon(TOTAL_TIMESTEPS)
        .logger(&mut grapher)
        .build()?;
    agent.learn(&mut env, TOTAL_TIMESTEPS)?;
    grapher.display();
    Ok(())
}

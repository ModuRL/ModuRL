use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use modurl::prelude::*;

#[path = "support/graphers.rs"]
mod graphers;
use graphers::DeterministicActorCriticGrapher;

const TOTAL_TIMESTEPS: usize = 1_000_000;

#[cfg(not(any(feature = "half-cheetah", feature = "hopper", feature = "walker2d")))]
compile_error!("enable exactly one MuJoCo environment feature: half-cheetah, hopper, or walker2d");

#[cfg(any(
    all(feature = "half-cheetah", feature = "hopper"),
    all(feature = "half-cheetah", feature = "walker2d"),
    all(feature = "hopper", feature = "walker2d"),
))]
compile_error!("enable exactly one MuJoCo environment feature: half-cheetah, hopper, or walker2d");

#[cfg(feature = "half-cheetah")]
use modurl_mojoco::HalfCheetahV5 as SelectedEnvironment;
#[cfg(all(not(feature = "half-cheetah"), feature = "hopper"))]
use modurl_mojoco::HopperV5 as SelectedEnvironment;
#[cfg(all(not(feature = "half-cheetah"), not(feature = "hopper")))]
use modurl_mojoco::Walker2dV5 as SelectedEnvironment;

#[cfg(feature = "half-cheetah")]
const ENVIRONMENT_NAME: &str = "HalfCheetah-v5";
#[cfg(all(not(feature = "half-cheetah"), feature = "hopper"))]
const ENVIRONMENT_NAME: &str = "Hopper-v5";
#[cfg(all(not(feature = "half-cheetah"), not(feature = "hopper")))]
const ENVIRONMENT_NAME: &str = "Walker2d-v5";

fn actor(
    variables: &VarMap,
    observation_size: usize,
    action_size: usize,
    device: &Device,
) -> candle_core::Result<MLP> {
    MLP::builder()
        .input_size(observation_size)
        .output_size(action_size)
        .vb(VarBuilder::from_varmap(variables, DType::F32, device))
        .hidden_layer_sizes(vec![64, 64])
        .activation(Box::new(Tensor::relu))
        .output_activation(Box::new(Tensor::tanh))
        .name("actor".to_owned())
        .build()
}

fn critic<'a>(
    online_variables: &'a VarMap,
    target_variables: &'a mut VarMap,
    observation_size: usize,
    action_size: usize,
    optimizer_parameters: &ParamsAdamW,
    device: &Device,
) -> Result<DeterministicCritic<'a, AdamW>, SACCriticError> {
    let network = |variables| {
        MLP::builder()
            .input_size(observation_size + action_size)
            .output_size(1)
            .vb(VarBuilder::from_varmap(variables, DType::F32, device))
            .hidden_layer_sizes(vec![64, 64])
            .activation(Box::new(Tensor::relu))
            .name("critic".to_owned())
            .build()
    };
    let online = network(online_variables)?;
    let target = network(target_variables)?;

    DeterministicCritic::builder()
        .online_network(Box::new(ScalarStateActionCritic::new(Box::new(online))))
        .target_network(Box::new(ScalarStateActionCritic::new(Box::new(target))))
        .online_vars(online_variables)
        .target_vars(target_variables)
        .optimizer(AdamW::new(
            online_variables.all_vars(),
            optimizer_parameters.clone(),
        )?)
        .build()
}

fn main() {
    let device = Device::cuda_if_available(0).unwrap();
    println!("Environment: {ENVIRONMENT_NAME}");
    println!("Using device: {device:?}");

    let mut env = VectorizedGymWrapper::from(vec![TimeLimitGym::new(
        SelectedEnvironment::builder()
            .device(&device)
            .build()
            .unwrap(),
        1_000,
    )]);
    let observation_space = env.observation_space();
    let environment_action_space = env.action_space();
    let observation_size = observation_space.shape()[0];
    let action_shape = environment_action_space.shape();
    let action_size = action_shape.iter().product();
    let action_space = BoxSpace::new_with_universal_bounds(action_shape, -1.0, 1.0, &device);
    let optimizer_parameters = ParamsAdamW {
        lr: 1e-3,
        weight_decay: 0.0,
        ..Default::default()
    };

    let online_actor_variables = VarMap::new();
    let mut target_actor_variables = VarMap::new();
    let online_actor = actor(
        &online_actor_variables,
        observation_size,
        action_size,
        &device,
    )
    .unwrap();
    let target_actor = actor(
        &target_actor_variables,
        observation_size,
        action_size,
        &device,
    )
    .unwrap();
    let actor_optimizer = AdamW::new(
        online_actor_variables.all_vars(),
        optimizer_parameters.clone(),
    )
    .unwrap();

    let online_critic_variables_1 = VarMap::new();
    let mut target_critic_variables_1 = VarMap::new();
    let critic_1 = critic(
        &online_critic_variables_1,
        &mut target_critic_variables_1,
        observation_size,
        action_size,
        &optimizer_parameters,
        &device,
    )
    .unwrap();
    let online_critic_variables_2 = VarMap::new();
    let mut target_critic_variables_2 = VarMap::new();
    let critic_2 = critic(
        &online_critic_variables_2,
        &mut target_critic_variables_2,
        observation_size,
        action_size,
        &optimizer_parameters,
        &device,
    )
    .unwrap();
    let mut grapher = DeterministicActorCriticGrapher::new();

    let mut agent = TD3Agent::builder()
        .online_actor(Box::new(online_actor))
        .target_actor(Box::new(target_actor))
        .online_actor_vars(&online_actor_variables)
        .target_actor_vars(&mut target_actor_variables)
        .actor_optimizer(actor_optimizer)
        .critics(vec![critic_1, critic_2])
        .action_space(action_space)
        .observation_space(observation_space)
        .device_strategy(ReplayDeviceStrategy::OneDevice(device))
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
        .build()
        .unwrap();

    agent.learn(&mut env, TOTAL_TIMESTEPS).unwrap();
    grapher.display();
}

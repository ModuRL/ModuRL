use burn::module::Param;
use burn::{
    nn::{Initializer, Tanh},
    optim::AdamWConfig,
    tensor::{DType, Device, Tensor},
};
use modurl::prelude::*;
use modurl_gym::classic_control::cartpole::CartPoleV1;

mod support;
use support::graphers::SACGrapher;

const DTYPE: DType = DType::F32;

fn sac_mlp(
    input_size: usize,
    output_size: usize,
    output_gain: f64,
    device: &Device,
) -> Result<MLP, ModelError> {
    MLP::builder()
        .input_size(input_size)
        .output_size(output_size)
        .hidden_layer_sizes(vec![64, 64])
        .activation(Tanh)
        .hidden_initializer(Initializer::Orthogonal {
            gain: 2.0_f64.sqrt(),
        })
        .output_initializer(Initializer::Orthogonal { gain: output_gain })
        .options((device, DTYPE))
        .build()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let total_timesteps = std::env::var("MODURL_SAC_TIMESTEPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(100_000);
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let device = Device::flex();
    #[cfg(feature = "cuda")]
    let device = Device::cuda(0);
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let device = Device::metal(burn::tensor::DeviceKind::DefaultDevice);
    let device = device.autodiff();
    let environment_device = Device::flex();
    let env = VectorizedGymWrapper::from(CartPoleV1::builder().seed(42).build()?);
    let mut env = DeviceMultiGymWrapper::new(env, environment_device, device.clone());
    let observation_space = env.observation_space();
    let action_space = env.action_space();
    let observation_size = observation_space.shape()[0];
    let action_count = action_space.get_possible_values();
    let policy = ProbabilisticPolicyModel::new(
        sac_mlp(observation_size, action_count, 0.01, &device)?,
        CategoricalDistribution,
    );
    let mut critics = Vec::new();
    for _ in 0..2 {
        critics.push(
            ActionValueCritic::builder()
                .online_network(DiscreteVectorHeadCritic::new(sac_mlp(
                    observation_size,
                    action_count,
                    1.0,
                    &device,
                )?))
                .optimizer(AdamWConfig::new().with_weight_decay(0.0).init())
                .learning_rate(3e-4)
                .build()?,
        );
    }
    let entropy = SACEntropyConfiguration::automatic_with_target_schedule(
        Param::from_tensor(Tensor::zeros([1], (&device, DTYPE))),
        AdamWConfig::new().with_weight_decay(0.0).init(),
        3e-4,
        move |progress| (1.0 - progress) * (action_count as f64).ln(),
    )?;
    let mut grapher = SACGrapher::new();
    let mut agent = SACAgent::builder()
        .dtype(DTYPE)
        .policy(policy)
        .actor_optimizer(AdamWConfig::new().with_weight_decay(0.0).init())
        .actor_learning_rate(3e-4)
        .critics(critics)
        .entropy_configuration(entropy)
        .action_space(action_space)
        .observation_space(observation_space)
        .aggregation_mode(ActionValueCriticAggregationMode::Mean)
        .entropy_change_penalty(0.5)
        .replay_capacity(100_000)
        .batch_size(64)
        .training_horizon(total_timesteps)
        .logger(&mut grapher)
        .replay_storage_config(ReplayStorageConfig::new(ReplayDeviceStrategy::OneDevice(
            device,
        )))
        .build()?;
    agent.learn(&mut env, total_timesteps)?;
    grapher.display();
    Ok(())
}

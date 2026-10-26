use burn::{
    module::{Module, Param},
    tensor::Float,
};
use burn::{
    nn::{Initializer, Tanh},
    optim::AdamWConfig,
    tensor::{DType, Device, Tensor},
};
use modurl::prelude::*;

mod support;
use support::{
    graphers::SACGrapher,
    mujoco::{self, ENVIRONMENT_NAME},
};

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

/// Produces Gaussian parameters `[batch_size, 2 * action_size]` from native observations `[batch_size, observation_size]`.
#[derive(Module, Debug)]
struct GaussianModule {
    mean: MLP,
    log_std: Param<Tensor<2>>,
}

impl Forward<2, 2> for GaussianModule {
    type InputKind = Float;
    type OutputKind = Float;
    type Error = ModelError;

    /// Broadcasts learned log standard deviations `[1, action_size]` over the batch axis and joins them with the means.
    /// Maps observations `[batch_size, observation_size]` to `[batch_size, 2 * action_size]`, preserving model dtype, device, and gradients.
    fn forward(&self, observations: Tensor<2>) -> Result<Tensor<2>, Self::Error> {
        let mean = self.mean.forward(observations)?;
        let log_std = self.log_std.val().expand(mean.dims()).clamp(-20.0, 2.0);
        Ok(Tensor::cat(vec![mean, log_std], 1))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let total_timesteps = std::env::var("MODURL_SAC_TIMESTEPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1_000_000);
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let device = Device::flex();
    #[cfg(feature = "cuda")]
    let device = Device::cuda(0);
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let device = Device::metal(burn::tensor::DeviceKind::DefaultDevice);
    let device = device.autodiff();
    println!("Environment: {ENVIRONMENT_NAME}");
    let env = VectorizedGymWrapper::from(TimeLimitGym::new(mujoco::build_environment(), 1_000));
    let mut env = DeviceMultiGymWrapper::new(env, Device::flex(), device.clone());
    let observation_space = env.observation_space();
    let action_space = env.action_space();
    let observation_size = observation_space.shape()[0];
    let action_size = action_space.shape().iter().product();
    let model = GaussianModule {
        mean: sac_mlp(observation_size, action_size, 0.01, &device)?,
        log_std: Param::from_tensor(Tensor::zeros([1, action_size], (&device, DTYPE))),
    };
    let distribution =
        TransformedDistribution::new(GaussianDistribution::new([action_size])?, TanhTransform);
    let policy = ProbabilisticPolicyModel::new(model, distribution);
    let mut critics = Vec::new();
    for _ in 0..2 {
        critics.push(
            ActionValueCritic::builder()
                .online_network(ScalarStateActionCritic::new(sac_mlp(
                    observation_size + action_size,
                    1,
                    1.0,
                    &device,
                )?))
                .optimizer(AdamWConfig::new().with_weight_decay(0.0).init())
                .learning_rate(3e-4)
                .build()?,
        );
    }
    let entropy = SACEntropyConfiguration::automatic(
        Param::from_tensor(Tensor::zeros([1], (&device, DTYPE))),
        AdamWConfig::new().with_weight_decay(0.0).init(),
        3e-4,
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
        .aggregation_mode(ActionValueCriticAggregationMode::Min)
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

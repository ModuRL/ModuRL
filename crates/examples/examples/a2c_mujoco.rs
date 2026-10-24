use burn::{
    module::{Module, Param},
    nn::{Initializer, activation::Activation},
    optim::AdamWConfig,
    tensor::{DType, Device, Float, Tensor},
};
use modurl::prelude::*;

mod support;
use support::{
    graphers::OnPolicyGrapher,
    mujoco::{self, ENVIRONMENT_NAME},
};

const TOTAL_TIMESTEPS: usize = 1_000_000;
const DTYPE: DType = DType::F32;

/// Produces Gaussian parameters shaped `[batch, 2 * action_size]` from
/// observations shaped `[batch, observation_size]`.
#[derive(Module, Debug)]
struct GaussianParameterModule {
    mean: MLP,
    log_std: Param<Tensor<2>>,
}

impl Forward<2, 2> for GaussianParameterModule {
    type InputKind = Float;
    type OutputKind = Float;
    type Error = ModelError;

    /// Maps observations `[batch, observation_size]` to Gaussian parameters `[batch, 2 * action_size]`.
    /// Broadcasts the learned log standard deviation `[1, action_size]` over the batch axis.
    fn forward(&self, observations: Tensor<2>) -> Result<Tensor<2>, Self::Error> {
        let mean = self.mean.forward(observations)?;
        let log_std = self.log_std.val().expand(mean.shape());
        Ok(Tensor::cat(vec![mean, log_std], 1))
    }
}

fn main() {
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let device = Device::flex();
    #[cfg(feature = "cuda")]
    let device = Device::cuda(0);
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let device = Device::metal(burn::tensor::DeviceKind::DefaultDevice);
    let device = device.autodiff();
    println!("Environment: {ENVIRONMENT_NAME}");
    println!("Using device: {device:?}");

    let env = NormalizeRewardGym::new(
        NormalizeObservationGym::new(RecordRawRewardGym::new(TimeLimitGym::new(
            mujoco::build_environment(),
            1_000,
        )))
        .with_clip(10.0),
        0.99,
    )
    .with_clip(10.0);
    let env = VectorizedGymWrapper::from(vec![env]);
    let mut env = DeviceMultiGymWrapper::new(env, Device::flex(), device.clone());
    let rollout_batch_size = 5 * env.num_envs();
    let observation_space = env.observation_space();
    let observation_size = observation_space.shape()[0];
    let action_space = env.action_space();
    let action_shape = action_space.shape();
    let action_size = action_shape.iter().product::<usize>();

    let actor = GaussianParameterModule {
        mean: MLP::builder()
            .input_size(observation_size)
            .output_size(action_size)
            .options((&device, DTYPE))
            .hidden_layer_sizes(vec![64, 64])
            .activation(Activation::Tanh)
            .hidden_initializer(Initializer::Orthogonal {
                gain: 2.0_f64.sqrt(),
            })
            .output_initializer(Initializer::Orthogonal { gain: 0.01 })
            .build()
            .unwrap(),
        log_std: Param::from_tensor(Tensor::zeros([1, action_size], (&device, DTYPE))),
    };

    let critic = MLP::builder()
        .input_size(observation_size)
        .output_size(1)
        .options((&device, DTYPE))
        .hidden_layer_sizes(vec![64, 64])
        .activation(Activation::Tanh)
        .hidden_initializer(Initializer::Orthogonal {
            gain: 2.0_f64.sqrt(),
        })
        .output_initializer(Initializer::Orthogonal { gain: 1.0 })
        .build()
        .unwrap();

    let optimizer_config = AdamWConfig::new().with_epsilon(1e-5).with_weight_decay(0.0);
    let networks = A2CNetworkInfo::separate(
        SeparateA2CNetwork::builder()
            .actor_network(ProbabilisticPolicyModel::new(
                actor,
                GaussianDistribution::new(action_shape.to_vec()).unwrap(),
            ))
            .critic_network(critic)
            .actor_optimizer(optimizer_config.init())
            .critic_optimizer(optimizer_config.init())
            .actor_learning_rate(7e-4)
            .critic_learning_rate(7e-4)
            .actor_lr_scheduler(LinearSchedule::new(7e-4, 0.0))
            .critic_lr_scheduler(LinearSchedule::new(7e-4, 0.0))
            .combined_loss(true)
            .build()
            .unwrap(),
    );

    let mut grapher = OnPolicyGrapher::a2c_mujoco(TOTAL_TIMESTEPS, ENVIRONMENT_NAME);
    let mut agent = A2CAgent::builder()
        .observation_space(observation_space)
        .action_space(action_space)
        .network_info(networks)
        // SB3 A2C collects five steps from each environment per update.
        .batch_size(rollout_batch_size)
        .training_horizon(TOTAL_TIMESTEPS)
        .logging_info(&mut grapher)
        .device(device)
        .build()
        .unwrap();

    agent.learn(&mut env, TOTAL_TIMESTEPS).unwrap();
    grapher.display();
}

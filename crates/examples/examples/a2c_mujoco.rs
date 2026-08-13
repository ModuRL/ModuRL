use candle_core::{DType, Module, Tensor};
use candle_nn::{AdamW, Init, Optimizer, ParamsAdamW, VarBuilder, VarMap};
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
struct GaussianParameterModule {
    mean: MLP,
    log_std: Tensor,
}

impl Module for GaussianParameterModule {
    /// Maps observations `[batch, observation_size]` to Gaussian parameters
    /// `[batch, 2 * action_size]`.
    fn forward(&self, observations: &Tensor) -> candle_core::Result<Tensor> {
        let mean = self.mean.forward(observations)?;
        let log_std = self.log_std.broadcast_as(mean.shape())?;
        Tensor::cat(&[mean, log_std], 1)
    }
}

fn main() {
    let device = candle_core::Device::cuda_if_available(0).unwrap();
    println!("Environment: {ENVIRONMENT_NAME}");
    println!("Using device: {device:?}");

    let env = NormalizeRewardGym::new(
        NormalizeObservationGym::new(RecordRawRewardGym::new(TimeLimitGym::new(
            mujoco::build_environment(&device),
            1_000,
        )))
        .with_clip(10.0),
        0.99,
    )
    .with_clip(10.0);
    let mut env = VectorizedGymWrapper::from(vec![env]);
    let rollout_batch_size = 5 * env.num_envs();
    let observation_size = env.observation_space().shape()[0];
    let action_space = env.action_space();
    let action_shape = action_space.shape();
    let action_size = action_shape.iter().product();

    let actor_vars = VarMap::new();
    let actor_vb = VarBuilder::from_varmap(&actor_vars, DTYPE, &device);
    let actor = GaussianParameterModule {
        mean: MLP::builder()
            .input_size(observation_size)
            .output_size(action_size)
            .vb(actor_vb.pp("mean"))
            .hidden_layer_sizes(vec![64, 64])
            .activation(Box::new(Tensor::tanh))
            .initializer(Box::new(OrthogonalMLPInitializer {
                hidden_gain: 2.0_f64.sqrt(),
                output_gain: 0.01,
            }))
            .build()
            .unwrap(),
        log_std: actor_vb
            .get_with_hints((1, action_size), "log_std", Init::Const(0.0))
            .unwrap(),
    };

    let critic_vars = VarMap::new();
    let critic = MLP::builder()
        .input_size(observation_size)
        .output_size(1)
        .vb(VarBuilder::from_varmap(&critic_vars, DTYPE, &device))
        .hidden_layer_sizes(vec![64, 64])
        .activation(Box::new(Tensor::tanh))
        .initializer(Box::new(OrthogonalMLPInitializer {
            hidden_gain: 2.0_f64.sqrt(),
            output_gain: 1.0,
        }))
        .build()
        .unwrap();

    let optimizer_config = ParamsAdamW {
        lr: 7e-4,
        eps: 1e-5,
        weight_decay: 0.0,
        ..Default::default()
    };
    let networks = A2CNetworkInfo::separate(
        SeparateA2CNetwork::builder()
            .actor_network(Box::new(ProbabilisticPolicyModel::with_distribution(
                Box::new(actor),
                GaussianDistribution::new(action_shape).unwrap(),
            )))
            .critic_network(Box::new(critic))
            .actor_optimizer(AdamW::new(actor_vars.all_vars(), optimizer_config.clone()).unwrap())
            .critic_optimizer(AdamW::new(critic_vars.all_vars(), optimizer_config).unwrap())
            .actor_lr_scheduler(Box::new(LinearSchedule::new(7e-4, 0.0)))
            .critic_lr_scheduler(Box::new(LinearSchedule::new(7e-4, 0.0)))
            .combined_loss(true)
            .build(),
    );

    let mut grapher = OnPolicyGrapher::a2c_mujoco(TOTAL_TIMESTEPS, ENVIRONMENT_NAME);
    let mut agent = A2CAgent::builder()
        .dtype(DTYPE)
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

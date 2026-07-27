use candle_core::{Module, Tensor};
use candle_nn::{AdamW, Init, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use modurl::prelude::*;

#[path = "support/graphers.rs"]
mod graphers;
use graphers::PPOMujocoGrapher;

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

/// Produces the parameter tensor required by `GaussianDistribution`.
///
/// For observations shaped `[batch, observation_size]`, `mean` produces
/// `[batch, action_size]`. `log_std` has shape `[1, action_size]`, making it
/// trainable but observation-independent. `forward` returns
/// `[mean_0, ..., mean_(A-1), log_std_0, ..., log_std_(A-1)]` in each batch row,
/// with shape `[batch, 2 * action_size]`.
struct GaussianParameterModule {
    mean: MLP,
    log_std: Tensor,
}

impl Module for GaussianParameterModule {
    /// Maps observations `[batch, observation_size]` to Gaussian parameters
    /// `[batch, 2 * action_size]`.
    fn forward(&self, observations: &Tensor) -> candle_core::Result<Tensor> {
        // The mean changes with each observation: [batch, action_size].
        let mean = self.mean.forward(observations)?;
        // Every observation shares one learned log standard deviation row.
        let log_std = self.log_std.broadcast_as(mean.shape())?;
        // GaussianDistribution splits this tensor into equal halves.
        Tensor::cat(&[mean, log_std], 1)
    }
}

fn main() {
    let device = candle_core::Device::cuda_if_available(0).unwrap();
    println!("Environment: {ENVIRONMENT_NAME}");
    println!("Using device: {device:?}");

    let env = NormalizeRewardGym::new(
        NormalizeObservationGym::new(RecordRawRewardGym::new(TimeLimitGym::new(
            SelectedEnvironment::builder()
                .device(&device)
                .build()
                .unwrap(),
            1_000,
        )))
        .with_clip(10.0),
        0.99,
    )
    .with_clip(10.0);
    let mut env = VectorizedGymWrapper::from(vec![env]);
    let observation_size = env.observation_space().shape()[0];
    let action_space = env.action_space();
    let action_shape = action_space.shape();
    let action_size = action_shape.iter().product();

    let actor_vars = VarMap::new();
    let actor_vb = VarBuilder::from_varmap(&actor_vars, candle_core::DType::F32, &device);
    let actor = GaussianParameterModule {
        // The mean network returns action_size values. GaussianParameterModule
        // appends another action_size values for log_std.
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
        // Because this tensor comes from actor_vb, actor_vars tracks it and the
        // actor optimizer updates it along with the mean network.
        log_std: actor_vb
            .get_with_hints((1, action_size), "log_std", Init::Const(0.0))
            .unwrap(),
    };

    let critic_vars = VarMap::new();
    let critic = MLP::builder()
        .input_size(observation_size)
        .output_size(1)
        .vb(VarBuilder::from_varmap(
            &critic_vars,
            candle_core::DType::F32,
            &device,
        ))
        .hidden_layer_sizes(vec![64, 64])
        .activation(Box::new(Tensor::tanh))
        .initializer(Box::new(OrthogonalMLPInitializer {
            hidden_gain: 2.0_f64.sqrt(),
            output_gain: 1.0,
        }))
        .build()
        .unwrap();

    let optimizer_config = ParamsAdamW {
        lr: 3e-4,
        eps: 1e-5,
        weight_decay: 0.0,
        ..Default::default()
    };
    let networks = PPONetworkInfo::Separate(
        SeparatePPONetwork::builder()
            // The policy interprets actor output as [mean, log_std], samples an
            // unsquashed Gaussian action, and evaluates its log probability.
            // BoxSpace separately clips the action sent to the environment.
            .actor_network(Box::new(ProbabilisticPolicyModel::with_distribution(
                Box::new(actor),
                GaussianDistribution::new(action_shape).unwrap(),
            )))
            .critic_network(Box::new(critic))
            .actor_optimizer(AdamW::new(actor_vars.all_vars(), optimizer_config.clone()).unwrap())
            .critic_optimizer(AdamW::new(critic_vars.all_vars(), optimizer_config).unwrap())
            .actor_lr_scheduler(Box::new(LinearSchedule::new(3e-4, 0.0)))
            .critic_lr_scheduler(Box::new(LinearSchedule::new(3e-4, 0.0)))
            .combined_loss(true)
            .build(),
    );

    let mut grapher = PPOMujocoGrapher::new(TOTAL_TIMESTEPS, ENVIRONMENT_NAME);
    {
        let mut agent = PPOAgent::builder()
            .action_space(action_space)
            .network_info(networks)
            .batch_size(2_048)
            .mini_batch_size(64)
            .num_epochs(10)
            .normalize_advantage(true)
            .clip_range(Box::new(ConstantSchedule::new(0.2)))
            .clip_value_loss(true)
            .gamma(0.99)
            .gae_lambda(0.95)
            .ent_coef(0.0)
            .vf_coef(0.5)
            .training_horizon(TOTAL_TIMESTEPS)
            .logging_info(&mut grapher)
            .device(device)
            .build();

        agent.learn(&mut env, TOTAL_TIMESTEPS).unwrap();
    }
    grapher.display();
}

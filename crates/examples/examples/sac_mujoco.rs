use candle_core::{DType, Device, Module, Tensor, Var};
use candle_nn::{AdamW, Init, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use modurl::prelude::*;

#[path = "support/graphers.rs"]
mod graphers;
use graphers::SACGrapher;

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

fn sac_mlp(
    vb: VarBuilder<'_>,
    input_size: usize,
    output_size: usize,
    output_gain: f64,
    name: &str,
) -> candle_core::Result<MLP> {
    MLP::builder()
        .input_size(input_size)
        .output_size(output_size)
        .vb(vb)
        .hidden_layer_sizes(vec![64, 64])
        .activation(Box::new(Tensor::tanh))
        .initializer(Box::new(OrthogonalMLPInitializer {
            hidden_gain: 2.0_f64.sqrt(),
            output_gain,
        }))
        .name(name.to_owned())
        .build()
}

struct GaussianActor {
    mean: MLP,
    log_std: Tensor,
}

impl Module for GaussianActor {
    /// Maps observations `[batch, observation_size]` to Gaussian parameters
    /// `[batch, 2 * action_size]`.
    fn forward(&self, observations: &Tensor) -> candle_core::Result<Tensor> {
        let mean = self.mean.forward(observations)?;
        let log_std = self.log_std.broadcast_as(mean.shape())?.clamp(-20.0, 2.0)?;
        Tensor::cat(&[mean, log_std], 1)
    }
}

fn scalar_critic<'a>(
    online_vars: &'a VarMap,
    target_vars: &'a mut VarMap,
    input_size: usize,
    optimizer_parameters: &ParamsAdamW,
    device: &Device,
) -> Result<SACCritic<'a, AdamW>, SACCriticError> {
    let online = sac_mlp(
        VarBuilder::from_varmap(online_vars, DType::F32, device),
        input_size,
        1,
        1.0,
        "mlp",
    )?;
    let target = sac_mlp(
        VarBuilder::from_varmap(target_vars, DType::F32, device),
        input_size,
        1,
        1.0,
        "mlp",
    )?;
    SACCritic::builder()
        .online_network(Box::new(ScalarStateActionCritic::new(Box::new(online))))
        .target_network(Box::new(ScalarStateActionCritic::new(Box::new(target))))
        .online_vars(online_vars)
        .target_vars(target_vars)
        .optimizer(AdamW::new(
            online_vars.all_vars(),
            optimizer_parameters.clone(),
        )?)
        .build()
}

fn main() {
    let total_timesteps = 1_000_000;
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
    let action_space = env.action_space();
    let observation_size = observation_space.shape()[0];
    let action_shape = action_space.shape();
    let action_size = action_shape.iter().product();
    let optimizer_parameters = ParamsAdamW {
        lr: 3e-4,
        weight_decay: 0.0,
        ..Default::default()
    };

    let actor_vars = VarMap::new();
    let actor_vb = VarBuilder::from_varmap(&actor_vars, DType::F32, &device);
    let mean = sac_mlp(
        actor_vb.pp("mean"),
        observation_size,
        action_size,
        0.01,
        "mlp",
    )
    .unwrap();
    let log_std = actor_vb
        .get_with_hints((1, action_size), "log_std", Init::Const(0.0))
        .unwrap();
    let distribution = TransformedDistribution::new(
        GaussianDistribution::new(action_shape).unwrap(),
        TanhTransform,
    );
    let policy = ProbabilisticPolicyModel::with_distribution(
        Box::new(GaussianActor { mean, log_std }),
        distribution,
    );
    let actor_optimizer = AdamW::new(actor_vars.all_vars(), optimizer_parameters.clone()).unwrap();

    let online_vars_1 = VarMap::new();
    let mut target_vars_1 = VarMap::new();
    let critic_1 = scalar_critic(
        &online_vars_1,
        &mut target_vars_1,
        observation_size + action_size,
        &optimizer_parameters,
        &device,
    )
    .unwrap();

    let online_vars_2 = VarMap::new();
    let mut target_vars_2 = VarMap::new();
    let critic_2 = scalar_critic(
        &online_vars_2,
        &mut target_vars_2,
        observation_size + action_size,
        &optimizer_parameters,
        &device,
    )
    .unwrap();

    let log_alpha = Var::from_vec(vec![0.0f32], (), &device).unwrap();
    let alpha_optimizer =
        AdamW::new(vec![log_alpha.clone()], optimizer_parameters.clone()).unwrap();
    let entropy = SACEntropyConfiguration::automatic(log_alpha, alpha_optimizer, None);
    let mut grapher = SACGrapher::new();

    let mut agent = SACAgent::builder()
        .policy(Box::new(policy))
        .actor_optimizer(actor_optimizer)
        .critics(vec![critic_1, critic_2])
        .entropy_configuration(entropy)
        .action_space(action_space)
        .observation_space(observation_space)
        .aggregation_mode(SACCriticAggregationMode::Min)
        .training_horizon(total_timesteps)
        .logger(&mut grapher)
        .device_strategy(SACDeviceStrategy::OneDevice(device))
        .build()
        .unwrap();

    agent.learn(&mut env, total_timesteps).unwrap();
    drop(agent);
    grapher.display();
}

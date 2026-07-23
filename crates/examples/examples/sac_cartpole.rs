use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use modurl::prelude::*;
use modurl_gym::classic_control::cartpole::CartPoleV1;

#[path = "support/graphers.rs"]
mod graphers;
use graphers::SACGrapher;

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

fn discrete_critic<'a>(
    online_vars: &'a VarMap,
    target_vars: &'a mut VarMap,
    observation_size: usize,
    action_count: usize,
    optimizer_parameters: &ParamsAdamW,
    device: &Device,
) -> Result<SACCritic<'a, AdamW>, SACCriticError> {
    let online = sac_mlp(
        VarBuilder::from_varmap(online_vars, DType::F32, device),
        observation_size,
        action_count,
        1.0,
        "mlp",
    )?;
    let target = sac_mlp(
        VarBuilder::from_varmap(target_vars, DType::F32, device),
        observation_size,
        action_count,
        1.0,
        "mlp",
    )?;
    SACCritic::builder()
        .online_network(Box::new(DiscreteVectorHeadCritic::new(Box::new(online))))
        .target_network(Box::new(DiscreteVectorHeadCritic::new(Box::new(target))))
        .online_vars(online_vars)
        .target_vars(target_vars)
        .optimizer(AdamW::new(
            online_vars.all_vars(),
            optimizer_parameters.clone(),
        )?)
        .build()
}

fn main() {
    let total_timesteps = std::env::var("MODURL_SAC_TIMESTEPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(100_000);
    let device = Device::cuda_if_available(0).unwrap();
    println!("Using device: {device:?}");
    let mut env = VectorizedGymWrapper::from(vec![CartPoleV1::builder().device(&device).build()]);
    let observation_space = env.observation_space();
    let action_space = env.action_space();
    let observation_size = observation_space.shape()[0];
    let action_count = action_space.shape()[0];
    let optimizer_parameters = ParamsAdamW {
        lr: 3e-4,
        // Candle exposes AdamW rather than Adam; zero decay matches the
        // optimizer used by standard SAC implementations.
        weight_decay: 0.0,
        ..Default::default()
    };

    let actor_vars = VarMap::new();
    let actor = sac_mlp(
        VarBuilder::from_varmap(&actor_vars, DType::F32, &device),
        observation_size,
        action_count,
        0.01,
        "actor",
    )
    .unwrap();
    let actor_optimizer = AdamW::new(actor_vars.all_vars(), optimizer_parameters.clone()).unwrap();
    let policy = ProbabilisticPolicyModel::<CategoricalDistribution>::new(Box::new(actor));

    let online_vars_1 = VarMap::new();
    let mut target_vars_1 = VarMap::new();
    let critic_1 = discrete_critic(
        &online_vars_1,
        &mut target_vars_1,
        observation_size,
        action_count,
        &optimizer_parameters,
        &device,
    )
    .unwrap();

    let online_vars_2 = VarMap::new();
    let mut target_vars_2 = VarMap::new();
    let critic_2 = discrete_critic(
        &online_vars_2,
        &mut target_vars_2,
        observation_size,
        action_count,
        &optimizer_parameters,
        &device,
    )
    .unwrap();

    let log_alpha = Var::from_vec(vec![0.0f32], (), &device).unwrap();
    let alpha_optimizer =
        AdamW::new(vec![log_alpha.clone()], optimizer_parameters.clone()).unwrap();
    // Begin near the categorical default, then let the ordinary parameter
    // schedule make the policy increasingly deterministic.
    let maximum_entropy = (action_count as f64).ln();
    let target_entropy = LinearSchedule::new(0.98 * maximum_entropy, 0.3 * maximum_entropy);
    let entropy = SACEntropyConfiguration::automatic(
        log_alpha,
        alpha_optimizer,
        Some(Box::new(target_entropy)),
    );
    let mut grapher = SACGrapher::new();

    let mut agent = SACAgent::builder()
        .policy(Box::new(policy))
        .actor_optimizer(actor_optimizer)
        .critics(vec![critic_1, critic_2])
        .entropy_configuration(entropy)
        .action_space(action_space)
        .observation_space(observation_space)
        .stabilization_configuration(SACStabilizationConfiguration::stable_discrete())
        .replay_capacity(100_000)
        .batch_size(64)
        .training_horizon(total_timesteps)
        .logger(&mut grapher)
        .device_strategy(SACDeviceStrategy::OneDevice(device))
        .build()
        .unwrap();

    agent.learn(&mut env, total_timesteps).unwrap();
    drop(agent);
    grapher.display();
}

use std::{env, error::Error, fmt::Debug, time::Instant};

use candle_core::{DType, Device, Module, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use modurl::prelude::*;
use modurl_gym::classic_control::{cartpole::CartPoleV1, pendulum::PendulumV1};
use serde_json::{Value, json};

const SEED: u64 = 42;
const PPO_ENV_COUNT: usize = 8;
const PPO_STEPS_PER_ENV: usize = 256;
const PPO_ROLLOUT_SIZE: usize = PPO_ENV_COUNT * PPO_STEPS_PER_ENV;
const PPO_MINI_BATCH_SIZE: usize = 64;
const PPO_EPOCHS: usize = 10;
const PPO_PARAMETERS: usize = 9_155;
const DQN_BATCH_SIZE: usize = 64;
const DQN_REPLAY_CAPACITY: usize = 10_000;
const DQN_UPDATE_FREQUENCY: usize = 4;
const DQN_TARGET_UPDATE_INTERVAL: usize = 1_000;
const DQN_PARAMETERS: usize = 4_610;
const SAC_BATCH_SIZE: usize = 64;
const SAC_REPLAY_CAPACITY: usize = 10_000;
const SAC_PARAMETERS: usize = 13_637;

#[derive(Debug)]
struct Args {
    algorithm: String,
    device: String,
    measured_steps: Option<usize>,
    warmup_steps: Option<usize>,
    threads: usize,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut parsed = Self {
            algorithm: "ppo".to_owned(),
            device: "cpu".to_owned(),
            measured_steps: None,
            warmup_steps: None,
            threads: 1,
        };
        let mut args = env::args().skip(1);
        while let Some(flag) = args.next() {
            let value = args
                .next()
                .ok_or_else(|| format!("missing value after {flag}"))?;
            match flag.as_str() {
                "--algorithm" => parsed.algorithm = value,
                "--device" => parsed.device = value,
                "--steps" => {
                    parsed.measured_steps = Some(
                        value
                            .parse()
                            .map_err(|_| format!("invalid --steps value: {value}"))?,
                    )
                }
                "--warmup-steps" => {
                    parsed.warmup_steps = Some(
                        value
                            .parse()
                            .map_err(|_| format!("invalid --warmup-steps value: {value}"))?,
                    )
                }
                "--threads" => {
                    parsed.threads = value
                        .parse()
                        .map_err(|_| format!("invalid --threads value: {value}"))?
                }
                _ => return Err(format!("unknown argument: {flag}")),
            }
        }
        if !matches!(parsed.algorithm.as_str(), "ppo" | "dqn" | "sac") {
            return Err(format!("unsupported algorithm: {}", parsed.algorithm));
        }
        if parsed.threads == 0 {
            return Err("--threads must be at least 1".to_owned());
        }
        let (steps, warmup) = parsed.resolved_steps();
        if steps == 0 || warmup == 0 {
            return Err("--steps and --warmup-steps must be nonzero".to_owned());
        }
        if parsed.algorithm == "ppo"
            && (!steps.is_multiple_of(PPO_ROLLOUT_SIZE) || !warmup.is_multiple_of(PPO_ROLLOUT_SIZE))
        {
            return Err(format!(
                "PPO --steps and --warmup-steps must be multiples of {PPO_ROLLOUT_SIZE}"
            ));
        }
        Ok(parsed)
    }

    fn resolved_steps(&self) -> (usize, usize) {
        let defaults = match self.algorithm.as_str() {
            "ppo" => (20_480, 2_048),
            "dqn" => (4_096, 1_024),
            "sac" => (1_024, 256),
            _ => unreachable!(),
        };
        (
            self.measured_steps.unwrap_or(defaults.0),
            self.warmup_steps.unwrap_or(defaults.1),
        )
    }
}

fn select_device(name: &str) -> Result<Device, String> {
    match name {
        "cpu" => Ok(Device::Cpu),
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(0).map_err(|error| format!("CUDA is unavailable: {error}"))
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err("CUDA requires `--features cuda`".to_owned())
            }
        }
        "metal" => {
            #[cfg(feature = "metal")]
            {
                Device::new_metal(0).map_err(|error| format!("Metal is unavailable: {error}"))
            }
            #[cfg(not(feature = "metal"))]
            {
                Err("Metal requires `--features metal`".to_owned())
            }
        }
        _ => Err(format!("unsupported device: {name}")),
    }
}

fn adam(lr: f64) -> ParamsAdamW {
    ParamsAdamW {
        lr,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-5,
        weight_decay: 0.0,
    }
}

fn parameter_count<'a>(maps: impl IntoIterator<Item = &'a VarMap>) -> usize {
    maps.into_iter()
        .flat_map(VarMap::all_vars)
        .map(|variable| variable.elem_count())
        .sum()
}

fn debug_result<T, E: Debug>(result: Result<T, E>) -> Result<T, Box<dyn Error>> {
    result.map_err(|error| format!("{error:?}").into())
}

fn ppo(
    args: &Args,
    device: &Device,
    measured_steps: usize,
    warmup_steps: usize,
) -> Result<Value, Box<dyn Error>> {
    let envs = debug_result(
        (0..PPO_ENV_COUNT)
            .map(|_| CartPoleV1::builder().device(device).build())
            .collect::<Result<Vec<_>, _>>(),
    )?;
    let mut env: VectorizedGymWrapper<CartPoleV1> = envs.into();
    let action_space = env.action_space();
    let actor_vars = VarMap::new();
    let actor = MLP::builder()
        .input_size(4)
        .output_size(2)
        .vb(VarBuilder::from_varmap(&actor_vars, DType::F32, device))
        .activation(Tensor::tanh)
        .hidden_layer_sizes(vec![64, 64])
        .initializer(OrthogonalMLPInitializer {
            hidden_gain: 2.0_f64.sqrt(),
            output_gain: 0.01,
        })
        .name("actor".to_owned())
        .build()?;
    let critic_vars = VarMap::new();
    let critic = MLP::builder()
        .input_size(4)
        .output_size(1)
        .vb(VarBuilder::from_varmap(&critic_vars, DType::F32, device))
        .activation(Tensor::tanh)
        .hidden_layer_sizes(vec![64, 64])
        .initializer(OrthogonalMLPInitializer {
            hidden_gain: 2.0_f64.sqrt(),
            output_gain: 1.0,
        })
        .name("critic".to_owned())
        .build()?;
    assert_eq!(parameter_count([&actor_vars, &critic_vars]), PPO_PARAMETERS);
    let networks = PPONetworkInfo::Separate(
        SeparatePPONetwork::builder()
            .actor_network(ProbabilisticPolicyModel::<CategoricalDistribution>::new(
                actor,
            ))
            .critic_network(critic)
            .actor_optimizer(AdamW::new(actor_vars.all_vars(), adam(3e-4))?)
            .critic_optimizer(AdamW::new(critic_vars.all_vars(), adam(3e-4))?)
            .combined_loss(true)
            .build(),
    );
    let mut agent = debug_result(
        PPOAgent::builder()
            .dtype(DType::F32)
            .action_space(action_space)
            .network_info(networks)
            .batch_size(PPO_ROLLOUT_SIZE)
            .mini_batch_size(PPO_MINI_BATCH_SIZE)
            .num_epochs(PPO_EPOCHS)
            .normalize_advantage(true)
            .ent_coef(0.0)
            .gamma(0.99)
            .gae_lambda(0.95)
            .vf_coef(0.5)
            .gradient_clip(0.5)
            .clip_range(ConstantSchedule::new(0.2))
            .clip_value_loss(false)
            .training_horizon(warmup_steps + measured_steps)
            .device(device.clone())
            .build(),
    )?;
    debug_result(agent.learn(&mut env, warmup_steps))?;
    device.synchronize()?;
    let start = Instant::now();
    debug_result(agent.learn(&mut env, measured_steps))?;
    device.synchronize()?;
    let elapsed = start.elapsed().as_secs_f64();
    Ok(result(
        args,
        "ppo",
        measured_steps,
        warmup_steps,
        elapsed,
        json!({
            "environment": "CartPole-v1", "environments": PPO_ENV_COUNT,
            "steps_per_environment": PPO_STEPS_PER_ENV, "rollout_size": PPO_ROLLOUT_SIZE,
            "hidden_layers": [64, 64], "trainable_parameters": PPO_PARAMETERS,
            "activation": "tanh", "dtype": "float32", "mini_batch_size": PPO_MINI_BATCH_SIZE,
            "ppo_epochs": PPO_EPOCHS, "learning_rate": 3e-4, "adam_epsilon": 1e-5,
            "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2,
            "value_loss_clipping": false, "advantage_normalization": "per-mini-batch",
            "entropy_coefficient": 0.0, "value_coefficient": 0.5,
            "max_gradient_norm": 0.5, "seed": SEED
        }),
    ))
}

fn dqn(
    args: &Args,
    device: &Device,
    measured_steps: usize,
    warmup_steps: usize,
) -> Result<Value, Box<dyn Error>> {
    let cartpole = debug_result(CartPoleV1::builder().device(device).build())?;
    let mut env = VectorizedGymWrapper::from(vec![cartpole]);
    let observation_space = env.observation_space();
    let online_vars = VarMap::new();
    let online = MLP::builder()
        .input_size(4)
        .output_size(2)
        .vb(VarBuilder::from_varmap(&online_vars, DType::F32, device))
        .activation(Tensor::tanh)
        .hidden_layer_sizes(vec![64, 64])
        .name("q".to_owned())
        .build()?;
    let mut target_vars = VarMap::new();
    let target = MLP::builder()
        .input_size(4)
        .output_size(2)
        .vb(VarBuilder::from_varmap(&target_vars, DType::F32, device))
        .activation(Tensor::tanh)
        .hidden_layer_sizes(vec![64, 64])
        .name("q".to_owned())
        .build()?;
    assert_eq!(parameter_count([&online_vars]), DQN_PARAMETERS);
    let mut agent = debug_result(
        DQNAgent::builder()
            .dtype(DType::F32)
            .action_space(Discrete::new(2))
            .observation_space(observation_space)
            .online_q_network(online)
            .target_q_network(target)
            .online_vars(&online_vars)
            .target_vars(&mut target_vars)
            .optimizer(AdamW::new(online_vars.all_vars(), adam(2.5e-4))?)
            .replay_capacity(DQN_REPLAY_CAPACITY)
            .batch_size(DQN_BATCH_SIZE)
            .training_start(warmup_steps + 1)
            .update_frequency(DQN_UPDATE_FREQUENCY)
            .target_update_interval(DQN_TARGET_UPDATE_INTERVAL)
            .training_horizon(warmup_steps + measured_steps)
            .epsilon_schedule(ConstantSchedule::new(0.1))
            .replay_storage_config(ReplayStorageConfig::new(ReplayDeviceStrategy::OneDevice(
                device.clone(),
            )))
            .build(),
    )?;
    debug_result(agent.learn(&mut env, warmup_steps))?;
    device.synchronize()?;
    let start = Instant::now();
    debug_result(agent.learn(&mut env, measured_steps))?;
    device.synchronize()?;
    let elapsed = start.elapsed().as_secs_f64();
    Ok(result(
        args,
        "dqn",
        measured_steps,
        warmup_steps,
        elapsed,
        json!({
            "environment": "CartPole-v1", "environments": 1, "hidden_layers": [64, 64],
            "trainable_parameters": DQN_PARAMETERS, "activation": "tanh", "dtype": "float32",
            "replay_capacity": DQN_REPLAY_CAPACITY, "batch_size": DQN_BATCH_SIZE,
            "training_start": warmup_steps, "updates_per_transition": 0.25,
            "update_frequency": DQN_UPDATE_FREQUENCY, "target_update_interval": DQN_TARGET_UPDATE_INTERVAL,
            "learning_rate": 2.5e-4, "adam_epsilon": 1e-5, "gamma": 0.99,
            "epsilon": 0.1, "target": "vanilla", "loss": "mse", "gradient_clipping": false,
            "n_step_return": 1, "seed": SEED
        }),
    ))
}

struct GaussianActor {
    network: MLP,
}

impl Module for GaussianActor {
    /// Maps observations `[batch, 3]` to Gaussian parameters `[batch, 2]`.
    fn forward(&self, observations: &Tensor) -> candle_core::Result<Tensor> {
        let output = self.network.forward(observations)?;
        let mean = output.narrow(1, 0, 1)?;
        let log_std = output.narrow(1, 1, 1)?.clamp(-20.0, 2.0)?;
        Tensor::cat(&[mean, log_std], 1)
    }
}

fn sac_critic<'a>(
    online_vars: &'a VarMap,
    target_vars: &'a mut VarMap,
    device: &Device,
) -> Result<SACCritic<'a, AdamW>, Box<dyn Error>> {
    let online = MLP::builder()
        .input_size(4)
        .output_size(1)
        .vb(VarBuilder::from_varmap(online_vars, DType::F32, device))
        .activation(Tensor::tanh)
        .hidden_layer_sizes(vec![64, 64])
        .name("critic".to_owned())
        .build()?;
    let target = MLP::builder()
        .input_size(4)
        .output_size(1)
        .vb(VarBuilder::from_varmap(target_vars, DType::F32, device))
        .activation(Tensor::tanh)
        .hidden_layer_sizes(vec![64, 64])
        .name("critic".to_owned())
        .build()?;
    debug_result(
        SACCritic::builder()
            .online_network(ScalarStateActionCritic::new(online))
            .target_network(ScalarStateActionCritic::new(target))
            .online_vars(online_vars)
            .target_vars(target_vars)
            .optimizer(AdamW::new(online_vars.all_vars(), adam(3e-4))?)
            .build(),
    )
}

fn sac(
    args: &Args,
    device: &Device,
    measured_steps: usize,
    warmup_steps: usize,
) -> Result<Value, Box<dyn Error>> {
    let pendulum = debug_result(PendulumV1::builder().device(device).build())?;
    let mut env = VectorizedGymWrapper::from(vec![TimeLimitGym::new(pendulum, 200)]);
    let observation_space = env.observation_space();
    let action_space = env.action_space();
    let actor_vars = VarMap::new();
    let actor = MLP::builder()
        .input_size(3)
        .output_size(2)
        .vb(VarBuilder::from_varmap(&actor_vars, DType::F32, device))
        .activation(Tensor::tanh)
        .hidden_layer_sizes(vec![64, 64])
        .name("actor".to_owned())
        .build()?;
    let distribution = TransformedDistribution::new(
        debug_result(GaussianDistribution::new(vec![1]))?,
        TanhTransform,
    );
    let policy =
        ProbabilisticPolicyModel::with_distribution(GaussianActor { network: actor }, distribution);
    let critic_vars_1 = VarMap::new();
    let mut target_vars_1 = VarMap::new();
    let critic_1 = sac_critic(&critic_vars_1, &mut target_vars_1, device)?;
    let critic_vars_2 = VarMap::new();
    let mut target_vars_2 = VarMap::new();
    let critic_2 = sac_critic(&critic_vars_2, &mut target_vars_2, device)?;
    let log_alpha = Var::zeros((), DType::F32, device)?;
    assert_eq!(
        parameter_count([&actor_vars, &critic_vars_1, &critic_vars_2]) + 1,
        SAC_PARAMETERS
    );
    let entropy = SACEntropyConfiguration::automatic_with_target_schedule(
        log_alpha.clone(),
        AdamW::new(vec![log_alpha], adam(3e-4))?,
        ConstantSchedule::new(-1.0),
    );
    let mut agent = debug_result(
        SACAgent::builder()
            .dtype(DType::F32)
            .policy(policy)
            .actor_optimizer(AdamW::new(actor_vars.all_vars(), adam(3e-4))?)
            .critics(vec![critic_1, critic_2])
            .entropy_configuration(entropy)
            .action_space(action_space)
            .observation_space(observation_space)
            .aggregation_mode(SACCriticAggregationMode::Min)
            .replay_capacity(SAC_REPLAY_CAPACITY)
            .batch_size(SAC_BATCH_SIZE)
            .training_start(warmup_steps + 1)
            .gamma(0.99)
            .tau(0.005)
            .training_horizon(warmup_steps + measured_steps)
            .replay_storage_config(ReplayStorageConfig::new(ReplayDeviceStrategy::OneDevice(
                device.clone(),
            )))
            .build(),
    )?;
    debug_result(agent.learn(&mut env, warmup_steps))?;
    device.synchronize()?;
    let start = Instant::now();
    debug_result(agent.learn(&mut env, measured_steps))?;
    device.synchronize()?;
    let elapsed = start.elapsed().as_secs_f64();
    Ok(result(
        args,
        "sac",
        measured_steps,
        warmup_steps,
        elapsed,
        json!({
            "environment": "Pendulum-v1", "time_limit": 200, "environments": 1,
            "actor_hidden_layers": [64, 64], "critic_hidden_layers": [64, 64],
            "trainable_parameters": SAC_PARAMETERS, "activation": "tanh", "dtype": "float32",
            "state_dependent_log_std": true, "log_std_bounds": [-20.0, 2.0],
            "replay_capacity": SAC_REPLAY_CAPACITY, "batch_size": SAC_BATCH_SIZE,
            "training_start": warmup_steps, "updates_per_transition": 1,
            "learning_rate": 3e-4, "adam_epsilon": 1e-5, "gamma": 0.99, "tau": 0.005,
            "critics": 2, "critic_aggregation": "min", "automatic_entropy_tuning": true,
            "initial_alpha": 1.0, "target_entropy": -1.0, "n_step_return": 1, "seed": SEED
        }),
    ))
}

fn result(
    args: &Args,
    algorithm: &str,
    steps: usize,
    warmup: usize,
    elapsed: f64,
    config: Value,
) -> Value {
    json!({
        "algorithm": algorithm, "framework": "modurl", "framework_version": "0.1.0 (workspace)",
        "backend": "candle 0.11.0", "device": args.device, "threads": args.threads,
        "measured_steps": steps, "warmup_steps": warmup, "elapsed_seconds": elapsed,
        "steps_per_second": steps as f64 / elapsed, "config": config
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse().map_err(|error| format!("argument error: {error}"))?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()?;
    let device = select_device(&args.device)?;
    if !matches!(device, Device::Cpu) {
        device.set_seed(SEED)?;
    }
    let (measured_steps, warmup_steps) = args.resolved_steps();
    let output = match args.algorithm.as_str() {
        "ppo" => ppo(&args, &device, measured_steps, warmup_steps)?,
        "dqn" => dqn(&args, &device, measured_steps, warmup_steps)?,
        "sac" => sac(&args, &device, measured_steps, warmup_steps)?,
        _ => unreachable!(),
    };
    println!("BENCH_RESULT={output}");
    Ok(())
}

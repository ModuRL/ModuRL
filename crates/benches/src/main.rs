use burn::{
    module::Module,
    nn::Initializer,
    optim::AdamWConfig,
    tensor::{DType, Device, Float, Int, Tensor},
};
use modurl::prelude::*;
use serde_json::{Value, json};
use std::{env, error::Error, fmt::Debug, path::PathBuf, time::Instant};

mod support;
use support::{
    atari,
    nature_cnn::{NatureCnn, NatureCnnError},
};

#[cfg(feature = "tracing")]
mod profile;

const SEED: u64 = 42;
const PPO_ENV_COUNT: usize = 32;
const PPO_ROLLOUT_SIZE: usize = PPO_ENV_COUNT * 128;
const DQN_REPLAY_CAPACITY: usize = 1_000_000;
const DQN_TRAINING_START: usize = 80_000;

#[derive(Debug)]
struct Args {
    algorithm: String,
    device: String,
    backend: String,
    trace: Option<PathBuf>,
    trace_sync: bool,
    rom: PathBuf,
    steps: usize,
    warmup_steps: usize,
    threads: usize,
}

impl Args {
    fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut algorithm = "ppo".to_owned();
        let mut device = "cuda".to_owned();
        let mut backend = "native".to_owned();
        let mut trace = None;
        let mut trace_sync = false;
        let mut rom = None;
        let mut steps = None;
        let mut threads = 1;
        let mut arguments = arguments.into_iter();
        while let Some(flag) = arguments.next() {
            let value = arguments
                .next()
                .ok_or_else(|| format!("missing value after {flag}"))?;
            match flag.as_str() {
                "--algorithm" => algorithm = value,
                "--device" => device = value,
                "--backend" => backend = value,
                "--trace" => trace = Some(PathBuf::from(value)),
                "--trace-sync" => {
                    trace_sync = value
                        .parse()
                        .map_err(|_| "--trace-sync requires true or false")?
                }
                "--rom" => rom = Some(PathBuf::from(value)),
                "--steps" => steps = Some(value.parse().map_err(|_| "invalid --steps")?),
                "--threads" => threads = value.parse().map_err(|_| "invalid --threads")?,
                _ => return Err(format!("unknown argument: {flag}")),
            }
        }
        let (default_steps, default_warmup) = match algorithm.as_str() {
            "ppo" => (40_960, PPO_ROLLOUT_SIZE * 10),
            "dqn" => (4_096, DQN_TRAINING_START + 4),
            _ => return Err(format!("unsupported algorithm: {algorithm}")),
        };
        let parsed = Self {
            algorithm,
            device,
            backend,
            trace,
            trace_sync,
            rom: rom.ok_or("--rom requires a Breakout ROM path")?,
            steps: steps.unwrap_or(default_steps),
            warmup_steps: default_warmup,
            threads,
        };
        if parsed.threads == 0 || parsed.steps == 0 {
            return Err("--threads and --steps must be nonzero".to_owned());
        }
        if parsed.trace.is_some() && parsed.algorithm != "ppo" {
            return Err("--trace requires --algorithm ppo".to_owned());
        }
        if parsed.trace_sync && parsed.trace.is_none() {
            return Err("--trace-sync requires --trace".to_owned());
        }
        if parsed.trace.is_some() && !cfg!(feature = "tracing") {
            return Err("--trace requires --features tracing".to_owned());
        }
        let interval = if parsed.algorithm == "ppo" {
            PPO_ROLLOUT_SIZE
        } else {
            4
        };
        if !parsed.steps.is_multiple_of(interval) {
            return Err(format!("--steps must be a multiple of {interval}"));
        }
        if !parsed.rom.is_file() {
            return Err(format!("ROM file does not exist: {}", parsed.rom.display()));
        }
        Ok(parsed)
    }
}

fn select_device(name: &str, backend: &str) -> Result<Device, String> {
    if backend == "libtorch" {
        #[cfg(feature = "libtorch")]
        {
            return match name {
                "cuda" => Ok(Device::new(burn::backend::DispatchDevice::LibTorch(
                    tch::Device::Cuda(0).into(),
                ))),
                "cpu" => Ok(Device::new(burn::backend::DispatchDevice::LibTorch(
                    tch::Device::Cpu.into(),
                ))),
                _ => Err("LibTorch benchmarks require --device cpu or cuda".to_owned()),
            };
        }
        #[cfg(not(feature = "libtorch"))]
        {
            return Err("LibTorch requires --features libtorch".to_owned());
        }
    }
    if backend != "native" {
        return Err(format!("unsupported backend: {backend}"));
    }
    match name {
        "cpu" => Ok(Device::flex()),
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                Ok(Device::cuda(0))
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err("CUDA requires --features cuda".to_owned())
            }
        }
        "metal" => {
            #[cfg(feature = "metal")]
            {
                Ok(Device::metal(burn::tensor::DeviceKind::DefaultDevice))
            }
            #[cfg(not(feature = "metal"))]
            {
                Err("Metal requires --features metal".to_owned())
            }
        }
        _ => Err(format!("unsupported device: {name}")),
    }
}

fn debug_result<T, E: Debug>(result: Result<T, E>) -> Result<T, Box<dyn Error>> {
    result.map_err(|error| format!("{error:?}").into())
}

#[derive(Module, Debug)]
struct AtariQNetwork {
    encoder: NatureCnn,
    head: MLP,
}

struct DQNTimingLogger {
    device: Device,
    warmup_steps: usize,
    start: Option<Instant>,
    sync_error: Option<burn::tensor::ExecutionError>,
    updates: usize,
}

impl<I> DQNLogger<I> for DQNTimingLogger {
    fn log(&mut self, _entry: &QLogEntry) {
        self.updates += 1;
    }

    fn log_collection(&mut self, entry: &QCollectionLogEntry<I>) {
        if entry.collection_timestep == self.warmup_steps {
            match self.device.sync() {
                Ok(()) => self.start = Some(Instant::now()),
                Err(error) => self.sync_error = Some(error),
            }
        }
    }
}

impl AtariQNetwork {
    fn new(device: &Device) -> Result<Self, ModelError> {
        let bound = 1.0 / 512.0f64.sqrt();
        Ok(Self {
            encoder: NatureCnn::new(device, false)?,
            head: MLP::builder()
                .input_size(512)
                .output_size(4)
                .hidden_layer_sizes(vec![])
                .output_initializer(Initializer::Uniform {
                    min: -bound,
                    max: bound,
                })
                .options((device, DType::F32))
                .build()?,
        })
    }
}

impl Forward<4, 2> for AtariQNetwork {
    type InputKind = Int;
    type OutputKind = Float;
    type Error = NatureCnnError;

    /// Maps U8 [batch_size, 4, 84, 84] to F32 Q values [batch_size, 4] on the model device.
    /// The encoder converts and normalizes observations; both model stages preserve gradients.
    fn forward(&self, observations: Tensor<4, Int>) -> Result<Tensor<2>, Self::Error> {
        Ok(self.head.forward(self.encoder.forward(observations)?)?)
    }
}

fn ppo(args: &Args, device: &Device) -> Result<f64, Box<dyn Error>> {
    let action_space = Discrete::new(4)?;
    let constructors = (0..PPO_ENV_COUNT)
        .map(|index| {
            let rom = args.rom.clone();
            move || atari::make_atari_env(rom, (SEED + index as u64) as i32, 400_000, false)
        })
        .collect();
    let envs = debug_result(MultithreadedVectorizedGymWrapper::new(
        constructors,
        atari::FrameObservationSpace,
        action_space.clone(),
    ))?;
    let mut envs = DeviceMultiGymWrapper::new(envs, Device::flex(), device.clone());
    let cnn = NatureCnn::new(device, true)?;
    let actor = MLP::builder()
        .input_size(512)
        .output_size(4)
        .hidden_layer_sizes(vec![])
        .output_initializer(Initializer::Orthogonal { gain: 0.01 })
        .options((device, DType::F32))
        .build()?;
    let critic = MLP::builder()
        .input_size(512)
        .output_size(1)
        .hidden_layer_sizes(vec![])
        .output_initializer(Initializer::Orthogonal { gain: 1.0 })
        .options((device, DType::F32))
        .build()?;
    assert_eq!(
        cnn.num_params() + actor.num_params() + critic.num_params(),
        1_686_693
    );
    let optimizer = AdamWConfig::new()
        .with_epsilon(1e-5)
        .with_weight_decay(0.0)
        .init();
    let networks = SharedPPONetwork::builder()
        .shared_network(cnn)
        .actor_head(ProbabilisticPolicyModel::new(
            actor,
            CategoricalDistribution,
        ))
        .critic_head(critic)
        .optimizer(optimizer)
        .learning_rate(2.5e-4)
        .build()?;
    let mut agent = debug_result(
        PPOAgent::builder()
            .observation_space(envs.observation_space())
            .action_space(action_space)
            .network_info(networks)
            .batch_size(PPO_ROLLOUT_SIZE)
            .mini_batch_size(PPO_ROLLOUT_SIZE / 4)
            .num_epochs(4)
            .gamma(0.99)
            .gae_lambda(0.95)
            .clip_range(ConstantSchedule::new(0.1))
            .clip_value_loss(true)
            .vf_coef(0.25)
            .ent_coef(0.01)
            .gradient_clip(0.5)
            .training_horizon(args.warmup_steps + args.steps)
            .build(),
    )?;
    debug_result(agent.learn(&mut envs, args.warmup_steps))?;
    device.sync()?;
    #[cfg(feature = "tracing")]
    if let Some(path) = &args.trace {
        return profile::record(path, device, args.trace_sync, || {
            debug_result(agent.learn(&mut envs, args.steps))
        });
    }
    let start = Instant::now();
    debug_result(agent.learn(&mut envs, args.steps))?;
    device.sync()?;
    Ok(start.elapsed().as_secs_f64())
}

fn dqn(args: &Args, device: &Device) -> Result<f64, Box<dyn Error>> {
    let envs = debug_result(VectorizedGymWrapper::new(vec![atari::make_atari_env(
        args.rom.clone(),
        SEED as i32,
        108_000,
        false,
    )]))?;
    let mut envs = DeviceMultiGymWrapper::new(envs, Device::flex(), device.clone());
    if envs.action_space().get_possible_values() != 4 {
        return Err("Breakout requires its four-action minimal action set".into());
    }
    let online = AtariQNetwork::new(device)?;
    assert_eq!(online.num_params(), 1_686_180);
    let horizon = args.warmup_steps + args.steps;
    let mut logger = DQNTimingLogger {
        device: device.clone(),
        warmup_steps: args.warmup_steps,
        start: None,
        sync_error: None,
        updates: 0,
    };
    let optimizer = AdamWConfig::new()
        .with_epsilon(1e-8)
        .with_weight_decay(0.0)
        .init();
    let mut agent = debug_result(
        DQNAgent::builder()
            .dtype(DType::F32)
            .action_space(envs.action_space())
            .observation_space(envs.observation_space())
            .online_q_network(online)
            .logger(&mut logger)
            .optimizer(optimizer)
            .learning_rate(1e-4)
            .replay_capacity(DQN_REPLAY_CAPACITY)
            .batch_size(32)
            .training_start(DQN_TRAINING_START + 1)
            .update_frequency(4)
            .target_update_interval(1_000)
            .gamma(0.99)
            .training_horizon(horizon)
            .epsilon_schedule(move |progress: f64| {
                1.0 - 0.99 * (progress * horizon as f64 / 1_000_000.0).min(1.0)
            })
            .replay_device(Device::flex())
            .build(),
    )?;
    debug_result(agent.learn(&mut envs, horizon))?;
    device.sync()?;
    if let Some(error) = logger.sync_error {
        return Err(error.into());
    }
    let start = logger
        .start
        .ok_or("DQN did not reach its warmup boundary")?;
    let elapsed = start.elapsed().as_secs_f64();
    if logger.updates != args.steps / 4 + 1 {
        return Err(format!(
            "DQN performed {} updates, expected {}",
            logger.updates,
            args.steps / 4 + 1
        )
        .into());
    }
    Ok(elapsed)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args =
        Args::parse(env::args().skip(1)).map_err(|error| format!("argument error: {error}"))?;
    #[cfg(feature = "libtorch")]
    if args.backend == "libtorch" {
        tch::set_num_threads(i32::try_from(args.threads)?);
        if args.device == "cuda" && (!tch::Cuda::is_available() || !tch::Cuda::cudnn_is_available())
        {
            return Err("LibTorch CUDA benchmarks require CUDA and cuDNN".into());
        }
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()?;
    let device = select_device(&args.device, &args.backend)?.autodiff();
    Device::flex().seed(SEED);
    device.seed(SEED);
    let workloads: Value = serde_json::from_str(include_str!("../workloads.json"))?;
    let elapsed = match args.algorithm.as_str() {
        "ppo" => ppo(&args, &device)?,
        "dqn" => dqn(&args, &device)?,
        _ => unreachable!(),
    };
    let output = json!({ "algorithm": args.algorithm, "framework": "modurl", "framework_version": "0.1.0 (workspace)",
        "backend": format!("burn 0.22.0 ({})", args.backend), "device": args.device, "threads": args.threads,
        "autotuning": if args.backend == "libtorch" || args.device == "cpu" { "not applicable" } else if cfg!(feature = "cuda") { "enabled" } else { "disabled" },
        "trace": args.trace,
        "timing_mode": if args.trace_sync { "synchronized trace" } else if args.trace.is_some() { "host trace" } else { "untraced" },
        "fusion": if args.backend == "libtorch" { "unavailable: LibTorch" } else if args.device == "cpu" { "unavailable: CPU Flex" } else { "enabled" },
        "environment_workers": if args.algorithm == "ppo" { 32 } else { 0 },
        "worker_kind": if args.algorithm == "ppo" { "threads" } else { "none" },
        "measured_steps": args.steps, "warmup_steps": args.warmup_steps, "elapsed_seconds": elapsed,
        "optimizer_steps": if args.algorithm == "ppo" { args.steps / PPO_ROLLOUT_SIZE * 16 } else { args.steps / 4 },
        "steps_per_second": args.steps as f64 / elapsed, "config": workloads[&args.algorithm] });
    println!("BENCH_RESULT={output}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ppo_defaults_to_ten_warmup_rollouts() {
        let args = Args::parse(
            ["--rom", concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml")]
                .into_iter()
                .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(args.warmup_steps, 40_960);
    }

    #[test]
    fn dqn_uses_fixed_warmup() {
        let args = Args::parse(
            [
                "--algorithm",
                "dqn",
                "--rom",
                concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"),
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(args.warmup_steps, 80_004);
    }

    #[test]
    fn warmup_override_is_rejected_as_unknown_argument() {
        let flag = ["--warmup", "-steps"].concat();
        let args = [
            "--rom".to_owned(),
            concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml").to_owned(),
            flag.clone(),
            "4096".to_owned(),
        ];
        assert_eq!(
            Args::parse(args).unwrap_err(),
            format!("unknown argument: {flag}")
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_backend_enables_fusion_and_autotuning() {
        assert!(std::any::type_name::<burn_cubecl::Cube>().contains("Fusion<"));
        assert!(matches!(
            burn_cubecl::kernel::conv::ConvStrategy::default(),
            burn_cubecl::kernel::conv::ConvStrategy::Autotune
        ));
    }

    #[cfg(feature = "libtorch")]
    #[test]
    fn libtorch_preserves_cpu_byte_inputs_and_model_gradients() {
        let device = select_device("cpu", "libtorch").unwrap().autodiff();
        let model = AtariQNetwork::new(&device).unwrap();
        let observations = Tensor::<4, Int>::zeros([2, 4, 84, 84], (&Device::flex(), DType::U8))
            .to_device(&device);
        assert_eq!(observations.dtype(), DType::U8);
        let output = model.forward(observations).unwrap();
        assert_eq!(output.dims(), [2, 4]);
        assert_eq!(output.dtype(), DType::F32);
        let gradients = burn::optim::GradientsParams::from_grads(output.mean().backward(), &model);
        assert!(!gradients.is_empty());
    }

    #[cfg(feature = "cuda")]
    struct ByteGym;

    #[cfg(feature = "cuda")]
    impl Gym<(), 4, 2> for ByteGym {
        type Error = std::convert::Infallible;
        type ObservationSpace = atari::FrameObservationSpace;
        type ActionSpace = Discrete;

        /// Returns a CPU U8 item [4, 84, 84].
        fn reset(&mut self) -> Result<ResetInfo<(), Tensor<3, Int>>, Self::Error> {
            Ok(ResetInfo {
                observation: Tensor::zeros([4, 84, 84], (&Device::flex(), DType::U8)),
                info: (),
            })
        }

        /// Accepts CPU Int actions [1] and returns a CPU U8 item [4, 84, 84] with a scalar F32 reward.
        fn step(
            &mut self,
            action: Tensor<1, Int>,
        ) -> Result<StepInfo<(), Tensor<3, Int>>, Self::Error> {
            assert_eq!(action.device(), Device::flex());
            Ok(StepInfo {
                observation: Tensor::full([4, 84, 84], 127, (&Device::flex(), DType::U8)),
                reward: 1.0,
                done: false,
                truncated: true,
                info: (),
            })
        }

        fn observation_space(&self) -> Self::ObservationSpace {
            atari::FrameObservationSpace
        }

        fn action_space(&self) -> Discrete {
            Discrete::new(4).unwrap()
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_dqn_trains_from_cpu_byte_replay_and_boolean_flags() {
        assert!(std::any::type_name::<burn_cubecl::Cube>().contains("Fusion<"));
        let device = Device::cuda(0).autodiff();
        let envs = VectorizedGymWrapper::new(vec![ByteGym]).unwrap();
        let mut envs = DeviceMultiGymWrapper::new(envs, Device::flex(), device.clone());
        let model = AtariQNetwork::new(&device).unwrap();
        let mut logger = DQNTimingLogger {
            device: device.clone(),
            warmup_steps: 2,
            start: None,
            sync_error: None,
            updates: 0,
        };
        let mut agent = DQNAgent::builder()
            .action_space(envs.action_space())
            .observation_space(envs.observation_space())
            .online_q_network(model)
            .optimizer(AdamWConfig::new().init())
            .learning_rate(1e-4)
            .replay_capacity(8)
            .batch_size(2)
            .training_start(1)
            .update_frequency(1)
            .target_update_interval(2)
            .training_horizon(4)
            .logger(&mut logger)
            .replay_device(Device::flex())
            .build()
            .unwrap();
        agent.learn(&mut envs, 4).unwrap();
        device.sync().unwrap();
        assert_eq!(logger.updates, 3);
        assert!(logger.start.is_some());
        assert!(logger.sync_error.is_none());
    }

    /// Checks U8 CPU observations [32, 4, 84, 84] and Int actions [32, 1] across native workers.
    #[test]
    fn parallel_native_workers_preserve_byte_frame_stacks() {
        let Some(rom) = env::var_os("MODURL_ALE_TEST_ROM") else {
            eprintln!("skipping: MODURL_ALE_TEST_ROM is not set");
            return;
        };
        let constructors = (0..PPO_ENV_COUNT)
            .map(|index| {
                let rom = PathBuf::from(rom.clone());
                move || atari::make_atari_env(rom, (SEED + index as u64) as i32, 400_000, false)
            })
            .collect();
        let mut envs = MultithreadedVectorizedGymWrapper::new(
            constructors,
            atari::FrameObservationSpace,
            Discrete::new(4).unwrap(),
        )
        .unwrap();
        let observations = envs.reset().unwrap();
        assert_eq!(observations.dims(), [PPO_ENV_COUNT, 4, 84, 84]);
        assert_eq!(observations.dtype(), DType::U8);
        assert_eq!(observations.device(), Device::flex());
        for _ in 0..4 {
            let step = envs
                .step(Tensor::<2, Int>::zeros(
                    [PPO_ENV_COUNT, 1],
                    (&Device::flex(), DType::U32),
                ))
                .unwrap();
            assert_eq!(step.observations.dtype(), DType::U8);
            assert_eq!(step.observations.dims(), [PPO_ENV_COUNT, 4, 84, 84]);
            assert_eq!(step.observations.device(), Device::flex());
        }
    }

    #[test]
    fn workload_preserves_native_observations_and_cpu_replay() {
        let workloads: Value = serde_json::from_str(include_str!("../workloads.json")).unwrap();
        assert_eq!(workloads["ppo"]["environments"], PPO_ENV_COUNT);
        assert_eq!(workloads["ppo"]["rollout_size"], PPO_ROLLOUT_SIZE);
        assert_eq!(workloads["dqn"]["replay_capacity"], DQN_REPLAY_CAPACITY);
        assert_eq!(workloads["dqn"]["training_start"], DQN_TRAINING_START);
        assert_eq!(workloads["dqn"]["replay_storage"], "cpu");
        for algorithm in ["ppo", "dqn"] {
            assert_eq!(workloads[algorithm]["observation_dtype"], "uint8");
        }
    }

    #[test]
    fn dqn_module_converts_byte_inputs_and_has_expected_parameter_count() {
        let device = Device::flex();
        let model = AtariQNetwork::new(&device).unwrap();
        assert_eq!(model.num_params(), 1_686_180);
        let observations = Tensor::<4, Int>::zeros([1, 4, 84, 84], (&device, DType::U8));
        let output = model.forward(observations.clone()).unwrap();
        assert_eq!(observations.dtype(), DType::U8);
        assert_eq!(output.dtype(), DType::F32);
        assert_eq!(output.dims(), [1, 4]);
    }
}

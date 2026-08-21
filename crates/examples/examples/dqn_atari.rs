//! DQN on an Atari game, matching CleanRL's `dqn_atari.py` configuration.
//!
//! The environment applies the paper's Atari preprocessing: random no-op
//! starts, action-repeat/max-pooling, episodic lives, reward clipping,
//! 84x84 grayscale frames, and a four-frame stack. The learner uses CleanRL's
//! convolutional architecture, Adam optimizer, replay warm-up, update/target
//! cadence, and epsilon schedule. It collects from eight environments to
//! improve throughput, unlike CleanRL's single-environment default.
//!
//! Run with a legally obtained Atari ROM:
//!
//! `cargo run --release -p examples --example dqn_atari --features atari-environment,cuda -- path/to/game.bin`
//!
//! Add `multithreading` to construct the eight ALE environments in worker
//! threads; without it, they are stepped sequentially on the main thread.

use std::{env, path::PathBuf};

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{
    AdamW, Conv2d, Conv2dConfig, Init, Linear, Optimizer, ParamsAdamW, VarBuilder, VarMap,
};
use modurl::{agents::ReplayDeviceStrategy, prelude::*};
use modurl_ale::{
    AtariGym, AtariInfo, AtariObsType,
    wrappers::{EpisodicLifeGym, FireResetGym, FireResetGymError, NoopResetGym, WarpGym},
};

mod support;
use support::graphers::DQNGrapher;

const SEED: i32 = 1;
const FRAME_SKIP: usize = 4;
const FRAME_STACK: usize = 4;
const NUM_ENVS: usize = 8;
const NOOP_MAX: u32 = 30;
const MAX_EPISODE_FRAMES: u32 = 400_000;

// Separate state and next-state storage at this capacity uses about 53 GiB.
const REPLAY_CAPACITY: usize = 1_000_000;
const BATCH_SIZE: usize = 32;
const GAMMA: f32 = 0.99;
const TRAINING_START: usize = 80_000;
const TARGET_UPDATE_INTERVAL: usize = 1_000;
const EPSILON_ANNEALING_STEPS: usize = 1_000_000;
const TRAINING_TIMESTEPS: usize = 10_000_000;
const LEARNING_RATE: f64 = 1e-4;

const AGENT_UPDATE_FREQUENCY: usize = 4;

/// Applies FIRE reset only to games whose minimal action set includes FIRE.
enum OptionalFireResetGym<G> {
    Plain(G),
    Fire(FireResetGym<G>),
}

impl<G> OptionalFireResetGym<G> {
    fn new(gym: G, enabled: bool) -> Self {
        if enabled {
            Self::Fire(FireResetGym::new(gym))
        } else {
            Self::Plain(gym)
        }
    }
}

impl<G, I> Gym<I> for OptionalFireResetGym<G>
where
    G: Gym<I>,
{
    type Error = FireResetGymError<G::Error>;
    type SpaceError = G::SpaceError;

    fn reset(&mut self) -> std::result::Result<ResetInfo<I>, Self::Error> {
        match self {
            Self::Plain(gym) => gym.reset().map_err(FireResetGymError::GymError),
            Self::Fire(gym) => gym.reset(),
        }
    }

    /// Steps with a scalar action shaped `[]`.
    fn step(&mut self, action: Tensor) -> std::result::Result<StepInfo<I>, Self::Error> {
        match self {
            Self::Plain(gym) => gym.step(action).map_err(FireResetGymError::GymError),
            Self::Fire(gym) => gym.step(action),
        }
    }

    fn action_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
        match self {
            Self::Plain(gym) => gym.action_space(),
            Self::Fire(gym) => gym.action_space(),
        }
    }

    fn observation_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
        match self {
            Self::Plain(gym) => gym.observation_space(),
            Self::Fire(gym) => gym.observation_space(),
        }
    }
}

fn make_atari_env(
    rom_path: PathBuf,
    seed: i32,
) -> impl Gym<
    RawRewardInfo<EpisodeStatisticsInfo<AtariInfo>>,
    SpaceError = candle_core::Error,
    Error: Send + Sync + std::fmt::Debug,
> {
    let mut env = AtariGym::builder()
        .rom_path(rom_path)
        .obs_type(AtariObsType::RGBScreen)
        .random_seed(seed)
        .repeat_action_probability(0.0)
        .build()
        .expect("failed to load the Atari ROM");
    let has_fire_action = env.minimal_action_set().contains(&1);

    let env = TimeLimitGym::new(env, MAX_EPISODE_FRAMES);
    let env = RecordEpisodeStatisticsGym::new(env);
    let env = NoopResetGym::new_with_noop_max(env, NOOP_MAX);
    let env = MaxAndSkipGym::new(env, FRAME_SKIP);
    let env = EpisodicLifeGym::new(env, &Device::Cpu);
    let env = OptionalFireResetGym::new(env, has_fire_action);
    let env = RecordRawRewardGym::new(env);
    let env = ClipRewardGym::new(env);
    let env = WarpGym::new(env);
    FrameStackGym::new(env, FRAME_STACK)
}

/// The convolutional Q-network used by CleanRL's Atari DQN.
struct AtariQNetwork {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    hidden: Linear,
    q_values: Linear,
}

// PyTorch's default Conv2d and Linear initialization used by CleanRL is
// Kaiming uniform with `a = sqrt(5)`, which simplifies to this bound.
fn cleanrl_linear(in_features: usize, out_features: usize, vb: VarBuilder) -> Result<Linear> {
    let bound = 1.0 / (in_features as f64).sqrt();
    let init = Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let weight = vb.get_with_hints((out_features, in_features), "weight", init)?;
    let bias = vb.get_with_hints(out_features, "bias", init)?;
    Ok(Linear::new(weight, Some(bias)))
}

fn cleanrl_conv2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    config: Conv2dConfig,
    vb: VarBuilder,
) -> Result<Conv2d> {
    let bound = 1.0 / ((in_channels * kernel_size * kernel_size) as f64).sqrt();
    let init = Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let weight = vb.get_with_hints(
        (
            out_channels,
            in_channels / config.groups,
            kernel_size,
            kernel_size,
        ),
        "weight",
        init,
    )?;
    let bias = vb.get_with_hints(out_channels, "bias", init)?;
    Ok(Conv2d::new(weight, Some(bias), config))
}

impl AtariQNetwork {
    fn new(action_count: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv1: cleanrl_conv2d(
                4,
                32,
                8,
                Conv2dConfig {
                    stride: 4,
                    ..Default::default()
                },
                vb.pp("conv1"),
            )?,
            conv2: cleanrl_conv2d(
                32,
                64,
                4,
                Conv2dConfig {
                    stride: 2,
                    ..Default::default()
                },
                vb.pp("conv2"),
            )?,
            conv3: cleanrl_conv2d(
                64,
                64,
                3,
                Conv2dConfig {
                    stride: 1,
                    ..Default::default()
                },
                vb.pp("conv3"),
            )?,
            hidden: cleanrl_linear(64 * 7 * 7, 512, vb.pp("hidden"))?,
            q_values: cleanrl_linear(512, action_count, vb.pp("q_values"))?,
        })
    }
}

impl Module for AtariQNetwork {
    /// Maps `[batch, 4, 84, 84]` observations to `[batch, action_count]`.
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let x = (input / 255.0)?;
        let x = self.conv1.forward(&x)?.relu()?;
        let x = self.conv2.forward(&x)?.relu()?;
        let x = self.conv3.forward(&x)?.relu()?;
        self.q_values
            .forward(&self.hidden.forward(&x.flatten_from(1)?)?.relu()?)
    }
}

fn main() {
    let rom_path = env::args_os().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        eprintln!("usage: cargo run --release -p examples --example dqn_atari --features atari-environment[,cuda,multithreading] -- path/to/game.bin");
        std::process::exit(2);
    });

    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let device = Device::Cpu;
    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0).expect("failed to create CUDA device");
    #[cfg(feature = "metal")]
    let device = Device::new_metal(0).expect("failed to create Metal device");
    #[cfg(any(feature = "cuda", feature = "metal"))]
    device
        .set_seed(SEED as u64)
        .expect("failed to seed accelerator device");

    let probe_env = make_atari_env(rom_path.clone(), SEED);
    let action_count = probe_env.action_space().shape()[0];
    drop(probe_env);
    let action_space = Discrete::new(action_count);

    #[cfg(feature = "multithreading")]
    let envs = {
        let constructors = (0..NUM_ENVS)
            .map(|environment_index| {
                let rom_path = rom_path.clone();
                move || make_atari_env(rom_path, SEED + environment_index as i32)
            })
            .collect();
        let observation_space = BoxSpace::new(
            Tensor::zeros((FRAME_STACK, 84, 84), DType::U8, &Device::Cpu).unwrap(),
            Tensor::full(u8::MAX, (FRAME_STACK, 84, 84), &Device::Cpu).unwrap(),
        );
        MultithreadedVectorizedGymWrapper::new(
            constructors,
            observation_space,
            action_space.clone(),
        )
    };
    #[cfg(not(feature = "multithreading"))]
    let envs = VectorizedGymWrapper::from(
        (0..NUM_ENVS)
            .map(|environment_index| {
                make_atari_env(rom_path.clone(), SEED + environment_index as i32)
            })
            .collect::<Vec<_>>(),
    );

    // Keep ALE and replay on CPU; the agent moves inference and samples as needed.
    let action_device = Device::Cpu;
    let observation_device = Device::Cpu;
    let mut envs = TensorMapMultiGymWrapper::new(
        envs,
        move |action: Tensor| action.to_device(&action_device),
        move |observation: Tensor| observation.to_device(&observation_device),
    );

    let online_vars = VarMap::new();
    let online_q_network = AtariQNetwork::new(
        action_count,
        VarBuilder::from_varmap(&online_vars, DType::F32, &device).pp("network"),
    )
    .expect("failed to build online Q-network");
    let mut target_vars = VarMap::new();
    let target_q_network = AtariQNetwork::new(
        action_count,
        VarBuilder::from_varmap(&target_vars, DType::F32, &device).pp("network"),
    )
    .expect("failed to build target Q-network");
    // With zero weight decay, AdamW has the same update as CleanRL's Adam.
    let optimizer = AdamW::new(
        online_vars.all_vars(),
        ParamsAdamW {
            lr: LEARNING_RATE,
            weight_decay: 0.0,
            ..Default::default()
        },
    )
    .expect("failed to build optimizer");

    #[cfg(feature = "cuda")]
    let device_strategy = ReplayDeviceStrategy::Hybrid {
        optimization_device: device.clone(),
        storage_device: Device::Cpu,
    };
    #[cfg(not(feature = "cuda"))]
    let device_strategy = ReplayDeviceStrategy::OneDevice(device.clone());

    let mut grapher = DQNGrapher::new();
    let mut agent = DQNAgent::builder()
        .dtype(DType::F32)
        .replay_dtype(DType::U8)
        .action_space(action_space)
        .observation_space(envs.observation_space())
        .online_q_network(online_q_network)
        .target_q_network(target_q_network)
        .online_vars(&online_vars)
        .target_vars(&mut target_vars)
        .optimizer(optimizer)
        .replay_capacity(REPLAY_CAPACITY)
        .environment_count(NUM_ENVS)
        .batch_size(BATCH_SIZE)
        .gamma(GAMMA)
        .training_start(TRAINING_START)
        .update_frequency(AGENT_UPDATE_FREQUENCY)
        .target_update_interval(TARGET_UPDATE_INTERVAL)
        .training_horizon(TRAINING_TIMESTEPS)
        .epsilon_schedule(|progress: f64| {
            let annealing =
                (progress * TRAINING_TIMESTEPS as f64 / EPSILON_ANNEALING_STEPS as f64).min(1.0);
            1.0 + (0.01 - 1.0) * annealing
        })
        .logger(&mut grapher)
        .device_strategy(device_strategy)
        .build()
        .expect("invalid DQN configuration");

    println!(
        "training CleanRL-style DQN for {TRAINING_TIMESTEPS} transitions across {NUM_ENVS} environments"
    );
    agent
        .learn(&mut envs, TRAINING_TIMESTEPS)
        .expect("DQN learning failed");
    grapher.display();
}

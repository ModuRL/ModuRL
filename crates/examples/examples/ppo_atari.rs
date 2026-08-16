//! PPO on an Atari game.
//!
//! This follows the Atari recreation from "The 37 Implementation Details of
//! Proximal Policy Optimization": eight actors, 128-step rollouts, four
//! minibatches and epochs, and the standard Atari preprocessing and Nature CNN.
//!
//! Run with a legally obtained Atari ROM:
//!
//! `cargo run --release -p examples --example ppo_atari --features atari-environment,multithreading -- path/to/game.bin`

use std::{env, path::PathBuf};

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{AdamW, Conv2d, Conv2dConfig, Linear, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use modurl::{
    agents::ppo::FakeOptimizer,
    init::{conv2d_ortho, linear_ortho},
    prelude::*,
};
use modurl_ale::{
    AtariGym, AtariInfo, AtariObsType,
    wrappers::{EpisodicLifeGym, FireResetGym, FireResetGymError, NoopResetGym, WarpGym},
};

mod support;
use support::graphers::OnPolicyGrapher;

const NUM_ENVS: usize = 8;
const NUM_STEPS: usize = 128;
const NUM_MINIBATCHES: usize = 4;
const UPDATE_EPOCHS: usize = 4;
const BATCH_SIZE: usize = NUM_ENVS * NUM_STEPS;
const REQUESTED_TIMESTEPS: usize = 10_000_000;
// PPO updates require complete rollouts, so discard the partial final batch.
const TRAINING_TIMESTEPS: usize = REQUESTED_TIMESTEPS / BATCH_SIZE * BATCH_SIZE;
const SEED: i32 = 1;
const LEARNING_RATE: f64 = 2.5e-4;
const ADAM_BETA1: f64 = 0.9;
const ADAM_BETA2: f64 = 0.999;
const ADAM_EPSILON: f64 = 1e-5;
const GAMMA: f32 = 0.99;
const GAE_LAMBDA: f32 = 0.95;
const CLIP_COEFFICIENT: f64 = 0.1;
const ENTROPY_COEFFICIENT: f32 = 0.01;
const VALUE_COEFFICIENT: f32 = 0.5;
const MAX_GRADIENT_NORM: f32 = 0.5;

const NOOP_MAX: u32 = 30;
const FRAME_SKIP: usize = 4;
const FRAME_STACK: usize = 4;
// This wrapper is inside the action-repeat wrapper, so the limit is measured
// in emulator frames rather than agent steps.
const MAX_EPISODE_FRAMES: u32 = 400_000;

/// Applies FIRE reset only to games whose action set includes FIRE.
enum OptionalFireResetGym<G> {
    Plain(G),
    Fire(FireResetGym<G>),
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

    /// Forwards one scalar Atari action shaped `[]`.
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

/// The shared Nature CNN used by both the policy and value heads.
struct NatureCnn {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    hidden: Linear,
}

impl NatureCnn {
    fn new(vb: VarBuilder) -> Result<Self> {
        let gain = 2.0f64.sqrt();
        Ok(Self {
            conv1: conv2d_ortho(
                4,
                32,
                8,
                Conv2dConfig {
                    stride: 4,
                    ..Default::default()
                },
                gain,
                vb.pp("conv1"),
            )?,
            conv2: conv2d_ortho(
                32,
                64,
                4,
                Conv2dConfig {
                    stride: 2,
                    ..Default::default()
                },
                gain,
                vb.pp("conv2"),
            )?,
            conv3: conv2d_ortho(
                64,
                64,
                3,
                Conv2dConfig {
                    stride: 1,
                    ..Default::default()
                },
                gain,
                vb.pp("conv3"),
            )?,
            hidden: linear_ortho(64 * 7 * 7, 512, gain, vb.pp("hidden"))?,
        })
    }
}

impl Module for NatureCnn {
    /// Maps a batch of stacked frames shaped `[batch, 4, 84, 84]` to
    /// features shaped `[batch, 512]`.
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let features = self.conv1.forward(input)?.relu()?;
        let features = self.conv2.forward(&features)?.relu()?;
        let features = self.conv3.forward(&features)?.relu()?;
        self.hidden.forward(&features.flatten_from(1)?)?.relu()
    }
}

fn main() {
    let rom_path = env::args_os().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        eprintln!(
            "usage: cargo run --release -p examples --example ppo_atari \
             --features atari-environment -- path/to/game.bin"
        );
        std::process::exit(2);
    });
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let device = {
        println!("CPU seed cannot be set");
        Device::Cpu
    };
    #[cfg(feature = "cuda")]
    let device = {
        let device = Device::new_cuda(0).expect("failed to create CUDA device");
        device
            .set_seed(SEED as u64)
            .expect("failed to seed the CUDA device");
        device
    };
    #[cfg(feature = "metal")]
    let device = {
        let device = Device::new_metal(0).expect("failed to create Metal device");
        device
            .set_seed(SEED as u64)
            .expect("failed to seed the Metal device");
        device
    };

    // ALE and Atari image preprocessing are CPU-native. Keep all eight
    // environments on the CPU and transfer one processed batch at the policy
    // boundary instead of synchronizing each frame and action separately.
    let environment_device = Device::Cpu;

    // Load one environment on the main thread to discover the ROM's minimal
    // action set. Worker environments are constructed inside their own threads
    // because ALE instances are intentionally not Send.
    let probe_env = make_atari_env(rom_path.clone(), SEED);
    let action_count = probe_env.action_space().shape()[0];
    drop(probe_env);
    let action_space = Discrete::new(action_count);

    #[cfg(feature = "multithreading")]
    let envs = {
        let constructors: Vec<_> = (0..NUM_ENVS)
            .map(|seed| {
                let rom_path = rom_path.clone();
                move || make_atari_env(rom_path, seed as i32 + SEED)
            })
            .collect();
        let observation_space = BoxSpace::new(
            Tensor::zeros((FRAME_STACK, 84, 84), DType::U8, &Device::Cpu)
                .expect("failed to build Atari observation space"),
            Tensor::full(u8::MAX, (FRAME_STACK, 84, 84), &Device::Cpu)
                .expect("failed to build Atari observation space"),
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
            .map(|seed| make_atari_env(rom_path.clone(), seed as i32 + SEED))
            .collect::<Vec<_>>(),
    );
    let action_device = environment_device.clone();
    let observation_device = device.clone();
    let mut envs = TensorMapMultiGymWrapper::new(
        envs,
        move |tensor: Tensor| tensor.to_device(&action_device),
        move |tensor: Tensor| {
            let tensor = tensor.to_device(&observation_device)?;
            if tensor.dtype() == DType::U8 {
                tensor.to_dtype(DType::F32)? / 255.0
            } else {
                Ok(tensor)
            }
        },
    );

    let variables = VarMap::new();
    let vb = VarBuilder::from_varmap(&variables, DType::F32, &device);
    let cnn = NatureCnn::new(vb.pp("network")).expect("failed to build Nature CNN");
    let actor =
        linear_ortho(512, action_count, 0.01, vb.pp("actor")).expect("failed to build actor head");
    let critic = linear_ortho(512, 1, 1.0, vb.pp("critic")).expect("failed to build critic head");

    // AdamW is equivalent to Adam when weight decay is zero.
    let optimizer = AdamW::new(
        variables.all_vars(),
        ParamsAdamW {
            lr: LEARNING_RATE,
            beta1: ADAM_BETA1,
            beta2: ADAM_BETA2,
            eps: ADAM_EPSILON,
            weight_decay: 0.0,
        },
    )
    .expect("failed to build Adam optimizer");

    // Schedule progress advances before each update. The offset preserves the
    // initial learning rate for the first update and reaches one update's
    // fraction of it for the final update.
    let update_count = TRAINING_TIMESTEPS / BATCH_SIZE;
    let learning_rate = move |progress: f64| {
        let fraction = (1.0 - progress + 1.0 / update_count as f64).clamp(0.0, 1.0);
        LEARNING_RATE * fraction
    };

    let networks: PPONetworkInfo<AdamW, _, FakeOptimizer> = PPONetworkInfo::Shared(
        SharedPPONetwork::builder()
            .shared_network(cnn)
            .actor_head(ProbabilisticPolicyModel::<CategoricalDistribution>::new(
                actor,
            ))
            .critic_head(critic)
            .optimizer(optimizer)
            .lr_scheduler(learning_rate)
            .build(),
    );

    let game_name = rom_path
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("atari");
    let mut logger = OnPolicyGrapher::ppo_atari(TRAINING_TIMESTEPS, game_name);
    {
        let mut agent = PPOAgent::builder()
            .action_space(Box::new(action_space))
            .network_info(networks)
            .clipped(true)
            .batch_size(BATCH_SIZE)
            .mini_batch_size(BATCH_SIZE / NUM_MINIBATCHES)
            .num_epochs(UPDATE_EPOCHS)
            .gamma(GAMMA)
            .gae_lambda(GAE_LAMBDA)
            .normalize_advantage(true)
            .normalize_returns(false)
            .clip_range(ConstantSchedule::new(CLIP_COEFFICIENT))
            .clip_value_loss(true)
            // The PPO objective defines value loss as 0.5 * MSE, while the
            // library's value-loss helper returns MSE directly.
            .vf_coef(0.5 * VALUE_COEFFICIENT)
            .ent_coef(ENTROPY_COEFFICIENT)
            .gradient_clip(MAX_GRADIENT_NORM)
            .training_horizon(TRAINING_TIMESTEPS)
            .device(device)
            .logging_info(&mut logger)
            .build()
            .expect("invalid PPO configuration");

        println!("training Atari for {TRAINING_TIMESTEPS} steps");
        agent
            .learn(&mut envs, TRAINING_TIMESTEPS)
            .expect("PPO training failed");
    }
    logger.display();
}

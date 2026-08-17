//! Plays an Atari game indefinitely using a saved PPO checkpoint.
//!
//! Run with a legally obtained ROM and a checkpoint produced by `ppo_atari`:
//!
//! `cargo run --release -p examples --example ppo_atari_play --features atari-environment,rendering,cuda -- path/to/game.bin path/to/checkpoint.safetensors`

use std::{
    env,
    path::PathBuf,
    thread,
    time::{Duration, Instant},
};

use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Linear, VarBuilder, VarMap};
use modurl::{
    init::{conv2d_ortho, linear_ortho},
    prelude::*,
};
use modurl_ale::{
    AtariGym, AtariInfo, AtariObsType,
    wrappers::{EpisodicLifeGym, FireResetGym, FireResetGymError, NoopResetGym, WarpGym},
};

const SEED: i32 = 1;
const NOOP_MAX: u32 = 30;
const FRAME_SKIP: usize = 4;
const FRAME_STACK: usize = 4;
const MAX_EPISODE_FRAMES: u32 = 400_000;
const ATARI_FRAME_RATE: f64 = 60.0;

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
) -> impl Gym<
    RawRewardInfo<EpisodeStatisticsInfo<AtariInfo>>,
    SpaceError = candle_core::Error,
    Error: std::fmt::Debug,
> {
    let mut env = AtariGym::builder()
        .rom_path(rom_path)
        .obs_type(AtariObsType::RGBScreen)
        .random_seed(SEED)
        .repeat_action_probability(0.0)
        .render(true)
        .render_every(FRAME_SKIP)
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
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let features = self.conv1.forward(input)?.relu()?;
        let features = self.conv2.forward(&features)?.relu()?;
        let features = self.conv3.forward(&features)?.relu()?;
        self.hidden.forward(&features.flatten_from(1)?)?.relu()
    }
}

fn main() {
    let mut args = env::args_os().skip(1).map(PathBuf::from);
    let (Some(rom_path), Some(checkpoint_path), None) = (args.next(), args.next(), args.next())
    else {
        eprintln!(
            "usage: cargo run --release -p examples --example ppo_atari_play \
             --features atari-environment,rendering[,cuda] -- \
             path/to/game.bin path/to/checkpoint.safetensors"
        );
        std::process::exit(2);
    };

    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let device = Device::Cpu;
    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0).expect("failed to create CUDA device");
    #[cfg(feature = "metal")]
    let device = Device::new_metal(0).expect("failed to create Metal device");

    let mut gym = make_atari_env(rom_path);
    let action_count = gym.action_space().shape()[0];

    let mut variables = VarMap::new();
    let vb = VarBuilder::from_varmap(&variables, DType::F32, &device);
    let cnn = NatureCnn::new(vb.pp("network")).expect("failed to build Nature CNN");
    let actor =
        linear_ortho(512, action_count, 0.01, vb.pp("actor")).expect("failed to build actor head");
    // Checkpoints also contain the critic, so instantiate it before loading to
    // ensure every saved variable has the same model layout as training.
    let _critic = linear_ortho(512, 1, 1.0, vb.pp("critic")).expect("failed to build critic head");
    variables
        .load(&checkpoint_path)
        .unwrap_or_else(|error| panic!("failed to load {}: {error}", checkpoint_path.display()));

    println!(
        "playing {} at real time; press Ctrl+C to stop",
        checkpoint_path.display()
    );
    let action_period = Duration::from_secs_f64(FRAME_SKIP as f64 / ATARI_FRAME_RATE);
    let mut state = gym.reset().expect("failed to reset Atari").state;
    let mut next_action_time = Instant::now();

    loop {
        let input = (state
            .unsqueeze(0)
            .and_then(|state| state.to_device(&device))
            .and_then(|state| state.to_dtype(DType::F32))
            .expect("failed to prepare observation")
            / 255.0)
            .expect("failed to normalize observation");
        let logits = actor
            .forward(&cnn.forward(&input).expect("CNN inference failed"))
            .expect("actor inference failed");
        let action = logits
            .argmax(D::Minus1)
            .and_then(|action| action.squeeze(0))
            .and_then(|action| action.to_device(&Device::Cpu))
            .expect("failed to select action");

        let step = gym.step(action).expect("Atari step failed");
        if let Some(episode) = step.info.inner.completed_episode {
            println!(
                "episode return: {}, length: {} agent steps",
                episode.episode_return,
                episode.episode_length.div_ceil(FRAME_SKIP)
            );
        }

        state = if step.done || step.truncated {
            gym.reset().expect("failed to reset Atari").state
        } else {
            step.state
        };

        next_action_time += action_period;
        let now = Instant::now();
        if next_action_time > now {
            thread::sleep(next_action_time - now);
        } else {
            next_action_time = now;
        }
    }
}

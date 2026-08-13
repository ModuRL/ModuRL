//! PPO on an Atari game.
//!
//! This follows the Atari recreation from "The 37 Implementation Details of
//! Proximal Policy Optimization": eight actors, 128-step rollouts, four
//! minibatches and epochs, and the standard Atari preprocessing and Nature CNN.
//!
//! Run with a legally obtained Atari ROM:
//!
//! `cargo run --release -p examples --example ppo_atari --features atari-environment -- path/to/game.bin`

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
    wrappers::{EpisodicLifeGym, FireResetGym, NoopResetGym, WarpGym},
};

const NUM_ENVS: usize = 8;
const NUM_STEPS: usize = 128;
const BATCH_SIZE: usize = NUM_ENVS * NUM_STEPS;
const REQUESTED_TIMESTEPS: usize = 10_000_000;
// Like the reference, train only on complete 1,024-transition rollouts.
const TRAINING_TIMESTEPS: usize = REQUESTED_TIMESTEPS / BATCH_SIZE * BATCH_SIZE;
const LEARNING_RATE: f64 = 2.5e-4;

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

/// Prints unclipped episode returns while PPO trains on sign-clipped rewards.
struct ScoreLogger {
    returns: [f32; NUM_ENVS],
    lengths: [usize; NUM_ENVS],
}

impl ScoreLogger {
    fn new() -> Self {
        Self {
            returns: [0.0; NUM_ENVS],
            lengths: [0; NUM_ENVS],
        }
    }
}

impl PPOLogger<RawRewardInfo<AtariInfo>> for ScoreLogger {
    fn log(&mut self, _info: &PPOLogEntry) {}

    fn log_collection(&mut self, info: &PPOCollectionLogEntry<RawRewardInfo<AtariInfo>>) {
        for (index, atari) in info.infos.iter().enumerate() {
            self.returns[index] += atari.raw_reward.unwrap_or(0.0);
            self.lengths[index] += 1;
        }

        for episode in &info.completed_episodes {
            let index = episode.environment_index;
            println!(
                "step {:>8}: episode return {:>6.1}, length {}",
                info.collection_timestep, self.returns[index], self.lengths[index]
            );
            self.returns[index] = 0.0;
            self.lengths[index] = 0;
        }
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
        device.set_seed(1).expect("failed to seed the CUDA device");
        device
    };
    #[cfg(feature = "metal")]
    let device = {
        let device = Device::new_metal(0).expect("failed to create Metal device");
        device.set_seed(1).expect("failed to seed the Metal device");
        device
    };

    let mut envs = Vec::with_capacity(NUM_ENVS);
    for seed in 0..NUM_ENVS {
        let env = AtariGym::builder()
            .rom_path(rom_path.clone())
            .obs_type(AtariObsType::RGBScreen)
            .device(device.clone())
            .random_seed(seed as i32 + 1)
            .repeat_action_probability(0.0)
            .build()
            .expect("failed to load the Atari ROM");

        // The order matches the Atari recreation. AtariGym emits values in
        // [0, 1], so WarpGym and the CNN keep that scale without another /255.
        let env = NoopResetGym::new(env);
        let env = MaxAndSkipGym::new(env, 4);
        let env = EpisodicLifeGym::new(env, &device);
        let env = FireResetGym::new(env);
        let env = RecordRawRewardGym::new(env);
        let env = ClipRewardGym::new(env);
        let env = WarpGym::new(env);
        let env = FrameStackGym::new(env, 4);
        envs.push(env);
    }

    let action_space = envs[0].action_space();
    let action_count = action_space.shape()[0];
    let mut envs = VectorizedGymWrapper::from(envs);

    let variables = VarMap::new();
    let vb = VarBuilder::from_varmap(&variables, DType::F32, &device);
    let cnn = NatureCnn::new(vb.pp("network")).expect("failed to build Nature CNN");
    let actor =
        linear_ortho(512, action_count, 0.01, vb.pp("actor")).expect("failed to build actor head");
    let critic = linear_ortho(512, 1, 1.0, vb.pp("critic")).expect("failed to build critic head");

    // AdamW with zero weight decay is Adam. epsilon=1e-5 matches Baselines.
    let optimizer = AdamW::new(
        variables.all_vars(),
        ParamsAdamW {
            lr: LEARNING_RATE,
            eps: 1e-5,
            weight_decay: 0.0,
            ..Default::default()
        },
    )
    .expect("failed to build Adam optimizer");

    // ModuRL advances schedule progress immediately before each update. This
    // one-batch offset makes its values identical to Baselines update 1..N.
    let update_count = TRAINING_TIMESTEPS / BATCH_SIZE;
    let learning_rate = move |progress: f64| {
        let fraction = (1.0 - progress + 1.0 / update_count as f64).clamp(0.0, 1.0);
        LEARNING_RATE * fraction
    };

    let networks: PPONetworkInfo<AdamW, _, FakeOptimizer> = PPONetworkInfo::Shared(
        SharedPPONetwork::builder()
            .shared_network(Box::new(cnn))
            .actor_head(Box::new(
                ProbabilisticPolicyModel::<CategoricalDistribution>::new(Box::new(actor)),
            ))
            .critic_head(Box::new(critic))
            .optimizer(optimizer)
            .lr_scheduler(Box::new(learning_rate))
            .build(),
    );

    let mut logger = ScoreLogger::new();
    let mut agent = PPOAgent::builder()
        .action_space(action_space)
        .network_info(networks)
        .batch_size(BATCH_SIZE)
        .mini_batch_size(BATCH_SIZE / 4)
        .num_epochs(4)
        .gamma(0.99)
        .gae_lambda(0.95)
        .normalize_advantage(true)
        .clip_range(Box::new(ConstantSchedule::new(0.1)))
        .clip_value_loss(true)
        // The recreation uses 0.5 * vf_coef * MSE. ModuRL's value loss is
        // MSE, so 0.25 gives the reference's vf_coef=0.5 effective weight.
        .vf_coef(0.25)
        .ent_coef(0.01)
        .gradient_clip(0.5)
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

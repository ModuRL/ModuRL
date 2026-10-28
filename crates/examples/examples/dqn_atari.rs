//! DQN on an Atari game, matching CleanRL's `dqn_atari.py` configuration.
//!
//! The environment applies the paper's Atari preprocessing: random no-op
//! starts, action-repeat/max-pooling, episodic lives, reward clipping,
//! 84x84 grayscale frames, and a four-frame stack. The learner uses CleanRL's
//! convolutional architecture, Adam optimizer, replay warm-up, update/target
//! cadence, epsilon schedule, and single-environment configuration.
//!
//! Run with a legally obtained Atari ROM:
//!
//! `cargo run --release -p examples --example dqn_atari --features atari-environment,cuda -- path/to/game.bin`
//!
use burn::{
    module::Module,
    nn::Initializer,
    optim::AdamWConfig,
    tensor::{DType, Device, Float, Int, Tensor},
};
use modurl::prelude::*;
use std::{env, path::PathBuf};
mod support;
use support::{
    atari::make_atari_env,
    graphers::DQNGrapher,
    nature_cnn::{NatureCnn, NatureCnnError},
};

const SEED: i32 = 1;
const NUM_ENVS: usize = 1;
const MAX_EPISODE_FRAMES: u32 = 108_000;

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

/// Maps native U8 frame stacks to one Q value per action with CleanRL's convolutional architecture.
#[derive(Module, Debug)]
struct AtariQNetwork {
    encoder: NatureCnn,
    q_values: MLP,
}

impl AtariQNetwork {
    fn new(action_count: usize, device: &Device) -> Result<Self, ModelError> {
        let bound = 1.0 / 512.0f64.sqrt();
        Ok(Self {
            encoder: NatureCnn::new(device, false)?,
            q_values: MLP::builder()
                .input_size(512)
                .output_size(action_count)
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

    /// Converts U8 observations [batch_size, 4, 84, 84] explicitly to normalized F32 features and returns [batch_size, action_count].
    /// Preserves batch and model gradients; inputs must use the parameter device.
    fn forward(&self, observations: Tensor<4, Int>) -> Result<Tensor<2>, Self::Error> {
        Ok(self.q_values.forward(self.encoder.forward(observations)?)?)
    }
}

fn main() {
    let rom_path = env::args_os().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        eprintln!("usage: cargo run --release -p examples --example dqn_atari --features atari-environment[,cuda] -- path/to/game.bin");
        std::process::exit(2);
    });
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let device = Device::flex();
    #[cfg(feature = "cuda")]
    let device = Device::cuda(0);
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let device = Device::metal(burn::tensor::DeviceKind::DefaultDevice);
    let device = device.autodiff();
    device.seed(SEED as u64);
    let envs = VectorizedGymWrapper::new(
        (0..NUM_ENVS)
            .map(|index| {
                make_atari_env(
                    rom_path.clone(),
                    SEED + index as i32,
                    MAX_EPISODE_FRAMES,
                    false,
                )
            })
            .collect(),
    )
    .unwrap();
    let mut envs = DeviceMultiGymWrapper::new(envs, Device::flex(), device.clone());
    let action_space = envs.action_space();
    let observation_space = envs.observation_space();
    let online_q_network = AtariQNetwork::new(action_space.get_possible_values(), &device).unwrap();
    let optimizer = AdamWConfig::new().with_weight_decay(0.0).init();
    let mut grapher = DQNGrapher::atari();
    let mut agent = DQNAgent::builder()
        .dtype(DType::F32)
        .action_space(action_space)
        .observation_space(observation_space)
        .online_q_network(online_q_network)
        .optimizer(optimizer)
        .learning_rate(LEARNING_RATE)
        .replay_capacity(REPLAY_CAPACITY)
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
        .replay_device(Device::flex())
        .build()
        .unwrap();
    println!(
        "training CleanRL-style DQN for {TRAINING_TIMESTEPS} transitions across {NUM_ENVS} environments"
    );
    agent.learn(&mut envs, TRAINING_TIMESTEPS).unwrap();
    grapher.display();
}

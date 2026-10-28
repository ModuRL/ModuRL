//! PPO on an Atari game.
//!
//! This follows the Atari recreation from "The 37 Implementation Details of
//! Proximal Policy Optimization": eight actors, 128-step rollouts, four
//! minibatches and epochs, and the standard Atari preprocessing and Nature CNN.
//!
//! Run with a legally obtained Atari ROM:
//!
//! `cargo run --release -p examples --example ppo_atari --features atari-environment,multithreading -- path/to/game.bin`

use std::{env, fs, path::PathBuf};

use burn::{
    module::Module,
    nn::Initializer,
    optim::AdamWConfig,
    tensor::{DType, Device},
};
use modurl::prelude::*;

mod support;
use support::{
    atari::{self, make_atari_env},
    graphers::OnPolicyGrapher,
    nature_cnn::NatureCnn,
};

const NUM_ENVS: usize = 8;
const NUM_STEPS: usize = 128;
const NUM_MINIBATCHES: usize = 4;
const UPDATE_EPOCHS: usize = 4;
const BATCH_SIZE: usize = NUM_ENVS * NUM_STEPS;
const REQUESTED_TIMESTEPS: usize = 10_000_000;
// PPO updates require complete rollouts, so discard the partial final batch.
const TRAINING_TIMESTEPS: usize = REQUESTED_TIMESTEPS / BATCH_SIZE * BATCH_SIZE;
const CHECKPOINT_INTERVAL: usize = 1_000_000;
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

// This wrapper is inside the action-repeat wrapper, so the limit is measured
// in emulator frames rather than agent steps.
const MAX_EPISODE_FRAMES: u32 = 400_000;

fn main() {
    let rom_path = env::args_os().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        eprintln!(
            "usage: cargo run --release -p examples --example ppo_atari \
             --features atari-environment -- path/to/game.bin"
        );
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

    // ALE and Atari image preprocessing are CPU-native. Keep all eight
    // environments on the CPU and transfer one processed batch at the policy
    // boundary instead of synchronizing each frame and action separately.
    let environment_device = Device::flex();

    // Load one environment on the main thread to discover the ROM's minimal
    // action set. Worker environments are constructed inside their own threads
    // because ALE instances are intentionally not Send.
    let action_space = {
        let probe_env = make_atari_env(rom_path.clone(), SEED, MAX_EPISODE_FRAMES, false);
        probe_env.action_space()
    };

    #[cfg(feature = "multithreading")]
    let envs = {
        let constructors: Vec<_> = (0..NUM_ENVS)
            .map(|seed| {
                let rom_path = rom_path.clone();
                move || make_atari_env(rom_path, seed as i32 + SEED, MAX_EPISODE_FRAMES, false)
            })
            .collect();
        let observation_space = atari::FrameObservationSpace;
        MultithreadedVectorizedGymWrapper::new(
            constructors,
            observation_space,
            action_space.clone(),
        )
        .unwrap()
    };

    #[cfg(not(feature = "multithreading"))]
    let envs = VectorizedGymWrapper::new(
        (0..NUM_ENVS)
            .map(|seed| {
                make_atari_env(
                    rom_path.clone(),
                    seed as i32 + SEED,
                    MAX_EPISODE_FRAMES,
                    false,
                )
            })
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let mut envs = DeviceMultiGymWrapper::new(envs, environment_device, device.clone());
    let observation_space = envs.observation_space();
    let cnn = NatureCnn::new(&device, true).expect("failed to build Nature CNN");
    let actor = MLP::builder()
        .input_size(512)
        .output_size(action_space.get_possible_values())
        .hidden_layer_sizes(vec![])
        .output_initializer(Initializer::Orthogonal { gain: 0.01 })
        .options((&device, DType::F32))
        .build()
        .unwrap();
    let critic = MLP::builder()
        .input_size(512)
        .output_size(1)
        .hidden_layer_sizes(vec![])
        .output_initializer(Initializer::Orthogonal { gain: 1.0 })
        .options((&device, DType::F32))
        .build()
        .unwrap();
    let optimizer = AdamWConfig::new()
        .with_beta_1(ADAM_BETA1 as f32)
        .with_beta_2(ADAM_BETA2 as f32)
        .with_epsilon(ADAM_EPSILON as f32)
        .with_weight_decay(0.0)
        .init();

    // Schedule progress advances before each update. The offset preserves the
    // initial learning rate for the first update and reaches one update's
    // fraction of it for the final update.
    let update_count = TRAINING_TIMESTEPS / BATCH_SIZE;
    let learning_rate = move |progress: f64| {
        let fraction = (1.0 - progress + 1.0 / update_count as f64).clamp(0.0, 1.0);
        LEARNING_RATE * fraction
    };

    let networks = SharedPPONetwork::builder()
        .shared_network(cnn)
        .actor_head(ProbabilisticPolicyModel::new(
            actor,
            CategoricalDistribution,
        ))
        .critic_head(critic)
        .optimizer(optimizer)
        .learning_rate(learning_rate)
        .build()
        .unwrap();

    let game_name = rom_path
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("atari");
    let checkpoint_dir = PathBuf::from("runs")
        .join("ppo_atari_checkpoints")
        .join(game_name);
    fs::create_dir_all(&checkpoint_dir).expect("failed to create Atari checkpoint directory");
    let mut logger = OnPolicyGrapher::ppo_atari(TRAINING_TIMESTEPS, game_name);

    let mut agent = PPOAgent::builder()
        .observation_space(observation_space)
        .action_space(action_space)
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
    let mut trained_timesteps = 0;
    for checkpoint_timestep in
        (CHECKPOINT_INTERVAL..TRAINING_TIMESTEPS).step_by(CHECKPOINT_INTERVAL)
    {
        // PPO can only stop after a complete rollout. Save the first fully
        // trained model at or beyond each nominal checkpoint boundary.
        let trained_checkpoint = checkpoint_timestep.div_ceil(BATCH_SIZE) * BATCH_SIZE;
        agent
            .learn(&mut envs, trained_checkpoint - trained_timesteps)
            .expect("PPO training failed");
        trained_timesteps = trained_checkpoint;

        let checkpoint_path = checkpoint_dir.join(format!("{game_name}-{checkpoint_timestep}.bpk"));
        agent
            .network()
            .models()
            .save_file(&checkpoint_path)
            .expect("failed to save Atari checkpoint");
        println!(
            "saved {}-step checkpoint after {trained_timesteps} trained steps to {}",
            checkpoint_timestep,
            checkpoint_path.display()
        );
    }

    agent
        .learn(&mut envs, TRAINING_TIMESTEPS - trained_timesteps)
        .expect("PPO training failed");

    let final_checkpoint_path =
        checkpoint_dir.join(format!("{game_name}-{REQUESTED_TIMESTEPS}.bpk"));
    agent
        .network()
        .models()
        .save_file(&final_checkpoint_path)
        .expect("failed to save final Atari checkpoint");
    println!(
        "saved {}-step checkpoint after {TRAINING_TIMESTEPS} trained steps to {}",
        REQUESTED_TIMESTEPS,
        final_checkpoint_path.display()
    );

    logger.display();
}

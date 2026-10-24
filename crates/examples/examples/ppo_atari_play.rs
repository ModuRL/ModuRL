//! Plays an Atari game indefinitely using a saved PPO checkpoint.
//!
//! Run with a legally obtained ROM and a checkpoint produced by `ppo_atari`:
//!
//! `cargo run --release -p examples --example ppo_atari_play --features atari-environment,rendering,cuda -- path/to/game.bin path/to/checkpoint.bpk`

use std::{
    env,
    path::PathBuf,
    thread,
    time::{Duration, Instant},
};

use burn::{
    module::Module,
    nn::Initializer,
    tensor::{DType, Device, Int, Tensor},
};
use modurl::prelude::*;
use modurl_ale::{
    AtariGym, AtariInfo, AtariObsType,
    wrappers::{EpisodicLifeGym, FireResetGym, FireResetGymError, NoopResetGym, WarpGym},
};

mod support;
use support::nature_cnn::NatureCnn;

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

impl<G, I> Gym<I, 3> for OptionalFireResetGym<G>
where
    G: Gym<I, 3, ActionSpace = Discrete, ObservationSpace = BoxSpace<3>>,
{
    type Error = FireResetGymError<G::Error>;
    type ObservationSpace = G::ObservationSpace;
    type ActionSpace = G::ActionSpace;

    fn reset(&mut self) -> std::result::Result<ResetInfo<I, Tensor<2>>, Self::Error> {
        match self {
            Self::Plain(gym) => gym.reset().map_err(FireResetGymError::GymError),
            Self::Fire(gym) => gym.reset(),
        }
    }

    /// Steps the environment using an action tensor with shape `[1]`.
    fn step(
        &mut self,
        action: Tensor<1, Int>,
    ) -> std::result::Result<StepInfo<I, Tensor<2>>, Self::Error> {
        match self {
            Self::Plain(gym) => gym.step(action).map_err(FireResetGymError::GymError),
            Self::Fire(gym) => gym.step(action),
        }
    }

    fn action_space(&self) -> Self::ActionSpace {
        match self {
            Self::Plain(gym) => gym.action_space(),
            Self::Fire(gym) => gym.action_space(),
        }
    }

    fn observation_space(&self) -> Self::ObservationSpace {
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
    4,
    ObservationSpace = BoxSpace<4>,
    ActionSpace = Discrete,
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
    let env = EpisodicLifeGym::new(env, &Device::flex());
    let env = OptionalFireResetGym::new(env, has_fire_action);
    let env = RecordRawRewardGym::new(env);
    let env = ClipRewardGym::new(env);
    let env = WarpGym::new(env);
    let observation_space =
        BoxSpace::new_with_universal_bounds([1, FRAME_STACK, 84, 84], 0.0, 255.0, &Device::flex());
    FrameStackGym::new(env, FRAME_STACK, observation_space)
}

fn main() {
    let mut args = env::args_os().skip(1).map(PathBuf::from);
    let (Some(rom_path), Some(checkpoint_path), None) = (args.next(), args.next(), args.next())
    else {
        eprintln!(
            "usage: cargo run --release -p examples --example ppo_atari_play \
             --features atari-environment,rendering[,cuda] -- \
             path/to/game.bin path/to/checkpoint.bpk"
        );
        std::process::exit(2);
    };

    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let device = Device::flex();
    #[cfg(feature = "cuda")]
    let device = Device::cuda(0);
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let device = Device::metal(burn::tensor::DeviceKind::DefaultDevice);
    let device = device.autodiff();
    device.seed(SEED as u64);

    let mut gym = make_atari_env(rom_path);
    let action_space = gym.action_space();
    let action_count = action_space.get_possible_values();

    let cnn = NatureCnn::new(&device).expect("failed to build Nature CNN");
    let actor = ProbabilisticPolicyModel::new(
        MLP::builder()
            .input_size(512)
            .output_size(action_count)
            .hidden_layer_sizes(vec![])
            .output_initializer(Initializer::Orthogonal { gain: 0.01 })
            .options((&device, DType::F32))
            .build()
            .unwrap(),
        CategoricalDistribution,
    );
    let critic = MLP::builder()
        .input_size(512)
        .output_size(1)
        .hidden_layer_sizes(vec![])
        .output_initializer(Initializer::Orthogonal { gain: 1.0 })
        .options((&device, DType::F32))
        .build()
        .unwrap();
    let (cnn, actor, _critic) = (cnn, actor, critic)
        .try_load_file(&checkpoint_path)
        .unwrap_or_else(|error| panic!("failed to load {}: {error}", checkpoint_path.display()));

    println!(
        "playing {} at real time; press Ctrl+C to stop",
        checkpoint_path.display()
    );
    let action_period = Duration::from_secs_f64(FRAME_SKIP as f64 / ATARI_FRAME_RATE);
    let mut observation = gym.reset().expect("failed to reset Atari").observation;
    let mut next_action_time = Instant::now();

    loop {
        let input = observation
            .clone()
            .unsqueeze::<4>()
            .to_device(&device)
            .cast(DType::F32)
            / 255.0;
        let action_neurons = actor
            .sample(cnn.forward(input).expect("CNN inference failed"))
            .expect("failed to sample action");
        let action = action_space
            .tensor_from_neurons(action_neurons)
            .expect("failed to select action")
            .squeeze::<1>()
            .to_device(&Device::flex());

        let step = gym.step(action).expect("Atari step failed");
        if let Some(episode) = step.info.inner.completed_episode {
            println!(
                "episode return: {}, length: {} agent steps",
                episode.episode_return,
                episode.episode_length.div_ceil(FRAME_SKIP)
            );
        }

        observation = if step.done || step.truncated {
            gym.reset().expect("failed to reset Atari").observation
        } else {
            step.observation
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

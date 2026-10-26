use burn::tensor::{DType, Device, Int, Tensor};
use modurl::{prelude::*, tensor_rank::TensorBatchTypes};
use modurl_ale::{
    AtariGym, AtariInfo, RgbScreen,
    wrappers::{EpisodicLifeGym, FireResetGym, FireResetGymError, NoopResetGym, WarpGym},
};
use std::path::PathBuf;

/// Four CPU U8 frames per observation, with batches [batch_size, 4, 84, 84].
#[derive(Clone, Debug)]
pub struct FrameObservationSpace;

impl TensorBatchTypes for FrameObservationSpace {
    type Kind = Int;
    type Item = Tensor<3, Int>;
    type Batch = Tensor<4, Int>;
}

impl ObservationSpace<4> for FrameObservationSpace {
    /// Checks native U8 observations [batch_size, 4, 84, 84] without changing kind or axes.
    fn contains(&self, observations: &Tensor<4, Int>) -> bool {
        observations.dims()[1..] == [4, 84, 84] && observations.dtype() == DType::U8
    }

    fn shape(&self) -> Vec<usize> {
        vec![4, 84, 84]
    }
}

enum OptionalFireResetGym<G> {
    Plain(G),
    Fire(FireResetGym<G>),
}

impl<G, I, const O: usize> Gym<I, O, 2> for OptionalFireResetGym<G>
where
    G: Gym<I, O, 2, ActionSpace = Discrete>,
{
    type Error = FireResetGymError<G::Error>;
    type ObservationSpace = G::ObservationSpace;
    type ActionSpace = Discrete;

    /// Resets native observations with the wrapped item shape and kind; FIRE games receive startup actions.
    fn reset(
        &mut self,
    ) -> Result<ResetInfo<I, <G::ObservationSpace as TensorBatchTypes>::Item>, Self::Error> {
        match self {
            Self::Plain(gym) => gym.reset().map_err(FireResetGymError::GymError),
            Self::Fire(gym) => gym.reset(),
        }
    }

    /// Forwards Int actions [1] and preserves the wrapped observation item shape, dtype, and kind.
    fn step(
        &mut self,
        action: Tensor<1, Int>,
    ) -> Result<StepInfo<I, <G::ObservationSpace as TensorBatchTypes>::Item>, Self::Error> {
        match self {
            Self::Plain(gym) => gym.step(action).map_err(FireResetGymError::GymError),
            Self::Fire(gym) => gym.step(action),
        }
    }

    fn observation_space(&self) -> Self::ObservationSpace {
        match self {
            Self::Plain(gym) => gym.observation_space(),
            Self::Fire(gym) => gym.observation_space(),
        }
    }

    fn action_space(&self) -> Discrete {
        match self {
            Self::Plain(gym) => gym.action_space(),
            Self::Fire(gym) => gym.action_space(),
        }
    }
}

/// Builds the standard Atari preprocessing stack on CPU with native U8 observations [4, 84, 84].
/// The time limit counts emulator frames. Actions are CPU Int [1]; raw metrics cover complete games.
pub fn make_atari_env(
    rom_path: PathBuf,
    seed: i32,
    max_episode_frames: u32,
    render: bool,
) -> impl Gym<
    RawRewardInfo<EpisodeStatisticsInfo<AtariInfo>>,
    4,
    ObservationSpace = FrameObservationSpace,
    ActionSpace = Discrete,
    Error: Send + Sync + std::fmt::Debug,
> {
    let builder = AtariGym::<RgbScreen>::builder()
        .rom_path(rom_path)
        .random_seed(seed)
        .repeat_action_probability(0.0);
    #[cfg(feature = "rendering")]
    let builder = builder.render(render).render_every(4);
    #[cfg(not(feature = "rendering"))]
    let _ = render;
    let mut env = builder.build().expect("failed to load the Atari ROM");
    let has_fire = env.minimal_action_set().contains(&1);
    let env = TimeLimitGym::new(env, max_episode_frames);
    let env = RecordEpisodeStatisticsGym::new(env);
    let env = NoopResetGym::new_with_noop_max(env, 30);
    let env = MaxAndSkipGym::new(env, 4);
    let env = EpisodicLifeGym::new(env, &Device::flex());
    let env = if has_fire {
        OptionalFireResetGym::Fire(FireResetGym::new(env))
    } else {
        OptionalFireResetGym::Plain(env)
    };
    let env = RecordRawRewardGym::new(env);
    let env = ClipRewardGym::new(env);
    let env = WarpGym::new(env);
    FrameStackGym::new(env, 4, FrameObservationSpace)
}

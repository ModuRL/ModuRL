use crate::ale::Ale;
use bon::bon;
use burn::tensor::{DType, Device, Int, Tensor, TensorData, TensorReadError};
use modurl::{
    gym::{Gym, ResetInfo, StepInfo},
    spaces::{Discrete, ObservationSpace},
    tensor_rank::{NextRank, PrevRank, TensorBatchTypes},
};
use std::{
    ffi::CString,
    fs::File,
    marker::PhantomData,
    path::{Path, PathBuf},
};

mod sealed {
    pub trait Sealed {}
}

/// Selects the native U8 observation layout. Only RAM, RGB, and grayscale modes are supported.
/// Each mode fixes the item and batch tensor ranks; it stores no tensors.
pub trait AtariObservationMode: TensorBatchTypes<Kind = Int> + sealed::Sealed {
    type Space;
    #[doc(hidden)]
    const FORMAT: u8;
}

/// RAM observations [ram_size], batched as [batch_size, ram_size].
pub struct Ram;

/// RGB observations [height, width, 3], batched as [batch_size, height, width, 3].
pub struct RgbScreen;

/// Grayscale observations [height, width], batched as [batch_size, height, width].
pub struct GrayscaleScreen;

macro_rules! observation_mode {
    ($mode:ty, $item:literal, $batch:literal, $format:literal) => {
        impl sealed::Sealed for $mode {}
        impl TensorBatchTypes for $mode {
            type Kind = Int;
            type Item = Tensor<$item, Int>;
            type Batch = Tensor<$batch, Int>;
        }
        impl AtariObservationMode for $mode {
            type Space = AtariObservationSpace<$batch>;
            const FORMAT: u8 = $format;
        }
    };
}

observation_mode!(Ram, 1, 2, 0);
observation_mode!(RgbScreen, 3, 4, 1);
observation_mode!(GrayscaleScreen, 2, 3, 2);

/// CPU U8 observation contract with one leading batch axis and byte values in 0..=255.
/// Rank B counts the batch axis and all observation axes; shape() excludes the batch axis.
#[derive(Clone, Debug)]
pub struct AtariObservationSpace<const B: usize> {
    shape: Vec<usize>,
}

impl<const B: usize> AtariObservationSpace<B> {
    pub(crate) fn new(shape: Vec<usize>) -> Self {
        const {
            assert!(B >= 2, "observations require batch and item axes");
        }
        assert_eq!(shape.len(), B - 1);
        assert!(shape.iter().all(|size| *size > 0));
        Self { shape }
    }
}

impl<const B: usize> TensorBatchTypes for AtariObservationSpace<B>
where
    Tensor<B, Int>: PrevRank,
{
    type Kind = Int;
    type Item = <Tensor<B, Int> as PrevRank>::Prev;
    type Batch = Tensor<B, Int>;
}

impl<const B: usize> ObservationSpace<B> for AtariObservationSpace<B>
where
    Tensor<B, Int>: PrevRank,
{
    /// Checks CPU U8 batches [batch_size, ...self.shape()], preserving all axes.
    fn contains(&self, observations: &Tensor<B, Int>) -> bool {
        observations.dims()[1..] == self.shape
            && observations.dtype() == DType::U8
            && observations.device() == Device::flex()
    }

    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }
}

/// Atari game metadata supplied with resets and transitions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtariInfo {
    pub lives: u32,
    /// Total emulator frames since ROM loading.
    pub frame_number: u32,
    /// Emulator frames since the last real game reset.
    pub episode_frame_number: u32,
}

/// CPU Atari environment with native U8 observations determined by mode M.
///
/// ```compile_fail
/// use modurl::gym::Gym;
/// use modurl_ale::{AtariGym, AtariInfo, Ram};
/// fn requires_rgb_rank<G: Gym<AtariInfo, 4, 2>>() {}
/// requires_rgb_rank::<AtariGym<Ram>>();
/// ```
/// Actions are Int [1] indices into the ROM's minimal action set. The environment remains thread-affine.
pub struct AtariGym<M: AtariObservationMode = RgbScreen> {
    _not_send_sync: PhantomData<*const ()>,
    mode: PhantomData<M>,
    ale: Ale,
    observation_space: M::Space,
    action_space: Discrete,
    frame_skip: usize,
    lives: u32,
    buffer_bytes: Vec<u8>,
    #[cfg(feature = "rendering")]
    renderer: Option<crate::renderer::Renderer>,
    #[cfg(feature = "rendering")]
    render_every: usize,
    #[cfg(feature = "rendering")]
    timesteps: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum AtariGymError {
    #[error("Atari ROM I/O failed: {0}")]
    IoError(#[source] std::io::Error),
    #[error("invalid Atari ROM path: {path}", path = .0.display())]
    InvalidRomPath(PathBuf),
    #[error("Atari tensor read failed: {0}")]
    Tensor(#[from] TensorReadError),
    #[error("repeat_action_probability must be finite and in 0..=1")]
    InvalidRepeatActionProbability,
    #[error("render_every must be nonzero")]
    ZeroRenderInterval,
    #[cfg(feature = "rendering")]
    #[error("Atari rendering failed: {0}")]
    Rendering(#[from] minifb::Error),
    #[error("Atari action must have shape [1], got {0:?}")]
    ActionShape(Vec<usize>),
    #[error("Atari actions must use CPU Flex")]
    ActionDevice,
    #[error("Atari action {action} must be in 0..{count}")]
    ActionIndex { action: i64, count: usize },
}

#[bon]
impl<M: AtariObservationMode> AtariGym<M> {
    /// Loads a supplied ROM and fixes the observation layout from mode M.
    /// Creates CPU U8 observations and a discrete minimal-action space. No ROMs are downloaded.
    /// Rejects unreadable paths and invalid sticky-action or rendering settings before loading the ROM.
    #[builder]
    pub fn builder<const D: usize, const B: usize>(
        rom_path: PathBuf,
        random_seed: Option<i32>,
        #[builder(default = 0.25)] repeat_action_probability: f32,
        #[cfg(feature = "rendering")]
        #[builder(default = false)]
        render: bool,
        #[cfg(feature = "rendering")]
        #[builder(default = 4)]
        render_every: usize,
    ) -> Result<Self, AtariGymError>
    where
        M: AtariObservationMode<
                Space = AtariObservationSpace<B>,
                Item = Tensor<D, Int>,
                Batch = Tensor<B, Int>,
            >,
        Tensor<D, Int>: NextRank<Next = Tensor<B, Int>>,
        Tensor<B, Int>: PrevRank<Prev = Tensor<D, Int>>,
    {
        const {
            assert!(B == D + 1, "batch rank must add one observation axis");
        }
        if !repeat_action_probability.is_finite()
            || !(0.0..=1.0).contains(&repeat_action_probability)
        {
            return Err(AtariGymError::InvalidRepeatActionProbability);
        }
        #[cfg(feature = "rendering")]
        if render_every == 0 {
            return Err(AtariGymError::ZeroRenderInterval);
        }
        validate_rom(&rom_path)?;
        let c_path = path_to_cstring(&rom_path)?;

        let mut ale = Ale::new();
        if let Some(seed) = random_seed {
            ale.set_int(c"random_seed", seed);
        }
        ale.set_float(c"repeat_action_probability", repeat_action_probability);
        ale.set_bool(c"color_averaging", false);
        ale.load_rom_file(&c_path);
        let shape = if const { M::FORMAT == 0 } {
            vec![ale.ram_size()]
        } else if const { M::FORMAT == 1 } {
            vec![ale.screen_height(), ale.screen_width(), 3]
        } else {
            vec![ale.screen_height(), ale.screen_width()]
        };
        let observation_len = shape.iter().product();
        let action_space = Discrete::new(ale.minimal_action_set().len());
        Ok(Self {
            _not_send_sync: PhantomData,
            mode: PhantomData,
            lives: ale.lives() as u32,
            #[cfg(feature = "rendering")]
            renderer: if render {
                Some(crate::renderer::Renderer::new(
                    ale.screen_width(),
                    ale.screen_height(),
                    "Atari Gym Renderer",
                    true,
                )?)
            } else {
                None
            },
            ale,
            observation_space: AtariObservationSpace::new(shape),
            action_space,
            frame_skip: 1,
            buffer_bytes: vec![0; observation_len],
            #[cfg(feature = "rendering")]
            render_every,
            #[cfg(feature = "rendering")]
            timesteps: 0,
        })
    }
}

impl<M: AtariObservationMode> AtariGym<M> {
    pub fn set_frame_skip(&mut self, frame_skip: usize) {
        assert!(frame_skip >= 1, "frame_skip must be at least 1");
        self.frame_skip = frame_skip;
    }

    fn info(&self) -> AtariInfo {
        AtariInfo {
            lives: self.lives,
            frame_number: self.ale.frame_number(),
            episode_frame_number: self.ale.episode_frame_number(),
        }
    }

    pub fn get_lives(&self) -> u32 {
        self.lives
    }

    pub fn minimal_action_set(&mut self) -> Vec<i32> {
        self.ale.minimal_action_set().to_vec()
    }

    #[cfg(feature = "rendering")]
    fn render(&mut self) -> Result<(), AtariGymError> {
        if let Some(renderer) = &mut self.renderer {
            let mut screen_vec = vec![0u8; self.ale.screen_width() * self.ale.screen_height() * 3];
            self.ale.get_screen_rgb(&mut screen_vec);
            renderer.set_buffer(
                &screen_vec
                    .chunks(3)
                    .map(|chunk| {
                        ((chunk[0] as u32) << 16) | ((chunk[1] as u32) << 8) | chunk[2] as u32
                    })
                    .collect::<Vec<u32>>(),
            );
            renderer.present()?;
        }
        Ok(())
    }
}

impl<M, const D: usize, const B: usize> AtariGym<M>
where
    M: AtariObservationMode<
            Space = AtariObservationSpace<B>,
            Item = Tensor<D, Int>,
            Batch = Tensor<B, Int>,
        >,
    Tensor<D, Int>: NextRank<Next = Tensor<B, Int>>,
    Tensor<B, Int>: PrevRank<Prev = Tensor<D, Int>>,
{
    /// Reads ALE bytes as CPU U8 observations of rank D with the mode's item shape; adds no batch axis.
    fn observation(&mut self) -> Tensor<D, Int> {
        if const { M::FORMAT == 0 } {
            self.ale.get_ram(&mut self.buffer_bytes);
        } else if const { M::FORMAT == 1 } {
            self.ale.get_screen_rgb(&mut self.buffer_bytes);
        } else {
            self.ale.get_screen_grayscale(&mut self.buffer_bytes);
        }
        Tensor::from_data(
            TensorData::new(self.buffer_bytes.clone(), self.observation_space.shape()),
            (&Device::flex(), DType::U8),
        )
    }

    /// Applies one minimal-action index and returns rank-D CPU U8 observations without a batch axis.
    /// Sums rewards across skipped frames and max-pools the final two available frames.
    fn step_index(
        &mut self,
        action: usize,
    ) -> Result<StepInfo<AtariInfo, Tensor<D, Int>>, AtariGymError> {
        let mapped_action = self.ale.minimal_action_set()[action];
        let mut observation: Option<Tensor<D, Int>> = None;
        let mut reward = 0.0f32;
        for i in 0..self.frame_skip {
            reward += self.ale.act(mapped_action) as f32;
            if i >= self.frame_skip.saturating_sub(2) {
                let next = self.observation();
                observation = Some(match observation {
                    Some(previous) => previous.max_pair(next),
                    None => next,
                });
            }
            if self.ale.is_game_over() {
                break;
            }
        }
        let done = self.ale.is_game_over();
        self.lives = self.ale.lives() as u32;
        let observation = observation.unwrap_or_else(|| self.observation());
        #[cfg(feature = "rendering")]
        {
            self.timesteps += 1;
            if self.timesteps.is_multiple_of(self.render_every) {
                self.render()?;
            }
        }
        Ok(StepInfo {
            observation,
            reward,
            done,
            truncated: false,
            info: self.info(),
        })
    }
}

impl<M, const D: usize, const B: usize> Gym<AtariInfo, B, 2> for AtariGym<M>
where
    M: AtariObservationMode<
            Space = AtariObservationSpace<B>,
            Item = Tensor<D, Int>,
            Batch = Tensor<B, Int>,
        >,
    Tensor<D, Int>: NextRank<Next = Tensor<B, Int>>,
    Tensor<B, Int>: PrevRank<Prev = Tensor<D, Int>>,
{
    type Error = AtariGymError;
    type ObservationSpace = AtariObservationSpace<B>;
    type ActionSpace = Discrete;

    /// Returns CPU U8 observations with mode-specific rank D and item shape after a real game reset.
    fn reset(&mut self) -> Result<ResetInfo<AtariInfo, Tensor<D, Int>>, Self::Error> {
        self.ale.reset_game();
        self.lives = self.ale.lives() as u32;
        Ok(ResetInfo {
            observation: self.observation(),
            info: self.info(),
        })
    }

    fn action_space(&self) -> Discrete {
        self.action_space.clone()
    }

    fn observation_space(&self) -> Self::ObservationSpace {
        self.observation_space.clone()
    }

    /// Accepts one CPU Int action [1] and returns the mode's rank-D CPU U8 item observation.
    /// Rejects shape, device, and action-index errors before advancing the emulator.
    fn step(
        &mut self,
        action: Tensor<1, Int>,
    ) -> Result<StepInfo<AtariInfo, Tensor<D, Int>>, Self::Error> {
        let action = action_index(action, self.action_space.get_possible_values())?;
        self.step_index(action)
    }
}

/// Reads one CPU Int action [1] and validates its index against the minimal-action count.
fn action_index(action: Tensor<1, Int>, count: usize) -> Result<usize, AtariGymError> {
    if action.dims() != [1] {
        return Err(AtariGymError::ActionShape(action.dims().to_vec()));
    }
    if action.device() != Device::flex() {
        return Err(AtariGymError::ActionDevice);
    }
    let action = action.try_into_scalar::<i64>()?;
    if action < 0 || action as usize >= count {
        return Err(AtariGymError::ActionIndex { action, count });
    }
    Ok(action as usize)
}

fn validate_rom(path: &Path) -> Result<(), AtariGymError> {
    let metadata = path.metadata().map_err(AtariGymError::IoError)?;
    if !metadata.is_file() {
        return Err(AtariGymError::IoError(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "ROM path is not a file",
        )));
    }
    File::open(path).map_err(AtariGymError::IoError)?;
    path_to_cstring(path).map(|_| ())
}

#[cfg(unix)]
fn path_to_cstring(path: &Path) -> Result<CString, AtariGymError> {
    use std::os::unix::ffi::OsStrExt;
    CString::new(path.as_os_str().as_bytes())
        .map_err(|_| AtariGymError::InvalidRomPath(path.to_owned()))
}

#[cfg(not(unix))]
fn path_to_cstring(path: &Path) -> Result<CString, AtariGymError> {
    let path_string = path
        .to_str()
        .ok_or_else(|| AtariGymError::InvalidRomPath(path.to_owned()))?;
    CString::new(path_string).map_err(|_| AtariGymError::InvalidRomPath(path.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_modes_preserve_byte_observation_ranks() {
        fn ram<M: AtariObservationMode<Item = Tensor<1, Int>, Batch = Tensor<2, Int>>>() {}
        fn rgb<M: AtariObservationMode<Item = Tensor<3, Int>, Batch = Tensor<4, Int>>>() {}
        fn gray<M: AtariObservationMode<Item = Tensor<2, Int>, Batch = Tensor<3, Int>>>() {}
        ram::<Ram>();
        rgb::<RgbScreen>();
        gray::<GrayscaleScreen>();
        let space = AtariObservationSpace::<4>::new(vec![210, 160, 3]);
        let observations = Tensor::<4, Int>::zeros([2, 210, 160, 3], (&Device::flex(), DType::U8));
        assert!(space.contains(&observations));
        assert!(!space.contains(&observations.cast(DType::I32)));
        assert!(!space.contains(&Tensor::zeros(
            [2, 160, 210, 3],
            (&Device::flex(), DType::U8)
        )));
        assert_eq!(space.shape(), [210, 160, 3]);
    }

    #[test]
    fn scalar_actions_and_byte_frame_pooling_preserve_native_types() {
        assert_eq!(
            action_index(Tensor::from_data([1i32], &Device::flex()), 3).unwrap(),
            1
        );
        assert!(matches!(
            action_index(Tensor::from_data([-1i32], &Device::flex()), 3),
            Err(AtariGymError::ActionIndex { .. })
        ));
        assert!(matches!(
            action_index(Tensor::from_data([3i32], &Device::flex()), 3),
            Err(AtariGymError::ActionIndex { .. })
        ));
        assert!(matches!(
            action_index(Tensor::from_data([0i32, 1], &Device::flex()), 3),
            Err(AtariGymError::ActionShape(_))
        ));
        let first = Tensor::<1, Int>::from_data([0u8, 255, 5], (&Device::flex(), DType::U8));
        let second = Tensor::<1, Int>::from_data([7u8, 254, 10], (&Device::flex(), DType::U8));
        let pooled = first.max_pair(second);
        assert_eq!(pooled.dtype(), DType::U8);
        assert_eq!(pooled.into_data().try_to_vec::<u8>().unwrap(), [7, 255, 10]);
    }

    #[test]
    fn constructor_rejects_invalid_settings_and_missing_roms() {
        assert!(matches!(
            AtariGym::<Ram>::builder()
                .rom_path(PathBuf::from("missing-rom.bin"))
                .repeat_action_probability(f32::NAN)
                .build(),
            Err(AtariGymError::InvalidRepeatActionProbability)
        ));
        assert!(matches!(
            AtariGym::<GrayscaleScreen>::builder()
                .rom_path(PathBuf::from("missing-rom.bin"))
                .repeat_action_probability(1.1)
                .build(),
            Err(AtariGymError::InvalidRepeatActionProbability)
        ));
        assert!(matches!(
            AtariGym::<RgbScreen>::builder()
                .rom_path(PathBuf::from("missing-rom.bin"))
                .build(),
            Err(AtariGymError::IoError(_))
        ));
        #[cfg(feature = "rendering")]
        assert!(matches!(
            AtariGym::<Ram>::builder()
                .rom_path(PathBuf::from("missing-rom.bin"))
                .render_every(0)
                .build(),
            Err(AtariGymError::ZeroRenderInterval)
        ));
    }
}

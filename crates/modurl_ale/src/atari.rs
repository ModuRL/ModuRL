use std::error::Error;
use std::ffi::CString;
use std::fmt;
use std::fs::File;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use crate::ale::Ale;
use bon::bon;
use candle_core::Tensor;
use modurl::{
    gym::{Gym, ResetInfo, StepInfo},
    spaces::{BoxSpace, Discrete},
};

#[derive(Clone, Copy)]
pub enum AtariObsType {
    RAM,
    RGBScreen,
    GrayscaleScreen,
}

/// Atari-specific state reported alongside resets and transitions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtariInfo {
    pub lives: u32,
    /// Total emulator frames since the ROM was loaded.
    pub frame_number: u32,
    /// Emulator frames since the most recent real game reset.
    pub episode_frame_number: u32,
}

pub struct AtariGym {
    // Keep auto-traits stable across feature combinations. Rendering is thread-affine, so the
    // environment must remain !Send and !Sync even when the rendering feature is disabled.
    _not_send_sync: PhantomData<*const ()>,
    ale: Ale,
    obs_type: AtariObsType,
    device: candle_core::Device,
    observation_space: BoxSpace,
    action_space: Discrete,
    frame_skip: usize,
    lives: u32,
    #[cfg(feature = "rendering")]
    renderer: Option<crate::renderer::Renderer>,
    #[cfg(feature = "rendering")]
    render_every: usize,
    #[cfg(feature = "rendering")]
    timesteps: usize,
}

#[derive(Debug)]
pub enum AtariGymError {
    IoError(std::io::Error),
    InvalidRomPath(PathBuf),
    CandleError(candle_core::Error),
}

impl fmt::Display for AtariGymError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IoError(error) => write!(formatter, "failed to read ROM: {error}"),
            Self::InvalidRomPath(path) => {
                write!(formatter, "invalid ROM path: {}", path.display())
            }
            Self::CandleError(error) => write!(formatter, "tensor error: {error}"),
        }
    }
}

impl Error for AtariGymError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::IoError(error) => Some(error),
            Self::CandleError(error) => Some(error),
            Self::InvalidRomPath(_) => None,
        }
    }
}

#[bon]
impl AtariGym {
    fn info(&self) -> AtariInfo {
        AtariInfo {
            lives: self.lives,
            frame_number: self.ale.frame_number(),
            episode_frame_number: self.ale.episode_frame_number(),
        }
    }

    #[builder]
    pub fn new(
        rom_path: PathBuf,
        obs_type: AtariObsType,
        device: candle_core::Device,
        random_seed: Option<i32>,
        /// Sticky-action probability. Defaults to 0.25 when unset;
        /// pass 0.0 for deterministic NoFrameskip-style behavior.
        repeat_action_probability: Option<f32>,
        #[cfg(feature = "rendering")]
        #[builder(default = false)]
        render: bool,
        #[cfg(feature = "rendering")]
        #[builder(default = 4)]
        render_every: usize,
    ) -> Result<Self, AtariGymError> {
        validate_rom(&rom_path)?;
        let c_path = path_to_cstring(&rom_path)?;

        let mut ale = Ale::new();
        if let Some(seed) = random_seed {
            ale.set_int(c"random_seed", seed);
        }

        let repeat_action_probability = repeat_action_probability.unwrap_or(0.25);
        ale.set_float(c"repeat_action_probability", repeat_action_probability);

        ale.set_bool(c"color_averaging", false);
        ale.load_rom_file(&c_path);

        let observation_space = Self::get_observation_space_initial(&ale, obs_type, &device)
            .map_err(AtariGymError::CandleError)?;
        let action_space = Self::get_action_space_initial(&mut ale);

        Ok(Self {
            _not_send_sync: PhantomData,
            lives: ale.lives() as u32,
            #[cfg(feature = "rendering")]
            renderer: if render {
                Some(crate::renderer::Renderer::new(
                    ale.screen_width(),
                    ale.screen_height(),
                    "Atari Gym Renderer",
                    true,
                ))
            } else {
                None
            },
            ale,
            obs_type,
            device,
            observation_space,
            action_space,
            frame_skip: 1,
            #[cfg(feature = "rendering")]
            render_every,
            #[cfg(feature = "rendering")]
            timesteps: 0,
        })
    }
}

impl AtariGym {
    pub fn set_frame_skip(&mut self, frame_skip: usize) {
        assert!(frame_skip >= 1, "frame_skip must be at least 1");
        self.frame_skip = frame_skip;
    }

    fn get_action_space_initial(ale: &mut Ale) -> Discrete {
        Discrete::new(ale.minimal_action_set().len())
    }

    fn get_observation_space_initial(
        ale: &Ale,
        obs_type: AtariObsType,
        device: &candle_core::Device,
    ) -> Result<BoxSpace, candle_core::Error> {
        Ok(match obs_type {
            AtariObsType::RAM => BoxSpace::new(
                Tensor::full(0.0f32, &[ale.ram_size()], device)?,
                Tensor::full(1.0f32, &[ale.ram_size()], device)?,
            ),
            AtariObsType::RGBScreen => {
                let (width, height) = (ale.screen_width(), ale.screen_height());
                BoxSpace::new(
                    Tensor::full(0.0f32, &[height, width, 3], device)?,
                    Tensor::full(1.0f32, &[height, width, 3], device)?,
                )
            }
            AtariObsType::GrayscaleScreen => {
                let (width, height) = (ale.screen_width(), ale.screen_height());
                BoxSpace::new(
                    Tensor::full(0.0f32, &[height, width], device)?,
                    Tensor::full(1.0f32, &[height, width], device)?,
                )
            }
        })
    }

    fn get_state(&self) -> Result<Tensor, candle_core::Error> {
        match self.obs_type {
            AtariObsType::RAM => {
                let mut ram_vec = vec![0u8; self.ale.ram_size()];
                self.ale.get_ram(&mut ram_vec);
                let ram: Vec<f32> = ram_vec.iter().map(|&x| x as f32 / 255.0).collect();
                Tensor::from_slice(&ram, &[ram.len()], &self.device)
            }
            AtariObsType::RGBScreen => {
                let (width, height) = (self.ale.screen_width(), self.ale.screen_height());
                let mut screen_vec = vec![0u8; width * height * 3];
                self.ale.get_screen_rgb(&mut screen_vec);
                let screen: Vec<f32> = screen_vec.iter().map(|&x| x as f32 / 255.0).collect();
                Tensor::from_slice(&screen, &[height, width, 3], &self.device)
            }
            AtariObsType::GrayscaleScreen => {
                let (width, height) = (self.ale.screen_width(), self.ale.screen_height());
                let mut screen_vec = vec![0u8; width * height];
                self.ale.get_screen_grayscale(&mut screen_vec);
                let screen: Vec<f32> = screen_vec.iter().map(|&x| x as f32 / 255.0).collect();
                Tensor::from_slice(&screen, &[height, width], &self.device)
            }
        }
    }

    fn step_usize(&mut self, action: usize) -> Result<StepInfo<AtariInfo>, candle_core::Error> {
        assert!(action < self.get_action_space().get_possible_values());
        let mapped_action = self.ale.minimal_action_set()[action];

        let mut state = None;
        let mut reward = 0.0f32;
        for i in 0..self.frame_skip {
            reward += self.ale.act(mapped_action) as f32;
            if i as i32 >= self.frame_skip as i32 - 2 {
                match state {
                    None => state = Some(self.get_state()?),
                    Some(ref mut s) => {
                        let new_state = self.get_state()?;
                        *s = Tensor::maximum(s, &new_state)?;
                    }
                }
            }
            if self.ale.is_game_over() {
                break;
            }
        }

        let done = self.ale.is_game_over();
        self.lives = self.ale.lives() as u32;
        if state.is_none() {
            state = Some(self.get_state()?);
        }

        #[cfg(feature = "rendering")]
        {
            self.timesteps += 1;
            if self.timesteps % self.render_every == 0 {
                self.render();
            }
        }

        Ok(StepInfo {
            state: state.unwrap(),
            reward,
            done,
            truncated: false,
            info: self.info(),
        })
    }

    #[cfg(feature = "rendering")]
    fn render(&mut self) {
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
            renderer.present();
        }
    }

    fn get_action_space(&self) -> &Discrete {
        &self.action_space
    }

    fn get_observation_space(&self) -> &BoxSpace {
        &self.observation_space
    }

    pub fn get_lives(&self) -> u32 {
        self.lives
    }

    pub fn minimal_action_set(&mut self) -> Vec<i32> {
        self.ale.minimal_action_set().to_vec()
    }
}

impl Gym<AtariInfo> for AtariGym {
    type Error = candle_core::Error;
    type SpaceError = candle_core::Error;

    fn reset(&mut self) -> Result<ResetInfo<AtariInfo>, Self::Error> {
        self.ale.reset_game();
        self.lives = self.ale.lives() as u32;
        Ok(ResetInfo {
            state: self.get_state()?,
            info: self.info(),
        })
    }

    fn action_space(&self) -> Box<dyn modurl::spaces::Space<Error = Self::SpaceError>> {
        Box::new(self.get_action_space().clone())
    }

    fn observation_space(&self) -> Box<dyn modurl::spaces::Space<Error = Self::SpaceError>> {
        Box::new(self.get_observation_space().clone())
    }

    /// Steps with one scalar discrete action shaped `[]`.
    fn step(&mut self, action: Tensor) -> Result<StepInfo<AtariInfo>, Self::Error> {
        let action = action.to_scalar::<u32>()? as usize;
        self.step_usize(action)
    }
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

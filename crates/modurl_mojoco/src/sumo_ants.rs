use bon::bon;
use candle_core::{Device, Tensor};
use modurl::{
    gym::{MultiGym, MultiGymStepInfo},
    spaces::{BoxSpace, Space},
};

use crate::{
    MujocoError,
    core::{MujocoCore, validate_noise_scale},
};

const ACTION_SIZE: usize = 8;
const QPOS_SIZE: usize = 15;
const QVEL_SIZE: usize = 14;
const PLAYER_OBSERVATION_SIZE: usize = QPOS_SIZE + QVEL_SIZE;

/// Per-player metadata from a [`SumoAnts`] transition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SumoAntsInfo {
    /// The row in the vectorized observation and action batches.
    pub player_index: usize,
    /// The player's torso distance from the center of the ring.
    pub distance_from_center: f32,
    /// Whether this player won on this transition.
    pub won: bool,
    /// Whether this player lost on this transition.
    pub lost: bool,
}

/// A competitive Ant game backed by one MuJoCo simulation.
///
/// Each vectorized batch row is a player in the shared game rather than an
/// independent environment. With `N` players, actions are shaped `[N, 8]` and
/// observations are shaped `[N, 29 * N]`. Every row terminates or truncates
/// together, and a terminal step automatically resets the shared simulation,
/// following [`MultiGym`] conventions.
pub struct SumoAnts {
    core: MujocoCore,
    device: Device,
    player_count: usize,
    observation_size: usize,
    initial_qpos: Vec<f64>,
    initial_qvel: Vec<f64>,
    win_reward: f64,
    standing_reward: f64,
    center_cost_weight: f64,
    control_cost_weight: f64,
    ring_radius: f64,
    minimum_torso_height: f64,
    reset_noise_scale: f64,
    max_episode_steps: usize,
    episode_steps: usize,
}

#[bon]
impl SumoAnts {
    /// Creates the shared multiplayer environment.
    #[builder]
    pub fn new(
        #[builder(default = &Device::Cpu)] device: &Device,
        #[builder(default = 2)] player_count: usize,
        #[builder(default = 5)] frame_skip: usize,
        #[builder(default = 100.0)] win_reward: f64,
        #[builder(default = 1.0)] standing_reward: f64,
        #[builder(default = 0.1)] center_cost_weight: f64,
        #[builder(default = 0.01)] control_cost_weight: f64,
        #[builder(default = 3.0)] ring_radius: f64,
        #[builder(default = 0.2)] minimum_torso_height: f64,
        #[builder(default = 0.01)] reset_noise_scale: f64,
        #[builder(default = 1_000)] max_episode_steps: usize,
        #[cfg(feature = "rendering")]
        #[builder(default = false)]
        render: bool,
    ) -> Result<Self, MujocoError> {
        #[cfg(not(feature = "rendering"))]
        let render = false;

        if !(2..=8).contains(&player_count) {
            return Err(MujocoError::InvalidInput(
                "player_count must be between 2 and 8".into(),
            ));
        }
        validate_non_negative("win_reward", win_reward)?;
        validate_non_negative("standing_reward", standing_reward)?;
        validate_non_negative("center_cost_weight", center_cost_weight)?;
        validate_non_negative("control_cost_weight", control_cost_weight)?;
        validate_positive("ring_radius", ring_radius)?;
        if !minimum_torso_height.is_finite() {
            return Err(MujocoError::InvalidInput(
                "minimum_torso_height must be finite".into(),
            ));
        }
        validate_noise_scale(reset_noise_scale)?;
        if max_episode_steps == 0 {
            return Err(MujocoError::InvalidInput(
                "max_episode_steps must be greater than zero".into(),
            ));
        }

        let spawn_radius = (ring_radius * 0.5).max(0.75);
        let model = build_model(player_count, ring_radius, spawn_radius);
        let core = MujocoCore::new(&model, frame_skip, device, render)?;
        let initial_qpos = initial_qpos(player_count, spawn_radius);
        let initial_qvel = vec![0.0; player_count * QVEL_SIZE];
        if core.nq() != initial_qpos.len()
            || core.nv() != initial_qvel.len()
            || core.nu() != player_count * ACTION_SIZE
        {
            return Err(MujocoError::InvalidInput(format!(
                "SumoAnts model shape mismatch: expected nq={}, nv={}, nu={}, got nq={}, nv={}, nu={}",
                initial_qpos.len(),
                initial_qvel.len(),
                player_count * ACTION_SIZE,
                core.nq(),
                core.nv(),
                core.nu()
            )));
        }

        Ok(Self {
            core,
            device: device.clone(),
            player_count,
            observation_size: PLAYER_OBSERVATION_SIZE * player_count,
            initial_qpos,
            initial_qvel,
            win_reward,
            standing_reward,
            center_cost_weight,
            control_cost_weight,
            ring_radius,
            minimum_torso_height,
            reset_noise_scale,
            max_episode_steps,
            episode_steps: 0,
        })
    }

    /// Re-seeds the reset-noise generator.
    pub fn seed(&mut self, seed: u64) {
        self.core.seed(seed);
    }

    /// Sets the exact shared MuJoCo state and returns every player observation.
    pub fn set_state(&mut self, qpos: &[f64], qvel: &[f64]) -> Result<Tensor, MujocoError> {
        self.core.set_state(qpos, qvel)?;
        self.episode_steps = 0;
        self.core.render()?;
        self.observations()
    }

    fn reset_game(&mut self) -> Result<Tensor, MujocoError> {
        self.core.reset_uniform_from(
            &self.initial_qpos,
            &self.initial_qvel,
            self.reset_noise_scale,
        )?;
        self.episode_steps = 0;
        self.core.render()?;
        self.observations()
    }

    fn observations(&self) -> Result<Tensor, MujocoError> {
        Tensor::stack(&self.player_observations()?, 0).map_err(MujocoError::from)
    }

    fn player_observations(&self) -> Result<Vec<Tensor>, MujocoError> {
        let qpos = self.core.qpos();
        let qvel = self.core.qvel();
        (0..self.player_count)
            .map(|player| {
                let self_qpos = player * QPOS_SIZE;
                let self_qvel = player * QVEL_SIZE;

                let mut observation = Vec::with_capacity(self.observation_size);
                observation.extend_from_slice(&qpos[self_qpos..self_qpos + QPOS_SIZE]);
                observation.extend_from_slice(&qvel[self_qvel..self_qvel + QVEL_SIZE]);
                for offset in 1..self.player_count {
                    let opponent = (player + offset) % self.player_count;
                    let opponent_qpos = opponent * QPOS_SIZE;
                    let opponent_qvel = opponent * QVEL_SIZE;
                    observation.extend(
                        (0..3).map(|axis| qpos[opponent_qpos + axis] - qpos[self_qpos + axis]),
                    );
                    observation
                        .extend_from_slice(&qpos[opponent_qpos + 3..opponent_qpos + QPOS_SIZE]);
                    observation.extend_from_slice(&qvel[opponent_qvel..opponent_qvel + QVEL_SIZE]);
                }
                debug_assert_eq!(observation.len(), self.observation_size);
                self.core.tensor(&observation)
            })
            .collect()
    }

    fn distance_from_center(&self, player: usize) -> f64 {
        let offset = player * QPOS_SIZE;
        self.core.qpos()[offset].hypot(self.core.qpos()[offset + 1])
    }

    fn player_lost(&self, player: usize) -> bool {
        let offset = player * QPOS_SIZE;
        let qpos = &self.core.qpos()[offset..offset + QPOS_SIZE];
        !qpos.iter().all(|value| value.is_finite())
            || qpos[0].hypot(qpos[1]) > self.ring_radius
            || qpos[2] < self.minimum_torso_height
    }
}

impl MultiGym<SumoAntsInfo> for SumoAnts {
    type Error = MujocoError;
    type SpaceError = candle_core::Error;

    /// Advances the one shared game with joint actions shaped `[player_count, 8]`.
    fn step(&mut self, action: Tensor) -> Result<MultiGymStepInfo<SumoAntsInfo>, Self::Error> {
        if action.dims() != [self.player_count, ACTION_SIZE] {
            return Err(MujocoError::InvalidInput(format!(
                "action shape mismatch: expected ({}, {ACTION_SIZE}), got {:?}",
                self.player_count,
                action.dims()
            )));
        }
        let action_values = self.core.step(&action.flatten_all()?)?;
        self.episode_steps += 1;
        self.core.render()?;

        let terminal_observations = self.player_observations()?;
        let distances = (0..self.player_count)
            .map(|player| self.distance_from_center(player))
            .collect::<Vec<_>>();
        let lost = (0..self.player_count)
            .map(|player| self.player_lost(player))
            .collect::<Vec<_>>();
        let terminated = lost.iter().any(|lost| *lost);
        let truncated = !terminated && self.episode_steps >= self.max_episode_steps;
        let episode_ended = terminated || truncated;

        let mut rewards = vec![0.0; self.player_count];
        for player in 0..self.player_count {
            let control_cost = action_values[player * ACTION_SIZE..(player + 1) * ACTION_SIZE]
                .iter()
                .map(|value| value * value)
                .sum::<f64>();
            rewards[player] = if lost[player] {
                0.0
            } else {
                self.standing_reward
            } - self.center_cost_weight * distances[player]
                - self.control_cost_weight * control_cost;
        }
        if terminated && lost.iter().any(|lost| !*lost) {
            for player in 0..self.player_count {
                rewards[player] += if lost[player] {
                    -self.win_reward
                } else {
                    self.win_reward
                };
            }
        }

        let infos = (0..self.player_count)
            .map(|player| SumoAntsInfo {
                player_index: player,
                distance_from_center: distances[player] as f32,
                won: terminated && !lost[player] && lost.iter().any(|lost| *lost),
                lost: terminated && lost[player] && lost.iter().any(|lost| !*lost),
            })
            .collect();
        let terminal_states = if episode_ended {
            terminal_observations.iter().cloned().map(Some).collect()
        } else {
            vec![None; self.player_count]
        };
        let states = if episode_ended {
            self.reset_game()?
        } else {
            Tensor::stack(&terminal_observations, 0)?
        };
        Ok(MultiGymStepInfo {
            states,
            rewards: self.core.tensor(&rewards)?,
            infos,
            dones: vec![terminated; self.player_count],
            truncateds: vec![truncated; self.player_count],
            terminal_states,
        })
    }

    fn observation_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
        Box::new(BoxSpace::new_unbounded(
            vec![self.observation_size],
            &self.device,
        ))
    }

    fn action_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
        Box::new(BoxSpace::new_with_universal_bounds(
            vec![ACTION_SIZE],
            -1.0,
            1.0,
            &self.device,
        ))
    }

    fn num_envs(&self) -> usize {
        self.player_count
    }

    fn reset(&mut self) -> Result<Tensor, Self::Error> {
        self.reset_game()
    }
}

fn initial_qpos(player_count: usize, spawn_radius: f64) -> Vec<f64> {
    let mut qpos = Vec::with_capacity(player_count * QPOS_SIZE);
    for player in 0..player_count {
        let angle =
            std::f64::consts::PI + std::f64::consts::TAU * player as f64 / player_count as f64;
        qpos.extend_from_slice(&[
            spawn_radius * angle.cos(),
            spawn_radius * angle.sin(),
            0.65,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            -1.0,
            0.0,
            -1.0,
            0.0,
            1.0,
        ]);
    }
    qpos
}

fn build_model(player_count: usize, ring_radius: f64, spawn_radius: f64) -> String {
    let mut model = format!(
        r#"<!-- Ant bodies adapted from Gymnasium's Ant-v5 model (MIT License). -->
<mujoco model="sumo_ants">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option integrator="RK4" timestep="0.01"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="1" condim="3" contype="1" density="5.0" friction="1 0.5 0.5" margin="0.01" rgba="0.8 0.6 0.4 1"/>
    <motor ctrllimited="true" ctrlrange="-1 1" gear="150"/>
  </default>
  <asset>
    <texture builtin="gradient" height="100" rgb1="0.4 0.6 0.8" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="checker" height="100" name="ring_texture" rgb1="0.12 0.12 0.12" rgb2="0.3 0.3 0.3" type="2d" width="100"/>
    <material name="ring_material" reflectance="0.2" shininess="0.5" specular="0.4" texrepeat="8 8" texture="ring_texture"/>
  </asset>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="0 0 -1" directional="true" pos="0 0 5" specular="0.1 0.1 0.1"/>
    <camera name="track" mode="fixed" pos="0 -8 6" xyaxes="1 0 0 0 0.6 0.8"/>
    <geom conaffinity="1" condim="3" contype="1" material="ring_material" name="ring" pos="0 0 0" size="{ring_radius} 0.1" type="cylinder"/>
    <geom conaffinity="1" condim="3" contype="1" name="lower_floor" pos="0 0 -0.55" rgba="0.04 0.04 0.04 1" size="20 20 0.1" type="box"/>
"#,
    );

    let colors = [
        "0.85 0.45 0.25 1",
        "0.35 0.55 0.95 1",
        "0.35 0.8 0.45 1",
        "0.85 0.75 0.25 1",
        "0.7 0.4 0.9 1",
        "0.2 0.8 0.8 1",
        "0.95 0.4 0.65 1",
        "0.65 0.65 0.65 1",
    ];
    for player in 0..player_count {
        let angle =
            std::f64::consts::PI + std::f64::consts::TAU * player as f64 / player_count as f64;
        let x = spawn_radius * angle.cos();
        let y = spawn_radius * angle.sin();
        let color = colors[player % colors.len()];
        model.push_str(&format!(
            r#"    <body name="player_{player}_torso" pos="{x} {y} 0.65">
      <freejoint name="player_{player}_root"/>
      <geom name="player_{player}_torso_geom" rgba="{color}" size="0.25" type="sphere"/>
      <body name="player_{player}_front_left_leg">
        <geom fromto="0 0 0 0.2 0.2 0" name="player_{player}_aux_1_geom" rgba="{color}" size="0.08" type="capsule"/>
        <body name="player_{player}_aux_1" pos="0.2 0.2 0">
          <joint axis="0 0 1" name="player_{player}_hip_1" range="-30 30" type="hinge"/>
          <geom fromto="0 0 0 0.2 0.2 0" name="player_{player}_left_leg_geom" rgba="{color}" size="0.08" type="capsule"/>
          <body pos="0.2 0.2 0">
            <joint axis="-1 1 0" name="player_{player}_ankle_1" range="30 70" type="hinge"/>
            <geom fromto="0 0 0 0.4 0.4 0" name="player_{player}_left_ankle_geom" rgba="{color}" size="0.08" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="player_{player}_front_right_leg">
        <geom fromto="0 0 0 -0.2 0.2 0" name="player_{player}_aux_2_geom" rgba="{color}" size="0.08" type="capsule"/>
        <body name="player_{player}_aux_2" pos="-0.2 0.2 0">
          <joint axis="0 0 1" name="player_{player}_hip_2" range="-30 30" type="hinge"/>
          <geom fromto="0 0 0 -0.2 0.2 0" name="player_{player}_right_leg_geom" rgba="{color}" size="0.08" type="capsule"/>
          <body pos="-0.2 0.2 0">
            <joint axis="1 1 0" name="player_{player}_ankle_2" range="-70 -30" type="hinge"/>
            <geom fromto="0 0 0 -0.4 0.4 0" name="player_{player}_right_ankle_geom" rgba="{color}" size="0.08" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="player_{player}_back_left_leg">
        <geom fromto="0 0 0 -0.2 -0.2 0" name="player_{player}_aux_3_geom" rgba="{color}" size="0.08" type="capsule"/>
        <body name="player_{player}_aux_3" pos="-0.2 -0.2 0">
          <joint axis="0 0 1" name="player_{player}_hip_3" range="-30 30" type="hinge"/>
          <geom fromto="0 0 0 -0.2 -0.2 0" name="player_{player}_back_leg_geom" rgba="{color}" size="0.08" type="capsule"/>
          <body pos="-0.2 -0.2 0">
            <joint axis="-1 1 0" name="player_{player}_ankle_3" range="-70 -30" type="hinge"/>
            <geom fromto="0 0 0 -0.4 -0.4 0" name="player_{player}_back_ankle_geom" rgba="{color}" size="0.08" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="player_{player}_back_right_leg">
        <geom fromto="0 0 0 0.2 -0.2 0" name="player_{player}_aux_4_geom" rgba="{color}" size="0.08" type="capsule"/>
        <body name="player_{player}_aux_4" pos="0.2 -0.2 0">
          <joint axis="0 0 1" name="player_{player}_hip_4" range="-30 30" type="hinge"/>
          <geom fromto="0 0 0 0.2 -0.2 0" name="player_{player}_back_right_leg_geom" rgba="{color}" size="0.08" type="capsule"/>
          <body pos="0.2 -0.2 0">
            <joint axis="1 1 0" name="player_{player}_ankle_4" range="30 70" type="hinge"/>
            <geom fromto="0 0 0 0.4 -0.4 0" name="player_{player}_back_right_ankle_geom" rgba="{color}" size="0.08" type="capsule"/>
          </body>
        </body>
      </body>
    </body>
"#,
        ));
    }

    model.push_str("  </worldbody>\n  <actuator>\n");
    for player in 0..player_count {
        for joint in [
            "hip_4", "ankle_4", "hip_1", "ankle_1", "hip_2", "ankle_2", "hip_3", "ankle_3",
        ] {
            model.push_str(&format!("    <motor joint=\"player_{player}_{joint}\"/>\n"));
        }
    }
    model.push_str("  </actuator>\n</mujoco>\n");
    model
}

fn validate_non_negative(name: &str, value: f64) -> Result<(), MujocoError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(MujocoError::InvalidInput(format!(
            "{name} must be finite and non-negative"
        )))
    }
}

fn validate_positive(name: &str, value: f64) -> Result<(), MujocoError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(MujocoError::InvalidInput(format!(
            "{name} must be finite and greater than zero"
        )))
    }
}

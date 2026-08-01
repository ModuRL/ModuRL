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

const PLAYER_COUNT: usize = 2;
const ACTION_SIZE: usize = 8;
const QPOS_SIZE: usize = 15;
const QVEL_SIZE: usize = 14;
const BODIES_PER_PLAYER: usize = 13;
const GEOMS_PER_PLAYER: usize = 13;
const OBSERVATION_SIZE: usize = 137;
const ARENA_GEOM_INDEX: usize = 3;
const FIRST_PLAYER_GEOM_INDEX: usize = 4;
const ARENA_HEIGHT: f64 = 0.5;
const GOAL_REWARD: f64 = 1_000.0;

/// Per-player metadata from a [`SumoAnts`] transition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SumoAntsInfo {
    /// The row in the observation and action batches.
    pub player_index: usize,
    /// The player's torso distance from the center of the ring.
    pub distance_from_center: f32,
    /// The center component of the movement reward.
    pub reward_center: f32,
    /// The positive control cost subtracted from the reward.
    pub reward_ctrl: f32,
    /// The positive contact cost subtracted from the reward.
    pub reward_contact: f32,
    /// The standing or fallen survival component.
    pub reward_survive: f32,
    /// The complete non-terminal AntFighter movement reward.
    pub reward_move: f32,
    /// The terminal goal-reward component.
    pub reward_remaining: f32,
    /// Whether the AntFighter's own standing test marked it done.
    pub agent_done: bool,
    /// Whether this player received the original environment's winner flag.
    pub won: bool,
    /// Whether this player received a terminal loss penalty.
    pub lost: bool,
}

/// A compatibility port of OpenAI's two-player `sumo-ants-v0` environment.
///
/// The two batch rows are the two players in one shared match. The model,
/// reset distribution, observations, rewards, contact-gated winner bonus, and
/// 500-step terminal condition follow the original Gym environment.
pub struct SumoAnts {
    core: MujocoCore,
    device: Device,
    min_radius: f64,
    max_radius: f64,
    current_max_radius: f64,
    radius: f64,
    reset_noise_scale: f64,
    move_reward_weight: f64,
    max_episode_steps: usize,
    episode_steps: usize,
    agent_contacts: bool,
}

#[bon]
impl SumoAnts {
    /// Creates a two-player `sumo-ants-v0` compatible environment.
    #[builder]
    pub fn new(
        #[builder(default = &Device::Cpu)] device: &Device,
        #[builder(default = 5)] frame_skip: usize,
        #[builder(default = 2.5)] min_radius: f64,
        #[builder(default = 4.5)] max_radius: f64,
        #[builder(default = 0.1)] reset_noise_scale: f64,
        #[builder(default = 1.0)] move_reward_weight: f64,
        #[builder(default = 500)] max_episode_steps: usize,
        #[cfg(feature = "rendering")]
        #[builder(default = false)]
        render: bool,
    ) -> Result<Self, MujocoError> {
        #[cfg(not(feature = "rendering"))]
        let render = false;

        validate_positive("min_radius", min_radius)?;
        validate_positive("max_radius", max_radius)?;
        if min_radius > max_radius {
            return Err(MujocoError::InvalidInput(
                "min_radius must not exceed max_radius".into(),
            ));
        }
        validate_noise_scale(reset_noise_scale)?;
        if !move_reward_weight.is_finite() {
            return Err(MujocoError::InvalidInput(
                "move_reward_weight must be finite".into(),
            ));
        }
        if max_episode_steps == 0 {
            return Err(MujocoError::InvalidInput(
                "max_episode_steps must be greater than zero".into(),
            ));
        }

        let core = MujocoCore::new(&build_model(), frame_skip, device, render)?;
        if core.nq() != PLAYER_COUNT * QPOS_SIZE
            || core.nv() != PLAYER_COUNT * QVEL_SIZE
            || core.nu() != PLAYER_COUNT * ACTION_SIZE
            || core.nbody() != 1 + PLAYER_COUNT * BODIES_PER_PLAYER
        {
            return Err(MujocoError::InvalidInput(format!(
                "SumoAnts model shape mismatch: got nq={}, nv={}, nu={}, nbody={}",
                core.nq(),
                core.nv(),
                core.nu(),
                core.nbody()
            )));
        }

        Ok(Self {
            core,
            device: device.clone(),
            min_radius,
            max_radius,
            current_max_radius: max_radius,
            radius: max_radius,
            reset_noise_scale,
            move_reward_weight,
            max_episode_steps,
            episode_steps: 0,
            agent_contacts: false,
        })
    }

    /// Re-seeds the reset generator.
    pub fn seed(&mut self, seed: u64) {
        self.core.seed(seed);
    }

    /// Applies the original radius curriculum and resets the match.
    pub fn reset_with_version(&mut self, version: f64) -> Result<Tensor, MujocoError> {
        if !version.is_finite() {
            return Err(MujocoError::InvalidInput("version must be finite".into()));
        }
        let curriculum_radius = self.min_radius + 0.1 * (0.001 * version).exp();
        self.current_max_radius = self.max_radius.min(curriculum_radius);
        self.reset_game()
    }

    /// Sets the exact shared MuJoCo state and starts a fresh match counter.
    pub fn set_state(&mut self, qpos: &[f64], qvel: &[f64]) -> Result<Tensor, MujocoError> {
        self.core.set_state(qpos, qvel)?;
        self.episode_steps = 0;
        self.agent_contacts = false;
        self.core.render()?;
        self.observations()
    }

    fn reset_game(&mut self) -> Result<Tensor, MujocoError> {
        self.episode_steps = 0;
        self.agent_contacts = false;
        self.radius = if self.min_radius == self.current_max_radius {
            self.min_radius
        } else {
            self.core
                .random_uniform(self.min_radius, self.current_max_radius)
        };
        self.core.set_geom_radius(ARENA_GEOM_INDEX, self.radius);
        self.core
            .reset_uniform_positions_normal_velocities(self.reset_noise_scale)?;

        let mut qpos = self.core.qpos().to_vec();
        let qvel = self.core.qvel().to_vec();
        let min_gap = 0.3 + self.min_radius / 2.0;
        for player in 0..PLAYER_COUNT {
            let x = if player % 2 == 0 {
                self.core.random_uniform(-self.radius + min_gap, -0.3)
            } else {
                self.core.random_uniform(0.3, self.radius - min_gap)
            };
            let y_limit = (self.radius * self.radius - x * x).sqrt();
            let y = self
                .core
                .random_uniform(-y_limit + min_gap, y_limit - min_gap);
            qpos[player * QPOS_SIZE] = x;
            qpos[player * QPOS_SIZE + 1] = y;
        }
        self.core.set_state(&qpos, &qvel)?;
        self.core.render()?;
        self.observations()
    }

    fn observations(&self) -> Result<Tensor, MujocoError> {
        Tensor::stack(&self.player_observations()?, 0).map_err(MujocoError::from)
    }

    fn player_observations(&self) -> Result<Vec<Tensor>, MujocoError> {
        let qpos = self.core.qpos();
        let qvel = self.core.qvel();
        (0..PLAYER_COUNT)
            .map(|player| {
                let opponent = 1 - player;
                let self_qpos = player * QPOS_SIZE;
                let self_qvel = player * QVEL_SIZE;
                let opponent_qpos = opponent * QPOS_SIZE;
                let body_start = 1 + player * BODIES_PER_PLAYER;

                let mut observation = Vec::with_capacity(OBSERVATION_SIZE);
                observation.extend_from_slice(&qpos[self_qpos..self_qpos + QPOS_SIZE]);
                observation.extend_from_slice(&qvel[self_qvel..self_qvel + QVEL_SIZE]);
                observation.extend(
                    self.core.cfrc_ext()[body_start..body_start + BODIES_PER_PLAYER]
                        .iter()
                        .flatten()
                        .map(|value| value.clamp(-1.0, 1.0)),
                );
                observation.extend_from_slice(&qpos[opponent_qpos..opponent_qpos + QPOS_SIZE]);
                observation
                    .extend((0..2).map(|axis| qpos[opponent_qpos + axis] - qpos[self_qpos + axis]));
                observation.extend_from_slice(&self.core.body_orientation(body_start));

                let own_distance = self.radius - self.distance_from_center(player);
                let opponent_distance = self.radius - self.distance_from_center(opponent);
                observation.extend_from_slice(&[
                    self.radius,
                    own_distance,
                    opponent_distance,
                    (self.max_episode_steps - self.episode_steps) as f64,
                ]);
                debug_assert_eq!(observation.len(), OBSERVATION_SIZE);
                self.core.tensor(&observation)
            })
            .collect()
    }

    fn distance_from_center(&self, player: usize) -> f64 {
        let offset = player * QPOS_SIZE;
        self.core.qpos()[offset].hypot(self.core.qpos()[offset + 1])
    }

    fn fallen(&self, player: usize) -> bool {
        self.core.qpos()[player * QPOS_SIZE + 2] <= ARENA_HEIGHT + 0.3
    }

    fn agent_done(&self, player: usize) -> bool {
        self.core.qpos()[player * QPOS_SIZE + 2] <= ARENA_HEIGHT + 0.28
    }

    fn standing(&self, player: usize) -> bool {
        self.core.qpos()[player * QPOS_SIZE + 2] >= ARENA_HEIGHT + 0.28
    }

    fn past_arena(&self, player: usize) -> bool {
        self.distance_from_center(player) > self.radius
    }

    fn players_touching(&self) -> bool {
        let first = FIRST_PLAYER_GEOM_INDEX;
        let middle = first + GEOMS_PER_PLAYER;
        let end = middle + GEOMS_PER_PLAYER;
        self.core.contacts().any(|(geom1, geom2, distance)| {
            distance < 0.0
                && (((first..middle).contains(&geom1) && (middle..end).contains(&geom2))
                    || ((first..middle).contains(&geom2) && (middle..end).contains(&geom1)))
        })
    }

    fn contact_cost(&self, player: usize) -> f64 {
        let body_start = 1 + player * BODIES_PER_PLAYER;
        (0.5e-6
            * self.core.cfrc_ext()[body_start..body_start + BODIES_PER_PLAYER]
                .iter()
                .flatten()
                .map(|force| force * force)
                .sum::<f64>())
        .min(10.0)
    }
}

impl MultiGym<SumoAntsInfo> for SumoAnts {
    type Error = MujocoError;
    type SpaceError = candle_core::Error;

    fn step(&mut self, action: Tensor) -> Result<MultiGymStepInfo<SumoAntsInfo>, Self::Error> {
        if action.dims() != [PLAYER_COUNT, ACTION_SIZE] {
            return Err(MujocoError::InvalidInput(format!(
                "action shape mismatch: expected ({PLAYER_COUNT}, {ACTION_SIZE}), got {:?}",
                action.dims()
            )));
        }
        let action_values = self.core.step_bounded(&action.flatten_all()?, -1.0, 1.0)?;
        self.episode_steps += 1;
        self.core.render()?;

        if self.players_touching() {
            self.agent_contacts = true;
        }

        let distances = (0..PLAYER_COUNT)
            .map(|player| self.distance_from_center(player))
            .collect::<Vec<_>>();
        let fallen = (0..PLAYER_COUNT)
            .map(|player| self.fallen(player))
            .collect::<Vec<_>>();
        let past_arena = (0..PLAYER_COUNT)
            .map(|player| self.past_arena(player))
            .collect::<Vec<_>>();
        let agent_dones = (0..PLAYER_COUNT)
            .map(|player| self.agent_done(player))
            .collect::<Vec<_>>();
        let finite_state = self
            .core
            .qpos()
            .iter()
            .chain(self.core.qvel())
            .all(|value| value.is_finite());

        let mut goal_rewards = [0.0; PLAYER_COUNT];
        let mut winners = [false; PLAYER_COUNT];
        let mut losers = [false; PLAYER_COUNT];
        let game_done = if fallen.iter().any(|value| *value) {
            for player in 0..PLAYER_COUNT {
                if fallen[player] {
                    goal_rewards[player] -= GOAL_REWARD;
                    losers[player] = true;
                } else if self.agent_contacts {
                    goal_rewards[player] += GOAL_REWARD;
                    winners[player] = true;
                }
            }
            true
        } else if past_arena.iter().any(|value| *value) {
            for player in 0..PLAYER_COUNT {
                if past_arena[player] {
                    goal_rewards[player] -= GOAL_REWARD;
                    losers[player] = true;
                } else if self.agent_contacts {
                    goal_rewards[player] += GOAL_REWARD;
                    winners[player] = true;
                }
            }
            true
        } else if self.episode_steps >= self.max_episode_steps {
            goal_rewards.fill(-GOAL_REWARD);
            losers.fill(true);
            true
        } else {
            false
        };
        let terminated = game_done || !finite_state || agent_dones.iter().all(|value| *value);

        let mut rewards = [0.0; PLAYER_COUNT];
        let mut infos = Vec::with_capacity(PLAYER_COUNT);
        for player in 0..PLAYER_COUNT {
            let control_cost = 0.1
                * action_values[player * ACTION_SIZE..(player + 1) * ACTION_SIZE]
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>();
            let contact_cost = self.contact_cost(player);
            let center_reward = -distances[player];
            let survive = if self.standing(player) { 5.0 } else { -5.0 };
            let move_reward = center_reward - control_cost - contact_cost + survive;
            rewards[player] = goal_rewards[player] + self.move_reward_weight * move_reward;
            infos.push(SumoAntsInfo {
                player_index: player,
                distance_from_center: distances[player] as f32,
                reward_center: center_reward as f32,
                reward_ctrl: control_cost as f32,
                reward_contact: contact_cost as f32,
                reward_survive: survive as f32,
                reward_move: move_reward as f32,
                reward_remaining: goal_rewards[player] as f32,
                agent_done: agent_dones[player],
                won: winners[player],
                lost: losers[player],
            });
        }

        let terminal_observations = self.player_observations()?;
        let terminal_states = if terminated {
            terminal_observations.iter().cloned().map(Some).collect()
        } else {
            vec![None; PLAYER_COUNT]
        };
        let states = if terminated {
            self.reset_game()?
        } else {
            Tensor::stack(&terminal_observations, 0)?
        };
        Ok(MultiGymStepInfo {
            states,
            rewards: self.core.tensor(&rewards)?,
            infos,
            dones: vec![terminated; PLAYER_COUNT],
            truncateds: vec![false; PLAYER_COUNT],
            terminal_states,
        })
    }

    fn observation_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
        Box::new(BoxSpace::new_unbounded(
            vec![OBSERVATION_SIZE],
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
        PLAYER_COUNT
    }

    fn reset(&mut self) -> Result<Tensor, Self::Error> {
        self.reset_game()
    }
}

fn build_model() -> String {
    let mut model = String::from(
        r#"<mujoco model="sumo_ants">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option integrator="RK4" timestep="0.003" solver="PGS" iterations="1000"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <default class="ant">
      <joint armature="1" damping="1" limited="true"/>
      <geom conaffinity="1" condim="3" density="5.0" friction="1 0.5 0.5" margin="0.01" rgba="0.8 0.6 0.4 1"/>
    </default>
  </default>
  <asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.1 0.5 0.1" rgb2="0.1 0.4 0.1" type="cube" width="127"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0.5 0.5" rgb2="0 0.6 0.6" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
  </asset>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom contype="1" conaffinity="1" friction="1 .1 .1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="20 20 0.125" type="plane"/>
    <geom fromto="4 -5 0 4 5 0" name="rightgoal" rgba="0.6 0 0 0" size=".03" type="cylinder"/>
    <geom fromto="-4 -5 0 -4 5 0" name="leftgoal" rgba="0.6 0 0 0" size=".03" type="cylinder"/>
    <geom conaffinity="1" condim="3" contype="1" friction="1 .1 .1" name="arena" size="4.5 .25" type="cylinder" pos="0 0 .25" rgba="0.3 0.3 0.5 1"/>
"#,
    );
    model.push_str(&ant_body(0, -1.0));
    model.push_str(&ant_body(1, 1.0));
    model.push_str("  </worldbody>\n  <actuator>\n");
    for player in 0..PLAYER_COUNT {
        for joint in [
            "hip_4", "ankle_4", "hip_1", "ankle_1", "hip_2", "ankle_2", "hip_3", "ankle_3",
        ] {
            model.push_str(&format!(
                "    <motor ctrllimited=\"true\" ctrlrange=\"-1 1\" joint=\"agent{player}_{joint}\" gear=\"150\"/>\n"
            ));
        }
    }
    model.push_str("  </actuator>\n</mujoco>\n");
    model
}

fn ant_body(player: usize, x: f64) -> String {
    format!(
        r#"    <body name="agent{player}_torso" pos="{x} 0 2.5" euler="0 0 180" childclass="ant">
      <geom name="agent{player}_torso_geom" pos="0 0 0" size="0.25" type="sphere"/>
      <joint armature="0" damping="0" limited="false" margin="0.01" name="agent{player}_root" pos="0 0 0" range="-30 30" type="free"/>
      <body name="agent{player}_front_left_leg" pos="0 0 0">
        <geom fromto="0 0 0 .2 .2 0" name="agent{player}_aux_1_geom" size=".08" type="capsule"/>
        <body name="agent{player}_aux_1" pos=".2 .2 0">
          <joint axis="0 0 1" name="agent{player}_hip_1" pos="0 0 0" range="-30 30" type="hinge"/>
          <geom fromto="0 0 0 .2 .2 0" name="agent{player}_left_leg_geom" size=".08" type="capsule"/>
          <body pos=".2 .2 0">
            <joint axis="-1 1 0" name="agent{player}_ankle_1" pos="0 0 0" range="30 70" type="hinge"/>
            <geom fromto="0 0 0 .4 .4 0" name="agent{player}_left_ankle_geom" size=".08" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="agent{player}_front_right_leg" pos="0 0 0">
        <geom fromto="0 0 0 -.2 .2 0" name="agent{player}_aux_2_geom" size=".08" type="capsule"/>
        <body name="agent{player}_aux_2" pos="-.2 .2 0">
          <joint axis="0 0 1" name="agent{player}_hip_2" pos="0 0 0" range="-30 30" type="hinge"/>
          <geom fromto="0 0 0 -.2 .2 0" name="agent{player}_right_leg_geom" size=".08" type="capsule"/>
          <body pos="-.2 .2 0">
            <joint axis="1 1 0" name="agent{player}_ankle_2" pos="0 0 0" range="-70 -30" type="hinge"/>
            <geom fromto="0 0 0 -.4 .4 0" name="agent{player}_right_ankle_geom" size=".08" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="agent{player}_back_leg" pos="0 0 0">
        <geom fromto="0 0 0 -.2 -.2 0" name="agent{player}_aux_3_geom" size=".08" type="capsule"/>
        <body name="agent{player}_aux_3" pos="-.2 -.2 0">
          <joint axis="0 0 1" name="agent{player}_hip_3" pos="0 0 0" range="-30 30" type="hinge"/>
          <geom fromto="0 0 0 -.2 -.2 0" name="agent{player}_back_leg_geom" size=".08" type="capsule"/>
          <body pos="-.2 -.2 0">
            <joint axis="-1 1 0" name="agent{player}_ankle_3" pos="0 0 0" range="-70 -30" type="hinge"/>
            <geom fromto="0 0 0 -.4 -.4 0" name="agent{player}_third_ankle_geom" size=".08" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="agent{player}_right_back_leg" pos="0 0 0">
        <geom fromto="0 0 0 .2 -.2 0" name="agent{player}_aux_4_geom" size=".08" type="capsule"/>
        <body name="agent{player}_aux_4" pos=".2 -.2 0">
          <joint axis="0 0 1" name="agent{player}_hip_4" pos="0 0 0" range="-30 30" type="hinge"/>
          <geom fromto="0 0 0 .2 -.2 0" name="agent{player}_rightback_leg_geom" size=".08" type="capsule"/>
          <body pos=".2 -.2 0">
            <joint axis="1 1 0" name="agent{player}_ankle_4" pos="0 0 0" range="30 70" type="hinge"/>
            <geom fromto="0 0 0 .4 -.4 0" name="agent{player}_fourth_ankle_geom" size=".08" type="capsule"/>
          </body>
        </body>
      </body>
    </body>
"#
    )
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

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};
    use modurl::gym::MultiGym;

    use super::SumoAnts;

    #[test]
    fn survivor_bonus_requires_and_uses_the_contact_latch() {
        let mut environment = SumoAnts::builder()
            .min_radius(4.5)
            .max_radius(4.5)
            .reset_noise_scale(0.0)
            .build()
            .unwrap();
        let qpos = [
            5.0, 0.0, 2.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            2.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        environment.set_state(&qpos, &[0.0; 28]).unwrap();
        environment.agent_contacts = true;

        let actions = Tensor::zeros((2, 8), DType::F32, &Device::Cpu).unwrap();
        let transition = environment.step(actions).unwrap();

        assert_eq!(transition.infos[0].reward_remaining, -1_000.0);
        assert_eq!(transition.infos[1].reward_remaining, 1_000.0);
        assert!(transition.infos[1].won);
    }
}

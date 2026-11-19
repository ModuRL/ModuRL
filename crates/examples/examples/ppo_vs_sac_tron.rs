//! Caller-owned discrete PPO and SAC training in a shared light-cycle arena.

use burn::{
    module::Param,
    nn::{Initializer, Tanh},
    optim::AdamWConfig,
    tensor::{DType, Device, Int, Tensor, TensorData},
};
use modurl::prelude::*;
use rand::{RngExt, SeedableRng, rngs::StdRng};

const SIDE: usize = 8;
const CELLS: usize = SIDE * SIDE;

#[derive(Debug, thiserror::Error)]
enum TronError {
    #[error("expected two integer action batches shaped [num_envs, 1] with values in 0..4")]
    InvalidActions,
}

#[derive(Clone)]
struct Slot {
    occupied: [bool; CELLS],
    heads: [[i32; 2]; 2],
}

impl Default for Slot {
    fn default() -> Self {
        Self {
            occupied: [false; CELLS],
            heads: [[0; 2]; 2],
        }
    }
}

struct Tron {
    slots: Vec<Slot>,
    rng: StdRng,
    device: Device,
}

impl Tron {
    fn new(num_envs: usize, seed: u64, device: Device) -> Self {
        Self {
            slots: vec![Slot::default(); num_envs],
            rng: StdRng::seed_from_u64(seed),
            device,
        }
    }

    fn cell([x, y]: [i32; 2]) -> Option<usize> {
        ((0..8).contains(&x) && (0..8).contains(&y)).then_some((y * 8 + x) as usize)
    }

    fn reset_slot(&mut self, index: usize) {
        let heads = [
            [self.rng.random_range(0..4), self.rng.random_range(0..8)],
            [self.rng.random_range(4..8), self.rng.random_range(0..8)],
        ];
        let mut slot = Slot {
            heads,
            ..Slot::default()
        };
        for head in heads {
            slot.occupied[Self::cell(head).unwrap()] = true;
        }
        self.slots[index] = slot;
    }

    fn observation(&self, slot: usize, agent: usize) -> Vec<f32> {
        let slot = &self.slots[slot];
        let mut values = vec![0.0; 3 * CELLS];
        for (value, occupied) in values.iter_mut().zip(slot.occupied) {
            *value = f32::from(occupied);
        }
        for (plane, head) in [slot.heads[agent], slot.heads[1 - agent]]
            .into_iter()
            .enumerate()
        {
            if let Some(cell) = Self::cell(head) {
                values[(plane + 1) * CELLS + cell] = 1.0;
            }
        }
        values
    }

    fn observations(&self, agent: usize) -> Tensor<2> {
        let values: Vec<_> = (0..self.num_envs())
            .flat_map(|slot| self.observation(slot, agent))
            .collect();
        Tensor::from_data(
            TensorData::new(values, [self.num_envs(), 192]),
            &self.device,
        )
    }
}

impl MultiAgentGym for Tron {
    type Error = TronError;
    type ObservationSpace = BoxSpace;
    type ActionSpace = Discrete;

    fn num_agents(&self) -> usize {
        2
    }
    fn num_envs(&self) -> usize {
        self.slots.len()
    }
    fn observation_space(&self, _agent: usize) -> BoxSpace {
        BoxSpace::new_with_universal_bounds([1, 192], 0.0, 1.0, &self.device)
    }
    fn action_space(&self, _agent: usize) -> Discrete {
        Discrete::new(4).unwrap()
    }
    fn reset(&mut self) -> Result<Vec<MultiAgentObservations>, TronError> {
        for slot in 0..self.num_envs() {
            self.reset_slot(slot);
        }
        Ok((0..2)
            .map(|agent| MultiAgentObservations {
                observations: self.observations(agent),
                slots: (0..self.num_envs()).collect(),
            })
            .collect())
    }

    /// Moves both heads simultaneously with absolute directions `[num_envs, 1]`: up, down, left, right.
    fn step(&mut self, actions: Vec<Tensor<2, Int>>) -> Result<Vec<MultiAgentStepInfo>, TronError> {
        if actions.len() != 2 || actions.iter().any(|a| a.dims() != [self.num_envs(), 1]) {
            return Err(TronError::InvalidActions);
        }
        let actions = actions
            .into_iter()
            .map(|a| {
                a.into_data()
                    .try_cast_as::<i64>()
                    .map_err(|_| TronError::InvalidActions)?
                    .try_to_vec::<i64>()
                    .map_err(|_| TronError::InvalidActions)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if actions.iter().flatten().any(|a| !(0..4).contains(a)) {
            return Err(TronError::InvalidActions);
        }
        let mut rewards = [Vec::new(), Vec::new()];
        let mut terminals = [Vec::new(), Vec::new()];
        let mut dones = Vec::new();
        for index in 0..self.num_envs() {
            let slot = &mut self.slots[index];
            let next = std::array::from_fn::<_, 2, _>(|agent| {
                let [dx, dy] = [[0, -1], [0, 1], [-1, 0], [1, 0]][actions[agent][index] as usize];
                [slot.heads[agent][0] + dx, slot.heads[agent][1] + dy]
            });
            let crash = next.map(|head| {
                Self::cell(head).is_none_or(|cell| slot.occupied[cell]) || next[0] == next[1]
            });
            let done = crash[0] || crash[1];
            let reward = if crash[0] == crash[1] {
                0.0
            } else if crash[0] {
                -1.0
            } else {
                1.0
            };
            rewards[0].push(reward);
            rewards[1].push(-reward);
            slot.heads = next;
            for head in next {
                if let Some(cell) = Self::cell(head) {
                    slot.occupied[cell] = true;
                }
            }
            dones.push(done);
            for (agent, terminal) in terminals.iter_mut().enumerate() {
                terminal.push(done.then(|| {
                    Tensor::from_data(
                        TensorData::new(self.observation(index, agent), [192]),
                        &self.device,
                    )
                }));
            }
            if done {
                self.reset_slot(index);
            }
        }
        Ok(rewards
            .into_iter()
            .zip(terminals)
            .enumerate()
            .map(|(agent, (rewards, terminal_observations))| {
                let observations = self.observations(agent);
                MultiAgentStepInfo {
                    next_observations: observations.clone(),
                    next_slots: (0..self.num_envs()).collect(),
                    step: MultiGymStepInfo {
                        observations,
                        rewards: Tensor::from_data(
                            TensorData::new(rewards, [self.num_envs(), 1]),
                            &self.device,
                        ),
                        infos: vec![(); self.num_envs()],
                        dones: dones.clone(),
                        truncateds: vec![false; self.num_envs()],
                        terminal_observations,
                    },
                }
            })
            .collect())
    }
}

fn mlp(output: usize, gain: f64, device: &Device) -> Result<MLP, ModelError> {
    MLP::builder()
        .input_size(192)
        .output_size(output)
        .hidden_layer_sizes(vec![128, 128])
        .activation(Tanh)
        .hidden_initializer(Initializer::Orthogonal {
            gain: 2.0_f64.sqrt(),
        })
        .output_initializer(Initializer::Orthogonal { gain })
        .options((device, DType::F32))
        .build()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let total_timesteps: usize = std::env::var("MODURL_TRON_TIMESTEPS")
        .map(|v| v.parse())
        .unwrap_or(Ok(200_000))?;
    let device = Device::flex().autodiff();
    device.seed(42);
    let mut env = Tron::new(8, 42, device.clone());
    let optimizer = AdamWConfig::new().with_weight_decay(0.0);
    let networks = SeparatePPONetwork::builder()
        .actor_network(ProbabilisticPolicyModel::new(
            mlp(4, 0.01, &device)?,
            CategoricalDistribution,
        ))
        .critic_network(mlp(1, 1.0, &device)?)
        .actor_optimizer(optimizer.init())
        .critic_optimizer(optimizer.init())
        .actor_learning_rate(3e-4)
        .critic_learning_rate(3e-4)
        .build()?;
    let mut ppo = PPOAgent::<_, _, _, TronError>::builder()
        .observation_space(env.observation_space(0))
        .action_space(env.action_space(0))
        .network_info(networks)
        .batch_size(1024)
        .mini_batch_size(64)
        .num_epochs(4)
        .normalize_advantage(true)
        .clip_range(ConstantSchedule::new(0.2))
        .gamma(0.99)
        .advantage_estimator(GAE::new(0.95)?)
        .ent_coef(0.01)
        .vf_coef(0.5)
        .training_horizon(total_timesteps)
        .build()?;
    let mut critics = Vec::new();
    for _ in 0..2 {
        critics.push(
            ActionValueCritic::builder()
                .online_network(DiscreteVectorHeadCritic::new(mlp(4, 1.0, &device)?))
                .optimizer(optimizer.init())
                .learning_rate(3e-4)
                .build()?,
        );
    }
    let entropy = SACEntropyConfiguration::automatic_with_target_schedule(
        Param::from_tensor(Tensor::zeros([1], (&device, DType::F32))),
        optimizer.init(),
        3e-4,
        |progress| (1.0 - progress) * 4.0_f64.ln(),
    )?;
    let mut sac = SACAgent::<_, _, _, TronError>::builder()
        .policy(ProbabilisticPolicyModel::new(
            mlp(4, 0.01, &device)?,
            CategoricalDistribution,
        ))
        .actor_optimizer(optimizer.init())
        .actor_learning_rate(3e-4)
        .critics(critics)
        .entropy_configuration(entropy)
        .action_space(env.action_space(1))
        .observation_space(env.observation_space(1))
        .aggregation_mode(ActionValueCriticAggregationMode::Mean)
        .entropy_change_penalty(0.5)
        .batch_size(64)
        .replay_capacity(50_000)
        .training_horizon(total_timesteps)
        .build()?;
    let mut obs = env.reset()?;
    let mut lengths = vec![0; env.num_envs()];
    let (mut episodes, mut ppo_wins, mut sac_wins, mut draws, mut total_length) = (0, 0, 0, 0, 0);
    let started = std::time::Instant::now();
    let mut timestep = 0;
    while timestep < total_timesteps {
        let ppo_t = ppo.act_for_training(&obs[0].observations, &mut ())?;
        let sac_t = sac.act_for_training(&obs[1].observations, &mut ())?;
        let mut steps = env
            .step(vec![
                ppo_t.to_environment_actions()?,
                sac_t.to_environment_actions()?,
            ])?
            .into_iter();
        let ppo_step = steps.next().unwrap();
        let sac_step = steps.next().unwrap();
        let rewards = ppo_step.step.rewards.to_data().try_to_vec::<f32>()?;
        for (row, &slot) in obs[0].slots.iter().enumerate() {
            let length = &mut lengths[slot];
            *length += 1;
            if ppo_step.step.dones[row] || ppo_step.step.truncateds[row] {
                episodes += 1;
                ppo_wins += usize::from(rewards[row] > 0.0);
                sac_wins += usize::from(rewards[row] < 0.0);
                draws += usize::from(rewards[row] == 0.0);
                total_length += *length;
                *length = 0;
                if episodes % 100 == 0 {
                    println!(
                        "steps={} episodes={episodes} ppo_wins={:.3} sac_wins={:.3} draws={:.3} mean_episode_length={:.2}",
                        timestep + obs[0].slots.len(),
                        ppo_wins as f64 / 100.0,
                        sac_wins as f64 / 100.0,
                        draws as f64 / 100.0,
                        total_length as f64 / 100.0
                    );
                    (ppo_wins, sac_wins, draws, total_length) = (0, 0, 0, 0);
                }
            }
        }
        let next = vec![
            MultiAgentObservations {
                observations: ppo_step.next_observations,
                slots: ppo_step.next_slots,
            },
            MultiAgentObservations {
                observations: sac_step.next_observations,
                slots: sac_step.next_slots,
            },
        ];
        ppo.learn_from_timesteps(&obs[0].observations, ppo_t, ppo_step.step, &obs[0].slots)?;
        sac.learn_from_timesteps(&obs[1].observations, sac_t, sac_step.step, &obs[1].slots)?;
        timestep += obs[0].slots.len();
        obs = next;
    }
    println!("Training elapsed: {:.2?}", started.elapsed());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(heads: [[i32; 2]; 2]) -> Tron {
        let mut env = Tron::new(1, 42, Device::flex());
        env.slots[0].heads = heads;
        for head in heads {
            env.slots[0].occupied[Tron::cell(head).unwrap()] = true;
        }
        env
    }

    fn step(env: &mut Tron, actions: [i64; 2]) -> Vec<MultiGymStepInfo> {
        let actions = actions.map(|a| Tensor::from_data([[a]], &env.device));
        env.step(actions.into())
            .unwrap()
            .into_iter()
            .map(|info| {
                assert_eq!(info.next_slots, (0..env.num_envs()).collect::<Vec<_>>());
                assert_eq!(info.next_observations.dims()[0], info.next_slots.len());
                info.step
            })
            .collect()
    }

    #[test]
    fn trails_and_perspective() {
        let mut env = env([[1, 1], [6, 6]]);
        let steps = step(&mut env, [3, 2]);
        assert_eq!(steps[0].dones, vec![false]);
        assert!(steps[0].terminal_observations[0].is_none());
        let a = env.observation(0, 0);
        let b = env.observation(0, 1);
        for cell in [9, 10, 54, 53] {
            assert_eq!(a[cell], 1.0);
        }
        assert_eq!(a[..64].iter().sum::<f32>(), 4.0);
        assert_eq!(a[64 + 10], 1.0);
        assert_eq!(a[128 + 53], 1.0);
        assert_eq!(a[64..128], b[128..]);
        assert_eq!(a[128..], b[64..128]);
    }

    #[test]
    fn wall_and_trail_crashes_have_joint_flags_and_opposite_rewards() {
        for trail in [false, true] {
            let mut env = env([[0, 1], [6, 6]]);
            if trail {
                env.slots[0].occupied[9] = true;
            }
            let steps = step(&mut env, [if trail { 3 } else { 2 }, 0]);
            assert_eq!(steps[0].dones, vec![true]);
            assert_eq!(steps[0].dones, steps[1].dones);
            assert_eq!(steps[0].truncateds, steps[1].truncateds);
            assert_eq!(
                steps[0].rewards.to_data().try_to_vec::<f32>().unwrap(),
                vec![-1.0]
            );
            assert_eq!(
                steps[1].rewards.to_data().try_to_vec::<f32>().unwrap(),
                vec![1.0]
            );
            assert!(steps.iter().all(|s| s.terminal_observations[0].is_some()));
        }
    }

    #[test]
    fn head_on_and_swaps_are_draws() {
        for heads in [[[2, 2], [4, 2]], [[2, 2], [3, 2]]] {
            let mut env = env(heads);
            for step in step(&mut env, [3, 2]) {
                assert_eq!(step.dones, vec![true]);
                assert_eq!(
                    step.rewards.to_data().try_to_vec::<f32>().unwrap(),
                    vec![0.0]
                );
            }
        }
    }

    #[test]
    fn invalid_actions_leave_state_unchanged() {
        let mut env = env([[1, 1], [6, 6]]);
        let before = env.observation(0, 0);
        for count in [0, 1, 3] {
            assert!(
                env.step(vec![Tensor::zeros([1, 1], &env.device); count])
                    .is_err()
            );
        }
        assert!(
            env.step(vec![Tensor::zeros([1, 2], &env.device); 2])
                .is_err()
        );
        for value in [-1, 4] {
            assert!(
                env.step(vec![Tensor::from_data([[value]], &env.device); 2])
                    .is_err()
            );
        }
        assert_eq!(env.observation(0, 0), before);
    }

    #[test]
    fn seeded_reset_has_distinct_marked_heads() {
        let mut a = Tron::new(8, 42, Device::flex());
        let mut b = Tron::new(8, 42, Device::flex());
        assert_eq!(
            a.reset().unwrap()[0].observations.to_data(),
            b.reset().unwrap()[0].observations.to_data()
        );
        for slot in &a.slots {
            assert_ne!(slot.heads[0], slot.heads[1]);
            assert_eq!(slot.occupied.iter().filter(|v| **v).count(), 2);
        }
    }
}

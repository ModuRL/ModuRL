//! Five independent learners sharing a compact-row, full-board Tron arena.

use burn::{
    nn::{Initializer, Tanh},
    optim::AdamWConfig,
    tensor::{DType, Device, Int, Tensor, TensorData},
};
use modurl::prelude::*;
use std::time::{Duration, Instant};

const PLAYERS: usize = 5;
const SIDE: usize = 32;
const CELLS: usize = SIDE * SIDE;
const FEATURES: usize = 3 * CELLS;
const MOVES: [[i32; 2]; 4] = [[0, -1], [0, 1], [-1, 0], [1, 0]];
const NAMES: [&str; PLAYERS] = ["PPO", "A2C", "SAC", "DQN", "DDQN"];

#[derive(Debug, thiserror::Error)]
enum TronError {
    #[error("expected five integer action batches [live_slots, 1] with values in 0..4")]
    InvalidActions,
    #[error("reset the arena before stepping")]
    NotReset,
    #[error("environment and step counts must be nonzero and their product must fit usize")]
    InvalidConfiguration,
    #[error("expected five agent batches")]
    InvalidAgentCount,
    #[error("a completed round is missing a placement")]
    IncompleteRound,
    #[error("space construction failed: {0}")]
    Space(#[from] SpaceError),
}

#[derive(Clone, Copy, Debug, Default)]
struct Finish {
    won: bool,
    placement: usize,
    length: usize,
}

#[derive(Clone)]
struct Slot {
    occupied: [bool; CELLS],
    heads: [[i32; 2]; PLAYERS],
    alive: [bool; PLAYERS],
    lengths: [usize; PLAYERS],
    finishes: [Option<Finish>; PLAYERS],
}

impl Slot {
    fn spawn(rotation: usize) -> Self {
        let heads = std::array::from_fn(|agent| {
            let angle = std::f64::consts::TAU * ((agent + rotation) % PLAYERS) as f64
                / PLAYERS as f64
                - std::f64::consts::FRAC_PI_2;
            [
                (15.5 + 11.0 * angle.cos()).round() as i32,
                (15.5 + 11.0 * angle.sin()).round() as i32,
            ]
        });
        let mut slot = Self {
            occupied: [false; CELLS],
            heads,
            alive: [true; PLAYERS],
            lengths: [0; PLAYERS],
            finishes: [None; PLAYERS],
        };
        for head in heads {
            if let Some(cell) = Self::cell(head) {
                slot.occupied[cell] = true;
            }
        }
        slot
    }

    fn cell([x, y]: [i32; 2]) -> Option<usize> {
        ((0..SIDE as i32).contains(&x) && (0..SIDE as i32).contains(&y))
            .then_some((y * SIDE as i32 + x) as usize)
    }

    fn advance(&mut self, actions: [usize; PLAYERS]) -> ([f32; PLAYERS], bool) {
        let acted = self.alive;
        let next: [_; PLAYERS] = std::array::from_fn(|agent| {
            let [dx, dy] = if acted[agent] {
                MOVES[actions[agent]]
            } else {
                [0, 0]
            };
            [self.heads[agent][0] + dx, self.heads[agent][1] + dy]
        });
        let crashes: [_; PLAYERS] = std::array::from_fn(|agent| {
            acted[agent]
                && (Self::cell(next[agent]).is_none_or(|cell| self.occupied[cell])
                    || (0..PLAYERS)
                        .any(|other| other != agent && acted[other] && next[agent] == next[other]))
        });
        for agent in 0..PLAYERS {
            if acted[agent] {
                self.lengths[agent] += 1;
                self.heads[agent] = next[agent];
                if let Some(cell) = Self::cell(next[agent]) {
                    self.occupied[cell] = true;
                }
                self.alive[agent] = !crashes[agent];
            }
        }
        let remaining = self.alive.iter().filter(|&&alive| alive).count();
        let ended = remaining <= 1;
        let rewards = std::array::from_fn(|agent| {
            if crashes[agent] {
                self.finishes[agent] = Some(Finish {
                    won: false,
                    placement: remaining + 1,
                    length: self.lengths[agent],
                });
                -1.0
            } else if ended && self.alive[agent] {
                self.finishes[agent] = Some(Finish {
                    won: true,
                    placement: 1,
                    length: self.lengths[agent],
                });
                1.0
            } else {
                0.0
            }
        });
        (rewards, ended)
    }

    fn observation(&self, agent: usize) -> Vec<f32> {
        let mut values = vec![0.0; FEATURES];
        for (value, occupied) in values.iter_mut().zip(self.occupied) {
            *value = f32::from(occupied);
        }
        for (other, &head) in self.heads.iter().enumerate() {
            if self.alive[other]
                && let Some(cell) = Self::cell(head)
            {
                let plane = if agent == other { 1 } else { 2 };
                values[plane * CELLS + cell] = 1.0;
            }
        }
        values
    }
}

struct Tron {
    slots: Vec<Slot>,
    rotations: Vec<usize>,
    completed: Vec<[Finish; PLAYERS]>,
    action_space: Discrete,
    device: Device,
    ready: bool,
}

impl Tron {
    fn new(num_envs: usize, device: Device) -> Result<Self, TronError> {
        if num_envs == 0 {
            return Err(TronError::InvalidConfiguration);
        }
        Ok(Self {
            slots: vec![Slot::spawn(0); num_envs],
            rotations: (0..num_envs).map(|slot| slot % PLAYERS).collect(),
            completed: Vec::new(),
            action_space: Discrete::new(4)?,
            device,
            ready: false,
        })
    }

    fn reset_slot(&mut self, index: usize) {
        self.slots[index] = Slot::spawn(self.rotations[index]);
        self.rotations[index] = (self.rotations[index] + 1) % PLAYERS;
    }

    fn live_slots(&self, agent: usize) -> Vec<usize> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(slot, state)| state.alive[agent].then_some(slot))
            .collect()
    }

    fn observations(&self, agent: usize, slots: &[usize]) -> Tensor<2> {
        let values: Vec<_> = slots
            .iter()
            .flat_map(|&slot| self.slots[slot].observation(agent))
            .collect();
        Tensor::from_data(
            TensorData::new(values, [slots.len(), FEATURES]),
            &self.device,
        )
    }

    fn current(&self, agent: usize) -> MultiAgentObservations {
        let slots = self.live_slots(agent);
        MultiAgentObservations {
            observations: self.observations(agent, &slots),
            slots,
        }
    }
}

impl MultiAgentGym for Tron {
    type Error = TronError;
    type ObservationSpace = BoxSpace;
    type ActionSpace = Discrete;

    fn num_agents(&self) -> usize {
        PLAYERS
    }
    fn num_envs(&self) -> usize {
        self.slots.len()
    }
    fn observation_space(&self, _agent: usize) -> BoxSpace {
        BoxSpace::new_with_universal_bounds([1, FEATURES], 0.0, 1.0, &self.device)
    }
    fn action_space(&self, _agent: usize) -> Discrete {
        self.action_space.clone()
    }

    fn reset(&mut self) -> Result<Vec<MultiAgentObservations>, TronError> {
        for slot in 0..self.num_envs() {
            self.reset_slot(slot);
        }
        self.completed.clear();
        self.ready = true;
        Ok((0..PLAYERS).map(|agent| self.current(agent)).collect())
    }

    /// Moves compact integer actions `[live_slots, 1]` simultaneously: up, down, left, right.
    fn step(&mut self, actions: Vec<Tensor<2, Int>>) -> Result<Vec<MultiAgentStepInfo>, TronError> {
        if !self.ready {
            return Err(TronError::NotReset);
        }
        let acting: [_; PLAYERS] = std::array::from_fn(|agent| self.live_slots(agent));
        if actions.len() != PLAYERS
            || actions
                .iter()
                .zip(&acting)
                .any(|(a, slots)| a.dims() != [slots.len(), 1])
        {
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
        let mut rewards: [_; PLAYERS] = std::array::from_fn(|_| Vec::new());
        let mut dones: [_; PLAYERS] = std::array::from_fn(|_| Vec::new());
        let mut terminals: [_; PLAYERS] = std::array::from_fn(|_| Vec::new());
        let mut cursors = [0; PLAYERS];
        for index in 0..self.num_envs() {
            let acted = self.slots[index].alive;
            let directions = std::array::from_fn(|agent| {
                if acted[agent] {
                    let direction = actions[agent][cursors[agent]] as usize;
                    cursors[agent] += 1;
                    direction
                } else {
                    0
                }
            });
            let (row_rewards, ended) = self.slots[index].advance(directions);
            for agent in 0..PLAYERS {
                if acted[agent] {
                    let done = ended || !self.slots[index].alive[agent];
                    rewards[agent].push(row_rewards[agent]);
                    dones[agent].push(done);
                    terminals[agent].push(done.then(|| {
                        Tensor::from_data(
                            TensorData::new(self.slots[index].observation(agent), [FEATURES]),
                            &self.device,
                        )
                    }));
                }
            }
            if ended {
                let mut finishes = [Finish::default(); PLAYERS];
                for (finish, target) in self.slots[index].finishes.iter().zip(&mut finishes) {
                    *target = finish.ok_or(TronError::IncompleteRound)?;
                }
                self.completed.push(finishes);
                self.reset_slot(index);
            }
        }
        Ok(rewards
            .into_iter()
            .zip(dones)
            .zip(terminals)
            .enumerate()
            .map(|(agent, ((rewards, dones), terminal_observations))| {
                let next = self.current(agent);
                let count = acting[agent].len();
                MultiAgentStepInfo {
                    next_observations: next.observations,
                    next_slots: next.slots,
                    step: MultiGymStepInfo {
                        observations: self.observations(agent, &acting[agent]),
                        rewards: Tensor::from_data(
                            TensorData::new(rewards, [count, 1]),
                            &self.device,
                        ),
                        infos: vec![(); count],
                        dones,
                        truncateds: vec![false; count],
                        terminal_observations,
                    },
                }
            })
            .collect())
    }
}

fn mlp(output: usize, gain: f64, device: &Device) -> Result<MLP, ModelError> {
    MLP::builder()
        .input_size(FEATURES)
        .output_size(output)
        .hidden_layer_sizes(vec![32, 32])
        .activation(Tanh)
        .output_initializer(Initializer::Orthogonal { gain })
        .options((device, DType::F32))
        .build()
}

struct Collection<T> {
    training: Option<T>,
    actions: Tensor<2, Int>,
}

fn collect<G>(
    agent: &mut G,
    observations: &MultiAgentObservations,
    device: &Device,
) -> Result<Collection<G::TrainingActions>, G::Error>
where
    G: for<'c> Agent<ObservationSpace = BoxSpace, ActionSpace = Discrete, ActContext<'c> = ()>,
{
    if observations.slots.is_empty() {
        return Ok(Collection {
            training: None,
            actions: Tensor::zeros([0, 1], device),
        });
    }
    let training = agent.act_for_training(&observations.observations, &mut ())?;
    let actions = training.to_environment_actions()?;
    Ok(Collection {
        training: Some(training),
        actions,
    })
}

fn learn<G: Agent<ObservationSpace = BoxSpace, ActionSpace = Discrete>>(
    agent: &mut G,
    observations: &MultiAgentObservations,
    training: Option<G::TrainingActions>,
    step: MultiAgentStepInfo,
) -> Result<MultiAgentObservations, G::Error> {
    if let Some(training) = training {
        agent.learn_from_timesteps(
            &observations.observations,
            training,
            step.step,
            &observations.slots,
        )?;
    }
    Ok(MultiAgentObservations {
        observations: step.next_observations,
        slots: step.next_slots,
    })
}

#[derive(Default)]
struct Statistics {
    rounds: usize,
    world_steps: usize,
    wins: [usize; PLAYERS],
    placements: [usize; PLAYERS],
    lengths: [usize; PLAYERS],
    rows: [usize; PLAYERS],
    partial_batches: [usize; PLAYERS],
    empty_batches: [usize; PLAYERS],
}

impl Statistics {
    fn record(&mut self, finishes: [Finish; PLAYERS]) {
        self.rounds += 1;
        for (agent, finish) in finishes.into_iter().enumerate() {
            self.wins[agent] += usize::from(finish.won);
            self.placements[agent] += finish.placement;
            self.lengths[agent] += finish.length;
        }
    }

    fn print(&self) {
        println!(
            "\nworld_steps={} completed_rounds={} (completed rounds only)",
            self.world_steps, self.rounds
        );
        println!("algorithm   win_rate  avg_place  avg_survival      rows  partial  empty");
        for (agent, name) in NAMES.iter().enumerate() {
            let denominator = self.rounds.max(1) as f64;
            println!(
                "{name:<10} {:>8.3} {:>10.2} {:>13.2} {:>9} {:>8} {:>6}",
                self.wins[agent] as f64 / denominator,
                self.placements[agent] as f64 / denominator,
                self.lengths[agent] as f64 / denominator,
                self.rows[agent],
                self.partial_batches[agent],
                self.empty_batches[agent]
            );
        }
    }
}

struct TrainingConfig {
    steps: usize,
    seconds: Option<Duration>,
    report_every: usize,
}

fn train(
    env: &mut Tron,
    mut observations: [MultiAgentObservations; PLAYERS],
    config: TrainingConfig,
) -> Result<Statistics, Box<dyn std::error::Error>> {
    let horizon = config
        .steps
        .checked_mul(env.num_envs())
        .filter(|&horizon| horizon > 0)
        .ok_or(TronError::InvalidConfiguration)?;
    let device = &env.device.clone();
    let optimizer = AdamWConfig::new().with_weight_decay(0.0);
    let networks = SeparatePPONetwork::builder()
        .actor_network(ProbabilisticPolicyModel::new(
            mlp(4, 0.01, device)?,
            CategoricalDistribution,
        ))
        .critic_network(mlp(1, 1.0, device)?)
        .actor_optimizer(optimizer.init())
        .critic_optimizer(optimizer.init())
        .actor_learning_rate(3e-4)
        .critic_learning_rate(3e-4)
        .build()?;
    let mut ppo = PPOAgent::<_, _, _, TronError>::builder()
        .observation_space(env.observation_space(0))
        .action_space(env.action_space(0))
        .network_info(networks)
        .batch_size(16)
        .mini_batch_size(8)
        .num_epochs(2)
        .normalize_advantage(true)
        .clip_range(ConstantSchedule::new(0.2))
        .advantage_estimator(GAE::new(0.95)?)
        .ent_coef(0.01)
        .training_horizon(horizon)
        .build()?;
    let networks = A2CNetworkInfo::separate(
        SeparateA2CNetwork::builder()
            .actor_network(ProbabilisticPolicyModel::new(
                mlp(4, 0.01, device)?,
                CategoricalDistribution,
            ))
            .critic_network(mlp(1, 1.0, device)?)
            .actor_optimizer(optimizer.init())
            .critic_optimizer(optimizer.init())
            .actor_learning_rate(7e-4)
            .critic_learning_rate(7e-4)
            .build()?,
    );
    let mut a2c = A2CAgent::<_, _, _, TronError>::builder()
        .observation_space(env.observation_space(1))
        .action_space(env.action_space(1))
        .network_info(networks)
        .batch_size(16)
        .ent_coef(0.01)
        .training_horizon(horizon)
        .build()?;
    let mut critics = Vec::new();
    for _ in 0..2 {
        critics.push(
            ActionValueCritic::builder()
                .online_network(DiscreteVectorHeadCritic::new(mlp(4, 1.0, device)?))
                .optimizer(optimizer.init())
                .learning_rate(3e-4)
                .build()?,
        );
    }
    let mut sac = SACAgent::<_, _, _, TronError>::builder()
        .policy(ProbabilisticPolicyModel::new(
            mlp(4, 0.01, device)?,
            CategoricalDistribution,
        ))
        .actor_optimizer(optimizer.init())
        .actor_learning_rate(3e-4)
        .critics(critics)
        .entropy_configuration(SACEntropyConfiguration::fixed(0.1)?)
        .observation_space(env.observation_space(2))
        .action_space(env.action_space(2))
        .batch_size(8)
        .replay_capacity(512)
        .training_start(8)
        .training_horizon(horizon)
        .build()?;
    let mut dqn = DQNAgent::<_, _, TronError>::builder()
        .observation_space(env.observation_space(3))
        .action_space(env.action_space(3))
        .online_q_network(mlp(4, 1.0, device)?)
        .optimizer(optimizer.init())
        .learning_rate(2.5e-4)
        .batch_size(8)
        .replay_capacity(512)
        .training_start(8)
        .update_frequency(4)
        .target_update_interval(64)
        .epsilon_schedule(LinearSchedule::new(1.0, 0.1))
        .training_horizon(horizon)
        .build()?;
    let mut ddqn = DDQNAgent::<_, _, TronError>::builder()
        .observation_space(env.observation_space(4))
        .action_space(env.action_space(4))
        .online_q_network(mlp(4, 1.0, device)?)
        .optimizer(optimizer.init())
        .learning_rate(2.5e-4)
        .batch_size(8)
        .replay_capacity(512)
        .training_start(8)
        .update_frequency(4)
        .target_update_interval(64)
        .epsilon_schedule(LinearSchedule::new(1.0, 0.1))
        .training_horizon(horizon)
        .build()?;
    let started = Instant::now();
    let mut stats = Statistics::default();
    for tick in 0..config.steps {
        if config
            .seconds
            .is_some_and(|seconds| started.elapsed() >= seconds)
        {
            break;
        }
        for (agent, obs) in observations.iter().enumerate() {
            let count = obs.slots.len();
            stats.rows[agent] += count;
            stats.partial_batches[agent] += usize::from(count > 0 && count < env.num_envs());
            stats.empty_batches[agent] += usize::from(count == 0);
        }
        let p = collect(&mut ppo, &observations[0], device)?;
        let a = collect(&mut a2c, &observations[1], device)?;
        let s = collect(&mut sac, &observations[2], device)?;
        let d = collect(&mut dqn, &observations[3], device)?;
        let q = collect(&mut ddqn, &observations[4], device)?;
        let [ps, as_, ss, ds, qs]: [_; PLAYERS] = env
            .step(vec![p.actions, a.actions, s.actions, d.actions, q.actions])?
            .try_into()
            .map_err(|_| TronError::InvalidAgentCount)?;
        observations = [
            learn(&mut ppo, &observations[0], p.training, ps)?,
            learn(&mut a2c, &observations[1], a.training, as_)?,
            learn(&mut sac, &observations[2], s.training, ss)?,
            learn(&mut dqn, &observations[3], d.training, ds)?,
            learn(&mut ddqn, &observations[4], q.training, qs)?,
        ];
        for finishes in env.completed.drain(..) {
            stats.record(finishes);
        }
        stats.world_steps = tick + 1;
        if config.report_every > 0 && stats.world_steps.is_multiple_of(config.report_every) {
            stats.print();
        }
    }
    stats.print();
    println!("Training elapsed: {:.2?}", started.elapsed());
    Ok(stats)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let steps = std::env::var("MODURL_TRON_FFA_STEPS")
        .map(|v| v.parse())
        .unwrap_or(Ok(256))?;
    let num_envs = std::env::var("MODURL_TRON_FFA_ENVS")
        .map(|v| v.parse())
        .unwrap_or(Ok(8))?;
    let seconds = std::env::var("MODURL_TRON_FFA_SECONDS")
        .ok()
        .map(|v| v.parse::<u64>())
        .transpose()?
        .map(Duration::from_secs);
    let device = Device::flex().autodiff();
    device.seed(42);
    let mut env = Tron::new(num_envs, device)?;
    let observations = env
        .reset()?
        .try_into()
        .map_err(|_| TronError::InvalidAgentCount)?;
    train(
        &mut env,
        observations,
        TrainingConfig {
            steps,
            seconds,
            report_every: 64,
        },
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arena(heads: [[i32; 2]; PLAYERS], alive: [bool; PLAYERS]) -> Tron {
        let mut env = Tron::new(1, Device::flex()).unwrap();
        env.reset().unwrap();
        env.slots[0].occupied.fill(false);
        env.slots[0].heads = heads;
        env.slots[0].alive = alive;
        for (agent, head) in heads.into_iter().enumerate() {
            env.slots[0].occupied[Slot::cell(head).unwrap()] = true;
            if !alive[agent] {
                env.slots[0].finishes[agent] = Some(Finish {
                    placement: agent + 1,
                    ..Finish::default()
                });
            }
        }
        env
    }

    fn step(env: &mut Tron, moves: [i64; PLAYERS]) -> Vec<MultiAgentStepInfo> {
        let actions = moves
            .into_iter()
            .enumerate()
            .map(|(agent, action)| {
                let count = env.live_slots(agent).len();
                Tensor::from_data(
                    TensorData::new(vec![action; count], [count, 1]),
                    &env.device,
                )
            })
            .collect();
        env.step(actions).unwrap()
    }

    fn reward(step: &MultiAgentStepInfo) -> Vec<f32> {
        step.step.rewards.to_data().try_to_vec().unwrap()
    }

    #[test]
    fn three_way_collision_keeps_dead_trails_and_compact_rows() {
        let mut env = arena(
            [[9, 10], [11, 10], [10, 9], [20, 20], [25, 25]],
            [true; PLAYERS],
        );
        let steps = step(&mut env, [3, 2, 1, 3, 3]);
        for result in &steps[..3] {
            assert_eq!(reward(result), [-1.0]);
            assert_eq!(result.step.dones, [true]);
            assert!(result.step.terminal_observations[0].is_some());
            assert!(result.next_slots.is_empty());
            assert_eq!(result.next_observations.dims(), [0, FEATURES]);
        }
        assert!(env.slots[0].occupied[Slot::cell([10, 10]).unwrap()]);
        assert_eq!(env.slots[0].finishes[0].unwrap().placement, 3);
        let next = step(&mut env, [0, 0, 0, 3, 3]);
        for result in &next[..3] {
            assert_eq!(result.step.rewards.dims(), [0, 1]);
        }
        env.slots[0].heads[3] = [31, 20];
        let final_step = step(&mut env, [0, 0, 0, 3, 3]);
        assert_eq!(reward(&final_step[3]), [-1.0]);
        assert_eq!(reward(&final_step[4]), [1.0]);
        assert!(final_step[3].step.dones[0] && final_step[4].step.dones[0]);
        assert!(final_step.iter().all(|s| s.next_slots == [0]));
        assert_eq!(env.completed[0][0].length, 1);
        assert_eq!(env.completed[0][3].placement, 2);
        assert_eq!(env.completed[0][4].placement, 1);
        assert_eq!(env.completed[0][4].length, 3);
    }

    #[test]
    fn swaps_walls_and_trails_kill_only_the_colliding_players() {
        let mut env = arena(
            [[4, 4], [5, 4], [0, 10], [20, 20], [25, 25]],
            [true; PLAYERS],
        );
        let results = step(&mut env, [3, 2, 2, 3, 3]);
        assert!(results[..3].iter().all(|s| reward(s) == [-1.0]));
        assert!(
            results[3..]
                .iter()
                .all(|s| reward(s) == [0.0] && !s.step.dones[0])
        );
        env.slots[0].heads[3] = [3, 4];
        let results = step(&mut env, [0, 0, 0, 3, 3]);
        assert_eq!(reward(&results[3]), [-1.0]);
        assert_eq!(reward(&results[4]), [1.0]);
    }

    #[test]
    fn simultaneous_last_deaths_are_a_tie_without_a_winner() {
        let mut env = arena(
            [[4, 4], [6, 4], [10, 10], [20, 20], [25, 25]],
            [true, true, false, false, false],
        );
        let results = step(&mut env, [3, 2, 0, 0, 0]);
        for result in &results[..2] {
            assert_eq!(reward(result), [-1.0]);
            assert!(result.step.dones[0]);
        }
        assert!(env.completed[0].iter().all(|f| !f.won));
        assert_eq!(env.completed[0][0].placement, env.completed[0][1].placement);
        assert!(results.iter().all(|s| s.next_slots == [0]));
    }

    #[test]
    fn full_board_planes_are_absolute_and_only_show_living_heads() {
        let env = arena(
            [[0, 0], [1, 0], [10, 10], [20, 20], [31, 31]],
            [true, false, true, true, true],
        );
        for agent in [0, 2, 3, 4] {
            let obs = env.slots[0].observation(agent);
            let own = Slot::cell(env.slots[0].heads[agent]).unwrap();
            assert_eq!(obs.len(), FEATURES);
            assert_eq!(obs[..CELLS].iter().sum::<f32>(), 5.0);
            assert_eq!(obs[CELLS + own], 1.0);
            assert_eq!(obs[CELLS..2 * CELLS].iter().sum::<f32>(), 1.0);
            assert_eq!(obs[2 * CELLS..].iter().sum::<f32>(), 3.0);
            assert_eq!(obs[2 * CELLS + 1], 0.0);
            for other in [0, 2, 3, 4] {
                let cell = Slot::cell(env.slots[0].heads[other]).unwrap();
                assert_eq!(obs[2 * CELLS + cell], f32::from(other != agent));
            }
        }
    }

    #[test]
    fn spawn_assignments_rotate_across_slots_and_rounds() {
        let mut env = Tron::new(PLAYERS, Device::flex()).unwrap();
        env.reset().unwrap();
        let initial = env.slots[0].heads;
        for agent in 0..PLAYERS {
            assert_eq!(env.slots[agent].heads[0], initial[agent]);
            assert_eq!(
                env.slots[agent].occupied.iter().filter(|&&v| v).count(),
                PLAYERS
            );
        }
        for round in 1..=PLAYERS {
            env.reset_slot(0);
            assert_eq!(env.slots[0].heads[0], initial[round % PLAYERS]);
        }
    }

    #[test]
    fn invalid_actions_are_recoverable_and_do_not_change_the_board() {
        assert!(Tron::new(0, Device::flex()).is_err());
        let mut env = Tron::new(1, Device::flex()).unwrap();
        assert!(matches!(env.step(vec![]), Err(TronError::NotReset)));
        env.reset().unwrap();
        let before = env.slots[0].observation(0);
        for count in [0, 4, 6] {
            assert!(
                env.step(vec![Tensor::zeros([1, 1], &env.device); count])
                    .is_err()
            );
        }
        for shape in [[0, 1], [2, 1], [1, 2]] {
            assert!(
                env.step(vec![Tensor::zeros(shape, &env.device); PLAYERS])
                    .is_err()
            );
        }
        for action in [-1i64, 4] {
            assert!(
                env.step(vec![Tensor::from_data([[action]], &env.device); PLAYERS])
                    .is_err()
            );
        }
        assert_eq!(env.slots[0].observation(0), before);
    }

    #[test]
    fn five_learners_train_through_partial_batches_and_resets() {
        let device = Device::flex().autodiff();
        device.seed(17);
        let mut env = Tron::new(2, device).unwrap();
        env.reset().unwrap();
        env.slots[0].heads[0] = [0, 0];
        for head in [[0, 0], [1, 0], [0, 1]] {
            env.slots[0].occupied[Slot::cell(head).unwrap()] = true;
        }
        let observations = std::array::from_fn(|agent| env.current(agent));
        let stats = train(
            &mut env,
            observations,
            TrainingConfig {
                steps: 24,
                seconds: None,
                report_every: 0,
            },
        )
        .unwrap();
        assert_eq!(stats.world_steps, 24);
        assert!(stats.rounds > 0);
        assert!(stats.rows.iter().all(|&rows| rows >= 16));
        assert!(stats.partial_batches[0] > 0);
        assert!(stats.rows.iter().any(|&rows| rows < 48));
    }
}

// Each example imports only the grapher used by that executable.
#![allow(dead_code)]

use std::{
    io::{self, Write},
    marker::PhantomData,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use candle_core::{D, Device, Tensor};
use modurl::prelude::*;
use modurl_logger::{Aggregation, AggregationConfig, Logger, TensorBoardLogger, TerminalLogger};

const DIAGNOSTIC_SMOOTHING_WINDOW: usize = 10;
const LOSS_SMOOTHING_WINDOW: usize = 100;
const REWARD_SMOOTHING_WINDOW: usize = 25;
const EPISODE_SMOOTHING_WINDOW: usize = 100;
const EPISODE_RETURN_METRIC: &str = "Episode Return";
const EPISODE_LENGTH_METRIC: &str = "Episode Length";

fn standard_aggregation() -> AggregationConfig {
    AggregationConfig::new(Aggregation::mean().with_rolling_window(DIAGNOSTIC_SMOOTHING_WINDOW))
}

fn with_metric_window(
    mut config: AggregationConfig,
    metrics: &[&str],
    rolling_window: usize,
) -> AggregationConfig {
    for metric in metrics {
        config = config.with_override(
            *metric,
            Aggregation::mean().with_rolling_window(rolling_window),
        );
    }
    config
}

fn with_episode_smoothing(config: AggregationConfig) -> AggregationConfig {
    with_metric_window(
        config,
        &[EPISODE_RETURN_METRIC, EPISODE_LENGTH_METRIC],
        EPISODE_SMOOTHING_WINDOW,
    )
}

pub struct DQNGrapher {
    terminal: TerminalLogger,
}

impl DQNGrapher {
    pub fn new() -> Self {
        let aggregation = with_episode_smoothing(with_metric_window(
            standard_aggregation(),
            &["DQN Loss"],
            LOSS_SMOOTHING_WINDOW,
        ));
        Self {
            terminal: TerminalLogger::new(aggregation).with_live_updates(),
        }
    }

    pub fn display(mut self) {
        self.terminal.display();
    }
}

impl<I> DQNLogger<I> for DQNGrapher {
    fn log(&mut self, entry: &QLogEntry) {
        let loss = entry.loss.mean_all().unwrap();
        let epsilon = Tensor::new(entry.epsilon as f32, &Device::Cpu).unwrap();
        let mean_q_value = entry.q_values.mean_all().unwrap();
        self.terminal
            .log(
                entry.collection_timestep,
                &[
                    ("DQN Loss", &loss),
                    ("Exploration Epsilon", &epsilon),
                    ("Mean Selected Q-Value", &mean_q_value),
                ],
            )
            .unwrap();
    }

    fn log_collection(&mut self, entry: &QCollectionLogEntry<I>) {
        // Update logs may have already advanced the monotonic logger to the batch endpoint.
        for episode in &entry.completed_episodes {
            let episode_return = Tensor::new(episode.episode_return, &Device::Cpu).unwrap();
            let episode_length = Tensor::new(episode.episode_length as f32, &Device::Cpu).unwrap();
            self.terminal
                .log(
                    entry.collection_timestep,
                    &[
                        (EPISODE_RETURN_METRIC, &episode_return),
                        (EPISODE_LENGTH_METRIC, &episode_length),
                    ],
                )
                .unwrap();
        }
    }
}

pub struct PPOGrapher {
    terminal: TerminalLogger,
}

impl PPOGrapher {
    pub fn new() -> Self {
        let aggregation = with_episode_smoothing(with_metric_window(
            standard_aggregation(),
            &["Actor Loss", "Critic Loss"],
            LOSS_SMOOTHING_WINDOW,
        ));
        Self {
            terminal: TerminalLogger::new(aggregation).with_live_updates(),
        }
    }

    pub fn display(mut self) {
        self.terminal.display();
    }
}

impl PPOLogger for PPOGrapher {
    fn log(&mut self, info: &PPOLogEntry) {
        let actor_loss = info.actor_loss.mean_all().unwrap();
        let critic_loss = info.critic_loss.mean_all().unwrap();
        let entropy = info.entropy.mean_all().unwrap();
        let kl_divergence = info.kl_divergence.mean_all().unwrap();
        let explained_variance = info.explained_variance.mean_all().unwrap();
        self.terminal
            .log(
                info.timestep,
                &[
                    ("Actor Loss", &actor_loss),
                    ("Critic Loss", &critic_loss),
                    ("Entropy", &entropy),
                    ("KL Divergence", &kl_divergence),
                    ("Explained Variance", &explained_variance),
                ],
            )
            .unwrap();
    }

    fn log_collection(&mut self, info: &PPOCollectionLogEntry) {
        for episode in &info.completed_episodes {
            let episode_return = Tensor::new(episode.episode_return, &Device::Cpu).unwrap();
            let episode_length = Tensor::new(episode.episode_length as f32, &Device::Cpu).unwrap();
            self.terminal
                .log(
                    episode.collection_timestep,
                    &[
                        (EPISODE_RETURN_METRIC, &episode_return),
                        (EPISODE_LENGTH_METRIC, &episode_length),
                    ],
                )
                .unwrap();
        }
    }
}

pub struct AgentEpisodeBoundaries;

#[cfg(feature = "atari-environment")]
pub struct RecordedEpisodeBoundaries;

pub struct OnPolicyGrapher<EpisodeSource = AgentEpisodeBoundaries> {
    timestep: usize,
    total_timesteps: usize,
    terminal: TerminalLogger,
    tensorboard: TensorBoardLogger,
    raw_reward_sum: f32,
    raw_reward_samples: usize,
    running_episode_returns: Vec<f32>,
    episode_source: PhantomData<EpisodeSource>,
}

impl OnPolicyGrapher<AgentEpisodeBoundaries> {
    pub fn ppo_mujoco(total_timesteps: usize, environment_name: &str) -> Self {
        Self::new(total_timesteps, environment_name, "ppo_mujoco")
    }

    pub fn a2c_mujoco(total_timesteps: usize, environment_name: &str) -> Self {
        Self::new(total_timesteps, environment_name, "a2c_mujoco")
    }
}

#[cfg(feature = "atari-environment")]
impl OnPolicyGrapher<RecordedEpisodeBoundaries> {
    pub fn ppo_atari(total_timesteps: usize, environment_name: &str) -> Self {
        Self::new(total_timesteps, environment_name, "ppo_atari")
    }
}

impl<EpisodeSource> OnPolicyGrapher<EpisodeSource> {
    fn new(total_timesteps: usize, environment_name: &str, run_name: &str) -> Self {
        let aggregation = with_episode_smoothing(with_metric_window(
            with_metric_window(
                standard_aggregation(),
                &["Actor Loss", "Critic Loss"],
                LOSS_SMOOTHING_WINDOW,
            ),
            &["Mean Raw Step Reward"],
            REWARD_SMOOTHING_WINDOW,
        ));
        let tensorboard_log_dir = tensorboard_log_dir(run_name, environment_name);
        let tensorboard = TensorBoardLogger::new(&tensorboard_log_dir, aggregation.clone())
            .expect("failed to create the TensorBoard event file");
        println!(
            "TensorBoard log directory: {}",
            tensorboard_log_dir.display()
        );
        println!("View all runs with: tensorboard --logdir runs/{run_name}");

        Self {
            timestep: 0,
            total_timesteps,
            terminal: TerminalLogger::new(aggregation).with_live_updates(),
            tensorboard,
            raw_reward_sum: 0.0,
            raw_reward_samples: 0,
            running_episode_returns: Vec::new(),
            episode_source: PhantomData,
        }
    }

    fn log_update(&mut self, info: &PPOLogEntry) {
        let new_timestep = info.timestep != self.timestep;
        if new_timestep {
            self.timestep = info.timestep;
            if self.raw_reward_samples > 0 {
                let mean_raw_reward = Tensor::new(
                    self.raw_reward_sum / self.raw_reward_samples as f32,
                    &Device::Cpu,
                )
                .unwrap();
                self.log_terminal_and_tensorboard(
                    info.timestep,
                    &[("Mean Raw Step Reward", &mean_raw_reward)],
                );
            }
            self.raw_reward_sum = 0.0;
            self.raw_reward_samples = 0;
        }

        let actor_loss = info.actor_loss.mean_all().unwrap();
        let critic_loss = info.critic_loss.mean_all().unwrap();
        let entropy = info.entropy.mean_all().unwrap();
        let kl_divergence = info.kl_divergence.mean_all().unwrap();
        let explained_variance = info.explained_variance.mean_all().unwrap();
        self.log_terminal_and_tensorboard(
            info.timestep,
            &[
                ("Actor Loss", &actor_loss),
                ("Critic Loss", &critic_loss),
                ("Entropy", &entropy),
                ("KL Divergence", &kl_divergence),
                ("Explained Variance", &explained_variance),
            ],
        );
        if new_timestep {
            self.progress();
        }
    }

    fn record_raw_reward(&mut self, reward: f32) {
        self.raw_reward_sum += reward;
        self.raw_reward_samples += 1;
    }

    fn record_episode(&mut self, timestep: usize, episode_return: f32, episode_length: usize) {
        let episode_return = Tensor::new(episode_return, &Device::Cpu).unwrap();
        let episode_length = Tensor::new(episode_length as f32, &Device::Cpu).unwrap();
        self.log_terminal_and_tensorboard(
            timestep,
            &[
                (EPISODE_RETURN_METRIC, &episode_return),
                (EPISODE_LENGTH_METRIC, &episode_length),
            ],
        );
        self.progress();
    }

    fn progress(&self) {
        let fraction = (self.timestep as f32 / self.total_timesteps as f32).min(1.0);
        let filled = (fraction * 40.0) as usize;
        print!(
            "\rTraining [{:<40}] {:>6.2}% ({}/{})",
            "=".repeat(filled),
            fraction * 100.0,
            self.timestep,
            self.total_timesteps
        );
        io::stdout().flush().unwrap();
    }

    pub fn display(mut self) {
        self.timestep = self.total_timesteps;
        self.tensorboard
            .finish()
            .expect("failed to finish the TensorBoard event file");
        self.terminal.display();
        self.progress();
        println!();
    }

    /// Logs named scalar metric tensors, each shaped `[]`.
    fn log_terminal_and_tensorboard(&mut self, timestep: usize, metrics: &[(&str, &Tensor)]) {
        self.terminal.log(timestep, metrics).unwrap();
        self.tensorboard.log(timestep, metrics).unwrap();
    }
}

fn tensorboard_log_dir(run_name: &str, environment_name: &str) -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time must be after the Unix epoch")
        .as_secs();
    PathBuf::from("runs").join(run_name).join(format!(
        "{environment_name}-{timestamp}-{}",
        std::process::id()
    ))
}

impl<I> PPOLogger<RawRewardInfo<I>> for OnPolicyGrapher<AgentEpisodeBoundaries> {
    fn log(&mut self, info: &PPOLogEntry) {
        self.log_update(info);
    }

    fn log_collection(&mut self, info: &PPOCollectionLogEntry<RawRewardInfo<I>>) {
        if self.running_episode_returns.len() != info.infos.len() {
            self.running_episode_returns = vec![0.0; info.infos.len()];
        }
        for (environment_index, env_info) in info.infos.iter().enumerate() {
            let raw_reward = env_info
                .raw_reward
                .expect("RecordRawRewardGym attaches every raw step reward");
            self.record_raw_reward(raw_reward);
            self.running_episode_returns[environment_index] += raw_reward;
        }
        for episode in &info.completed_episodes {
            let episode_return = self.running_episode_returns[episode.environment_index];
            self.running_episode_returns[episode.environment_index] = 0.0;
            self.record_episode(
                episode.collection_timestep,
                episode_return,
                episode.episode_length,
            );
        }
    }
}

#[cfg(feature = "atari-environment")]
impl<I> PPOLogger<RawRewardInfo<EpisodeStatisticsInfo<I>>>
    for OnPolicyGrapher<RecordedEpisodeBoundaries>
{
    fn log(&mut self, info: &PPOLogEntry) {
        self.log_update(info);
    }

    fn log_collection(
        &mut self,
        info: &PPOCollectionLogEntry<RawRewardInfo<EpisodeStatisticsInfo<I>>>,
    ) {
        for env_info in &info.infos {
            let raw_reward = env_info
                .raw_reward
                .expect("RecordRawRewardGym attaches every raw step reward");
            self.record_raw_reward(raw_reward);

            if let Some(episode) = env_info.inner.completed_episode {
                self.record_episode(
                    info.collection_timestep,
                    episode.episode_return,
                    episode.episode_length,
                );
            }
        }
    }
}

const SAC_UPDATE_LOG_INTERVAL: usize = 1_000;

pub struct SACGrapher {
    terminal: TerminalLogger,
}

impl SACGrapher {
    pub fn new() -> Self {
        let aggregation = with_episode_smoothing(with_metric_window(
            with_metric_window(
                standard_aggregation(),
                &["Critic Loss", "Actor Loss", "Entropy Change Loss"],
                LOSS_SMOOTHING_WINDOW,
            ),
            &["Mean Collection Reward"],
            REWARD_SMOOTHING_WINDOW,
        ));
        Self {
            terminal: TerminalLogger::new(aggregation).with_live_updates(),
        }
    }

    pub fn display(mut self) {
        self.terminal.display();
    }
}

impl<I> SACLogger<I> for SACGrapher {
    fn log_update(&mut self, entry: &SACLogEntry) {
        if !entry
            .collection_timestep
            .is_multiple_of(SAC_UPDATE_LOG_INTERVAL)
        {
            return;
        }
        let critic_loss = Tensor::stack(&entry.critic_losses, 0)
            .unwrap()
            .mean_all()
            .unwrap();
        let actor_loss = entry.actor_loss.mean_all().unwrap();
        let entropy_coefficient = entry.alpha.mean_all().unwrap();
        let policy_entropy = entry
            .policy_log_probabilities
            .mul(&entry.policy_weights)
            .unwrap()
            .sum(D::Minus1)
            .unwrap()
            .neg()
            .unwrap()
            .mean_all()
            .unwrap();
        let expected_policy_q = entry
            .policy_q_values
            .mul(&entry.policy_weights)
            .unwrap()
            .sum(D::Minus1)
            .unwrap()
            .mean_all()
            .unwrap();
        let replay_reward = entry.replay_rewards.mean_all().unwrap();
        let bellman_target = entry.bellman_targets.mean_all().unwrap();
        let alpha_loss = entry
            .alpha_loss
            .as_ref()
            .map(|loss| loss.mean_all().unwrap());
        let entropy_change_loss = entry
            .entropy_change_loss
            .as_ref()
            .map(|loss| loss.mean_all().unwrap());
        let target_entropy = entry
            .target_entropy
            .map(|value| Tensor::new(value, &Device::Cpu).unwrap());

        let mut metrics = vec![
            ("Critic Loss", &critic_loss),
            ("Actor Loss", &actor_loss),
            ("Entropy Coefficient", &entropy_coefficient),
            ("Policy Entropy", &policy_entropy),
            ("Expected Soft Q", &expected_policy_q),
            ("Mean Soft Bellman Target", &bellman_target),
            ("Mean Replay Reward", &replay_reward),
        ];
        if let Some(alpha_loss) = &alpha_loss {
            metrics.push(("Entropy Coefficient Loss", alpha_loss));
        }
        if let Some(entropy_change_loss) = &entropy_change_loss {
            metrics.push(("Entropy Change Loss", entropy_change_loss));
        }
        if let Some(target_entropy) = &target_entropy {
            metrics.push(("Target Entropy", target_entropy));
        }
        self.terminal
            .log(entry.collection_timestep, &metrics)
            .unwrap();
    }

    fn log_collection(&mut self, entry: &SACCollectionLogEntry<I>) {
        if entry
            .collection_timestep
            .is_multiple_of(SAC_UPDATE_LOG_INTERVAL)
        {
            let mean_step_reward = entry.collection_rewards.mean_all().unwrap();
            self.terminal
                .log(
                    entry.collection_timestep,
                    &[("Mean Collection Reward", &mean_step_reward)],
                )
                .unwrap();
        }

        // Update metrics may already have advanced this monotonic logger, so
        // completed episodes are intentionally recorded at the batch time.
        for episode in &entry.completed_episodes {
            let episode_return = Tensor::new(episode.episode_return, &Device::Cpu).unwrap();
            let episode_length = Tensor::new(episode.episode_length as f32, &Device::Cpu).unwrap();
            self.terminal
                .log(
                    entry.collection_timestep,
                    &[
                        (EPISODE_RETURN_METRIC, &episode_return),
                        (EPISODE_LENGTH_METRIC, &episode_length),
                    ],
                )
                .unwrap();
        }
    }
}

const DETERMINISTIC_ACTOR_CRITIC_UPDATE_LOG_INTERVAL: usize = 1_000;

pub struct DeterministicActorCriticGrapher {
    terminal: TerminalLogger,
}

impl DeterministicActorCriticGrapher {
    pub fn new() -> Self {
        let aggregation = with_episode_smoothing(with_metric_window(
            with_metric_window(
                standard_aggregation(),
                &["Critic Loss", "Actor Loss"],
                LOSS_SMOOTHING_WINDOW,
            ),
            &["Mean Collection Reward"],
            REWARD_SMOOTHING_WINDOW,
        ));
        Self {
            terminal: TerminalLogger::new(aggregation).with_live_updates(),
        }
    }

    fn log_update(&mut self, entry: &DeterministicActorCriticLogEntry) {
        if !entry
            .collection_timestep
            .is_multiple_of(DETERMINISTIC_ACTOR_CRITIC_UPDATE_LOG_INTERVAL)
        {
            return;
        }

        let critic_loss = Tensor::stack(&entry.critic_losses, 0)
            .unwrap()
            .mean_all()
            .unwrap();
        let replay_q = Tensor::stack(&entry.critic_q_values, 0)
            .unwrap()
            .mean_all()
            .unwrap();
        let bellman_target = entry.bellman_targets.mean_all().unwrap();
        let replay_reward = entry.replay_rewards.mean_all().unwrap();
        let exploration_noise = Tensor::new(
            entry.exploration_noise_standard_deviation as f32,
            &Device::Cpu,
        )
        .unwrap();
        let actor_loss = entry
            .actor_loss
            .as_ref()
            .map(|loss| loss.mean_all().unwrap());
        let policy_q = entry
            .policy_q_values
            .as_ref()
            .map(|values| values.mean_all().unwrap());

        let mut metrics = vec![
            ("Critic Loss", &critic_loss),
            ("Mean Replay Q", &replay_q),
            ("Mean Bellman Target", &bellman_target),
            ("Mean Replay Reward", &replay_reward),
            ("Exploration Noise", &exploration_noise),
        ];
        if let Some(actor_loss) = &actor_loss {
            metrics.push(("Actor Loss", actor_loss));
        }
        if let Some(policy_q) = &policy_q {
            metrics.push(("Mean Policy Q", policy_q));
        }
        self.terminal
            .log(entry.collection_timestep, &metrics)
            .unwrap();
    }

    fn log_collection_metrics<I>(&mut self, entry: &DeterministicActorCriticCollectionLogEntry<I>) {
        if entry
            .collection_timestep
            .is_multiple_of(DETERMINISTIC_ACTOR_CRITIC_UPDATE_LOG_INTERVAL)
        {
            let mean_step_reward = entry.collection_rewards.mean_all().unwrap();
            self.terminal
                .log(
                    entry.collection_timestep,
                    &[("Mean Collection Reward", &mean_step_reward)],
                )
                .unwrap();
        }

        // See the SAC grapher: completed episodes use the enclosing batch time
        // because optimization metrics may already have advanced the logger.
        for episode in &entry.completed_episodes {
            let episode_return = Tensor::new(episode.episode_return, &Device::Cpu).unwrap();
            let episode_length = Tensor::new(episode.episode_length as f32, &Device::Cpu).unwrap();
            self.terminal
                .log(
                    entry.collection_timestep,
                    &[
                        (EPISODE_RETURN_METRIC, &episode_return),
                        (EPISODE_LENGTH_METRIC, &episode_length),
                    ],
                )
                .unwrap();
        }
    }

    pub fn display(mut self) {
        self.terminal.display();
    }
}

impl<I> DDPGLogger<I> for DeterministicActorCriticGrapher {
    fn log(&mut self, entry: &DeterministicActorCriticLogEntry) {
        self.log_update(entry);
    }

    fn log_collection(&mut self, entry: &DeterministicActorCriticCollectionLogEntry<I>) {
        self.log_collection_metrics(entry);
    }
}

impl<I> TD3Logger<I> for DeterministicActorCriticGrapher {
    fn log(&mut self, entry: &DeterministicActorCriticLogEntry) {
        self.log_update(entry);
    }

    fn log_collection(&mut self, entry: &DeterministicActorCriticCollectionLogEntry<I>) {
        self.log_collection_metrics(entry);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dqn_episode_metrics_do_not_go_backwards_after_a_batch_update() {
        let device = Device::Cpu;
        let mut grapher = DQNGrapher::new();
        DQNLogger::<()>::log(
            &mut grapher,
            &QLogEntry {
                loss: Tensor::new(1.0f32, &device).unwrap(),
                epsilon: 0.1,
                learning_rate: 1e-4,
                q_values: Tensor::new(&[1.0f32], &device).unwrap(),
                replay_rewards: Tensor::new(&[1.0f32], &device).unwrap(),
                update_index: 0,
                collection_timestep: 80_000,
            },
        );
        DQNLogger::<()>::log_collection(
            &mut grapher,
            &QCollectionLogEntry {
                collection_rewards: Tensor::new(&[1.0f32], &device).unwrap(),
                infos: vec![()],
                epsilon: 0.1,
                collection_timestep: 80_000,
                completed_episodes: vec![QEpisodeLogEntry {
                    environment_index: 3,
                    episode_return: 10.0,
                    episode_length: 100,
                    terminated: true,
                    truncated: false,
                    collection_timestep: 79_996,
                }],
            },
        );
    }

    #[test]
    fn shared_smoothing_windows_have_expected_lengths() {
        let aggregation = with_episode_smoothing(with_metric_window(
            with_metric_window(standard_aggregation(), &["loss"], LOSS_SMOOTHING_WINDOW),
            &["reward"],
            REWARD_SMOOTHING_WINDOW,
        ));
        let mut logger = TerminalLogger::new(aggregation);
        for timestep in 0..=100 {
            let value = Tensor::new(timestep as f32, &Device::Cpu).unwrap();
            logger
                .log(
                    timestep,
                    &[
                        ("diagnostic", &value),
                        ("loss", &value),
                        ("reward", &value),
                        (EPISODE_RETURN_METRIC, &value),
                        (EPISODE_LENGTH_METRIC, &value),
                    ],
                )
                .unwrap();
        }
        logger.finish().unwrap();

        assert_eq!(logger.series("diagnostic").unwrap().last().unwrap().1, 95.5);
        assert_eq!(logger.series("loss").unwrap().last().unwrap().1, 50.5);
        assert_eq!(logger.series("reward").unwrap().last().unwrap().1, 88.0);
        assert_eq!(
            logger
                .series(EPISODE_RETURN_METRIC)
                .unwrap()
                .last()
                .unwrap()
                .1,
            50.5
        );
        assert_eq!(
            logger
                .series(EPISODE_LENGTH_METRIC)
                .unwrap()
                .last()
                .unwrap()
                .1,
            50.5
        );
    }

    #[test]
    fn deterministic_grapher_uses_batch_timestep_for_completed_episodes() {
        let mut grapher = DeterministicActorCriticGrapher::new();
        let update_metric = Tensor::new(1.0_f32, &Device::Cpu).unwrap();
        grapher
            .terminal
            .log(6_000, &[("Update", &update_metric)])
            .unwrap();

        grapher.log_collection_metrics(&DeterministicActorCriticCollectionLogEntry {
            collection_rewards: Tensor::zeros(2, candle_core::DType::F32, &Device::Cpu).unwrap(),
            infos: vec![(), ()],
            collection_timestep: 6_000,
            completed_episodes: vec![
                DeterministicActorCriticEpisodeLogEntry {
                    environment_index: 0,
                    episode_return: 1.0,
                    episode_length: 10,
                    terminated: true,
                    truncated: false,
                    collection_timestep: 5_999,
                },
                DeterministicActorCriticEpisodeLogEntry {
                    environment_index: 1,
                    episode_return: -1.0,
                    episode_length: 10,
                    terminated: true,
                    truncated: false,
                    collection_timestep: 6_000,
                },
            ],
            replay_len: 6_000,
        });
        grapher.terminal.finish().unwrap();
        let returns = grapher.terminal.series(EPISODE_RETURN_METRIC).unwrap();
        let lengths = grapher.terminal.series(EPISODE_LENGTH_METRIC).unwrap();
        assert_eq!(returns.last().unwrap().0, 6_000);
        assert_eq!(lengths.last().unwrap().0, 6_000);
        assert_eq!(returns.last().unwrap().1, 0.0);
        assert_eq!(lengths.last().unwrap().1, 10.0);
    }
}

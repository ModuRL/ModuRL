// Each example imports one grapher from this shared module.
#![allow(dead_code)]

use std::{
    io::{self, Write},
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use candle_core::{D, Device, Tensor};
use modurl::prelude::*;
use modurl_logger::{Aggregation, AggregationConfig, Logger, TensorBoardLogger, TerminalLogger};

pub struct DQNGrapher {
    terminal: TerminalLogger,
}

impl DQNGrapher {
    pub fn new() -> Self {
        Self {
            terminal: TerminalLogger::new(
                AggregationConfig::new(Aggregation::mean())
                    .with_override("DQN Loss", Aggregation::mean().with_rolling_window(100))
                    .with_override(
                        "Mean Selected Q-Value",
                        Aggregation::mean().with_rolling_window(100),
                    )
                    .with_override(
                        "Episode Return",
                        Aggregation::mean().with_rolling_window(25),
                    )
                    .with_override(
                        "Episode Length",
                        Aggregation::mean().with_rolling_window(25),
                    ),
            )
            .with_live_updates(),
        }
    }

    pub fn display(mut self) {
        self.terminal.display();
    }
}

impl DQNLogger for DQNGrapher {
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

    fn log_collection(&mut self, entry: &QCollectionLogEntry) {
        for episode in &entry.completed_episodes {
            let episode_return = Tensor::new(episode.episode_return, &Device::Cpu).unwrap();
            let episode_length = Tensor::new(episode.episode_length as f32, &Device::Cpu).unwrap();
            self.terminal
                .log(
                    episode.collection_timestep,
                    &[
                        ("Episode Return", &episode_return),
                        ("Episode Length", &episode_length),
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
        Self {
            terminal: TerminalLogger::new(AggregationConfig::new(
                Aggregation::mean().with_rolling_window(5),
            ))
            .with_live_updates(),
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
                        ("Episode Returns", &episode_return),
                        ("Episode Lengths", &episode_length),
                    ],
                )
                .unwrap();
        }
    }
}

pub struct PPOMujocoGrapher {
    timestep: usize,
    total_timesteps: usize,
    terminal: TerminalLogger,
    tensorboard: TensorBoardLogger,
    raw_reward_sum: f32,
    raw_reward_samples: usize,
    running_episode_returns: Vec<f32>,
}

impl PPOMujocoGrapher {
    pub fn new(total_timesteps: usize, environment_name: &str) -> Self {
        let aggregation = AggregationConfig::new(Aggregation::mean().with_rolling_window(10));
        let tensorboard_log_dir = tensorboard_log_dir(environment_name);
        let tensorboard = TensorBoardLogger::new(&tensorboard_log_dir, aggregation.clone())
            .expect("failed to create the TensorBoard event file");
        println!(
            "TensorBoard log directory: {}",
            tensorboard_log_dir.display()
        );
        println!("View all runs with: tensorboard --logdir runs/ppo_mujoco");

        Self {
            timestep: 0,
            total_timesteps,
            terminal: TerminalLogger::new(aggregation).with_live_updates(),
            tensorboard,
            raw_reward_sum: 0.0,
            raw_reward_samples: 0,
            running_episode_returns: Vec::new(),
        }
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
    fn log_metrics(&mut self, timestep: usize, metrics: &[(&str, &Tensor)]) {
        self.terminal.log(timestep, metrics).unwrap();
        self.tensorboard.log(timestep, metrics).unwrap();
    }
}

fn tensorboard_log_dir(environment_name: &str) -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time must be after the Unix epoch")
        .as_secs();
    PathBuf::from("runs").join("ppo_mujoco").join(format!(
        "{environment_name}-{timestamp}-{}",
        std::process::id()
    ))
}

impl PPOLogger<RawRewardInfo<()>> for PPOMujocoGrapher {
    fn log(&mut self, info: &PPOLogEntry) {
        let new_timestep = info.timestep != self.timestep;
        if new_timestep {
            self.timestep = info.timestep;
            if self.raw_reward_samples > 0 {
                let mean_raw_reward = Tensor::new(
                    self.raw_reward_sum / self.raw_reward_samples as f32,
                    &Device::Cpu,
                )
                .unwrap();
                self.log_metrics(info.timestep, &[("Mean Raw Step Reward", &mean_raw_reward)]);
            }
            self.raw_reward_sum = 0.0;
            self.raw_reward_samples = 0;
        }

        let actor_loss = info.actor_loss.mean_all().unwrap();
        let critic_loss = info.critic_loss.mean_all().unwrap();
        let entropy = info.entropy.mean_all().unwrap();
        let kl_divergence = info.kl_divergence.mean_all().unwrap();
        let explained_variance = info.explained_variance.mean_all().unwrap();
        self.log_metrics(
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

    fn log_collection(&mut self, info: &PPOCollectionLogEntry<RawRewardInfo<()>>) {
        if self.running_episode_returns.len() != info.infos.len() {
            self.running_episode_returns = vec![0.0; info.infos.len()];
        }
        for (episode_return, env_info) in self.running_episode_returns.iter_mut().zip(&info.infos) {
            let raw_reward = env_info
                .raw_reward
                .expect("step info must have a raw reward");
            self.raw_reward_sum += raw_reward;
            self.raw_reward_samples += 1;
            *episode_return += raw_reward;
        }
        for episode in &info.completed_episodes {
            let episode_return = Tensor::new(
                self.running_episode_returns[episode.environment_index],
                &Device::Cpu,
            )
            .unwrap();
            self.log_metrics(
                episode.collection_timestep,
                &[("Episodic Return", &episode_return)],
            );
            self.running_episode_returns[episode.environment_index] = 0.0;
            self.progress();
        }
    }
}

const SAC_UPDATE_LOG_INTERVAL: usize = 1_000;

pub struct SACGrapher {
    terminal: TerminalLogger,
}

impl SACGrapher {
    pub fn new() -> Self {
        let aggregation = AggregationConfig::new(Aggregation::mean())
            .with_override("Critic Loss", Aggregation::mean().with_rolling_window(100))
            .with_override("Actor Loss", Aggregation::mean().with_rolling_window(100))
            .with_override(
                "Entropy Change Loss",
                Aggregation::mean().with_rolling_window(100),
            )
            .with_override(
                "Expected Soft Q",
                Aggregation::mean().with_rolling_window(10),
            )
            .with_override(
                "Mean Soft Bellman Target",
                Aggregation::mean().with_rolling_window(10),
            )
            .with_override(
                "Episode Return",
                Aggregation::mean().with_rolling_window(25),
            )
            .with_override(
                "Episode Length",
                Aggregation::mean().with_rolling_window(25),
            );
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

        for episode in &entry.completed_episodes {
            let episode_return = Tensor::new(episode.episode_return, &Device::Cpu).unwrap();
            let episode_length = Tensor::new(episode.episode_length as f32, &Device::Cpu).unwrap();
            self.terminal
                .log(
                    entry.collection_timestep,
                    &[
                        ("Episode Return", &episode_return),
                        ("Episode Length", &episode_length),
                    ],
                )
                .unwrap();
        }
    }
}

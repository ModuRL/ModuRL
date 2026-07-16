use std::io::{self, Write};
use std::sync::atomic::{AtomicUsize, Ordering};

use candle_core::{Module, Tensor};
use candle_nn::{AdamW, Init, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use modurl::prelude::*;
use modurl_mojoco::HalfCheetahV5;
use textplots::{Chart, Plot};

const TOTAL_TIMESTEPS: usize = 1_000_000;
const LOG_STD_PRINT_EVERY: usize = 100_000;

/// CleanRL's observation-independent Gaussian log standard deviation plus a
/// state-dependent mean network.
struct GaussianParameterModule {
    mean: MLP,
    log_std: Tensor,
    forward_calls: AtomicUsize,
}

impl Module for GaussianParameterModule {
    fn forward(&self, observations: &Tensor) -> candle_core::Result<Tensor> {
        if self.forward_calls.fetch_add(1, Ordering::Relaxed) % LOG_STD_PRINT_EVERY == 0 {
            let log_std = self
                .log_std
                .flatten_all()?
                .to_device(&candle_core::Device::Cpu)?
                .to_vec1::<f32>()?;
            println!("\nlog_std: {log_std:?}");
        }
        let mean = self.mean.forward(observations)?;
        let log_std = self.log_std.broadcast_as(mean.shape())?;
        Tensor::cat(&[mean, log_std], 1)
    }
}

struct Grapher {
    timestep: usize,
    samples: usize,
    metrics: [Vec<f32>; 5],
    raw_step_rewards: Vec<f32>,
    raw_reward_sum: f32,
    raw_reward_samples: usize,
    episode_returns: Vec<f32>,
    running_episode_returns: Vec<f32>,
}

impl Grapher {
    fn new() -> Self {
        Self {
            timestep: 0,
            samples: 0,
            metrics: std::array::from_fn(|_| Vec::new()),
            raw_step_rewards: Vec::new(),
            raw_reward_sum: 0.0,
            raw_reward_samples: 0,
            episode_returns: Vec::new(),
            running_episode_returns: Vec::new(),
        }
    }

    fn finish_update(&mut self) {
        if self.samples == 0 {
            return;
        }
        for values in &mut self.metrics {
            *values.last_mut().unwrap() /= self.samples as f32;
        }
        self.samples = 0;
    }

    fn progress(&self) {
        let fraction = (self.timestep as f32 / TOTAL_TIMESTEPS as f32).min(1.0);
        let filled = (fraction * 40.0) as usize;
        print!(
            "\rTraining [{:<40}] {:>6.2}% ({}/{})",
            "=".repeat(filled),
            fraction * 100.0,
            self.timestep,
            TOTAL_TIMESTEPS
        );
        io::stdout().flush().unwrap();
    }

    fn plot(data: &[f32], label: &str) {
        let window = 10.min(data.len());
        let smoothed = data
            .windows(window)
            .map(|values| values.iter().sum::<f32>() / window as f32)
            .enumerate()
            .map(|(x, y)| (x as f32, y))
            .collect::<Vec<_>>();
        println!("{label}:");
        Chart::new(180, 30, 0.0, smoothed.len() as f32)
            .lineplot(&textplots::Shape::Lines(&smoothed))
            .display();
    }

    fn display(mut self) {
        self.finish_update();
        self.timestep = TOTAL_TIMESTEPS;
        self.progress();
        println!();
        for (values, label) in self.metrics.iter().zip([
            "Actor Loss",
            "Critic Loss",
            "Entropy",
            "KL Divergence",
            "Explained Variance",
        ]) {
            Self::plot(values, label);
        }
        Self::plot(&self.raw_step_rewards, "Mean Raw Step Reward");
        Self::plot(&self.episode_returns, "Episodic Return");
    }
}

impl PPOLogger<RawRewardInfo<()>> for Grapher {
    fn log(&mut self, info: &PPOLogEntry) {
        if info.timestep != self.timestep {
            self.finish_update();
            self.timestep = info.timestep;
            self.progress();
            for values in &mut self.metrics {
                values.push(0.0);
            }
            if self.raw_reward_samples > 0 {
                self.raw_step_rewards
                    .push(self.raw_reward_sum / self.raw_reward_samples as f32);
            }
            self.raw_reward_sum = 0.0;
            self.raw_reward_samples = 0;
        }

        let values = [
            &info.actor_loss,
            &info.critic_loss,
            &info.entropy,
            &info.kl_divergence,
            &info.explained_variance,
        ];
        for (history, value) in self.metrics.iter_mut().zip(values) {
            *history.last_mut().unwrap() += value.mean_all().unwrap().to_scalar::<f32>().unwrap();
        }
        self.samples += 1;
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
            self.episode_returns
                .push(self.running_episode_returns[episode.environment_index]);
            self.running_episode_returns[episode.environment_index] = 0.0;
        }
    }
}

fn main() {
    let device = candle_core::Device::cuda_if_available(0).unwrap();
    println!("Using device: {device:?}");

    let env = NormalizeRewardGym::new(
        NormalizeObservationGym::new(RecordRawRewardGym::new(TimeLimitGym::new(
            HalfCheetahV5::builder().device(&device).build().unwrap(),
            1_000,
        )))
        .with_clip(10.0),
        0.99,
    )
    .with_clip(10.0);
    let mut env = VectorizedGymWrapper::from(vec![env]);
    let observation_size = env.observation_space().shape()[0];
    let action_space = env.action_space();
    let action_size = action_space.shape()[0];

    let actor_vars = VarMap::new();
    let actor_vb = VarBuilder::from_varmap(&actor_vars, candle_core::DType::F32, &device);
    let actor = GaussianParameterModule {
        mean: MLP::builder()
            .input_size(observation_size)
            .output_size(action_size)
            .vb(actor_vb.pp("mean"))
            .hidden_layer_sizes(vec![64, 64])
            .activation(Box::new(Tensor::tanh))
            .initializer(Box::new(OrthogonalMLPInitializer {
                hidden_gain: 2.0_f64.sqrt(),
                output_gain: 0.01,
            }))
            .build()
            .unwrap(),
        log_std: actor_vb
            .get_with_hints((1, action_size), "log_std", Init::Const(0.0))
            .unwrap(),
        forward_calls: AtomicUsize::new(0),
    };

    let critic_vars = VarMap::new();
    let critic = MLP::builder()
        .input_size(observation_size)
        .output_size(1)
        .vb(VarBuilder::from_varmap(
            &critic_vars,
            candle_core::DType::F32,
            &device,
        ))
        .hidden_layer_sizes(vec![64, 64])
        .activation(Box::new(Tensor::tanh))
        .initializer(Box::new(OrthogonalMLPInitializer {
            hidden_gain: 2.0_f64.sqrt(),
            output_gain: 1.0,
        }))
        .build()
        .unwrap();

    let optimizer_config = ParamsAdamW {
        lr: 3e-4,
        eps: 1e-5,
        weight_decay: 0.0,
        ..Default::default()
    };
    let networks = PPONetworkInfo::Separate(
        SeparatePPONetwork::builder()
            .actor_network(Box::new(
                ProbabilisticPolicyModel::<GuassianDistribution>::new(Box::new(actor)),
            ))
            .critic_network(Box::new(critic))
            .actor_optimizer(AdamW::new(actor_vars.all_vars(), optimizer_config.clone()).unwrap())
            .critic_optimizer(AdamW::new(critic_vars.all_vars(), optimizer_config).unwrap())
            .actor_lr_scheduler(Box::new(LinearSchedule::new(3e-4, 0.0)))
            .critic_lr_scheduler(Box::new(LinearSchedule::new(3e-4, 0.0)))
            .combined_loss(true)
            .build(),
    );

    let mut grapher = Grapher::new();
    let mut agent = PPOAgent::builder()
        .action_space(action_space)
        .network_info(networks)
        .batch_size(2_048)
        .mini_batch_size(64)
        .num_epochs(10)
        .normalize_advantage(true)
        .clip_range(Box::new(ConstantSchedule::new(0.2)))
        .clip_value_loss(true)
        .gamma(0.99)
        .gae_lambda(0.95)
        .ent_coef(0.0)
        .vf_coef(0.5)
        .training_horizon(TOTAL_TIMESTEPS)
        .logging_info(&mut grapher)
        .device(device)
        .build();

    agent.learn(&mut env, TOTAL_TIMESTEPS).unwrap();
    drop(agent);
    grapher.display();
}

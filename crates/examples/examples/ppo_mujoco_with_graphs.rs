use std::io::{self, Write};

use candle_core::{Module, Tensor};
use candle_nn::{AdamW, Init, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use modurl::prelude::*;
use textplots::{Chart, Plot};

const TOTAL_TIMESTEPS: usize = 1_000_000;

#[cfg(not(any(feature = "half-cheetah", feature = "hopper", feature = "walker2d")))]
compile_error!("enable exactly one MuJoCo environment feature: half-cheetah, hopper, or walker2d");

#[cfg(any(
    all(feature = "half-cheetah", feature = "hopper"),
    all(feature = "half-cheetah", feature = "walker2d"),
    all(feature = "hopper", feature = "walker2d"),
))]
compile_error!("enable exactly one MuJoCo environment feature: half-cheetah, hopper, or walker2d");

#[cfg(feature = "half-cheetah")]
use modurl_mojoco::HalfCheetahV5 as SelectedEnvironment;
#[cfg(all(not(feature = "half-cheetah"), feature = "hopper"))]
use modurl_mojoco::HopperV5 as SelectedEnvironment;
#[cfg(all(not(feature = "half-cheetah"), not(feature = "hopper")))]
use modurl_mojoco::Walker2dV5 as SelectedEnvironment;

#[cfg(feature = "half-cheetah")]
const ENVIRONMENT_NAME: &str = "HalfCheetah-v5";
#[cfg(all(not(feature = "half-cheetah"), feature = "hopper"))]
const ENVIRONMENT_NAME: &str = "Hopper-v5";
#[cfg(all(not(feature = "half-cheetah"), not(feature = "hopper")))]
const ENVIRONMENT_NAME: &str = "Walker2d-v5";

/// Produces the parameter tensor required by `GaussianDistribution`.
///
/// For observations shaped `[batch, observation_size]`, `mean` produces
/// `[batch, action_size]`. `log_std` has shape `[1, action_size]`, making it
/// trainable but observation-independent. `forward` returns
/// `[mean_0, ..., mean_(A-1), log_std_0, ..., log_std_(A-1)]` in each batch row,
/// with shape `[batch, 2 * action_size]`.
struct GaussianParameterModule {
    mean: MLP,
    log_std: Tensor,
}

impl Module for GaussianParameterModule {
    fn forward(&self, observations: &Tensor) -> candle_core::Result<Tensor> {
        // The mean changes with each observation: [batch, action_size].
        let mean = self.mean.forward(observations)?;
        // Every observation shares one learned log standard deviation row.
        let log_std = self.log_std.broadcast_as(mean.shape())?;
        // GaussianDistribution splits this tensor into equal halves.
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
    println!("Environment: {ENVIRONMENT_NAME}");
    println!("Using device: {device:?}");

    let env = NormalizeRewardGym::new(
        NormalizeObservationGym::new(RecordRawRewardGym::new(TimeLimitGym::new(
            SelectedEnvironment::builder()
                .device(&device)
                .build()
                .unwrap(),
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
        // The mean network returns action_size values. GaussianParameterModule
        // appends another action_size values for log_std.
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
        // Because this tensor comes from actor_vb, actor_vars tracks it and the
        // actor optimizer updates it along with the mean network.
        log_std: actor_vb
            .get_with_hints((1, action_size), "log_std", Init::Const(0.0))
            .unwrap(),
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
            // The policy interprets actor output as [mean, log_std], samples an
            // unsquashed Gaussian action, and evaluates its log probability.
            // BoxSpace separately clips the action sent to the environment.
            .actor_network(Box::new(
                ProbabilisticPolicyModel::<GaussianDistribution>::new(Box::new(actor)),
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

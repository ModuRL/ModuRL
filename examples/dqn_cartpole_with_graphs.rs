use candle_core::{DType, Device};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use modurl::prelude::*;
use modurl_gym::classic_control::cartpole::CartPoleV1;
use textplots::{Chart, Plot};

struct DQNGrapher {
    losses: Vec<f32>,
    epsilons: Vec<f32>,
    mean_q_values: Vec<f32>,
    episode_returns: Vec<f32>,
    episode_lengths: Vec<f32>,
}

impl DQNGrapher {
    fn new() -> Self {
        Self {
            losses: Vec::new(),
            epsilons: Vec::new(),
            mean_q_values: Vec::new(),
            episode_returns: Vec::new(),
            episode_lengths: Vec::new(),
        }
    }

    fn plot(data: &[f32], label: &str) {
        if data.is_empty() {
            println!("No data collected for {label}.");
            return;
        }

        println!("Graph for {label}:");
        Chart::new(180, 30, 0.0, data.len() as f32)
            .lineplot(&textplots::Shape::Lines(
                &(0..data.len())
                    .map(|index| (index as f32, data[index]))
                    .collect::<Vec<_>>(),
            ))
            .display();
    }

    fn moving_average(data: &[f32], window: usize) -> Vec<f32> {
        if data.len() < window {
            return data.to_vec();
        }

        data.windows(window)
            .map(|values| values.iter().sum::<f32>() / window as f32)
            .collect()
    }

    fn display_graphs(&self) {
        Self::plot(&Self::moving_average(&self.losses, 100), "DQN Loss");
        Self::plot(&self.epsilons, "Exploration Epsilon");
        Self::plot(
            &Self::moving_average(&self.mean_q_values, 100),
            "Mean Selected Q-Value",
        );
        Self::plot(
            &Self::moving_average(&self.episode_returns, 25),
            "Episode Return",
        );
        Self::plot(
            &Self::moving_average(&self.episode_lengths, 25),
            "Episode Length",
        );
    }
}

impl DQNLogger for DQNGrapher {
    fn log(&mut self, entry: &QLogEntry) {
        self.losses.push(entry.loss.to_vec0::<f32>().unwrap());
        self.epsilons.push(entry.epsilon as f32);
        self.mean_q_values
            .push(entry.q_values.mean_all().unwrap().to_vec0::<f32>().unwrap());
    }

    fn log_collection(&mut self, entry: &QCollectionLogEntry) {
        for episode in &entry.completed_episodes {
            println!(
                "CartPole episode finished at step {}: length = {}, return = {}",
                episode.collection_timestep, episode.episode_length, episode.episode_return,
            );
            self.episode_returns.push(episode.episode_return);
            self.episode_lengths.push(episode.episode_length as f32);
        }
    }
}

fn main() {
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let device = Device::Cpu;
    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0).unwrap();
    #[cfg(feature = "metal")]
    let device = Device::new_metal(0).unwrap();

    println!("Using device: {device:?}");

    let envs = vec![CartPoleV1::builder().device(&device).build()];
    let mut env = VectorizedGymWrapper::from(envs);
    let observation_space = env.observation_space();

    let online_var_map = VarMap::new();
    let online_q_network = MLP::builder()
        .input_size(observation_space.shape()[0])
        .output_size(2)
        .vb(VarBuilder::from_varmap(
            &online_var_map,
            DType::F32,
            &device,
        ))
        .activation(Box::new(tanh))
        .hidden_layer_sizes(vec![64, 64])
        .build()
        .expect("failed to build the online Q-network");

    let mut target_var_map = VarMap::new();
    let target_q_network = MLP::builder()
        .input_size(observation_space.shape()[0])
        .output_size(2)
        .vb(VarBuilder::from_varmap(
            &target_var_map,
            DType::F32,
            &device,
        ))
        .activation(Box::new(tanh))
        .hidden_layer_sizes(vec![64, 64])
        .build()
        .expect("failed to build the target Q-network");

    let optimizer = AdamW::new(
        online_var_map.all_vars(),
        ParamsAdamW {
            lr: 2.5e-4,
            ..Default::default()
        },
    )
    .expect("failed to build the optimizer");

    let mut grapher = DQNGrapher::new();
    let mut agent = DQNAgent::builder()
        .action_space(Discrete::new(2))
        .observation_space(observation_space)
        .online_q_network(Box::new(online_q_network))
        .target_q_network(Box::new(target_q_network))
        .online_vars(&online_var_map)
        .target_vars(&mut target_var_map)
        .optimizer(optimizer)
        .replay_capacity(10_000)
        .batch_size(128)
        .training_start(10_000)
        .update_frequency(10)
        .target_update_interval(500)
        .training_horizon(500_000)
        .epsilon_schedule(Box::new(|progress: f64| {
            let exploration_progress = (progress / 0.5).min(1.0);
            1.0 + (0.05 - 1.0) * exploration_progress
        }))
        .logger(&mut grapher)
        .device_strategy(QLearningDeviceStrategy::OneDevice(device))
        .build()
        .expect("DQN configuration should be valid");

    agent.learn(&mut env, 500_000).expect("DQN learning failed");
    drop(agent);
    grapher.display_graphs();
}

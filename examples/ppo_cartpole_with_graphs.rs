use candle_core::Device;
use candle_nn::{Optimizer, VarBuilder, VarMap};
use candle_optimisers::adam::{Adam, ParamsAdam};
use modurl::actors::ppo::PPOLogger;
use modurl::gym::{VectorizedGym, VectorizedGymWrapper};
use modurl::tensor_operations::tanh;
use modurl::{
    actors::{Actor, ppo::PPOActor},
    distributions::CategoricalDistribution,
    models::{MLP, probabilistic_model::MLPProbabilisticActor},
};
use modurl_gym::classic_control::cartpole::CartPoleV1;
use textplots::{Chart, Plot};

struct PPOGrapher {
    last_timestep: usize,
    samples_on_this_step: u32,
    actor_losses: Vec<f32>,
    critic_losses: Vec<f32>,
    entropies: Vec<f32>,
    kl_divs: Vec<f32>,
    explained_variances: Vec<f32>,
    rewards: Vec<f32>,
}

impl PPOGrapher {
    fn new() -> Self {
        Self {
            last_timestep: 0,
            samples_on_this_step: 0,
            actor_losses: vec![],
            critic_losses: vec![],
            entropies: vec![],
            kl_divs: vec![],
            explained_variances: vec![],
            rewards: vec![],
        }
    }

    fn add_to_running_total(&mut self, info: &modurl::actors::ppo::PPOLogEntry) {
        self.samples_on_this_step += 1;
        let policy_loss = info
            .actor_loss
            .mean_all()
            .unwrap()
            .to_vec0::<f32>()
            .unwrap();
        let value_loss = info
            .critic_loss
            .mean_all()
            .unwrap()
            .to_vec0::<f32>()
            .unwrap();
        let entropy = info.entropy.mean_all().unwrap().to_vec0::<f32>().unwrap();
        let kl_div = info
            .kl_divergence
            .mean_all()
            .unwrap()
            .to_vec0::<f32>()
            .unwrap();
        let explained_variance = info
            .explained_variance
            .mean_all()
            .unwrap()
            .to_vec0::<f32>()
            .unwrap();
        let reward = info.rewards.mean_all().unwrap().to_vec0::<f32>().unwrap();

        let self_fields = vec![
            &mut self.actor_losses,
            &mut self.critic_losses,
            &mut self.entropies,
            &mut self.kl_divs,
            &mut self.explained_variances,
            &mut self.rewards,
        ];
        let new_values = vec![
            policy_loss,
            value_loss,
            entropy,
            kl_div,
            explained_variance,
            reward,
        ];
        for (field, new_value) in self_fields.into_iter().zip(new_values.into_iter()) {
            let current_total = field.last().cloned().unwrap_or(0.0);
            let updated_total = current_total + new_value;
            if field.len() == 0 {
                field.push(updated_total);
            } else {
                *field.last_mut().unwrap() = updated_total;
            }
        }
    }

    fn add_new_step(&mut self, info: &modurl::actors::ppo::PPOLogEntry) {
        self.divide_last_by_samples();
        self.last_timestep = info.timestep;
        let fields = vec![
            &mut self.actor_losses,
            &mut self.critic_losses,
            &mut self.entropies,
            &mut self.kl_divs,
            &mut self.explained_variances,
            &mut self.rewards,
        ];
        for field in fields.into_iter() {
            field.push(0.0);
        }
        self.add_to_running_total(info);
    }

    fn divide_last_by_samples(&mut self) {
        let fields = vec![
            &mut self.actor_losses,
            &mut self.critic_losses,
            &mut self.entropies,
            &mut self.kl_divs,
            &mut self.explained_variances,
            &mut self.rewards,
        ];
        for field in fields.into_iter() {
            field.last_mut().map(|last_value| {
                *last_value /= self.samples_on_this_step as f32;
            });
        }
        self.samples_on_this_step = 0;
    }

    fn textplot_graph(data: &Vec<f32>, label: &str) {
        println!("Graph for {}:", label);
        Chart::new(180, 30, 0.0, data.len() as f32)
            .lineplot(&textplots::Shape::Lines(
                &(0..data.len())
                    .map(|x| (x as f32, data[x]))
                    .collect::<Vec<(f32, f32)>>(),
            ))
            .display();
    }

    fn display_graphs(&mut self, rolling_window_size: usize) {
        self.divide_last_by_samples();

        let mut variables = vec![
            self.actor_losses.clone(),
            self.critic_losses.clone(),
            self.entropies.clone(),
            self.kl_divs.clone(),
            self.explained_variances.clone(),
            self.rewards.clone(),
        ];
        for var in variables.iter_mut() {
            let len = var.len();
            if len >= rolling_window_size {
                let mut smoothed = vec![];
                for i in 0..=len - rolling_window_size {
                    let window = &var[i..i + rolling_window_size];
                    let window_avg: f32 = window.iter().sum::<f32>() / rolling_window_size as f32;
                    smoothed.push(window_avg);
                }
                *var = smoothed;
            }
        }

        Self::textplot_graph(&variables[0], "Actor Loss");
        Self::textplot_graph(&variables[1], "Critic Loss");
        Self::textplot_graph(&variables[2], "Entropy");
        Self::textplot_graph(&variables[3], "KL Divergence");
        Self::textplot_graph(&variables[4], "Explained Variance");
        let episode_lengths: Vec<f32> = variables[5]
            .iter()
            .map(|&r| if r != 1.0 { -r / (r - 1.0) } else { 1.0 })
            .collect();
        Self::textplot_graph(&episode_lengths, "Episode Lengths");
    }
}

impl PPOLogger for PPOGrapher {
    fn log(&mut self, info: &modurl::actors::ppo::PPOLogEntry) {
        if info.timestep == self.last_timestep {
            self.add_to_running_total(info);
        } else {
            self.add_new_step(info);
        }
    }
}

fn main() {
    ppo_cartpole();
}

fn ppo_cartpole() {
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let device = Device::Cpu;
    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0).unwrap();
    #[cfg(feature = "metal")]
    let device = Device::new_metal(0).unwrap();

    #[cfg(any(feature = "cuda", feature = "metal"))]
    device.set_seed(42).unwrap();
    println!("Using device: {:?}", device);

    let mut envs = vec![];
    for _ in 0..8 {
        let env = CartPoleV1::builder().device(&device).build();
        envs.push(env);
    }
    let mut vec_env: VectorizedGymWrapper<CartPoleV1> = envs.into();

    let observation_space = vec_env.observation_space();
    let action_space = vec_env.action_space();
    let var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &device);

    // Actor network: 2x64, tanh activation
    let actor_network = MLP::builder()
        .input_size(observation_space.shape().iter().product())
        .output_size(action_space.shape().iter().product::<usize>())
        .vb(vb.clone())
        .activation(Box::new(tanh))
        .hidden_layer_sizes(vec![64, 64])
        .name("actor_network".to_string())
        .build()
        .unwrap();
    let mut config = ParamsAdam::default();
    // Optimizers: both with lr=3e-4
    config.lr = 3e-4;
    let actor_optimizer =
        Adam::new(var_map.all_vars(), config.clone()).expect("Failed to create Adam");

    let critic_var_map = VarMap::new();
    let critic_vb = VarBuilder::from_varmap(&critic_var_map, candle_core::DType::F32, &device);

    // Critic network: 2x64, tanh activation
    let critic_network = MLP::builder()
        .input_size(observation_space.shape().iter().product())
        .output_size(1)
        .vb(critic_vb)
        .activation(Box::new(tanh))
        .hidden_layer_sizes(vec![64, 64])
        .name("critic_network".to_string())
        .build()
        .unwrap();

    config.lr = 3e-4;
    let critic_optimizer =
        Adam::new(critic_var_map.all_vars(), config.clone()).expect("Failed to create Adam");

    let mut logger = PPOGrapher::new();

    // PPO config
    // Stable baselines3 config:
    let mut actor = PPOActor::builder()
        .action_space(action_space)
        .actor_network(Box::new(
            MLPProbabilisticActor::<CategoricalDistribution>::new(actor_network),
        ))
        .critic_network(Box::new(critic_network))
        .critic_optimizer(critic_optimizer)
        .actor_optimizer(actor_optimizer)
        .batch_size(2048)
        .mini_batch_size(64)
        .normalize_advantage(true)
        .ent_coef(0.005)
        .gamma(0.99)
        .vf_coef(0.5)
        .clip_range(0.2)
        .clipped(true)
        .gae_lambda(0.95)
        .num_epochs(10)
        .device(device)
        .logging_info(&mut logger)
        .build();

    actor.learn(&mut vec_env, 100_000).unwrap();

    logger.display_graphs(5);
}

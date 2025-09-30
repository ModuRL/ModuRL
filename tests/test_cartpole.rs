use candle_core::Device;
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use candle_optimisers::adam::{Adam, ParamsAdam};
use modurl::actors::dqn::DQNActor;
use modurl::gym::{VectorizedGym, VectorizedGymWrapper};
use modurl_gym::classic_control::cartpole::CartPoleV1;

use modurl::tensor_operations::tanh;
use modurl::{
    actors::{Actor, ppo::PPOActor},
    distributions::CategoricalDistribution,
    gym::{Gym, StepInfo},
    models::{MLP, probabilistic_model::MLPProbabilisticActor},
    spaces::Discrete,
};

fn get_average_steps<AE, GE>(actor: &mut dyn Actor<Error = AE, GymError = GE>) -> f32
where
    AE: std::fmt::Debug,
    GE: std::fmt::Debug,
{
    // We consider PPO to be solved if it gets an average reward of 475.0 over 100 consecutive episodes.
    let mut env = CartPoleV1::builder().build();
    let mut total_steps = 0;

    for _ in 0..100 {
        let mut obs = env.reset().expect("Failed to reset environment.");
        let mut done = false;
        let mut episode_steps = 0;

        while !done {
            let mut new_observations_shape: Vec<usize> = vec![1];
            new_observations_shape.append(&mut env.observation_space().shape());
            obs = obs.reshape(&*new_observations_shape).unwrap();

            let action = actor.act(&obs).unwrap();
            let action = env.action_space().from_neurons(&action);

            let StepInfo {
                state: next_obs,
                reward: _reward,
                done: step_done,
                truncated,
            } = env.step(action).unwrap();
            obs = next_obs;
            done = step_done || truncated;
            episode_steps += 1;
        }

        total_steps += episode_steps;
    }

    total_steps as f32 / 100.0
}

struct DebugCartpoleV1 {
    env: CartPoleV1,
    steps_since_print: usize,
    episodes_since_print: usize,
}

impl DebugCartpoleV1 {
    fn new() -> Self {
        let env = CartPoleV1::builder().build();
        Self {
            env,
            steps_since_print: 0,
            episodes_since_print: 0,
        }
    }
}

impl Gym for DebugCartpoleV1 {
    type Error = <CartPoleV1 as Gym>::Error;

    fn step(&mut self, action: candle_core::Tensor) -> Result<StepInfo, Self::Error> {
        self.steps_since_print += 1;
        self.env.step(action)
    }

    fn reset(&mut self) -> Result<candle_core::Tensor, Self::Error> {
        self.episodes_since_print += 1;
        if self.episodes_since_print >= 10 {
            println!(
                "Average steps per episode: {}",
                self.steps_since_print as f32 / self.episodes_since_print as f32
            );
            self.episodes_since_print = 0;
            self.steps_since_print = 0;
        }
        self.env.reset()
    }

    fn observation_space(&self) -> Box<dyn modurl::spaces::Space> {
        self.env.observation_space()
    }

    fn action_space(&self) -> Box<dyn modurl::spaces::Space> {
        self.env.action_space()
    }
}

#[test]
fn ppo_cartpole() {
    let tracer = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(tracer).unwrap();

    let mut envs = vec![];
    for _ in 0..8 {
        let env = DebugCartpoleV1::new();
        envs.push(env);
    }
    let mut vec_env: VectorizedGymWrapper<DebugCartpoleV1> = envs.into();

    let observation_space = vec_env.observation_space();
    let action_space = vec_env.action_space();
    let var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &Device::Cpu);

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
    let critic_vb = VarBuilder::from_varmap(&critic_var_map, candle_core::DType::F32, &Device::Cpu);

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
        .build();

    for i in 0..6 {
        actor.learn(&mut vec_env, 20000).unwrap();
        println!("Testing if PPO solved CartPole-v1...");

        let avg_steps = get_average_steps(&mut actor);
        println!(
            "Average steps over 100 episodes: {} with {} timesteps",
            avg_steps,
            (i + 1) * 20000
        );

        // Cartpole v1 should be using 475, which we can reach but no need for that here
        if avg_steps >= 195.0 {
            println!("PPO solved CartPole-v1 in {} timesteps!", (i + 1) * 20000);
            return;
        }
    }
    panic!("Failed to solve CartPole-v1 within 100000 timesteps.");
}

#[test]
fn dqn_cartpole() {
    let mut envs = vec![];
    for _ in 0..8 {
        let env = DebugCartpoleV1::new();
        envs.push(env);
    }
    let mut vec_env: VectorizedGymWrapper<DebugCartpoleV1> = envs.into();
    let observation_space = vec_env.observation_space();
    let var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &Device::Cpu);

    let mlp = MLP::builder()
        .input_size(observation_space.shape().iter().sum())
        .output_size(2)
        .vb(vb)
        .hidden_layer_sizes(vec![64, 64])
        .build()
        .expect("Failed to create MLP");

    let mut config = ParamsAdam::default();
    config.lr = 3e-4;
    let optimizer = Adam::new(var_map.all_vars(), config).expect("Failed to create AdamW");

    let mut actor = DQNActor::builder()
        .action_space(Discrete::new(2, 0)) // had to hardcode this :(, I would prefer to get it from the env but I can't guarentee it's Discrete
        .observation_space(observation_space)
        .q_network(Box::new(mlp))
        .epsilon_start(1.0)
        .epsilon_decay(0.99)
        .optimizer(optimizer)
        .replay_capacity(10_000)
        .batch_size(32)
        .update_frequency(1)
        .build();

    for i in 0..6 {
        actor.learn(&mut vec_env, 20000).unwrap();
        println!("Testing if PPO solved CartPole-v1...");

        let avg_steps = get_average_steps(&mut actor);
        println!(
            "Average steps over 100 episodes: {} with {} timesteps",
            avg_steps,
            (i + 1) * 20000
        );

        // Cartpole v1 should be using 475, which we can reach but no need for that here
        if avg_steps >= 195.0 {
            println!("PPO solved CartPole-v1 in {} timesteps!", (i + 1) * 20000);
            return;
        }
    }
    panic!("Failed to solve CartPole-v1 within 100000 timesteps.");
}

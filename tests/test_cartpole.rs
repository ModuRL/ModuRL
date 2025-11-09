use candle_core::Device;
use candle_nn::{Optimizer, VarBuilder, VarMap};
use candle_optimisers::adam::{Adam, ParamsAdam};
use modurl::actors::ddqn::DDQNActor;
use modurl::actors::dqn::DQNActor;
use modurl::actors::ppo::SeparatePPONetwork;
use modurl::gym::{VectorizedGym, VectorizedGymWrapper};
use modurl::tensor_operations::tanh;
use modurl::{
    actors::{Actor, ppo::PPOActor},
    distributions::CategoricalDistribution,
    gym::{Gym, StepInfo},
    models::{MLP, probabilistic_model::ProbabilisticActorModel},
    spaces::Discrete,
};
use modurl_gym::classic_control::cartpole::CartPoleV1;

fn get_average_steps<AE, GE, SE>(
    actor: &mut dyn Actor<Error = AE, GymError = GE, SpaceError = SE>,
    device: &Device,
) -> f32
where
    AE: std::fmt::Debug,
    GE: std::fmt::Debug,
    SE: std::fmt::Debug,
{
    let envs = vec![CartPoleV1::builder().device(device).build()];
    let mut vec_env: VectorizedGymWrapper<CartPoleV1> = envs.into();
    let mut total_steps = 0;

    let mut total_done_count = 0;
    let target_episodes = 100;
    let mut states = vec_env.reset().unwrap();
    while total_done_count < target_episodes {
        let action = actor.act(&states).unwrap();
        let step_info = vec_env.step(action).unwrap();
        states = step_info.states;
        let step_dones = step_info.dones;
        let step_terminateds = step_info.truncateds;
        for i in 0..vec_env.num_envs() {
            if step_dones[i] || step_terminateds[i] {
                total_done_count += 1;
            }
        }
        total_steps += vec_env.num_envs();
    }

    total_steps as f32 / target_episodes as f32
}

struct DebugCartpoleV1 {
    env: CartPoleV1,
    steps_since_print: usize,
    episodes_since_print: usize,
}

impl DebugCartpoleV1 {
    fn new(device: &Device) -> Self {
        let env = CartPoleV1::builder().device(device).build();
        Self {
            env,
            steps_since_print: 0,
            episodes_since_print: 0,
        }
    }
}

impl Gym for DebugCartpoleV1 {
    type Error = <CartPoleV1 as Gym>::Error;
    type SpaceError = <CartPoleV1 as Gym>::SpaceError;

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

    fn observation_space(&self) -> Box<dyn modurl::spaces::Space<Error = Self::SpaceError>> {
        self.env.observation_space()
    }

    fn action_space(&self) -> Box<dyn modurl::spaces::Space<Error = Self::SpaceError>> {
        self.env.action_space()
    }
}

#[test]
fn ppo_cartpole() {
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let device = Device::Cpu;
    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0).unwrap();
    #[cfg(feature = "metal")]
    let device = Device::new_metal(0).unwrap();

    #[cfg(any(feature = "cuda", feature = "metal"))]
    device.set_seed(42).unwrap();

    let mut envs = vec![];
    for _ in 0..8 {
        let env = DebugCartpoleV1::new(&device);
        envs.push(env);
    }
    let mut vec_env: VectorizedGymWrapper<DebugCartpoleV1> = envs.into();

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

    let ppo_network_info = modurl::actors::ppo::PPONetworkInfo::Separate(
        SeparatePPONetwork::builder()
            .actor_network(Box::new(
                ProbabilisticActorModel::<CategoricalDistribution>::new(Box::new(actor_network)),
            ))
            .critic_network(Box::new(critic_network))
            .actor_optimizer(actor_optimizer)
            .critic_optimizer(critic_optimizer)
            .build(),
    );

    // PPO config
    // Stable baselines3 config:
    let mut actor = PPOActor::builder()
        .action_space(action_space)
        .network_info(ppo_network_info)
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
        .device(device.clone())
        .build();

    for i in 0..6 {
        actor.learn(&mut vec_env, 20000).unwrap();
        println!("Testing if PPO solved CartPole-v1...");

        // TODO! make a way to make actor deterministic for testing
        let avg_steps = get_average_steps(&mut actor, &device);
        println!(
            "PPO averaged {} steps over 100 episodes with {} timesteps",
            avg_steps,
            (i + 1) * 20000
        );

        // Cartpole v1 should be using 475, which we can reach but no need for that here
        if avg_steps >= 195.0 {
            println!("PPO solved CartPole-v1 in {} timesteps!", (i + 1) * 20000);
            return;
        }
    }
    panic!("PPO failed to solve CartPole-v1 within 100000 timesteps.");
}

#[test]
fn dqn_cartpole() {
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let device = Device::Cpu;
    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0).unwrap();
    #[cfg(feature = "metal")]
    let device = Device::new_metal(0).unwrap();

    #[cfg(any(feature = "cuda", feature = "metal"))]
    device.set_seed(42).unwrap();

    let mut envs = vec![];
    for _ in 0..8 {
        let env = DebugCartpoleV1::new(&device);
        envs.push(env);
    }

    let mut vec_env: VectorizedGymWrapper<DebugCartpoleV1> = envs.into();
    let observation_space = vec_env.observation_space();
    let var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &device);

    let mlp = MLP::builder()
        .input_size(observation_space.shape().iter().sum())
        .output_size(2)
        .vb(vb)
        .hidden_layer_sizes(vec![64, 64])
        .build()
        .expect("Failed to create MLP");

    let mut config = ParamsAdam::default();
    config.lr = 1e-3;
    let optimizer = Adam::new(var_map.all_vars(), config).expect("Failed to create AdamW");

    let mut actor = DQNActor::builder()
        .action_space(Discrete::new(2)) // had to hardcode this :(, I would prefer to get it from the env but I can't guarentee it's Discrete
        .observation_space(observation_space)
        .q_network(Box::new(mlp))
        .epsilon_start(1.0)
        .epsilon_decay(0.99)
        .optimizer(optimizer)
        .replay_capacity(10_000)
        .batch_size(32)
        .update_frequency(1)
        .device(device.clone())
        .build();

    // we'll give dqn more chances since it's more unstable
    // Hopefully it doesn't actually need this many to pass
    for i in 0..10 {
        actor.learn(&mut vec_env, 20000).unwrap();
        println!("Testing if DQN solved CartPole-v1...");

        // TODO! make a way to make actor deterministic for testing
        let avg_steps = get_average_steps(&mut actor, &device);
        println!(
            "DQN averaged {} steps over 100 episodes with {} timesteps",
            avg_steps,
            (i + 1) * 20000
        );

        // Cartpole v1 should be using 475, which we can reach but no need for that here
        if avg_steps >= 195.0 {
            println!("DQN solved CartPole-v1 in {} timesteps!", (i + 1) * 20000);
            return;
        }
    }
    panic!("DQN failed to solve CartPole-v1 within 100000 timesteps.");
}

#[test]
fn ddqn_cartpole() {
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let device = Device::Cpu;
    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0).unwrap();
    #[cfg(feature = "metal")]
    let device = Device::new_metal(0).unwrap();

    #[cfg(any(feature = "cuda", feature = "metal"))]
    device.set_seed(42).unwrap();

    let mut envs = vec![];
    for _ in 0..8 {
        let env = DebugCartpoleV1::new(&device);
        envs.push(env);
    }

    let mut vec_env: VectorizedGymWrapper<DebugCartpoleV1> = envs.into();
    let observation_space = vec_env.observation_space();

    let online_var_map = VarMap::new();
    let online_vb = VarBuilder::from_varmap(&online_var_map, candle_core::DType::F32, &device);
    let online_mlp = MLP::builder()
        .input_size(observation_space.shape().iter().sum())
        .output_size(2)
        .vb(online_vb)
        .hidden_layer_sizes(vec![64, 64])
        .build()
        .expect("Failed to create MLP");

    let mut target_var_map = VarMap::new();
    let target_vb = VarBuilder::from_varmap(&target_var_map, candle_core::DType::F32, &device);
    let target_mlp = MLP::builder()
        .input_size(observation_space.shape().iter().sum())
        .output_size(2)
        .vb(target_vb)
        .hidden_layer_sizes(vec![64, 64])
        .build()
        .expect("Failed to create MLP");

    let mut config = ParamsAdam::default();
    config.lr = 1e-3;
    // Online is the only one being optimized
    let optimizer = Adam::new(online_var_map.all_vars(), config).expect("Failed to create AdamW");

    let mut actor = DDQNActor::builder()
        .action_space(Discrete::new(2)) // had to hardcode this :(, I would prefer to get it from the env but I can't guarentee it's Discrete
        .observation_space(observation_space)
        .online_q_network(Box::new(online_mlp))
        .target_q_network(Box::new(target_mlp))
        .online_vars(&online_var_map)
        .target_vars(&mut target_var_map)
        .epsilon_start(1.0)
        .epsilon_decay(0.99)
        .optimizer(optimizer)
        .replay_capacity(10_000)
        .batch_size(32)
        .update_frequency(1)
        .device(device.clone())
        .build();

    // we'll give ddqn more chances since it's more unstable
    // Hopefully it doesn't actually need this many to pass
    for i in 0..10 {
        actor.learn(&mut vec_env, 20000).unwrap();
        println!("Testing if DDQN solved CartPole-v1...");

        // TODO! make a way to make actor deterministic for testing
        let avg_steps = get_average_steps(&mut actor, &device);
        println!(
            "DDQN averaged {} steps over 100 episodes with {} timesteps",
            avg_steps,
            (i + 1) * 20000
        );

        // Cartpole v1 should be using 475, which we can reach but no need for that here
        if avg_steps >= 195.0 {
            println!("DDQN solved CartPole-v1 in {} timesteps!", (i + 1) * 20000);
            return;
        }
    }
    panic!("DDQN failed to solve CartPole-v1 within 100000 timesteps.");
}

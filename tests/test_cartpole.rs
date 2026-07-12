use candle_core::Device;
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use modurl::prelude::*;
use modurl_gym::classic_control::cartpole::CartPoleV1;

fn get_average_steps<AE, GE, SE>(
    agent: &mut dyn Agent<Error = AE, GymError = GE, SpaceError = SE>,
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
        let action = agent.act(&states).unwrap();
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
#[ignore = "slow solve test; run manually for review prep"]
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

    // Actor network: 2x64, tanh activation, CleanRL init (sqrt(2) hidden, 0.01 head)
    let actor_network = MLP::builder()
        .input_size(observation_space.shape()[0])
        .output_size(action_space.shape()[0])
        .vb(vb.clone())
        .activation(Box::new(tanh))
        .hidden_layer_sizes(vec![64, 64])
        .initializer(Box::new(OrthogonalMLPInitializer {
            hidden_gain: 2.0f64.sqrt(),
            output_gain: 0.01,
        }))
        .name("actor_network".to_string())
        .build()
        .unwrap();
    // Optimizers: both with lr=3e-4
    let config = ParamsAdamW {
        lr: 3e-4,
        ..Default::default()
    };
    let actor_optimizer =
        AdamW::new(var_map.all_vars(), config.clone()).expect("Failed to create AdamW");

    let critic_var_map = VarMap::new();
    let critic_vb = VarBuilder::from_varmap(&critic_var_map, candle_core::DType::F32, &device);

    // Critic network: 2x64, tanh activation, CleanRL init (sqrt(2) hidden, 1.0 head)
    let critic_network = MLP::builder()
        .input_size(observation_space.shape()[0])
        .output_size(1)
        .vb(critic_vb)
        .activation(Box::new(tanh))
        .hidden_layer_sizes(vec![64, 64])
        .initializer(Box::new(OrthogonalMLPInitializer {
            hidden_gain: 2.0f64.sqrt(),
            output_gain: 1.0,
        }))
        .name("critic_network".to_string())
        .build()
        .unwrap();

    let critic_optimizer =
        AdamW::new(critic_var_map.all_vars(), config.clone()).expect("Failed to create AdamW");

    let ppo_network_info = modurl::agents::ppo::PPONetworkInfo::Separate(
        SeparatePPONetwork::builder()
            .actor_network(Box::new(
                ProbabilisticPolicyModel::<CategoricalDistribution>::new(Box::new(actor_network)),
            ))
            .critic_network(Box::new(critic_network))
            .actor_optimizer(actor_optimizer)
            .critic_optimizer(critic_optimizer)
            .build(),
    );

    // PPO config
    // Stable baselines3 config:
    let mut agent = PPOAgent::builder()
        .action_space(action_space)
        .network_info(ppo_network_info)
        .batch_size(2048)
        .mini_batch_size(64)
        .normalize_advantage(true)
        .ent_coef(0.005)
        .gamma(0.99)
        .vf_coef(0.5)
        .clip_range(Box::new(modurl::parameter_schedule::ConstantSchedule::new(
            0.2,
        )))
        .clipped(true)
        .gae_lambda(0.95)
        .num_epochs(10)
        .device(device.clone())
        .build();

    for i in 0..6 {
        agent.learn(&mut vec_env, 20000).unwrap();
        println!("Testing if PPO solved CartPole-v1...");

        // TODO! make a way to make agent deterministic for testing
        let avg_steps = get_average_steps(&mut agent, &device);
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

/// Same solve requirement as `ppo_cartpole`, but through the Shared network path
/// (shared trunk + actor/critic heads + single optimizer + combined loss) that the
/// Atari harness uses, which `ppo_cartpole`'s Separate path does not exercise.
#[test]
#[ignore = "slow solve test; run manually for review prep"]
fn ppo_cartpole_shared() {
    use modurl::agents::ppo::{FakeOptimizer, PPONetworkInfo, SharedPPONetwork};

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

    // Shared trunk: obs -> 64 latent features, CleanRL init (sqrt(2) throughout)
    let shared_network = MLP::builder()
        .input_size(observation_space.shape()[0])
        .output_size(64)
        .vb(vb.clone())
        .activation(Box::new(tanh))
        .output_activation(Box::new(tanh))
        .hidden_layer_sizes(vec![64])
        .initializer(Box::new(OrthogonalMLPInitializer {
            hidden_gain: 2.0f64.sqrt(),
            output_gain: 2.0f64.sqrt(),
        }))
        .name("shared_trunk".to_string())
        .build()
        .unwrap();

    // Linear heads on the shared latent, CleanRL gains (0.01 policy, 1.0 value)
    let actor_head =
        modurl::init::linear_ortho(64, action_space.shape()[0], 0.01, vb.pp("actor_head")).unwrap();
    let critic_head = modurl::init::linear_ortho(64, 1, 1.0, vb.pp("critic_head")).unwrap();

    let config = ParamsAdamW {
        lr: 3e-4,
        ..Default::default()
    };
    // Single optimizer over trunk + both heads, as in the Atari harness
    let optimizer = AdamW::new(var_map.all_vars(), config).expect("Failed to create AdamW");

    let ppo_network_info: PPONetworkInfo<
        AdamW,
        modurl::models::probabilistic_model::ProbabilisticPolicyModelError<candle_core::Error>,
        FakeOptimizer,
    > = PPONetworkInfo::Shared(
        SharedPPONetwork::builder()
            .actor_head(Box::new(
                ProbabilisticPolicyModel::<CategoricalDistribution>::new(Box::new(actor_head)),
            ))
            .critic_head(Box::new(critic_head))
            .shared_network(Box::new(shared_network))
            .optimizer(optimizer)
            .build(),
    );

    // Experiment overrides so the shared-path failure can be probed without code edits
    let vf_coef = std::env::var("TEST_VF_COEF")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.5);
    let gradient_clip = std::env::var("TEST_GRAD_CLIP")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.5);
    println!(
        "Shared PPO experiment config: vf_coef={} gradient_clip={}",
        vf_coef, gradient_clip
    );

    let mut agent = PPOAgent::builder()
        .action_space(action_space)
        .network_info(ppo_network_info)
        .batch_size(2048)
        .mini_batch_size(64)
        .normalize_advantage(true)
        .ent_coef(0.005)
        .gamma(0.99)
        .vf_coef(vf_coef)
        .clip_range(Box::new(modurl::parameter_schedule::ConstantSchedule::new(
            0.2,
        )))
        .clipped(true)
        .gae_lambda(0.95)
        .num_epochs(10)
        .gradient_clip(gradient_clip)
        .device(device.clone())
        .build();

    for i in 0..6 {
        agent.learn(&mut vec_env, 20000).unwrap();
        println!("Testing if shared-network PPO solved CartPole-v1...");

        let avg_steps = get_average_steps(&mut agent, &device);
        println!(
            "Shared PPO averaged {} steps over 100 episodes with {} timesteps",
            avg_steps,
            (i + 1) * 20000
        );

        if avg_steps >= 195.0 {
            println!(
                "Shared PPO solved CartPole-v1 in {} timesteps!",
                (i + 1) * 20000
            );
            return;
        }
    }
    panic!("Shared-network PPO failed to solve CartPole-v1 within 120000 timesteps.");
}

/// Same as `ppo_cartpole_shared`, but through `MultithreadedVectorizedGymWrapper` -
/// the vector-env path the Atari harness actually uses, which the other CartPole
/// tests (sync wrapper) never exercise under learning load. A data-alignment bug
/// in the threaded path would corrupt advantages while all other tests pass.
#[test]
#[ignore = "slow solve test; run manually for review prep"]
#[cfg(feature = "multithreading")]
fn ppo_cartpole_shared_multithreaded() {
    use candle_core::Tensor;
    use modurl::agents::ppo::{FakeOptimizer, PPONetworkInfo, SharedPPONetwork};
    use modurl::gym::MultithreadedVectorizedGymWrapper;

    let device = Device::Cpu;

    let env_constructors: Vec<_> = (0..8)
        .map(|_| {
            let device = device.clone();
            move || DebugCartpoleV1::new(&device)
        })
        .collect();

    let probe_env = CartPoleV1::builder().device(&device).build();
    let obs_space_shape: usize = probe_env.observation_space().shape()[0];
    let obs_space = modurl::spaces::BoxSpace::new(
        Tensor::full(-1000.0f32, &[obs_space_shape], &device).unwrap(),
        Tensor::full(1000.0f32, &[obs_space_shape], &device).unwrap(),
    );
    let action_space_concrete = Discrete::new(2);

    let mut vec_env = MultithreadedVectorizedGymWrapper::new(
        env_constructors,
        obs_space,
        action_space_concrete.clone(),
    );

    let var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &device);

    let shared_network = MLP::builder()
        .input_size(obs_space_shape)
        .output_size(64)
        .vb(vb.clone())
        .activation(Box::new(tanh))
        .output_activation(Box::new(tanh))
        .hidden_layer_sizes(vec![64])
        .initializer(Box::new(OrthogonalMLPInitializer {
            hidden_gain: 2.0f64.sqrt(),
            output_gain: 2.0f64.sqrt(),
        }))
        .name("shared_trunk_mt".to_string())
        .build()
        .unwrap();
    let actor_head = modurl::init::linear_ortho(64, 2, 0.01, vb.pp("actor_head_mt")).unwrap();
    let critic_head = modurl::init::linear_ortho(64, 1, 1.0, vb.pp("critic_head_mt")).unwrap();

    let config = ParamsAdamW {
        lr: 3e-4,
        ..Default::default()
    };
    let optimizer = AdamW::new(var_map.all_vars(), config).expect("Failed to create AdamW");

    let ppo_network_info: PPONetworkInfo<
        AdamW,
        modurl::models::probabilistic_model::ProbabilisticPolicyModelError<candle_core::Error>,
        FakeOptimizer,
    > = PPONetworkInfo::Shared(
        SharedPPONetwork::builder()
            .actor_head(Box::new(
                ProbabilisticPolicyModel::<CategoricalDistribution>::new(Box::new(actor_head)),
            ))
            .critic_head(Box::new(critic_head))
            .shared_network(Box::new(shared_network))
            .optimizer(optimizer)
            .build(),
    );

    let mut agent = PPOAgent::builder()
        .action_space(Box::new(action_space_concrete))
        .network_info(ppo_network_info)
        .batch_size(2048)
        .mini_batch_size(64)
        .normalize_advantage(true)
        .ent_coef(0.005)
        .gamma(0.99)
        .vf_coef(0.5)
        .clip_range(Box::new(modurl::parameter_schedule::ConstantSchedule::new(
            0.2,
        )))
        .clipped(true)
        .gae_lambda(0.95)
        .num_epochs(10)
        .device(device.clone())
        .build();

    for i in 0..6 {
        agent.learn(&mut vec_env, 20000).unwrap();
        println!("Testing if multithreaded shared PPO solved CartPole-v1...");

        let avg_steps = get_average_steps(&mut agent, &device);
        println!(
            "MT Shared PPO averaged {} steps over 100 episodes with {} timesteps",
            avg_steps,
            (i + 1) * 20000
        );

        if avg_steps >= 195.0 {
            println!(
                "MT Shared PPO solved CartPole-v1 in {} timesteps!",
                (i + 1) * 20000
            );
            return;
        }
    }
    panic!("Multithreaded shared PPO failed to solve CartPole-v1 within 120000 timesteps.");
}

#[test]
#[ignore = "slow solve test; run manually for review prep"]
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
    let online_var_map = VarMap::new();
    let online_vb = VarBuilder::from_varmap(&online_var_map, candle_core::DType::F32, &device);

    let online_q_network = MLP::builder()
        .input_size(observation_space.shape()[0])
        .output_size(2)
        .vb(online_vb)
        .hidden_layer_sizes(vec![64, 64])
        .build()
        .expect("Failed to create MLP");

    let mut target_var_map = VarMap::new();
    let target_vb = VarBuilder::from_varmap(&target_var_map, candle_core::DType::F32, &device);

    let target_q_network = MLP::builder()
        .input_size(observation_space.shape()[0])
        .output_size(2)
        .vb(target_vb)
        .hidden_layer_sizes(vec![64, 64])
        .build()
        .expect("Failed to create MLP");

    let config = ParamsAdamW {
        lr: 1e-3,
        ..Default::default()
    };
    let optimizer = AdamW::new(online_var_map.all_vars(), config).expect("Failed to create AdamW");

    let mut agent = DQNAgent::builder()
        .action_space(Discrete::new(2)) // had to hardcode this :(, I would prefer to get it from the env but I can't guarentee it's Discrete
        .observation_space(observation_space)
        .online_q_network(Box::new(online_q_network))
        .target_q_network(Box::new(target_q_network))
        .online_vars(&online_var_map)
        .target_vars(&mut target_var_map)
        .optimizer(optimizer)
        .replay_capacity(10_000)
        .batch_size(32)
        .update_frequency(1)
        .device_strategy(QLearningDeviceStrategy::OneDevice(device.clone()))
        .build()
        .expect("DQN configuration should be valid");

    // we'll give dqn more chances since it's more unstable
    // Hopefully it doesn't actually need this many to pass
    for i in 0..10 {
        agent.learn(&mut vec_env, 20000).unwrap();
        println!("Testing if DQN solved CartPole-v1...");

        // TODO! make a way to make agent deterministic for testing
        let avg_steps = get_average_steps(&mut agent, &device);
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
#[ignore = "slow solve test; run manually for review prep"]
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
        .input_size(observation_space.shape()[0])
        .output_size(2)
        .vb(online_vb)
        .hidden_layer_sizes(vec![64, 64])
        .build()
        .expect("Failed to create MLP");

    let mut target_var_map = VarMap::new();
    let target_vb = VarBuilder::from_varmap(&target_var_map, candle_core::DType::F32, &device);
    let target_mlp = MLP::builder()
        .input_size(observation_space.shape()[0])
        .output_size(2)
        .vb(target_vb)
        .hidden_layer_sizes(vec![64, 64])
        .build()
        .expect("Failed to create MLP");

    let config = ParamsAdamW {
        lr: 1e-3,
        ..Default::default()
    };
    // Online is the only one being optimized
    let optimizer = AdamW::new(online_var_map.all_vars(), config).expect("Failed to create AdamW");

    let mut agent = DDQNAgent::builder()
        .action_space(Discrete::new(2)) // had to hardcode this :(, I would prefer to get it from the env but I can't guarentee it's Discrete
        .observation_space(observation_space)
        .online_q_network(Box::new(online_mlp))
        .target_q_network(Box::new(target_mlp))
        .online_vars(&online_var_map)
        .target_vars(&mut target_var_map)
        .epsilon_schedule(Box::new(LinearSchedule::new(1.0, 0.1)))
        .optimizer(optimizer)
        .replay_capacity(10_000)
        .batch_size(32)
        .update_frequency(1)
        .device_strategy(QLearningDeviceStrategy::OneDevice(device.clone()))
        .build()
        .expect("DDQN configuration should be valid");

    // we'll give ddqn more chances since it's more unstable
    // Hopefully it doesn't actually need this many to pass
    for i in 0..10 {
        agent.learn(&mut vec_env, 20000).unwrap();
        println!("Testing if DDQN solved CartPole-v1...");

        // TODO! make a way to make agent deterministic for testing
        let avg_steps = get_average_steps(&mut agent, &device);
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

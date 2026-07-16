use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use modurl::prelude::*;
use modurl_gym::classic_control::cartpole::CartPoleV1;

// This is the logger shape shown in docs/src/understand-q-learning-training.md.
#[allow(dead_code)]
struct DocumentationDQNLogger;

impl DQNLogger for DocumentationDQNLogger {
    fn log(&mut self, entry: &QLogEntry) {
        let _ = (entry.update_index, &entry.loss);
    }

    fn log_collection(&mut self, entry: &QCollectionLogEntry) {
        for episode in &entry.completed_episodes {
            let _ = (
                episode.environment_index,
                episode.episode_return,
                episode.episode_length,
                episode.terminated,
                episode.truncated,
                episode.collection_timestep,
            );
        }
    }
}

// DDQN has the same logger entry types, and collection logging remains optional.
#[allow(dead_code)]
struct DocumentationDDQNLogger;

impl DDQNLogger for DocumentationDDQNLogger {
    fn log(&mut self, entry: &QLogEntry) {
        let _ = (&entry.replay_rewards, entry.collection_timestep);
    }
}

// This is the complete program shown in docs/src/getting-started.md. Keep this
// test compile-only: running it would turn a documentation check into training.
#[allow(dead_code)]
fn getting_started_program() {
    let device = Device::Cpu;

    let envs = (0..4)
        .map(|_| CartPoleV1::builder().device(&device).build())
        .collect::<Vec<_>>();
    let mut env = VectorizedGymWrapper::from(envs);

    let observation_space = env.observation_space();
    let action_space = env.action_space();

    let actor_var_map = VarMap::new();
    let actor_vb = VarBuilder::from_varmap(&actor_var_map, DType::F32, &device);
    let actor_network = MLP::builder()
        .input_size(observation_space.shape()[0])
        .output_size(action_space.shape()[0])
        .vb(actor_vb)
        .activation(Box::new(Tensor::tanh))
        .hidden_layer_sizes(vec![64, 64])
        .name("actor".to_string())
        .build()
        .expect("failed to build actor network");

    let critic_var_map = VarMap::new();
    let critic_vb = VarBuilder::from_varmap(&critic_var_map, DType::F32, &device);
    let critic_network = MLP::builder()
        .input_size(observation_space.shape()[0])
        .output_size(1)
        .vb(critic_vb)
        .activation(Box::new(Tensor::tanh))
        .hidden_layer_sizes(vec![64, 64])
        .name("critic".to_string())
        .build()
        .expect("failed to build critic network");

    let mut optimizer_config = ParamsAdamW::default();
    optimizer_config.lr = 3e-4;

    let actor_optimizer = AdamW::new(actor_var_map.all_vars(), optimizer_config.clone())
        .expect("failed to build actor optimizer");
    let critic_optimizer = AdamW::new(critic_var_map.all_vars(), optimizer_config)
        .expect("failed to build critic optimizer");

    let policy = ProbabilisticPolicyModel::<CategoricalDistribution>::new(Box::new(actor_network));

    let network_info = PPONetworkInfo::Separate(
        SeparatePPONetwork::builder()
            .actor_network(Box::new(policy))
            .critic_network(Box::new(critic_network))
            .actor_optimizer(actor_optimizer)
            .critic_optimizer(critic_optimizer)
            .build(),
    );

    let mut agent = PPOAgent::builder()
        .action_space(action_space)
        .network_info(network_info)
        .batch_size(1024)
        .mini_batch_size(64)
        .clip_range(Box::new(ConstantSchedule::new(0.2)))
        .training_horizon(10_000)
        .device(device)
        .build();

    agent.learn(&mut env, 10_000).expect("PPO learning failed");
    println!("Training complete.");
}

// This is the complete DQN program shown in docs/src/dqn.md. Keep this test
// compile-only: running it would turn a documentation check into training.
#[allow(dead_code)]
fn dqn_program() {
    let device = Device::Cpu;
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
        .device_strategy(QLearningDeviceStrategy::OneDevice(device.clone()))
        .build()
        .expect("DQN configuration should be valid");

    agent.learn(&mut env, 500_000).expect("DQN learning failed");
    println!("Training complete.");
}

// This is the DQN program with the DDQN-specific construction shown in
// docs/src/ddqn.md. Keep this test compile-only for the same reason.
#[allow(dead_code)]
fn ddqn_program() {
    let device = Device::Cpu;
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

    let mut agent = DDQNAgent::builder()
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
        .device_strategy(QLearningDeviceStrategy::OneDevice(device.clone()))
        .build()
        .expect("DDQN configuration should be valid");

    agent
        .learn(&mut env, 500_000)
        .expect("DDQN learning failed");
    println!("Training complete.");
}

// This is the PPO builder configuration shown in docs/src/understand-ppo-training.md.
#[allow(dead_code)]
fn understand_ppo_training_configuration() {
    let device = Device::Cpu;

    let envs = (0..4)
        .map(|_| CartPoleV1::builder().device(&device).build())
        .collect::<Vec<_>>();
    let mut env = VectorizedGymWrapper::from(envs);

    let observation_space = env.observation_space();
    let action_space = env.action_space();

    let actor_var_map = VarMap::new();
    let actor_vb = VarBuilder::from_varmap(&actor_var_map, DType::F32, &device);
    let actor_network = MLP::builder()
        .input_size(observation_space.shape()[0])
        .output_size(action_space.shape()[0])
        .vb(actor_vb)
        .activation(Box::new(Tensor::tanh))
        .hidden_layer_sizes(vec![64, 64])
        .name("actor".to_string())
        .build()
        .expect("failed to build actor network");

    let critic_var_map = VarMap::new();
    let critic_vb = VarBuilder::from_varmap(&critic_var_map, DType::F32, &device);
    let critic_network = MLP::builder()
        .input_size(observation_space.shape()[0])
        .output_size(1)
        .vb(critic_vb)
        .activation(Box::new(Tensor::tanh))
        .hidden_layer_sizes(vec![64, 64])
        .name("critic".to_string())
        .build()
        .expect("failed to build critic network");

    let mut optimizer_config = ParamsAdamW::default();
    optimizer_config.lr = 3e-4;
    let actor_optimizer = AdamW::new(actor_var_map.all_vars(), optimizer_config.clone())
        .expect("failed to build actor optimizer");
    let critic_optimizer = AdamW::new(critic_var_map.all_vars(), optimizer_config)
        .expect("failed to build critic optimizer");

    let policy = ProbabilisticPolicyModel::<CategoricalDistribution>::new(Box::new(actor_network));
    let network_info = PPONetworkInfo::Separate(
        SeparatePPONetwork::builder()
            .actor_network(Box::new(policy))
            .critic_network(Box::new(critic_network))
            .actor_optimizer(actor_optimizer)
            .critic_optimizer(critic_optimizer)
            .build(),
    );

    let mut agent = PPOAgent::builder()
        .action_space(action_space)
        .network_info(network_info)
        .batch_size(2048)
        .mini_batch_size(64)
        .ent_coef(0.005)
        .clip_range(Box::new(ConstantSchedule::new(0.2)))
        .num_epochs(10)
        .training_horizon(120_000)
        .device(device)
        .build();

    agent.learn(&mut env, 120_000).expect("PPO learning failed");
}

struct CounterEnv {
    state: i32,
    device: Device,
}

impl CounterEnv {
    fn new(device: Device) -> Self {
        Self { state: 0, device }
    }

    fn observation(&self) -> candle_core::Result<Tensor> {
        Tensor::from_vec(vec![self.state as f32], (1,), &self.device)
    }
}

impl Gym for CounterEnv {
    type Error = candle_core::Error;
    type SpaceError = candle_core::Error;

    fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
        self.state = 0;
        Ok(ResetInfo {
            state: self.observation()?,
            info: (),
        })
    }

    fn step(&mut self, action: Tensor) -> Result<StepInfo, Self::Error> {
        match action.to_vec0::<u32>()? {
            0 => self.state -= 1,
            1 => self.state += 1,
            _ => panic!("action is outside the action space"),
        }

        let done = self.state.abs() >= 4;
        Ok(StepInfo {
            state: self.observation()?,
            reward: 1.0,
            done,
            truncated: false,
            info: (),
        })
    }

    fn observation_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
        Box::new(BoxSpace::new_with_universal_bounds(
            vec![1],
            -4.0,
            4.0,
            &self.device,
        ))
    }

    fn action_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
        Box::new(Discrete::new(2))
    }
}

#[test]
fn custom_gym_can_be_vectorized() {
    let device = Device::Cpu;
    let envs = (0..4)
        .map(|_| CounterEnv::new(device.clone()))
        .collect::<Vec<_>>();
    let mut env = VectorizedGymWrapper::from(envs);

    let _states = env.reset().expect("counter environment resets");
    let actions = Tensor::zeros((4,), DType::U32, &device).expect("valid actions");
    let step = env.step(actions).expect("counter environment steps");
    let _transition_next_states = step
        .transition_next_states()
        .expect("transition states have a consistent shape");
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
fn cuda_device() -> candle_core::Result<Device> {
    Device::new_cuda(0)
}

#[cfg(feature = "metal")]
#[allow(dead_code)]
fn metal_device() -> candle_core::Result<Device> {
    Device::new_metal(0)
}

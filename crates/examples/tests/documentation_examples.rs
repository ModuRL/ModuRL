mod getting_started {
    use burn::{
        nn::Tanh,
        optim::AdamWConfig,
        tensor::{DType, Device},
    };
    use modurl::prelude::*;
    use modurl_gym::classic_control::cartpole::CartPoleV1;

    fn main() {
        let device = Device::flex().autodiff();

        let envs = (0..4)
            .map(|index| CartPoleV1::builder().seed(42 + index).build().unwrap())
            .collect::<Vec<_>>();
        let mut env = DeviceMultiGymWrapper::new(
            VectorizedGymWrapper::new(envs).unwrap(),
            Device::flex(),
            device.clone(),
        );

        let observation_space = env.observation_space();
        let action_space = env.action_space();

        let actor_network = MLP::builder()
            .input_size(observation_space.shape()[0])
            .output_size(action_space.get_possible_values())
            .options((&device, DType::F32))
            .activation(Tanh)
            .hidden_layer_sizes(vec![64, 64])
            .build()
            .expect("failed to build actor network");

        let critic_network = MLP::builder()
            .input_size(observation_space.shape()[0])
            .output_size(1)
            .options((&device, DType::F32))
            .activation(Tanh)
            .hidden_layer_sizes(vec![64, 64])
            .build()
            .expect("failed to build critic network");

        let optimizer_config = AdamWConfig::new();
        let actor_optimizer = optimizer_config.init();
        let critic_optimizer = optimizer_config.init();

        let policy = ProbabilisticPolicyModel::new(actor_network, CategoricalDistribution);

        let network_info = SeparatePPONetwork::builder()
            .actor_network(policy)
            .critic_network(critic_network)
            .actor_optimizer(actor_optimizer)
            .critic_optimizer(critic_optimizer)
            .actor_learning_rate(3e-4)
            .critic_learning_rate(3e-4)
            .build()
            .unwrap();

        let mut agent = PPOAgent::builder()
            .observation_space(observation_space)
            .action_space(action_space)
            .network_info(network_info)
            .batch_size(1024)
            .mini_batch_size(64)
            .clip_range(ConstantSchedule::new(0.2))
            .training_horizon(10_000)
            .build()
            .unwrap();

        agent.learn(&mut env, 10_000).expect("PPO learning failed");
        println!("Training complete.");
    }

    pub(super) fn compile_program() {
        let _ = main as fn();
    }
}

mod dqn {
    use burn::{
        optim::AdamWConfig,
        tensor::{DType, Device},
    };
    use modurl::prelude::*;
    use modurl_gym::classic_control::cartpole::CartPoleV1;

    fn main() {
        let device = Device::flex().autodiff();

        let environment_device = Device::flex();
        let envs = vec![CartPoleV1::builder().seed(42).build().unwrap()];
        let env = VectorizedGymWrapper::new(envs).unwrap();
        let mut env = DeviceMultiGymWrapper::new(env, environment_device, device.clone());
        let observation_space = env.observation_space();
        let online_q_network = MLP::builder()
            .input_size(observation_space.shape()[0])
            .output_size(env.action_space().get_possible_values())
            .options((&device, DType::F32))
            .hidden_layer_sizes(vec![64, 64])
            .build()
            .expect("failed to build the online Q-network");
        let optimizer = AdamWConfig::new().init();

        let mut agent = DQNAgent::builder()
            .dtype(DType::F32)
            .action_space(env.action_space())
            .observation_space(observation_space)
            .online_q_network(online_q_network)
            .optimizer(optimizer)
            .learning_rate(2.5e-4)
            .replay_capacity(10_000)
            .batch_size(128)
            .training_start(10_000)
            .update_frequency(10)
            .target_update_interval(500)
            .training_horizon(500_000)
            .epsilon_schedule(|progress: f64| {
                let exploration_progress = (progress / 0.5).min(1.0);
                1.0 + (0.05 - 1.0) * exploration_progress
            })
            .build()
            .expect("DQN configuration should be valid");

        agent.learn(&mut env, 500_000).expect("DQN learning failed");
        println!("Training complete.");
    }

    pub(super) fn compile_program() {
        let _ = main as fn();
    }
}

mod ddqn {
    use burn::{
        optim::AdamWConfig,
        tensor::{DType, Device},
    };
    use modurl::prelude::*;
    use modurl_gym::classic_control::cartpole::CartPoleV1;

    fn main() {
        let device = Device::flex().autodiff();

        let environment_device = Device::flex();
        let envs = vec![CartPoleV1::builder().seed(42).build().unwrap()];
        let env = VectorizedGymWrapper::new(envs).unwrap();
        let mut env = DeviceMultiGymWrapper::new(env, environment_device, device.clone());
        let observation_space = env.observation_space();
        let online_q_network = MLP::builder()
            .input_size(observation_space.shape()[0])
            .output_size(env.action_space().get_possible_values())
            .options((&device, DType::F32))
            .hidden_layer_sizes(vec![64, 64])
            .build()
            .expect("failed to build the online Q-network");
        let optimizer = AdamWConfig::new().init();

        let mut agent = DDQNAgent::builder()
            .action_space(Discrete::new(2).unwrap())
            .observation_space(observation_space)
            .online_q_network(online_q_network)
            .optimizer(optimizer)
            .learning_rate(2.5e-4)
            .replay_capacity(10_000)
            .batch_size(128)
            .training_start(10_000)
            .update_frequency(10)
            .target_update_interval(500)
            .training_horizon(500_000)
            .epsilon_schedule(|progress: f64| {
                let exploration_progress = (progress / 0.5).min(1.0);
                1.0 + (0.05 - 1.0) * exploration_progress
            })
            .build()
            .expect("DDQN configuration should be valid");

        agent
            .learn(&mut env, 500_000)
            .expect("DDQN learning failed");
    }
    pub(super) fn compile_program() {
        let _ = main as fn();
    }
}
mod counter_env {
    use burn::tensor::{Device, Int, Tensor, TensorReadError};
    use modurl::prelude::*;

    #[derive(Debug, thiserror::Error)]
    pub enum CounterEnvError {
        #[error("action {0} is outside the action space")]
        InvalidAction(i32),
        #[error("reading the action failed: {0}")]
        Read(#[from] TensorReadError),
    }

    pub struct CounterEnv {
        state: i32,
        device: Device,
    }

    impl CounterEnv {
        pub fn new(device: Device) -> Self {
            Self { state: 0, device }
        }

        /// Returns the position as an unbatched observation `[1]` with one feature.
        fn observation(&self) -> Tensor<1> {
            Tensor::from_data([self.state as f32], &self.device)
        }
    }

    impl Gym for CounterEnv {
        type Error = CounterEnvError;
        type ObservationSpace = BoxSpace<2>;
        type ActionSpace = Discrete;

        /// Returns the initial position as unbatched `[1]` on the environment device.
        fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
            self.state = 0;
            Ok(ResetInfo {
                observation: self.observation(),
                info: (),
            })
        }

        /// Consumes one scalar integer action `[1]` and returns an unbatched position `[1]`.
        /// Actions must be readable on the host; observations use the environment device.
        fn step(&mut self, action: Tensor<1, Int>) -> Result<StepInfo, Self::Error> {
            match action.try_into_scalar::<i32>()? {
                0 => self.state -= 1,
                1 => self.state += 1,
                action => return Err(CounterEnvError::InvalidAction(action)),
            }

            let done = self.state.abs() >= 4;
            Ok(StepInfo {
                observation: self.observation(),
                reward: 1.0,
                done,
                truncated: false,
                info: (),
            })
        }

        fn observation_space(&self) -> Self::ObservationSpace {
            BoxSpace::new_with_universal_bounds([1, 1], -4.0, 4.0, &self.device)
        }

        fn action_space(&self) -> Self::ActionSpace {
            Discrete::new(2).unwrap()
        }
    }
}

mod deterministic_logger {
    use modurl::prelude::*;

    pub struct ConsoleLogger;

    fn log_update(entry: &DeterministicActorCriticLogEntry) {
        if !entry.update_index.is_multiple_of(1_000) {
            return;
        }

        let critic_loss = entry.critic_losses[0]
            .clone()
            .cast(burn::tensor::DType::F32)
            .into_scalar::<f32>();
        let actor_loss = entry.actor_loss.as_ref().map(|loss| {
            loss.clone()
                .cast(burn::tensor::DType::F32)
                .into_scalar::<f32>()
        });

        println!(
            "step={} update={} critic_loss={critic_loss:.4} \
         actor_updated={} actor_loss={actor_loss:?}",
            entry.collection_timestep, entry.update_index, entry.actor_updated,
        );
    }

    fn log_collection<I>(entry: &DeterministicActorCriticCollectionLogEntry<I>) {
        for episode in &entry.completed_episodes {
            println!(
                "step={} env={} return={} length={} terminated={} truncated={}",
                episode.collection_timestep,
                episode.environment_index,
                episode.episode_return,
                episode.episode_length,
                episode.terminated,
                episode.truncated,
            );
        }
    }

    impl<I> DDPGLogger<I> for ConsoleLogger {
        fn log(&mut self, entry: &DeterministicActorCriticLogEntry) {
            log_update(entry);
        }

        fn log_collection(&mut self, entry: &DeterministicActorCriticCollectionLogEntry<I>) {
            log_collection(entry);
        }
    }

    impl<I> TD3Logger<I> for ConsoleLogger {
        fn log(&mut self, entry: &DeterministicActorCriticLogEntry) {
            log_update(entry);
        }

        fn log_collection(&mut self, entry: &DeterministicActorCriticCollectionLogEntry<I>) {
            log_collection(entry);
        }
    }
}

mod ppo_configuration {
    use burn::{
        nn::Tanh,
        optim::AdamWConfig,
        tensor::{DType, Device},
    };
    use modurl::prelude::*;
    use modurl_gym::classic_control::cartpole::CartPoleV1;

    fn configure() {
        let device = Device::flex().autodiff();

        let envs = (0..4)
            .map(|index| CartPoleV1::builder().seed(42 + index).build().unwrap())
            .collect::<Vec<_>>();
        let mut env = DeviceMultiGymWrapper::new(
            VectorizedGymWrapper::new(envs).unwrap(),
            Device::flex(),
            device.clone(),
        );

        let observation_space = env.observation_space();
        let action_space = env.action_space();

        let actor_network = MLP::builder()
            .input_size(observation_space.shape()[0])
            .output_size(action_space.get_possible_values())
            .options((&device, DType::F32))
            .activation(Tanh)
            .hidden_layer_sizes(vec![64, 64])
            .build()
            .expect("failed to build actor network");

        let critic_network = MLP::builder()
            .input_size(observation_space.shape()[0])
            .output_size(1)
            .options((&device, DType::F32))
            .activation(Tanh)
            .hidden_layer_sizes(vec![64, 64])
            .build()
            .expect("failed to build critic network");

        let optimizer_config = AdamWConfig::new();
        let actor_optimizer = optimizer_config.init();
        let critic_optimizer = optimizer_config.init();

        let policy = ProbabilisticPolicyModel::new(actor_network, CategoricalDistribution);

        let network_info = SeparatePPONetwork::builder()
            .actor_network(policy)
            .critic_network(critic_network)
            .actor_optimizer(actor_optimizer)
            .critic_optimizer(critic_optimizer)
            .actor_learning_rate(3e-4)
            .critic_learning_rate(3e-4)
            .build()
            .unwrap();

        let mut agent = PPOAgent::builder()
            .observation_space(observation_space)
            .action_space(action_space)
            .network_info(network_info)
            .batch_size(2048)
            .mini_batch_size(64)
            .ent_coef(0.005)
            .clip_range(ConstantSchedule::new(0.2))
            .training_horizon(120_000)
            .num_epochs(10)
            .build()
            .unwrap();

        agent.learn(&mut env, 120_000).expect("PPO learning failed");
        println!("Training complete.");
    }

    pub(super) fn compile_program() {
        let _ = configure as fn();
    }
}
use burn::tensor::{Device, Int, Tensor};
use modurl::prelude::*;

#[test]
fn documentation_programs_compile_without_running_training() {
    getting_started::compile_program();
    dqn::compile_program();
    ddqn::compile_program();
    ppo_configuration::compile_program();
    let _ = deterministic_logger::ConsoleLogger;
}

#[test]
fn custom_gym_can_be_vectorized() {
    let device = Device::flex();
    let envs = (0..4)
        .map(|_| counter_env::CounterEnv::new(device.clone()))
        .collect();
    let mut env = VectorizedGymWrapper::new(envs).unwrap();
    assert_eq!(env.reset().unwrap().dims(), [4, 1]);
    let actions = Tensor::<2, Int>::zeros([4, 1], &device);
    let step = env.step(actions).unwrap();
    assert_eq!(step.transition_next_observations().dims(), [4, 1]);
}

#[test]
fn mirrored_book_snippets_match() {
    let source = include_str!("documentation_examples.rs");
    for (book, block_index) in [
        (
            include_str!("../../../docs/src/getting-started.md"),
            usize::MAX,
        ),
        (include_str!("../../../docs/src/dqn.md"), 0),
        (include_str!("../../../docs/src/ddqn.md"), 0),
        (
            include_str!("../../../docs/src/custom-gym-environment.md"),
            0,
        ),
        (
            include_str!("../../../docs/src/understand-deterministic-actor-critic-training.md"),
            0,
        ),
    ] {
        let snippets: Vec<_> = book
            .split("```rust")
            .skip(1)
            .map(|block| {
                block
                    .split_once('\n')
                    .unwrap()
                    .1
                    .split("```")
                    .next()
                    .unwrap()
            })
            .collect();
        let snippet = if block_index == usize::MAX {
            snippets.last().unwrap()
        } else {
            &snippets[block_index]
        };
        assert!(
            without_whitespace(source).contains(&without_whitespace(snippet)),
            "book snippet differs from its compiled mirror"
        );
    }
}

fn without_whitespace(text: &str) -> String {
    let compact: String = text
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    compact
        .replace(",)", ")")
        .replace(",]", "]")
        .replace(",}", "}")
}

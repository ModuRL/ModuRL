use burn::{
    nn::Tanh,
    optim::SgdConfig,
    tensor::{DType, Device},
};
use modurl::{agents::action_value_critic::ScalarStateActionCritic, prelude::*};
use modurl_gym::classic_control::pendulum::PendulumV1;

#[derive(Default)]
struct RecordingLogger {
    actor_updates: Vec<bool>,
}

impl TD3Logger for RecordingLogger {
    fn log(&mut self, entry: &DeterministicActorCriticLogEntry) {
        self.actor_updates.push(entry.actor_updated);
    }
}

#[test]
fn td3_trains_with_cpu_environment_and_separate_replay_device() {
    let cpu = Device::flex();
    let device = cpu.clone().autodiff();
    let mut env =
        VectorizedGymWrapper::from(TimeLimitGym::new(PendulumV1::builder().build().unwrap(), 2));
    let actor = MLP::builder()
        .input_size(3)
        .output_size(1)
        .hidden_layer_sizes(vec![4])
        .output_activation(Tanh)
        .options((&device, DType::F32))
        .build()
        .unwrap();
    let critic = ActionValueCritic::builder()
        .online_network(ScalarStateActionCritic::new(
            MLP::builder()
                .input_size(4)
                .output_size(1)
                .hidden_layer_sizes(vec![4])
                .options((&device, DType::F32))
                .build()
                .unwrap(),
        ))
        .optimizer(SgdConfig::new().init())
        .learning_rate(0.001)
        .build()
        .unwrap();
    let mut logger = RecordingLogger::default();
    let mut agent = TD3Agent::builder()
        .online_actor(actor)
        .critics(vec![critic])
        .actor_optimizer(SgdConfig::new().init())
        .actor_learning_rate(0.001)
        .action_space(env.action_space())
        .observation_space(env.observation_space())
        .replay_device(cpu)
        .replay_capacity(8)
        .batch_size(2)
        .training_start(0)
        .training_horizon(4)
        .exploration_noise(0.0)
        .logger(&mut logger)
        .build()
        .unwrap();
    agent.learn(&mut env, 4).unwrap();
    let observations = env.reset().unwrap();
    let actions = agent.act_deterministic(&observations).unwrap();
    assert_eq!(actions.dims(), [1, 1]);
    assert_eq!(actions.dtype(), DType::F32);
    assert_eq!(logger.actor_updates, [false, true, false]);
}

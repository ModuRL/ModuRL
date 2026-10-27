use burn::{
    nn::Initializer,
    optim::SgdConfig,
    tensor::{DType, Device, Tensor},
};
use modurl::prelude::*;
use modurl_mujoco::HalfCheetahV5;

fn environments() -> VectorizedGymWrapper<TimeLimitGym<HalfCheetahV5>> {
    VectorizedGymWrapper::new(
        (0..2)
            .map(|index| {
                let mut environment = HalfCheetahV5::builder().build().unwrap();
                environment.seed(index);
                TimeLimitGym::new(environment, 2)
            })
            .collect(),
    )
    .unwrap()
}

fn model(output_size: usize, device: &Device) -> MLP {
    MLP::builder()
        .input_size(17)
        .output_size(output_size)
        .hidden_layer_sizes(vec![4])
        .hidden_initializer(Initializer::Constant { value: 0.1 })
        .output_initializer(Initializer::Zeros)
        .bias_initializer(Initializer::Zeros)
        .options((device, DType::F32))
        .build()
        .unwrap()
}

#[derive(Default)]
struct TrainingLogger {
    updates: usize,
    episodes: usize,
}

impl PPOLogger for TrainingLogger {
    fn log(&mut self, entry: &PPOLogEntry) {
        assert!(entry.actor_loss.clone().into_scalar::<f32>().is_finite());
        assert!(entry.critic_loss.clone().into_scalar::<f32>().is_finite());
        assert_eq!(entry.advantages.dtype(), DType::F32);
        self.updates += 1;
    }

    fn log_collection(&mut self, entry: &PPOCollectionLogEntry) {
        for episode in &entry.completed_episodes {
            assert!(!episode.terminated);
            assert!(episode.truncated);
            assert!(episode.episode_return.is_finite());
        }
        self.episodes += entry.completed_episodes.len();
    }
}

#[test]
fn vectorization_preserves_axes_and_terminal_observations() {
    let device = Device::flex();
    let mut environments = environments();
    assert_eq!(environments.reset().unwrap().dims(), [2, 17]);
    let first = environments.step(Tensor::zeros([2, 6], &device)).unwrap();
    assert_eq!(first.observations.dims(), [2, 17]);
    assert_eq!(first.rewards.dims(), [2, 1]);
    assert_eq!(first.rewards.dtype(), DType::F32);
    assert!(first.terminal_observations.iter().all(Option::is_none));
    let second = environments.step(Tensor::zeros([2, 6], &device)).unwrap();
    assert_eq!(second.dones, [false, false]);
    assert_eq!(second.truncateds, [true, true]);
    for (index, terminal) in second.terminal_observations.iter().enumerate() {
        let terminal = terminal.as_ref().unwrap();
        assert_eq!(terminal.dims(), [17]);
        assert_eq!(terminal.dtype(), DType::F32);
        assert_eq!(terminal.device(), device);
        let reset = second
            .observations
            .clone()
            .slice_dim(0, index..index + 1)
            .squeeze_dim::<1>(0);
        assert_ne!(terminal.clone().into_data(), reset.into_data());
    }
}

#[test]
fn a2c_trains_with_mujoco_and_time_limit_truncations() {
    let device = Device::flex().autodiff();
    let mut environments =
        DeviceMultiGymWrapper::new(environments(), Device::flex(), device.clone());
    let networks = SeparateA2CNetwork::builder()
        .actor_network(ProbabilisticPolicyModel::new(
            model(12, &device),
            GaussianDistribution::default(),
        ))
        .critic_network(model(1, &device))
        .actor_optimizer(SgdConfig::new().init())
        .critic_optimizer(SgdConfig::new().init())
        .actor_learning_rate(1e-3)
        .critic_learning_rate(1e-3)
        .build()
        .unwrap();
    let mut logger = TrainingLogger::default();
    let mut agent = A2CAgent::builder()
        .observation_space(environments.observation_space())
        .action_space(environments.action_space())
        .network_info(networks.into())
        .batch_size(4)
        .training_horizon(8)
        .logging_info(&mut logger)
        .build()
        .unwrap();
    agent.learn(&mut environments, 8).unwrap();
    assert_eq!(logger.updates, 2);
    assert_eq!(logger.episodes, 4);
}

#[test]
fn ppo_trains_with_mujoco_and_time_limit_truncations() {
    let device = Device::flex().autodiff();
    let mut environments =
        DeviceMultiGymWrapper::new(environments(), Device::flex(), device.clone());
    let networks = SeparatePPONetwork::builder()
        .actor_network(ProbabilisticPolicyModel::new(
            model(12, &device),
            GaussianDistribution::default(),
        ))
        .critic_network(model(1, &device))
        .actor_optimizer(SgdConfig::new().init())
        .critic_optimizer(SgdConfig::new().init())
        .actor_learning_rate(1e-3)
        .critic_learning_rate(1e-3)
        .build()
        .unwrap();
    let mut logger = TrainingLogger::default();
    let mut agent = PPOAgent::builder()
        .observation_space(environments.observation_space())
        .action_space(environments.action_space())
        .network_info(networks)
        .batch_size(4)
        .mini_batch_size(4)
        .num_epochs(1)
        .training_horizon(8)
        .logging_info(&mut logger)
        .build()
        .unwrap();
    agent.learn(&mut environments, 8).unwrap();
    assert_eq!(logger.updates, 2);
    assert_eq!(logger.episodes, 4);
}

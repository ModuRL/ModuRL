use std::sync::Mutex;
use burn::{
    module::Module,
    nn::Tanh,
    optim::SgdConfig,
    tensor::{DType, Device, Float, Tensor},
};
use modurl::prelude::*;
use modurl_gym::classic_control::cartpole::CartPoleV1;

#[derive(Module, Debug)]
struct Float64Model {
    model: MLP,
}

impl Forward<2, 2> for Float64Model {
    type InputKind = Float;
    type OutputKind = Float;
    type Error = ModelError;

    /// Explicitly casts native F32 observations [batch_size, features] to F64 and returns F64 predictions [batch_size, outputs].
    fn forward(&self, observations: Tensor<2>) -> Result<Tensor<2>, Self::Error> {
        self.model.forward(observations.cast(DType::F64))
    }
}

static RNG_LOCK: Mutex<()> = Mutex::new(());

fn model(input: usize, output: usize, device: &Device) -> Float64Model {
    Float64Model {
        model: MLP::builder()
            .input_size(input)
            .output_size(output)
            .hidden_layer_sizes(vec![8])
            .activation(Tanh)
            .options((device, DType::F64))
            .build()
            .unwrap(),
    }
}

fn cartpole() -> VectorizedGymWrapper<CartPoleV1> {
    VectorizedGymWrapper::from(CartPoleV1::builder().build().unwrap())
}

#[test]
fn dqn_updates_f64_networks_from_native_f32_observations() {
    let _guard = RNG_LOCK.lock().unwrap();
    let device = Device::flex().autodiff();
    let mut env = cartpole();
    let mut agent = DQNAgent::builder()
        .dtype(DType::F64)
        .online_q_network(model(4, 2, &device))
        .optimizer(SgdConfig::new().init())
        .learning_rate(0.01)
        .action_space(env.action_space())
        .observation_space(env.observation_space())
        .replay_storage_config(ReplayStorageConfig::new(ReplayDeviceStrategy::Hybrid {
            storage_device: Device::flex(),
            optimization_device: device,
        }))
        .replay_capacity(8)
        .batch_size(2)
        .training_start(0)
        .training_horizon(4)
        .build()
        .unwrap();
    agent.learn(&mut env, 4).unwrap();
    assert_eq!(agent.act(&env.reset().unwrap()).unwrap().dims(), [1, 1]);
}

#[test]
fn ddqn_updates_f64_networks_from_native_f32_observations() {
    let _guard = RNG_LOCK.lock().unwrap();
    let device = Device::flex().autodiff();
    let mut env = cartpole();
    let mut agent = DDQNAgent::builder()
        .dtype(DType::F64)
        .online_q_network(model(4, 2, &device))
        .optimizer(SgdConfig::new().init())
        .learning_rate(0.01)
        .action_space(env.action_space())
        .observation_space(env.observation_space())
        .replay_storage_config(ReplayStorageConfig::new(ReplayDeviceStrategy::Hybrid {
            storage_device: Device::flex(),
            optimization_device: device,
        }))
        .replay_capacity(8)
        .batch_size(2)
        .training_start(0)
        .training_horizon(4)
        .build()
        .unwrap();
    agent.learn(&mut env, 4).unwrap();
    assert_eq!(agent.act(&env.reset().unwrap()).unwrap().dims(), [1, 1]);
}

#[test]
fn ppo_updates_f64_networks_from_native_f32_observations() {
    let _guard = RNG_LOCK.lock().unwrap();
    let device = Device::flex().autodiff();
    let mut env = OutputMapMultiGymWrapper::new(
        cartpole(),
        Ok::<_, std::convert::Infallible>,
        |mut step: MultiGymStepInfo| {
            step.rewards = step.rewards.cast(DType::F64);
            Ok::<_, std::convert::Infallible>(step)
        },
    );
    let networks = SeparatePPONetwork::builder()
        .actor_network(ProbabilisticPolicyModel::new(
            model(4, 2, &device),
            CategoricalDistribution,
        ))
        .critic_network(model(4, 1, &device))
        .actor_optimizer(SgdConfig::new().init())
        .critic_optimizer(SgdConfig::new().init())
        .actor_learning_rate(0.01)
        .critic_learning_rate(0.01)
        .build()
        .unwrap();
    let mut agent = PPOAgent::builder()
        .network_info(networks)
        .action_space(env.action_space())
        .observation_space(env.observation_space())
        .batch_size(4)
        .mini_batch_size(2)
        .num_epochs(1)
        .training_horizon(4)
        .device(device)
        .build()
        .unwrap();
    agent.learn(&mut env, 4).unwrap();
    assert_eq!(env.reset().unwrap().dtype(), DType::F32);
}

#[test]
fn a2c_updates_f64_networks_from_native_f32_observations() {
    let _guard = RNG_LOCK.lock().unwrap();
    let device = Device::flex().autodiff();
    let mut env = OutputMapMultiGymWrapper::new(
        cartpole(),
        Ok::<_, std::convert::Infallible>,
        |mut step: MultiGymStepInfo| {
            step.rewards = step.rewards.cast(DType::F64);
            Ok::<_, std::convert::Infallible>(step)
        },
    );
    let networks = SeparateA2CNetwork::builder()
        .actor_network(ProbabilisticPolicyModel::new(
            model(4, 2, &device),
            CategoricalDistribution,
        ))
        .critic_network(model(4, 1, &device))
        .actor_optimizer(SgdConfig::new().init())
        .critic_optimizer(SgdConfig::new().init())
        .actor_learning_rate(0.01)
        .critic_learning_rate(0.01)
        .build()
        .unwrap();
    let mut agent = A2CAgent::builder()
        .network_info(networks.into())
        .action_space(env.action_space())
        .observation_space(env.observation_space())
        .batch_size(4)
        .training_horizon(4)
        .device(device)
        .build()
        .unwrap();
    agent.learn(&mut env, 4).unwrap();
    assert_eq!(env.reset().unwrap().dtype(), DType::F32);
}

#[test]
fn sac_updates_f64_networks_from_native_f32_observations() {
    let _guard = RNG_LOCK.lock().unwrap();
    let device = Device::flex().autodiff();
    let mut env = cartpole();
    let critic = SACCritic::builder()
        .online_network(DiscreteVectorHeadCritic::new(model(4, 2, &device)))
        .optimizer(SgdConfig::new().init())
        .learning_rate(0.01)
        .build()
        .unwrap();
    let mut agent = SACAgent::builder()
        .dtype(DType::F64)
        .policy(ProbabilisticPolicyModel::new(
            model(4, 2, &device),
            CategoricalDistribution,
        ))
        .critics(vec![critic])
        .actor_optimizer(SgdConfig::new().init())
        .actor_learning_rate(0.01)
        .entropy_configuration(SACEntropyConfiguration::fixed(0.1).unwrap())
        .action_space(env.action_space())
        .observation_space(env.observation_space())
        .replay_storage_config(ReplayStorageConfig::new(ReplayDeviceStrategy::Hybrid {
            storage_device: Device::flex(),
            optimization_device: device,
        }))
        .replay_capacity(8)
        .batch_size(2)
        .training_start(0)
        .training_horizon(4)
        .build()
        .unwrap();
    agent.learn(&mut env, 4).unwrap();
    assert_eq!(env.reset().unwrap().dtype(), DType::F32);
}

#[derive(Module, Debug)]
struct Float64ContinuousCritic {
    model: MLP,
}

impl ActionValueCriticNetwork for Float64ContinuousCritic {
    type ObservationKind = Float;
    type ActionKind = Float;
    type Error = ActionValueCriticError<ModelError>;

    /// Casts observations [batch_size, 3] explicitly to F64 and evaluates F64 actions [batch_size, 1], returning Q values [batch_size, 1].
    fn replay_values(
        &self,
        observations: &Tensor<2>,
        actions: &Tensor<2>,
    ) -> Result<Tensor<2>, Self::Error> {
        <modurl::agents::action_value_critic::ScalarStateActionCritic<MLP> as ActionValueCriticNetwork<2, 2, 3>>::replay_values(
            &modurl::agents::action_value_critic::ScalarStateActionCritic::new(self.model.clone()), &observations.clone().cast(DType::F64), actions)
    }

    /// Casts observations [batch_size, 3] explicitly to F64 and evaluates candidates [batch_size, candidates, 1], returning [batch_size, candidates, 1].
    fn policy_values(
        &self,
        observations: &Tensor<2>,
        actions: &Tensor<3>,
    ) -> Result<Tensor<3>, Self::Error> {
        <modurl::agents::action_value_critic::ScalarStateActionCritic<MLP> as ActionValueCriticNetwork<2, 2, 3>>::policy_values(
            &modurl::agents::action_value_critic::ScalarStateActionCritic::new(self.model.clone()), &observations.clone().cast(DType::F64), actions)
    }

    /// Evaluates F64 candidates [batch_size, candidates, 1] after casting observations [batch_size, 3]; preserves action gradients and omits critic gradients.
    fn actor_values(
        &self,
        observations: &Tensor<2>,
        actions: &Tensor<3>,
    ) -> Result<Tensor<3>, Self::Error> {
        <modurl::agents::action_value_critic::ScalarStateActionCritic<MLP> as ActionValueCriticNetwork<2, 2, 3>>::actor_values(
            &modurl::agents::action_value_critic::ScalarStateActionCritic::new(self.model.clone()), &observations.clone().cast(DType::F64), actions)
    }
}

fn continuous_critic(device: &Device) -> ActionValueCritic<Float64ContinuousCritic> {
    ActionValueCritic::builder()
        .online_network(Float64ContinuousCritic {
            model: model(4, 1, device).model,
        })
        .optimizer(SgdConfig::new().init())
        .learning_rate(0.01)
        .build()
        .unwrap()
}

#[test]
fn ddpg_updates_f64_networks_from_native_f32_observations() {
    let _guard = RNG_LOCK.lock().unwrap();
    let device = Device::flex().autodiff();
    let mut env = VectorizedGymWrapper::from(
        modurl_gym::classic_control::pendulum::PendulumV1::builder()
            .build()
            .unwrap(),
    );
    let mut agent = DDPGAgent::builder()
        .dtype(DType::F64)
        .online_actor(model(3, 1, &device))
        .critic(continuous_critic(&device))
        .actor_optimizer(SgdConfig::new().init())
        .actor_learning_rate(0.01)
        .action_space(env.action_space())
        .observation_space(env.observation_space())
        .replay_storage_config(ReplayStorageConfig::new(ReplayDeviceStrategy::Hybrid {
            storage_device: Device::flex(),
            optimization_device: device,
        }))
        .replay_capacity(8)
        .batch_size(2)
        .training_start(0)
        .training_horizon(4)
        .build()
        .unwrap();
    agent.learn(&mut env, 4).unwrap();
    let observations = env.reset().unwrap();
    assert_eq!(observations.dtype(), DType::F32);
    assert_eq!(
        agent.act_deterministic(&observations).unwrap().dtype(),
        DType::F64
    );
}

#[test]
fn td3_updates_f64_networks_from_native_f32_observations() {
    let _guard = RNG_LOCK.lock().unwrap();
    let device = Device::flex().autodiff();
    let mut env = VectorizedGymWrapper::from(
        modurl_gym::classic_control::pendulum::PendulumV1::builder()
            .build()
            .unwrap(),
    );
    let mut agent = TD3Agent::builder()
        .dtype(DType::F64)
        .online_actor(model(3, 1, &device))
        .critics(vec![continuous_critic(&device), continuous_critic(&device)])
        .actor_optimizer(SgdConfig::new().init())
        .actor_learning_rate(0.01)
        .action_space(env.action_space())
        .observation_space(env.observation_space())
        .replay_storage_config(ReplayStorageConfig::new(ReplayDeviceStrategy::Hybrid {
            storage_device: Device::flex(),
            optimization_device: device,
        }))
        .replay_capacity(8)
        .batch_size(2)
        .training_start(0)
        .training_horizon(4)
        .build()
        .unwrap();
    agent.learn(&mut env, 4).unwrap();
    let observations = env.reset().unwrap();
    assert_eq!(observations.dtype(), DType::F32);
    assert_eq!(
        agent.act_deterministic(&observations).unwrap().dtype(),
        DType::F64
    );
}

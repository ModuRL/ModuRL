use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use modurl::prelude::*;
use modurl_gym::classic_control::cartpole::CartPoleV1;

const DTYPE: DType = DType::F64;

fn mlp(
    variables: &VarMap,
    input_size: usize,
    output_size: usize,
    name: &str,
) -> candle_core::Result<MLP> {
    MLP::builder()
        .input_size(input_size)
        .output_size(output_size)
        .vb(VarBuilder::from_varmap(variables, DTYPE, &Device::Cpu))
        .hidden_layer_sizes(vec![8])
        .activation(Tensor::tanh)
        .initializer(OrthogonalMLPInitializer {
            hidden_gain: 2.0_f64.sqrt(),
            output_gain: 1.0,
        })
        .name(name.to_owned())
        .build()
}

fn cartpole() -> VectorizedGymWrapper<CartPoleV1> {
    VectorizedGymWrapper::from(vec![
        CartPoleV1::builder().device(&Device::Cpu).build().unwrap(),
    ])
}

#[test]
fn dqn_updates_f64_networks_from_native_f32_observations() {
    let mut env = cartpole();
    let observation_space = env.observation_space();
    let online_vars = VarMap::new();
    let mut target_vars = VarMap::new();
    let online = mlp(&online_vars, 4, 2, "q").unwrap();
    let target = mlp(&target_vars, 4, 2, "q").unwrap();
    let optimizer = AdamW::new(online_vars.all_vars(), ParamsAdamW::default()).unwrap();

    let mut agent = DQNAgent::builder()
        .action_space(Discrete::new(2))
        .observation_space(observation_space)
        .online_q_network(online)
        .target_q_network(target)
        .online_vars(&online_vars)
        .target_vars(&mut target_vars)
        .optimizer(optimizer)
        .epsilon_schedule(ConstantSchedule::new(0.0))
        .replay_capacity(8)
        .batch_size(2)
        .training_start(0)
        .update_frequency(1)
        .target_update_interval(2)
        .training_horizon(4)
        .device_strategy(ReplayDeviceStrategy::OneDevice(Device::Cpu))
        .dtype(DTYPE)
        .build()
        .unwrap();

    agent.learn(&mut env, 4).unwrap();
    assert!(
        online_vars
            .all_vars()
            .iter()
            .all(|variable| variable.dtype() == DTYPE)
    );
}

#[test]
fn ddqn_updates_f64_networks_from_native_f32_observations() {
    let mut env = cartpole();
    let observation_space = env.observation_space();
    let online_vars = VarMap::new();
    let mut target_vars = VarMap::new();
    let online = mlp(&online_vars, 4, 2, "q").unwrap();
    let target = mlp(&target_vars, 4, 2, "q").unwrap();
    let optimizer = AdamW::new(online_vars.all_vars(), ParamsAdamW::default()).unwrap();

    let mut agent = DDQNAgent::builder()
        .action_space(Discrete::new(2))
        .observation_space(observation_space)
        .online_q_network(online)
        .target_q_network(target)
        .online_vars(&online_vars)
        .target_vars(&mut target_vars)
        .optimizer(optimizer)
        .epsilon_schedule(ConstantSchedule::new(0.0))
        .replay_capacity(8)
        .batch_size(2)
        .training_start(0)
        .update_frequency(1)
        .target_update_interval(2)
        .training_horizon(4)
        .device_strategy(ReplayDeviceStrategy::OneDevice(Device::Cpu))
        .dtype(DTYPE)
        .build()
        .unwrap();

    agent.learn(&mut env, 4).unwrap();
}

#[test]
fn ppo_updates_f64_networks_from_native_f32_observations() {
    let mut env = cartpole();
    let action_space = env.action_space();
    let actor_vars = VarMap::new();
    let critic_vars = VarMap::new();
    let actor = mlp(&actor_vars, 4, 2, "actor").unwrap();
    let critic = mlp(&critic_vars, 4, 1, "critic").unwrap();
    let actor_optimizer = AdamW::new(actor_vars.all_vars(), ParamsAdamW::default()).unwrap();
    let critic_optimizer = AdamW::new(critic_vars.all_vars(), ParamsAdamW::default()).unwrap();
    let networks = PPONetworkInfo::Separate(
        SeparatePPONetwork::builder()
            .actor_network(ProbabilisticPolicyModel::<CategoricalDistribution>::new(
                actor,
            ))
            .critic_network(critic)
            .actor_optimizer(actor_optimizer)
            .critic_optimizer(critic_optimizer)
            .build(),
    );

    let mut agent = PPOAgent::builder()
        .action_space(action_space)
        .network_info(networks)
        .batch_size(4)
        .mini_batch_size(2)
        .num_epochs(1)
        .training_horizon(4)
        .device(Device::Cpu)
        .dtype(DTYPE)
        .build()
        .unwrap();

    agent.learn(&mut env, 4).unwrap();
    assert!(
        actor_vars
            .all_vars()
            .iter()
            .chain(critic_vars.all_vars().iter())
            .all(|variable| variable.dtype() == DTYPE)
    );
}

#[test]
fn a2c_updates_f64_networks_from_native_f32_observations() {
    let mut env = cartpole();
    let action_space = env.action_space();
    let actor_vars = VarMap::new();
    let critic_vars = VarMap::new();
    let actor = mlp(&actor_vars, 4, 2, "actor").unwrap();
    let critic = mlp(&critic_vars, 4, 1, "critic").unwrap();
    let networks = A2CNetworkInfo::separate(
        SeparateA2CNetwork::builder()
            .actor_network(ProbabilisticPolicyModel::<CategoricalDistribution>::new(
                actor,
            ))
            .critic_network(critic)
            .actor_optimizer(AdamW::new(actor_vars.all_vars(), ParamsAdamW::default()).unwrap())
            .critic_optimizer(AdamW::new(critic_vars.all_vars(), ParamsAdamW::default()).unwrap())
            .build(),
    );

    let mut agent = A2CAgent::builder()
        .action_space(action_space)
        .network_info(networks)
        .batch_size(4)
        .training_horizon(4)
        .device(Device::Cpu)
        .dtype(DTYPE)
        .build()
        .unwrap();

    agent.learn(&mut env, 4).unwrap();
}

fn discrete_critic<'a>(
    online_vars: &'a VarMap,
    target_vars: &'a mut VarMap,
) -> SACCritic<'a, AdamW> {
    let online = mlp(online_vars, 4, 2, "critic").unwrap();
    let target = mlp(target_vars, 4, 2, "critic").unwrap();
    SACCritic::builder()
        .online_network(DiscreteVectorHeadCritic::new(online))
        .target_network(DiscreteVectorHeadCritic::new(target))
        .online_vars(online_vars)
        .target_vars(target_vars)
        .optimizer(AdamW::new(online_vars.all_vars(), ParamsAdamW::default()).unwrap())
        .build()
        .unwrap()
}

#[test]
fn sac_updates_f64_networks_from_native_f32_observations() {
    let mut env = cartpole();
    let observation_space = env.observation_space();
    let action_space = env.action_space();
    let actor_vars = VarMap::new();
    let actor = mlp(&actor_vars, 4, 2, "actor").unwrap();
    let actor_optimizer = AdamW::new(actor_vars.all_vars(), ParamsAdamW::default()).unwrap();
    let policy = ProbabilisticPolicyModel::<CategoricalDistribution>::new(actor);
    let online_vars = VarMap::new();
    let mut target_vars = VarMap::new();
    let critic = discrete_critic(&online_vars, &mut target_vars);

    let mut agent = SACAgent::builder()
        .policy(policy)
        .actor_optimizer(actor_optimizer)
        .critics(vec![critic])
        .entropy_configuration(SACEntropyConfiguration::<AdamW>::fixed(0.2))
        .action_space(action_space)
        .observation_space(observation_space)
        .replay_capacity(8)
        .batch_size(2)
        .training_start(0)
        .training_horizon(4)
        .device_strategy(ReplayDeviceStrategy::OneDevice(Device::Cpu))
        .dtype(DTYPE)
        .stabilization_configuration(SACStabilizationConfiguration::stable_discrete())
        .build()
        .unwrap();

    agent.learn(&mut env, 4).unwrap();
    assert!(
        actor_vars
            .all_vars()
            .iter()
            .chain(online_vars.all_vars().iter())
            .all(|variable| variable.dtype() == DTYPE)
    );
}

struct NativeF32ContinuousEnv {
    state: f32,
}

fn continuous_critic<'a>(
    online_vars: &'a VarMap,
    target_vars: &'a mut VarMap,
) -> SACCritic<'a, AdamW> {
    let online = mlp(online_vars, 2, 1, "critic").unwrap();
    let target = mlp(target_vars, 2, 1, "critic").unwrap();
    SACCritic::builder()
        .online_network(ScalarStateActionCritic::new(online))
        .target_network(ScalarStateActionCritic::new(target))
        .online_vars(online_vars)
        .target_vars(target_vars)
        .optimizer(AdamW::new(online_vars.all_vars(), ParamsAdamW::default()).unwrap())
        .build()
        .unwrap()
}

impl Gym for NativeF32ContinuousEnv {
    type Error = candle_core::Error;
    type SpaceError = candle_core::Error;

    fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
        self.state = 0.0;
        Ok(ResetInfo {
            state: Tensor::new(&[self.state], &Device::Cpu)?,
            info: (),
        })
    }

    /// Steps with one continuous action shaped `[1]`.
    fn step(&mut self, action: Tensor) -> Result<StepInfo, Self::Error> {
        let action = action.to_dtype(DType::F32)?.to_vec1::<f32>()?[0];
        self.state += action;
        Ok(StepInfo {
            state: Tensor::new(&[self.state], &Device::Cpu)?,
            reward: 1.0,
            done: false,
            truncated: false,
            info: (),
        })
    }

    fn observation_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
        Box::new(BoxSpace::new_unbounded(vec![1], &Device::Cpu))
    }

    fn action_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
        Box::new(BoxSpace::new_with_universal_bounds(
            vec![1],
            -1.0,
            1.0,
            &Device::Cpu,
        ))
    }
}

#[test]
fn ddpg_updates_f64_networks_from_native_f32_observations() {
    let mut env = VectorizedGymWrapper::from(vec![NativeF32ContinuousEnv { state: 0.0 }]);
    let online_actor_vars = VarMap::new();
    let mut target_actor_vars = VarMap::new();
    let online_actor = mlp(&online_actor_vars, 1, 1, "actor").unwrap();
    let target_actor = mlp(&target_actor_vars, 1, 1, "actor").unwrap();
    let actor_optimizer = AdamW::new(online_actor_vars.all_vars(), ParamsAdamW::default()).unwrap();
    let online_critic_vars = VarMap::new();
    let mut target_critic_vars = VarMap::new();
    let critic = continuous_critic(&online_critic_vars, &mut target_critic_vars);

    let mut agent = DDPGAgent::builder()
        .online_actor(online_actor)
        .target_actor(target_actor)
        .online_actor_vars(&online_actor_vars)
        .target_actor_vars(&mut target_actor_vars)
        .actor_optimizer(actor_optimizer)
        .critic(critic)
        .action_space(BoxSpace::new_with_universal_bounds(
            vec![1],
            -1.0,
            1.0,
            &Device::Cpu,
        ))
        .observation_space(env.observation_space())
        .replay_capacity(8)
        .batch_size(2)
        .training_start(0)
        .training_horizon(4)
        .device_strategy(ReplayDeviceStrategy::OneDevice(Device::Cpu))
        .dtype(DTYPE)
        .build()
        .unwrap();

    agent.learn(&mut env, 4).unwrap();
    assert!(
        online_actor_vars
            .all_vars()
            .iter()
            .chain(online_critic_vars.all_vars().iter())
            .all(|variable| variable.dtype() == DTYPE)
    );
}

#[test]
fn td3_updates_f64_networks_from_native_f32_observations() {
    let mut env = VectorizedGymWrapper::from(vec![NativeF32ContinuousEnv { state: 0.0 }]);
    let online_actor_vars = VarMap::new();
    let mut target_actor_vars = VarMap::new();
    let online_actor = mlp(&online_actor_vars, 1, 1, "actor").unwrap();
    let target_actor = mlp(&target_actor_vars, 1, 1, "actor").unwrap();
    let actor_optimizer = AdamW::new(online_actor_vars.all_vars(), ParamsAdamW::default()).unwrap();
    let online_critic_vars_1 = VarMap::new();
    let mut target_critic_vars_1 = VarMap::new();
    let online_critic_vars_2 = VarMap::new();
    let mut target_critic_vars_2 = VarMap::new();
    let critic_1 = continuous_critic(&online_critic_vars_1, &mut target_critic_vars_1);
    let critic_2 = continuous_critic(&online_critic_vars_2, &mut target_critic_vars_2);

    let mut agent = TD3Agent::builder()
        .online_actor(online_actor)
        .target_actor(target_actor)
        .online_actor_vars(&online_actor_vars)
        .target_actor_vars(&mut target_actor_vars)
        .actor_optimizer(actor_optimizer)
        .critics(vec![critic_1, critic_2])
        .action_space(BoxSpace::new_with_universal_bounds(
            vec![1],
            -1.0,
            1.0,
            &Device::Cpu,
        ))
        .observation_space(env.observation_space())
        .replay_capacity(8)
        .batch_size(2)
        .training_start(0)
        .training_horizon(4)
        .device_strategy(ReplayDeviceStrategy::OneDevice(Device::Cpu))
        .dtype(DTYPE)
        .build()
        .unwrap();

    agent.learn(&mut env, 4).unwrap();
}

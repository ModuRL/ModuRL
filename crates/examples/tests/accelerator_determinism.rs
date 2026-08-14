#![cfg(any(feature = "cuda", feature = "metal"))]

use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use modurl::{prelude::*, sampling::shuffle_with_device_rng};
use modurl_gym::classic_control::cartpole::CartPoleV1;

#[cfg(feature = "metal")]
fn accelerator_device() -> Device {
    Device::new_metal(0).unwrap()
}

#[cfg(all(feature = "cuda", not(feature = "metal")))]
fn accelerator_device() -> Device {
    Device::new_cuda(0).unwrap()
}

#[test]
fn mlp_initialization_is_device_seeded() {
    let device = accelerator_device();
    let input = Tensor::rand(0.0_f32, 1.0, &[1, 4], &device).unwrap();
    let mut last_output: Option<Tensor> = None;

    for iteration in 0..10 {
        device.set_seed(42).unwrap();
        let variables = VarMap::new();
        let network = MLP::builder()
            .input_size(4)
            .output_size(2)
            .vb(VarBuilder::from_varmap(&variables, DType::F32, &device))
            .hidden_layer_sizes(vec![8, 8])
            .build()
            .unwrap();

        let current_output = network.forward(&input).unwrap();
        if let Some(last_output) = &last_output {
            let max_diff = last_output
                .sub(&current_output)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();

            assert!(
                max_diff < 1e-6,
                "outputs differ at iteration {iteration} by {max_diff}"
            );
        }
        last_output = Some(current_output);
    }
}

#[test]
fn shuffle_is_device_seeded() {
    let device = accelerator_device();
    let mut first: Vec<u32> = (0..100).collect();
    let mut second = first.clone();

    device.set_seed(42).unwrap();
    shuffle_with_device_rng(&mut first, &device).unwrap();
    device.set_seed(42).unwrap();
    shuffle_with_device_rng(&mut second, &device).unwrap();

    assert_eq!(first, second);
}

#[test]
fn cartpole_tensors_are_device_deterministic() {
    let device = accelerator_device();
    let mut last_state: Option<Tensor> = None;

    for iteration in 0..10 {
        device.set_seed(42).unwrap();
        let mut env = CartPoleV1::builder().device(&device).build().unwrap();
        let action_space = env.action_space();
        let mut state = env.reset().unwrap().state;

        for _ in 0..10 {
            let action = action_space.sample(&device).unwrap();
            let step = env.step(action).unwrap();
            state = step.state;
            if step.done {
                break;
            }
        }

        if let Some(last_state) = &last_state {
            let state_values = state.to_vec1::<f32>().unwrap();
            let last_values = last_state.to_vec1::<f32>().unwrap();
            for (actual, expected) in state_values.iter().zip(last_values.iter()) {
                assert_eq!(
                    actual, expected,
                    "states differ at iteration {iteration}: {actual} vs {expected}"
                );
            }
        }
        last_state = Some(state);
    }
}

struct DummyEnv {
    step_count: usize,
    device: Device,
}

impl DummyEnv {
    fn new(device: Device) -> Self {
        Self {
            step_count: 0,
            device,
        }
    }
}

impl Gym for DummyEnv {
    type Error = candle_core::Error;
    type SpaceError = candle_core::Error;

    /// Steps with one scalar discrete action shaped `[]`.
    fn step(&mut self, _action: Tensor) -> Result<StepInfo, Self::Error> {
        self.step_count += 1;
        Ok(StepInfo {
            state: Tensor::rand(0.0_f32, 1.0, &[4], &self.device)?,
            reward: self.step_count as f32,
            done: self.step_count >= 5,
            truncated: false,
            info: (),
        })
    }

    fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
        self.step_count = 0;
        Ok(ResetInfo {
            state: Tensor::rand(0.0_f32, 1.0, &[4], &self.device)?,
            info: (),
        })
    }

    fn observation_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
        Box::new(BoxSpace::new(
            Tensor::full(0.0_f32, &[4], &self.device).unwrap(),
            Tensor::full(1.0_f32, &[4], &self.device).unwrap(),
        ))
    }

    fn action_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
        Box::new(Discrete::new(2))
    }
}

#[test]
fn ppo_training_is_device_deterministic() {
    const SAMPLE_COUNT: usize = 25;

    let device = accelerator_device();
    let mut last_actions: Option<Vec<Vec<u32>>> = None;

    for iteration in 0..5 {
        device.set_seed(42).unwrap();

        let envs = (0..8)
            .map(|_| DummyEnv::new(device.clone()))
            .collect::<Vec<_>>();
        let mut env: VectorizedGymWrapper<DummyEnv> = envs.into();
        let observation_space = env.observation_space();
        let action_space = env.action_space();

        let actor_variables = VarMap::new();
        let actor_network = MLP::builder()
            .input_size(observation_space.shape()[0])
            .output_size(action_space.shape()[0])
            .vb(VarBuilder::from_varmap(
                &actor_variables,
                DType::F32,
                &device,
            ))
            .activation(Tensor::tanh)
            .hidden_layer_sizes(vec![8, 8])
            .name("actor_network".to_string())
            .build()
            .unwrap();

        let optimizer_parameters = ParamsAdamW {
            lr: 3e-4,
            ..Default::default()
        };
        let actor_optimizer =
            AdamW::new(actor_variables.all_vars(), optimizer_parameters.clone()).unwrap();

        let critic_variables = VarMap::new();
        let critic_network = MLP::builder()
            .input_size(observation_space.shape()[0])
            .output_size(1)
            .vb(VarBuilder::from_varmap(
                &critic_variables,
                DType::F32,
                &device,
            ))
            .activation(Tensor::tanh)
            .hidden_layer_sizes(vec![8, 8])
            .name("critic_network".to_string())
            .build()
            .unwrap();
        let critic_optimizer =
            AdamW::new(critic_variables.all_vars(), optimizer_parameters).unwrap();

        let networks = PPONetworkInfo::Separate(
            SeparatePPONetwork::builder()
                .actor_optimizer(actor_optimizer)
                .critic_optimizer(critic_optimizer)
                .actor_network(ProbabilisticPolicyModel::<CategoricalDistribution>::new(
                    actor_network,
                ))
                .critic_network(critic_network)
                .build(),
        );

        let mut agent = PPOAgent::builder()
            .action_space(action_space)
            .network_info(networks)
            .batch_size(2048)
            .mini_batch_size(64)
            .ent_coef(0.01)
            .vf_coef(0.5)
            .clip_range(ConstantSchedule::new(0.2))
            .gae_lambda(0.95)
            .num_epochs(10)
            .training_horizon(10_000)
            .device(device.clone())
            .build()
            .unwrap();

        agent.learn(&mut env, 10_000).expect("PPO learning failed");

        let mut actions = Vec::new();
        let mut states = env.reset().unwrap();
        for _ in 0..SAMPLE_COUNT {
            let action = agent.act(&states).unwrap();
            actions.push(action.to_vec1::<u32>().unwrap());
            states = env.step(action).unwrap().states;
        }

        if let Some(last_actions) = &last_actions {
            assert_eq!(
                last_actions, &actions,
                "PPO actions differ at iteration {iteration}"
            );
        }
        last_actions = Some(actions);
    }
}

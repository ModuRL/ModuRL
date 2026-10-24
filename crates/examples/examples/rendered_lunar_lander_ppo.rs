use burn::{
    module::Module,
    nn::{Initializer, activation::Activation},
    optim::AdamWConfig,
    tensor::{DType, Device, Int, Tensor},
};

use modurl::prelude::*;
use modurl_gym::box_2d::lunar_lander::LunarLanderV3;

const DTYPE: DType = DType::F32;

struct DebugLunarLander {
    env: LunarLanderV3,
    reward_sum: f32,
    episodes_since_print: u32,
}

impl DebugLunarLander {
    fn new(env: LunarLanderV3) -> Self {
        Self {
            env,
            reward_sum: 0.0,
            episodes_since_print: 0,
        }
    }
}

impl Gym for DebugLunarLander {
    type Error = <LunarLanderV3 as Gym>::Error;
    type ObservationSpace = <LunarLanderV3 as Gym>::ObservationSpace;
    type ActionSpace = <LunarLanderV3 as Gym>::ActionSpace;

    fn action_space(&self) -> Self::ActionSpace {
        self.env.action_space()
    }

    fn observation_space(&self) -> Self::ObservationSpace {
        self.env.observation_space()
    }

    fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
        self.episodes_since_print += 1;
        if self.episodes_since_print >= 10 {
            println!(
                "Average reward over last 10 episodes: {}",
                self.reward_sum / 10.0
            );
            self.reward_sum = 0.0;
            self.episodes_since_print = 0;
        }
        self.env.reset()
    }

    /// Steps with one scalar Int action shaped `[1]`.
    fn step(&mut self, action: Tensor<1, Int>) -> Result<StepInfo, Self::Error> {
        let step_info = self.env.step(action)?;
        self.reward_sum += step_info.reward;
        Ok(step_info)
    }
}

fn main() {
    let device = Device::flex();
    let device = device.autodiff();
    let env1 = DebugLunarLander::new(
        LunarLanderV3::builder()
            .render(true)
            .rng_device(Device::flex())
            .build()
            .unwrap(),
    );
    let mut envs = vec![env1];
    for _ in 0..15 {
        let env = DebugLunarLander::new(
            LunarLanderV3::builder()
                .rng_device(Device::flex())
                .build()
                .unwrap(),
        );
        envs.push(env);
    }

    let env = VectorizedGymWrapper::from(envs);
    let mut env = DeviceMultiGymWrapper::new(env, Device::flex(), device.clone());
    let observation_space = env.observation_space();
    let action_space = env.action_space();

    // Actor network: 2x64, tanh activation
    let actor_network = MLP::builder()
        .input_size(observation_space.shape()[0])
        .output_size(action_space.get_possible_values())
        .options((&device, DTYPE))
        .activation(Activation::Tanh)
        .hidden_layer_sizes(vec![64, 64])
        .hidden_initializer(Initializer::Orthogonal {
            gain: 2.0f64.sqrt(),
        })
        .output_initializer(Initializer::Orthogonal { gain: 0.01 })
        .build()
        .unwrap();

    // Learning rate - common range for Lunar Lander is 1e-4 to 5e-4
    let config = AdamWConfig::new().with_weight_decay(0.01);
    let actor_optimizer = config.init();

    // Critic network: 2x64, tanh activation
    let critic_network = MLP::builder()
        .input_size(observation_space.shape()[0])
        .output_size(1)
        .options((&device, DTYPE))
        .activation(Activation::Tanh)
        .hidden_layer_sizes(vec![64, 64])
        .hidden_initializer(Initializer::Orthogonal {
            gain: 2.0f64.sqrt(),
        })
        .output_initializer(Initializer::Orthogonal { gain: 1.0 })
        .build()
        .unwrap();

    let critic_optimizer = config.init();

    let ppo_network_info = SeparatePPONetwork::builder()
        .actor_network(ProbabilisticPolicyModel::new(
            actor_network,
            CategoricalDistribution,
        ))
        .critic_network(critic_network)
        .actor_optimizer(actor_optimizer)
        .critic_optimizer(critic_optimizer)
        .actor_learning_rate(3e-4)
        .critic_learning_rate(3e-4)
        .actor_lr_scheduler(LinearSchedule::new(3e-4, 3e-5))
        .critic_lr_scheduler(LinearSchedule::new(3e-4, 3e-5))
        .build()
        .unwrap();

    // PPO config - Optimized hyperparameters for Lunar Lander
    let mut agent = PPOAgent::builder()
        .observation_space(observation_space)
        .action_space(action_space)
        .network_info(ppo_network_info)
        .batch_size(2048)
        .mini_batch_size(64)
        .normalize_advantage(true)
        .ent_coef(0.01)
        .gamma(0.999)
        .vf_coef(0.5)
        .clip_range(ConstantSchedule::new(0.2))
        .clipped(true)
        .gae_lambda(0.98)
        .num_epochs(4)
        .training_horizon(10_000_000)
        .device(device)
        .build()
        .unwrap();

    agent.learn(&mut env, 10_000_000).unwrap();

    let (actor, critic) = agent.network().models();
    actor.save_file("ppo_lunar_lander_actor_vars.bpk").unwrap();
    critic
        .save_file("ppo_lunar_lander_critic_vars.bpk")
        .unwrap();
}

use candle_core::{Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};

use modurl::prelude::*;
use modurl_gym::box_2d::lunar_lander::LunarLanderV3;

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
    type SpaceError = <LunarLanderV3 as Gym>::SpaceError;

    fn action_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
        self.env.action_space()
    }

    fn observation_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
        self.env.observation_space()
    }

    fn reset(&mut self) -> Result<Tensor, Self::Error> {
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

    fn step(&mut self, action: Tensor) -> Result<StepInfo, Self::Error> {
        let step_info = self.env.step(action)?;
        self.reward_sum += step_info.reward;
        Ok(step_info)
    }
}

fn main() {
    let device = Device::cuda_if_available(0).unwrap();
    let env1 = DebugLunarLander::new(
        LunarLanderV3::builder()
            .render(true)
            .device(device.clone())
            .build(),
    );
    let mut envs = vec![env1];
    for _ in 0..15 {
        let env = DebugLunarLander::new(LunarLanderV3::builder().device(device.clone()).build());
        envs.push(env);
    }

    let mut env = VectorizedGymWrapper::from(envs);
    let observation_space = env.observation_space();
    let action_space = env.action_space();
    let actor_var_map = VarMap::new();
    let actor_vb = VarBuilder::from_varmap(&actor_var_map, candle_core::DType::F32, &device);

    // Actor network: 2x64, tanh activation
    let actor_network = MLP::builder()
        .input_size(observation_space.shape()[0])
        .output_size(action_space.shape()[0])
        .vb(actor_vb)
        .activation(Box::new(Tensor::tanh))
        .hidden_layer_sizes(vec![64, 64])
        .initializer(Box::new(OrthogonalMLPInitializer {
            hidden_gain: 2.0f64.sqrt(),
            output_gain: 0.01,
        }))
        .name("actor_network".to_string())
        .build()
        .unwrap();

    // Learning rate - common range for Lunar Lander is 1e-4 to 5e-4
    let config = ParamsAdamW {
        lr: 3e-4,
        ..Default::default()
    };
    let actor_optimizer =
        AdamW::new(actor_var_map.all_vars(), config.clone()).expect("Failed to create AdamW");

    let critic_var_map = VarMap::new();
    let critic_vb = VarBuilder::from_varmap(&critic_var_map, candle_core::DType::F32, &device);

    // Critic network: 2x64, tanh activation
    let critic_network = MLP::builder()
        .input_size(observation_space.shape()[0])
        .output_size(1)
        .vb(critic_vb)
        .activation(Box::new(Tensor::tanh))
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

    let ppo_network_info = PPONetworkInfo::Separate(
        SeparatePPONetwork::builder()
            .actor_network(Box::new(
                ProbabilisticPolicyModel::<CategoricalDistribution>::new(Box::new(actor_network)),
            ))
            .critic_network(Box::new(critic_network))
            .actor_optimizer(actor_optimizer)
            .critic_optimizer(critic_optimizer)
            .actor_lr_scheduler(Box::new(LinearSchedule::new(3e-4, 3e-5)))
            .critic_lr_scheduler(Box::new(LinearSchedule::new(3e-4, 3e-5)))
            .build(),
    );

    // PPO config - Optimized hyperparameters for Lunar Lander
    let mut agent = PPOAgent::builder()
        .action_space(action_space)
        .network_info(ppo_network_info)
        .batch_size(2048)
        .mini_batch_size(64)
        .normalize_advantage(true)
        .ent_coef(0.01)
        .gamma(0.999)
        .vf_coef(0.5)
        .clip_range(Box::new(ConstantSchedule::new(0.2)))
        .clipped(true)
        .gae_lambda(0.98)
        .num_epochs(4)
        .training_horizon(10_000_000)
        .device(device)
        .build();

    agent.learn(&mut env, 10_000_000).unwrap();

    actor_var_map.save("ppo_lunar_lander_actor_vars").unwrap();
    critic_var_map.save("ppo_lunar_lander_critic_vars").unwrap();
}

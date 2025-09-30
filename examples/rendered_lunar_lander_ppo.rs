/*
use candle_core::{Device, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use candle_optimisers::adam::{Adam, ParamsAdam};

use modurl::gym::StepInfo;
use modurl::tensor_operations::tanh;
use modurl::{
    actors::{Actor, ppo::PPOActor},
    distributions::CategoricalDistribution,
    gym::Gym,
    models::{MLP, probabilistic_model::MLPProbabilisticActor},
};
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
    type Error = candle_core::Error;
    fn get_name(&self) -> &str {
        self.env.get_name()
    }

    fn action_space(&self) -> Box<dyn modurl::spaces::Space> {
        self.env.action_space()
    }

    fn observation_space(&self) -> Box<dyn modurl::spaces::Space> {
        self.env.observation_space()
    }

    fn reset(&mut self) -> Result<Tensor, candle_core::Error> {
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
    let env = LunarLanderV3::builder().render(true).build();
    let mut env = DebugLunarLander::new(env);
    let observation_space = env.observation_space();
    let action_space = env.action_space();
    let actor_var_map = VarMap::new();
    let actor_vb = VarBuilder::from_varmap(&actor_var_map, candle_core::DType::F32, &Device::Cpu);

    // Actor network: 2x64, tanh activation
    let actor_network = MLP::builder()
        .input_size(observation_space.shape().iter().sum())
        .output_size(action_space.shape().iter().sum::<usize>())
        .vb(actor_vb)
        .activation(Box::new(tanh))
        .hidden_layer_sizes(vec![64, 64])
        .name("actor_network".to_string())
        .build()
        .unwrap();

    let mut config = ParamsAdam::default();
    // Learning rate - common range for Lunar Lander is 1e-4 to 5e-4
    config.lr = 3e-4;
    let actor_optimizer =
        Adam::new(actor_var_map.all_vars(), config.clone()).expect("Failed to create Adam");

    let critic_var_map = VarMap::new();
    let critic_vb = VarBuilder::from_varmap(&critic_var_map, candle_core::DType::F32, &Device::Cpu);

    // Critic network: 2x64, tanh activation
    let critic_network = MLP::builder()
        .input_size(observation_space.shape().iter().sum())
        .output_size(1)
        .vb(critic_vb)
        .activation(Box::new(tanh))
        .hidden_layer_sizes(vec![64, 64])
        .name("critic_network".to_string())
        .build()
        .unwrap();

    config.lr = 3e-4;
    let critic_optimizer =
        Adam::new(critic_var_map.all_vars(), config.clone()).expect("Failed to create Adam");

    // PPO config - Optimized hyperparameters for Lunar Lander
    let mut actor = PPOActor::builder()
        .observation_space(observation_space)
        .action_space(action_space)
        .actor_network(Box::new(
            MLPProbabilisticActor::<CategoricalDistribution>::new(actor_network),
        ))
        .critic_network(Box::new(critic_network))
        .critic_optimizer(critic_optimizer)
        .actor_optimizer(actor_optimizer)
        .batch_size(2048)
        .mini_batch_size(64)
        .normalize_advantage(true)
        .ent_coef(0.01)
        .gamma(0.999)
        .vf_coef(0.5)
        .clip_range(0.2)
        .clipped(true)
        .gae_lambda(0.98)
        .num_epochs(4)
        .actor_lr_scheduler(Box::new(|t| (1.0 - t * 0.9) * 3e-4))
        .critic_lr_scheduler(Box::new(|t| (1.0 - t * 0.9) * 3e-4))
        .build();

    actor.learn(&mut env, 1000000).unwrap();

    actor_var_map.save("ppo_lunar_lander_actor_vars").unwrap();
    critic_var_map.save("ppo_lunar_lander_critic_vars").unwrap();
}
*/

fn main() {}

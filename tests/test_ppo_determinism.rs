#[cfg(any(feature = "cuda", feature = "metal"))]
use candle_core::{Device, Tensor};
#[cfg(any(feature = "cuda", feature = "metal"))]
use candle_nn::{Optimizer, VarBuilder, VarMap};
#[cfg(any(feature = "cuda", feature = "metal"))]
use candle_optimisers::adam::{Adam, ParamsAdam};
#[cfg(any(feature = "cuda", feature = "metal"))]
use modurl::{
    actors::{Actor, ppo::PPOActor},
    distributions::CategoricalDistribution,
    gym::{Gym, VectorizedGym, VectorizedGymWrapper},
    models::{MLP, probabilistic_model::MLPProbabilisticActor},
    tensor_operations::tanh,
};
#[cfg(any(feature = "cuda", feature = "metal"))]
use modurl_gym::classic_control::cartpole::CartPoleV1;

#[cfg(any(feature = "cuda", feature = "metal"))]
#[test]
fn test_ppo_determinism() {
    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0).unwrap();
    #[cfg(feature = "metal")]
    let device = Device::new_metal(0).unwrap();

    const SAMPLE_COUNT: usize = 25;

    let mut last_actions: Option<Vec<Tensor>> = None;

    for i in 0..5 {
        // Reset seed before each iteration
        device.set_seed(42).unwrap();

        // Create fresh PPO actor
        let mut envs = vec![];
        for _ in 0..8 {
            envs.push(CartPoleV1::builder().device(&device).build());
        }
        let mut vec_env: VectorizedGymWrapper<CartPoleV1> = envs.into();

        let observation_space = vec_env.observation_space();
        let action_space = vec_env.action_space();

        // Actor network
        let actor_var_map = VarMap::new();
        let actor_vb = VarBuilder::from_varmap(&actor_var_map, candle_core::DType::F32, &device);
        let actor_network = MLP::builder()
            .input_size(observation_space.shape().iter().product())
            .output_size(action_space.shape().iter().product::<usize>())
            .vb(actor_vb)
            .activation(Box::new(tanh))
            .hidden_layer_sizes(vec![64, 64])
            .name("actor_network".to_string())
            .build()
            .unwrap();

        let mut config = ParamsAdam::default();
        config.lr = 3e-4;
        let actor_optimizer = Adam::new(actor_var_map.all_vars(), config.clone()).unwrap();

        // Critic network
        let critic_var_map = VarMap::new();
        let critic_vb = VarBuilder::from_varmap(&critic_var_map, candle_core::DType::F32, &device);
        let critic_network = MLP::builder()
            .input_size(observation_space.shape().iter().product())
            .output_size(1)
            .vb(critic_vb)
            .activation(Box::new(tanh))
            .hidden_layer_sizes(vec![64, 64])
            .name("critic_network".to_string())
            .build()
            .unwrap();

        let critic_optimizer = Adam::new(critic_var_map.all_vars(), config.clone()).unwrap();

        // Create PPO actor
        let mut actor = PPOActor::builder()
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
            .ent_coef(0.005)
            .gamma(0.99)
            .vf_coef(0.5)
            .clip_range(0.2)
            .clipped(true)
            .gae_lambda(0.95)
            .num_epochs(10)
            .device(device.clone())
            .build();

        // train for some timesteps
        actor
            .learn(&mut vec_env, 20000)
            .expect("PPO learning failed");

        // Get initial state and take one action
        let initial_state = vec_env.reset().unwrap();
        let mut actions = vec![];

        for _ in 0..SAMPLE_COUNT {
            let current_action = actor.act(&initial_state).unwrap();
            actions.push(current_action.clone());
            vec_env.step(current_action).unwrap();
        }

        // Compare with previous iteration
        if let Some(last_actions) = &last_actions {
            for (last_action, current_action) in last_actions.iter().zip(actions.iter()) {
                let max_diff = last_action
                    .ne(&*current_action)
                    .unwrap()
                    .max_all()
                    .unwrap()
                    .to_scalar::<u8>()
                    .unwrap();

                assert!(max_diff == 0, "PPO actions differ at iteration {i}");
            }
        }
        last_actions = Some(actions);
    }
}

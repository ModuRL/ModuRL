# DQN

This page builds a DQN agent for CartPole. It uses one vectorized CartPole
environment, two identically shaped Q-networks, an epsilon schedule, and an
experience replay buffer. You need Rust, Cargo, and the dependencies from
[Getting Started](./getting-started.md).

The program trains for 500,000 environment transitions, so the first complete
run takes longer than the PPO quick-start. Lower `training_horizon` and the
argument to `learn` together when you only want to check that the program runs.

## The Q-Networks

Create a Q-network for the online parameters and an identically shaped one for
the target parameters. CartPole has four observation values and two discrete
actions, so the network reads the environment's observation shape and produces
two Q-values.

`DQNAgent` needs a `Discrete` action space. `CartPoleV1` exposes the action
space through the general `Space` trait, so this example supplies the known
CartPole action count with `Discrete::new(2)`.

## Complete Program

Place this program in `src/main.rs`:

```rust,ignore
use candle_core::{DType, Device};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use modurl::prelude::*;
use modurl_gym::classic_control::cartpole::CartPoleV1;

fn main() {
    let device = Device::Cpu;
    let envs = vec![CartPoleV1::builder().device(&device).build()];
    let mut env = VectorizedGymWrapper::from(envs);
    let observation_space = env.observation_space();

    let online_var_map = VarMap::new();
    let online_q_network = MLP::builder()
        .input_size(observation_space.shape()[0])
        .output_size(2)
        .vb(VarBuilder::from_varmap(&online_var_map, DType::F32, &device))
        .hidden_layer_sizes(vec![64, 64])
        .build()
        .expect("failed to build the online Q-network");

    let mut target_var_map = VarMap::new();
    let target_q_network = MLP::builder()
        .input_size(observation_space.shape()[0])
        .output_size(2)
        .vb(VarBuilder::from_varmap(&target_var_map, DType::F32, &device))
        .hidden_layer_sizes(vec![64, 64])
        .build()
        .expect("failed to build the target Q-network");

    let optimizer = AdamW::new(
        online_var_map.all_vars(),
        ParamsAdamW {
            lr: 2.5e-4,
            ..Default::default()
        },
    )
    .expect("failed to build the optimizer");

    let mut agent = DQNAgent::builder()
        .action_space(Discrete::new(2))
        .observation_space(observation_space)
        .online_q_network(Box::new(online_q_network))
        .target_q_network(Box::new(target_q_network))
        .online_vars(&online_var_map)
        .target_vars(&mut target_var_map)
        .optimizer(optimizer)
        .replay_capacity(10_000)
        .batch_size(128)
        .training_start(10_000)
        .update_frequency(10)
        .target_update_interval(500)
        .training_horizon(500_000)
        .epsilon_schedule(Box::new(|progress: f64| {
            let exploration_progress = (progress / 0.5).min(1.0);
            1.0 + (0.05 - 1.0) * exploration_progress
        }))
        .device_strategy(QLearningDeviceStrategy::OneDevice(device.clone()))
        .build()
        .expect("DQN configuration should be valid");

    agent.learn(&mut env, 500_000).expect("DQN learning failed");
    println!("Training complete.");
}
```

The online network is the only network passed to `AdamW`; the target network is
not optimized directly. At construction, the agent copies the online parameters
to the target network. It repeats that copy every 500 transitions.

The epsilon schedule decreases exploration from `1.0` to `0.05` during the
first half of the 500,000-transition horizon. The agent collects 10,000
transitions before its first update, then samples 128 replay entries every 10
transitions.

Run the program with:

```sh
cargo run
```

When training finishes, it prints `Training complete.`. Read [Value-Based
Training](./q-learning.md) for the DQN and DDQN distinction, or [Double
DQN](./ddqn.md) to use the alternative target calculation. To record and
interpret training metrics, read [Understand a Q-Learning Training
Run](./understand-q-learning-training.md).

## Graph a Training Run

Run the repository example when you want terminal graphs instead of the minimal
program above:

```sh
cargo run --example dqn_cartpole_with_graphs
```

The example prints each completed CartPole episode's collection step, length,
and return during training. After its 500,000-transition run, it plots DQN loss,
exploration epsilon, mean selected Q-values, episode returns, and episode
lengths. The update metrics come from replay batches; the episode graphs come
from the current collection stream. Read [Understand a Q-Learning Training
Run](./understand-q-learning-training.md) for the distinction.

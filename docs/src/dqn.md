# DQN

This page builds a DQN agent for CartPole. It uses one vectorized CartPole
environment, an owned Q-network, an epsilon schedule, and an
experience replay buffer. You need Rust, Cargo, and the dependencies from
[Getting Started](./getting-started.md).

The program trains for 500,000 environment transitions, so the first complete
run takes longer than the PPO quick-start. Lower `training_horizon` and the
argument to `learn` together when you only want to check that the program runs.

## The Q-Networks

Create an online Q-network. The agent creates its detached target copy.
CartPole has four observation values and two discrete actions, so the network reads the environment's observation shape and produces
two Q-values.

`DQNAgent` needs a `Discrete` action space. `CartPoleV1` supplies that concrete type through `env.action_space()`.

## Complete Program

Enable Burn `0.22.0` features `std`, `optim`, `flex`, and `autodiff` in the application.
The code below shows the Burn agent API. It requires the CartPole environment migration before it can run.
The current CartPole executable still uses Candle.

```rust,ignore
use burn::{optim::AdamWConfig, tensor::{DType, Device}};
use modurl::prelude::*;
use modurl_gym::classic_control::cartpole::CartPoleV1;

fn main() {
    let device = Device::flex().autodiff();

    let environment_device = Device::flex();
    let envs = vec![CartPoleV1::builder().rng_device(&environment_device).build().unwrap()];
    let env = VectorizedGymWrapper::from(envs);
    let mut env = DeviceMultiGymWrapper::new(env, environment_device, device.clone());
    let observation_space = env.observation_space();
    let online_q_network = MLP::builder()
        .input_size(observation_space.shape()[0])
        .output_size(env.action_space().get_possible_values())
        .options((&device, DType::F32))
        .hidden_layer_sizes(vec![64, 64])
        .build()
        .expect("failed to build the online Q-network");
    let optimizer = AdamWConfig::new().init();

    let mut agent = DQNAgent::builder()
        .dtype(DType::F32)
        .action_space(env.action_space())
        .observation_space(observation_space)
        .online_q_network(online_q_network)
        .optimizer(optimizer)
        .learning_rate(2.5e-4)
        .replay_capacity(10_000)
        .batch_size(128)
        .training_start(10_000)
        .update_frequency(10)
        .target_update_interval(500)
        .training_horizon(500_000)
        .epsilon_schedule(|progress: f64| {
            let exploration_progress = (progress / 0.5).min(1.0);
            1.0 + (0.05 - 1.0) * exploration_progress
        })
        .replay_storage_config(ReplayStorageConfig::new(ReplayDeviceStrategy::OneDevice(device)))
        .build()
        .expect("DQN configuration should be valid");

    agent.learn(&mut env, 500_000).expect("DQN learning failed");
    println!("Training complete.");
}
```

The Burn optimizer updates the owned online model. The target model has no gradient graph.
The agent copies the online model at construction and every 500 transitions.

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
cargo run --example dqn_cartpole
```

The example prints each completed CartPole episode's collection step, length,
and return during training. After its 500,000-transition run, it plots DQN loss,
exploration epsilon, mean selected Q-values, episode returns, and episode
lengths. The update metrics come from replay batches; the episode graphs come
from the current collection stream. Read [Understand a Q-Learning Training
Run](./understand-q-learning-training.md) for the distinction.

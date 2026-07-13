# Double DQN

Double DQN, or DDQN, uses the same setup, builder fields, and training loop as
DQN. Its difference is the next-state target: the online Q-network chooses the
next action, and the target Q-network evaluates that selected action. This
separation avoids using the same values for both decisions.

Start with the complete [DQN CartPole program](./dqn.md). Keep its environment,
two Q-networks, two `VarMaps`, optimizer, replay configuration, epsilon schedule,
and `learn` call unchanged. Then replace the agent construction with this
version:

```rust,ignore
let mut agent = DDQNAgent::builder()
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
    .expect("DDQN configuration should be valid");

agent.learn(&mut env, 500_000).expect("DDQN learning failed");
```

`DDQNAgent` has the same configuration validation as `DQNAgent`: the replay
capacity must be at least the batch size; the replay capacity, batch size,
update frequency, target-update interval, and training horizon must be nonzero;
and gamma and epsilon values must stay in the inclusive `0.0..=1.0` range.

This is a change in the agent's learning target, not a change in the Q-network
shape or device setup. Use [Value-Based Training](./q-learning.md) for the
shared configuration rules. To record and interpret training metrics, read
[Understand a Q-Learning Training Run](./understand-q-learning-training.md).

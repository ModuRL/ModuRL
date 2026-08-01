# A2C

`A2CAgent` implements synchronous Advantage Actor-Critic by configuring an
inner PPO agent. The wrapper disables policy and value clipping, performs one
optimization epoch, and uses the complete rollout as one minibatch. These
algorithm-defining settings cannot be changed through the A2C builder.

The remaining defaults follow Stable-Baselines3 A2C: `gamma` is `0.99`,
`gae_lambda` is `1.0`, advantages and returns are not normalized, `vf_coef` is
`0.5`, `ent_coef` is `0.0`, and gradient clipping is `0.5`. The application
still supplies the optimizer as part of its network configuration.

Use `SharedA2CNetwork` when the actor and critic share a model, or
`SeparateA2CNetwork` when they have independent models and optimizers. Wrap the
result with `A2CNetworkInfo::shared` or `A2CNetworkInfo::separate` before
building the agent. A2C also provides typed errors, log entries, and the
`A2CLogger` trait. Pass an A2C logger directly to the agent's `logging_info`
builder field.

`batch_size` is the total number of transitions in one rollout, across every
vectorized environment. Choose a value divisible by the environment count so
the agent performs exactly one full-batch update.

The MuJoCo example uses a separate Gaussian actor and critic. Run it with one
of the supported environment features:

```console
cargo run -p examples --example a2c_mujoco --features ant
cargo run -p examples --example a2c_mujoco --features half-cheetah
cargo run -p examples --example a2c_mujoco --features hopper
cargo run -p examples --example a2c_mujoco --features walker2d
```

Choose one command for the environment you want to train. Read
[PPO](./ppo.md) for the underlying on-policy training model.

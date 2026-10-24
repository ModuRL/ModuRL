# Run on CUDA or Metal

First run the CPU version of your program. Then enable one Candle backend
feature and construct a device for that backend.

## CUDA

Enable CUDA on your direct `candle-core` dependency in `Cargo.toml`:

```toml
candle-core = { version = "0.11", features = ["cuda"] }
```

In a program, replace `Device::Cpu` with:

```rust,ignore
let device = Device::new_cuda(0)?;
```

`0` selects the first CUDA device. The CUDA runtime and a Candle build with CUDA
support must be available on the machine.

## Metal

Enable Metal on your direct `candle-core` dependency in `Cargo.toml`:

```toml
candle-core = { version = "0.11", features = ["metal"] }
```

In a program, replace `Device::Cpu` with:

```rust,ignore
let device = Device::new_metal(0)?;
```

`0` selects the first Metal device. Metal builds require a supported Apple
platform.

## Transfer at the Environment Batch Boundary

CPU environments always return CPU observations. Set `rng_device` only to
choose where classic-control and Box2D random draws occur; it does not change
observation placement. MuJoCo uses its existing seeded CPU RNG and has no
device setting. Wrap vectorized environments to transfer actions and observations
once per batch, and build model parameters on the agent device.

```rust,ignore
let env = CartPoleV1::builder().rng_device(&device).build().unwrap();
let mut envs = DeviceMultiGymWrapper::new(
    VectorizedGymWrapper::from(vec![env]), Device::Cpu, device.clone(),
);
let vb = VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &device);
```

If the selected device is unavailable, Candle returns an error when the program
constructs it. Fix the backend installation or return to `Device::Cpu`.

## Split Replay Storage From Optimization

Replay-based agents can keep a large replay buffer on the CPU while running
models and optimization on an accelerator:

```rust,ignore
use burn::tensor::Device;
use modurl::prelude::*;

let optimization_device = Device::cuda(0).autodiff();
let storage_device = Device::flex();
let env = CartPoleV1::builder().rng_device(&storage_device).build()?;
let mut envs = DeviceMultiGymWrapper::new(
    VectorizedGymWrapper::from(env), storage_device.clone(), optimization_device.clone(),
);
let replay_storage_config = ReplayStorageConfig::new(ReplayDeviceStrategy::Hybrid {
    optimization_device: optimization_device.clone(),
    storage_device,
});

let mut agent = SACAgent::builder()
    .policy(policy)
    .actor_optimizer(actor_optimizer)
    .actor_learning_rate(3e-4)
    .critics(critics)
    .entropy_configuration(entropy)
    .action_space(envs.action_space())
    .observation_space(envs.observation_space())
    .training_horizon(total_timesteps)
    .replay_storage_config(replay_storage_config)
    .build()?;
```

`ReplayStorageConfig` selects replay and optimization devices. SAC stores the native observation kind and dtype from the first batch. It does not
move an environment or model parameters for you.

Build SAC's actor, critics, target critics, optimizers, and automatic entropy
variable on `optimization_device`. Use `DeviceMultiGymWrapper` to connect CPU
environments to that device. For
DDPG or TD3, this includes both the online and target actors as well as every
critic pair. You do not create any of these components on `storage_device`.

Internally, the agent transfers detached transitions to `storage_device` when
it adds them to replay. It transfers sampled replay batches back to
`optimization_device` before each update.

This strategy trades transfer time for accelerator memory. Start with
`ReplayDeviceStrategy::OneDevice` and measure the run. Switch to `Hybrid` when
replay memory is the limiting resource and the transfer cost is acceptable.

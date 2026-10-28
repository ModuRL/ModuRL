# Run on CUDA or Metal

First run the CPU version of your program. Then enable one backend
feature and construct a device for that backend.

## CUDA

Enable CUDA on your direct `burn` dependency in `Cargo.toml`:

```toml
burn = { version = "=0.22.0", default-features = false, features = ["std", "optim", "autodiff", "cuda", "flex"] }
```

In a program, replace `Device::flex()` with:

```rust,ignore
let device = Device::cuda(0).autodiff();
```

`0` selects the first CUDA device. The CUDA runtime and a build with CUDA
support must be available on the machine.

## Metal

Enable Metal on your direct `burn` dependency in `Cargo.toml`:

```toml
burn = { version = "=0.22.0", default-features = false, features = ["std", "optim", "autodiff", "metal", "flex"] }
```

In a program, replace `Device::flex()` with:

```rust,ignore
let device = Device::metal(burn::tensor::DeviceKind::DefaultDevice).autodiff();
```

The default device selects the system Metal device. Metal builds require a supported Apple
platform.

## Transfer at the Environment Batch Boundary

CPU environments always return CPU observations. Classic-control and Box2D
environments own host RNGs; set `.seed(u64)` for reproducibility. Omitted
seeds use OS entropy and are independent of `Device::seed`. MuJoCo uses its
existing seeded CPU RNG and has no device setting. Wrap vectorized environments
to transfer actions and observations once per batch, and build model parameters
on the agent device.

```rust,ignore
let env = CartPoleV1::builder().seed(42).build().unwrap();
let mut envs = DeviceMultiGymWrapper::new(
    VectorizedGymWrapper::from(env), Device::flex(), device.clone(),
);
let model = MLP::builder().input_size(4).output_size(2)
    .hidden_layer_sizes(vec![64, 64]).options((&device, burn::tensor::DType::F32)).build()?;
```

The selected backend requires its runtime and drivers.
If initialization fails, fix the backend installation or return to `Device::flex()`.

## Split Replay Storage From Optimization

Replay-based agents can keep a large replay buffer on the CPU while running
models and optimization on an accelerator:

```rust,ignore
use burn::tensor::Device;
use modurl::prelude::*;

let optimization_device = Device::cuda(0).autodiff();
let storage_device = Device::flex();
let env = CartPoleV1::builder().seed(42).build()?;
let mut envs = DeviceMultiGymWrapper::new(
    VectorizedGymWrapper::from(env), storage_device.clone(), optimization_device.clone(),
);

let mut agent = SACAgent::builder()
    .policy(policy)
    .actor_optimizer(actor_optimizer)
    .actor_learning_rate(3e-4)
    .critics(critics)
    .entropy_configuration(entropy)
    .action_space(envs.action_space())
    .observation_space(envs.observation_space())
    .training_horizon(total_timesteps)
    .replay_device(storage_device)
    .build()?;
```

`.replay_device(..)` selects the replay storage device; optimization runs on the device of the model parameters. SAC stores the native observation kind and dtype from the first batch. It does not
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
the default (replay on the model device) and measure the run. Set `.replay_device(..)` when
replay memory is the limiting resource and the transfer cost is acceptable.

# Run on CUDA or Metal

First run the CPU version of your program. Then enable one Candle backend
feature and construct a device for that backend.

## CUDA

Replace the `modurl` dependency line in your project's `Cargo.toml` with:

```toml
modurl = { version = "0.1", features = ["cuda"] }
```

In a program, replace `Device::Cpu` with:

```rust,ignore
let device = Device::new_cuda(0)?;
```

`0` selects the first CUDA device. The CUDA runtime and a Candle build with CUDA
support must be available on the machine.

## Metal

Replace the `modurl` dependency line in your project's `Cargo.toml` with:

```toml
modurl = { version = "0.1", features = ["metal"] }
```

In a program, replace `Device::Cpu` with:

```rust,ignore
let device = Device::new_metal(0)?;
```

`0` selects the first Metal device. Metal builds require a supported Apple
platform.

## Keep Values on One Device

Pass the same `device` to the environment builder and to `VarBuilder`. That
places environment observations and model parameters on the same backend.

```rust,ignore
let env = CartPoleV1::builder().device(&device).build();
let vb = VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &device);
```

If the selected device is unavailable, Candle returns an error when the program
constructs it. Fix the backend installation or return to `Device::Cpu`.

## Split Replay Storage From Optimization

Replay-based agents can keep a large replay buffer on the CPU while running
models and optimization on an accelerator:

```rust,ignore
let optimization_device = Device::new_cuda(0)?;
let storage_device = Device::Cpu;

let env = CartPoleV1::builder()
    .device(&optimization_device)
    .build();

let actor_vb = VarBuilder::from_varmap(
    &actor_vars,
    candle_core::DType::F32,
    &optimization_device,
);

let device_strategy = ReplayDeviceStrategy::Hybrid {
    optimization_device: optimization_device.clone(),
    storage_device,
};

let mut agent = SACAgent::builder()
    // Build the actor, critics, target critics, optimizers, and entropy
    // variable on optimization_device.
    .device_strategy(device_strategy)
    // Keep the remaining SAC configuration unchanged.
    .build()?;
```

`ReplayDeviceStrategy` moves replay entries and sampled batches. It does not
move an environment or model parameters for you.

Set up everything you create for SAC on `optimization_device`: the environment,
actor, critics, target critics, optimizers, and automatic entropy variable. You
do not create any of these components on `storage_device`.

Internally, the agent transfers detached transitions to `storage_device` when
it adds them to replay. It transfers sampled replay batches back to
`optimization_device` before each update.

This strategy trades transfer time for accelerator memory. Start with
`ReplayDeviceStrategy::OneDevice` and measure the run. Switch to `Hybrid` when
replay memory is the limiting resource and the transfer cost is acceptable.

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

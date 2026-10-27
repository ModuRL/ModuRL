# modurl_gym

`modurl_gym` is a Rust library crate for running Gymnasium-compatible classic-control and Box2D environments with ModuRL.

> [!WARNING]
> ModuRL is early in development. Public APIs may change between revisions.

## Add the dependencies

```toml
[dependencies]
burn = { version = "=0.22.0", default-features = false, features = ["std", "flex"] }
modurl = "0.1.0"
modurl_gym = "0.1.0"
```

## Example

Create a CartPole environment, sample an action, and take one step:

```rust
use burn::tensor::Device;
use modurl::prelude::*;
use modurl_gym::classic_control::cartpole::CartPoleV1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::flex();
    let mut env = CartPoleV1::builder().seed(7).build()?;

    let observation = env.reset()?.observation;
    let action = env.action_space().sample_batch(1, &device)?.squeeze_dims::<1>(&[0]);
    let transition = env.step(action)?;

    println!("observation shape: {:?}", observation.dims());
    println!("reward: {}", transition.reward);
    Ok(())
}
```

## Environments

Discrete actions have shape `[1]` and kind `Int`; vectorized actions have shape `[batch_size, 1]`.
Continuous actions retain their component shapes. Observation space bounds include a leading size-one batch axis.

All environments produce observations and space bounds on CPU. Each environment
owns a host RNG; set `.seed(u64)` for reproducible behavior. If omitted, the RNG
uses OS entropy and is independent of `Device::seed`. Use
`DeviceMultiGymWrapper` around vectorized environments to transfer whole action
and observation batches to an accelerator.

| Module | Environment | Action space | Observation shape |
| --- | --- | --- | --- |
| `classic_control::acrobot` | `AcrobotV1` | Discrete, 3 actions | `[6]` |
| `classic_control::cartpole` | `CartPoleV1` | Discrete, 2 actions | `[4]` |
| `classic_control::mountain_car` | `MountainCarV0` | Discrete, 3 actions | `[2]` |
| `classic_control::pendulum` | `PendulumV1` | Continuous `[1]` | `[3]` |
| `box_2d::bipedal_walker` | `BipedalWalkerV3` | Continuous `[4]` | `[24]` |
| `box_2d::lunar_lander` | `LunarLanderV3` | Discrete, 4 actions | `[8]` |

`BipedalWalkerV3` implements the standard environment. The hardcore obstacle variant is not included.

The environment structs expose Gymnasium's unwrapped dynamics. Apply registry time limits explicitly with ModuRL's `TimeLimitGym` wrapper.

## Cargo features

The crate has no default Cargo features.

| Feature | Effect |
| --- | --- |
| `rendering` | Adds `render` builder options and renders environments through `minifb`. |
| `logging` | Emits the CartPole post-termination warning through the `log` crate. |

## Documentation

See the [environment guide](../../docs/src/environments.md) for time limits, vectorization, and custom environments.

Build the API documentation locally:

```console
cargo doc -p modurl_gym --no-deps --open
```

## License

`modurl_gym` is available under the [MIT License](LICENSE).

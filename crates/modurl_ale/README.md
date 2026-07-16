# modurl_ale

`modurl_ale` provides Atari 2600 environments for [ModuRL](https://github.com/ModuRL/ModuRL) using the Arcade Learning Environment (ALE).

This is a Rust crate, but it is not a pure-Rust implementation: it builds and links the bundled C++ ALE/Stella emulator code that provides the Atari 2600 backend. The public API and ModuRL integration are Rust; the native emulator is an upstream dependency included in this repository.

No ROMs are included or downloaded. You must supply a filesystem path to a ROM you are legally entitled to use:

```rust,no_run
use std::path::PathBuf;
use candle_core::Device;
use modurl_ale::{AtariGym, AtariObsType};

let mut env = AtariGym::builder()
    .rom_path(PathBuf::from("/path/to/your/game.bin"))
    .obs_type(AtariObsType::RAM)
    .device(Device::Cpu)
    .repeat_action_probability(0.0)
    .build()?;
env.set_frame_skip(4);
# Ok::<(), modurl_ale::AtariGymError>(())
```

Observations are normalized to `0.0..=1.0`. The environment exposes ALE's ROM-specific minimal action set as a ModuRL `Discrete` space and max-pools the final two frames when frame skipping is enabled. RAM, RGB, and grayscale observations are supported, along with sticky-action probability, seeding, lives, reset, device selection, and optional `minifb` rendering.

Enable display rendering with `features = ["rendering"]`. The `cuda`, `cudnn`, and `metal` features forward directly to the corresponding Candle backend.

## Licensing and ROMs

This crate is `GPL-2.0-only` and contains GPL-covered ALE/Stella native code. Distributed binaries that link this crate must comply with GPL-2.0; private use does not require publication. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md). This is not legal advice.

Commercial ROMs and other ROM images are strictly user-supplied. Do not commit ROMs to this repository. Invalid or unsupported ROM handling retains limitations inherited from the older ALE 0.6 backend and may not always produce a recoverable Rust error.

Building requires CMake and a working C/C++ toolchain.

## Testing

Unit and documentation tests require no ROM. To opt into the real-backend smoke test, set `MODURL_ALE_TEST_ROM` to a local ROM path before running `cargo test --test rom_opt_in`. The ROM is neither copied nor uploaded.

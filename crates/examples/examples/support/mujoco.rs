use candle_core::Device;

// Cargo features are additive. When several environment features are enabled
// (for example by `--all-features`), select the first in this documented
// priority order so every example remains buildable. The umbrella
// `mujoco-environment` feature defaults to Ant when no specific environment is
// selected.
#[cfg(any(
    feature = "ant",
    not(any(feature = "half-cheetah", feature = "hopper", feature = "walker2d"))
))]
use modurl_mujoco::AntV5 as SelectedEnvironment;
#[cfg(all(not(feature = "ant"), feature = "half-cheetah"))]
use modurl_mujoco::HalfCheetahV5 as SelectedEnvironment;
#[cfg(all(
    not(feature = "ant"),
    not(feature = "half-cheetah"),
    feature = "hopper"
))]
use modurl_mujoco::HopperV5 as SelectedEnvironment;
#[cfg(all(
    not(feature = "ant"),
    not(feature = "half-cheetah"),
    not(feature = "hopper"),
    feature = "walker2d"
))]
use modurl_mujoco::Walker2dV5 as SelectedEnvironment;

#[cfg(any(
    feature = "ant",
    not(any(feature = "half-cheetah", feature = "hopper", feature = "walker2d"))
))]
pub const ENVIRONMENT_NAME: &str = "Ant-v5";
#[cfg(all(not(feature = "ant"), feature = "half-cheetah"))]
pub const ENVIRONMENT_NAME: &str = "HalfCheetah-v5";
#[cfg(all(
    not(feature = "ant"),
    not(feature = "half-cheetah"),
    feature = "hopper"
))]
pub const ENVIRONMENT_NAME: &str = "Hopper-v5";
#[cfg(all(
    not(feature = "ant"),
    not(feature = "half-cheetah"),
    not(feature = "hopper"),
    feature = "walker2d"
))]
pub const ENVIRONMENT_NAME: &str = "Walker2d-v5";

pub fn build_environment(device: &Device) -> SelectedEnvironment {
    let builder = SelectedEnvironment::builder().device(device);
    #[cfg(feature = "rendering")]
    let builder = builder.render(true);
    builder.build().unwrap()
}

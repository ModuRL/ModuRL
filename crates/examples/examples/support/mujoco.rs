use candle_core::Device;

#[cfg(not(any(
    feature = "ant",
    feature = "half-cheetah",
    feature = "hopper",
    feature = "walker2d"
)))]
compile_error!(
    "enable exactly one MuJoCo environment feature: ant, half-cheetah, hopper, or walker2d"
);

#[cfg(any(
    all(feature = "ant", feature = "half-cheetah"),
    all(feature = "ant", feature = "hopper"),
    all(feature = "ant", feature = "walker2d"),
    all(feature = "half-cheetah", feature = "hopper"),
    all(feature = "half-cheetah", feature = "walker2d"),
    all(feature = "hopper", feature = "walker2d"),
))]
compile_error!(
    "enable exactly one MuJoCo environment feature: ant, half-cheetah, hopper, or walker2d"
);

#[cfg(feature = "ant")]
use modurl_mojoco::AntV5 as SelectedEnvironment;
#[cfg(all(not(feature = "ant"), feature = "half-cheetah"))]
use modurl_mojoco::HalfCheetahV5 as SelectedEnvironment;
#[cfg(all(
    not(feature = "ant"),
    not(feature = "half-cheetah"),
    feature = "hopper"
))]
use modurl_mojoco::HopperV5 as SelectedEnvironment;
#[cfg(all(
    not(feature = "ant"),
    not(feature = "half-cheetah"),
    not(feature = "hopper"),
    feature = "walker2d"
))]
use modurl_mojoco::Walker2dV5 as SelectedEnvironment;

#[cfg(feature = "ant")]
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

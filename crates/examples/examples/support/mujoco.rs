use candle_core::Device;

#[cfg(not(any(
    feature = "ant",
    feature = "half-cheetah",
    feature = "hopper",
    feature = "humanoid",
    feature = "sumo-ants",
    feature = "walker2d"
)))]
compile_error!(
    "enable exactly one MuJoCo environment feature: ant, half-cheetah, hopper, humanoid, sumo-ants, or walker2d"
);

#[cfg(any(
    all(feature = "ant", feature = "half-cheetah"),
    all(feature = "ant", feature = "hopper"),
    all(feature = "ant", feature = "humanoid"),
    all(feature = "ant", feature = "sumo-ants"),
    all(feature = "ant", feature = "walker2d"),
    all(feature = "half-cheetah", feature = "hopper"),
    all(feature = "half-cheetah", feature = "humanoid"),
    all(feature = "half-cheetah", feature = "sumo-ants"),
    all(feature = "half-cheetah", feature = "walker2d"),
    all(feature = "hopper", feature = "humanoid"),
    all(feature = "hopper", feature = "sumo-ants"),
    all(feature = "hopper", feature = "walker2d"),
    all(feature = "humanoid", feature = "sumo-ants"),
    all(feature = "humanoid", feature = "walker2d"),
    all(feature = "sumo-ants", feature = "walker2d"),
))]
compile_error!(
    "enable exactly one MuJoCo environment feature: ant, half-cheetah, hopper, humanoid, sumo-ants, or walker2d"
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
    feature = "humanoid"
))]
use modurl_mojoco::HumanoidV5 as SelectedEnvironment;
#[cfg(all(
    not(feature = "ant"),
    not(feature = "half-cheetah"),
    not(feature = "hopper"),
    not(feature = "humanoid"),
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
    feature = "humanoid"
))]
pub const ENVIRONMENT_NAME: &str = "Humanoid-v5";
#[cfg(all(
    not(feature = "ant"),
    not(feature = "half-cheetah"),
    not(feature = "hopper"),
    not(feature = "humanoid"),
    feature = "walker2d"
))]
pub const ENVIRONMENT_NAME: &str = "Walker2d-v5";
#[cfg(all(
    not(feature = "ant"),
    not(feature = "half-cheetah"),
    not(feature = "hopper"),
    not(feature = "humanoid"),
    not(feature = "walker2d"),
    feature = "sumo-ants"
))]
pub const ENVIRONMENT_NAME: &str = "SumoAnts";

#[cfg(not(feature = "sumo-ants"))]
pub fn build_environment(device: &Device) -> SelectedEnvironment {
    SelectedEnvironment::builder()
        .device(device)
        .build()
        .unwrap()
}

#[cfg(feature = "sumo-ants")]
pub fn build_environment(
    device: &Device,
) -> modurl::gym::StackedMultiGym<modurl_mojoco::SumoAnts, modurl_mojoco::SumoAntsInfo> {
    const GAME_COUNT: usize = 4;

    let games = (0..GAME_COUNT)
        .map(|game_index| {
            let builder = modurl_mojoco::SumoAnts::builder().device(device);
            #[cfg(feature = "rendering")]
            let builder = builder.render(game_index == 0);
            let mut game = builder.build().unwrap();
            game.seed(game_index as u64);
            game
        })
        .collect();
    modurl::gym::StackedMultiGym::try_new(games).unwrap()
}

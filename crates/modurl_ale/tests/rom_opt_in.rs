use std::path::PathBuf;

use candle_core::Device;
use modurl::gym::Gym;
use modurl_ale::{AtariGym, AtariObsType};

#[test]
fn user_supplied_rom_can_reset_and_step() {
    let Some(path) = std::env::var_os("MODURL_ALE_TEST_ROM") else {
        eprintln!("skipping: MODURL_ALE_TEST_ROM is not set");
        return;
    };
    let mut env = AtariGym::builder()
        .rom_path(PathBuf::from(path))
        .obs_type(AtariObsType::RAM)
        .device(Device::Cpu)
        .repeat_action_probability(0.0)
        .build()
        .unwrap();
    let state = env.reset().unwrap();
    assert_eq!(state.state.dims(), &[128]);
}

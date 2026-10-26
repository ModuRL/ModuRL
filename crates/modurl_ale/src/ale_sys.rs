pub(crate) use crate::bindings::root::{
    ALE_del as ale_del, ALE_new as ale_new, act, ale::ALEInterface as AleInterface, game_over,
    getEpisodeFrameNumber as get_episode_frame_number, getFrameNumber as get_frame_number,
    getMinimalActionSet as get_minimal_action_set, getMinimalActionSize as get_minimal_action_size,
    getRAM as get_ram, getRAMSize as get_ram_size, getScreen,
    getScreenGrayscale as get_screen_grayscale, getScreenHeight as get_screen_height,
    getScreenWidth as get_screen_width, lives, loadROM as load_rom, reset_game,
    setBool as set_bool, setFloat as set_float, setInt as set_int,
};

unsafe extern "C" {
    pub(crate) fn modurl_ale_get_rgb_palette(ale: *mut AleInterface, colors: *mut u32);
}

#[cfg(any(feature = "rendering", test))]
pub(crate) use crate::bindings::root::getScreenRGB;

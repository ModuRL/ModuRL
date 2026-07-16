#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]

pub(crate) use crate::bindings::root::{
    ALE_del, ALE_new, act, ale::ALEInterface, game_over, getEpisodeFrameNumber, getFrameNumber,
    getMinimalActionSet, getMinimalActionSize, getRAM, getRAMSize, getScreenGrayscale,
    getScreenHeight, getScreenRGB, getScreenWidth, lives, loadROM, reset_game, setBool, setFloat,
    setInt,
};

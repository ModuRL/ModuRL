use std::ffi::CStr;
use std::os::raw::c_int;
use std::ptr::null_mut;

use crate::ale_sys;

pub(crate) struct Ale {
    ptr: *mut ale_sys::ALEInterface,
    minimal_actions: Vec<i32>,
}

impl Ale {
    pub(crate) fn new() -> Self {
        let ptr = unsafe { ale_sys::ALE_new() };
        assert!(ptr != null_mut());
        Self {
            ptr,
            minimal_actions: vec![],
        }
    }

    pub(crate) fn set_int(&mut self, key: &CStr, value: i32) {
        unsafe { ale_sys::setInt(self.ptr, key.as_ptr(), value as c_int) };
    }

    pub(crate) fn set_float(&mut self, key: &CStr, value: f32) {
        unsafe { ale_sys::setFloat(self.ptr, key.as_ptr(), value) };
    }

    pub(crate) fn set_bool(&mut self, key: &CStr, value: bool) {
        unsafe { ale_sys::setBool(self.ptr, key.as_ptr(), value) };
    }

    pub(crate) fn load_rom_file(&mut self, rom_file: &CStr) {
        unsafe { ale_sys::loadROM(self.ptr, rom_file.as_ptr()) };
    }

    pub(crate) fn act(&mut self, action: i32) -> i32 {
        unsafe { ale_sys::act(self.ptr, action) }
    }

    pub(crate) fn is_game_over(&mut self) -> bool {
        unsafe { ale_sys::game_over(self.ptr) }
    }

    pub(crate) fn reset_game(&mut self) {
        unsafe { ale_sys::reset_game(self.ptr) };
    }

    pub(crate) fn minimal_action_set(&mut self) -> &[i32] {
        let size = unsafe { ale_sys::getMinimalActionSize(self.ptr) };
        assert!(size >= 0);
        self.minimal_actions.resize(size as usize, 0);
        unsafe { ale_sys::getMinimalActionSet(self.ptr, self.minimal_actions.as_mut_ptr()) };
        &self.minimal_actions
    }

    pub(crate) fn lives(&mut self) -> i32 {
        unsafe { ale_sys::lives(self.ptr) }
    }

    pub(crate) fn frame_number(&self) -> u32 {
        let frame_number = unsafe { ale_sys::getFrameNumber(self.ptr) };
        u32::try_from(frame_number).expect("ALE returned a negative frame number")
    }

    pub(crate) fn episode_frame_number(&self) -> u32 {
        let frame_number = unsafe { ale_sys::getEpisodeFrameNumber(self.ptr) };
        u32::try_from(frame_number).expect("ALE returned a negative episode frame number")
    }

    pub(crate) fn get_ram(&self, ram: &mut [u8]) {
        assert!(ram.len() >= self.ram_size());
        // SAFETY: `self.ptr` remains valid for `self`'s lifetime. In the pinned ALE revision,
        // `getRAM` only reads ALE's RAM and writes into the separate, sufficiently large buffer.
        unsafe { ale_sys::getRAM(self.ptr, ram.as_mut_ptr()) };
    }

    pub(crate) fn ram_size(&self) -> usize {
        // SAFETY: `self.ptr` remains valid for `self`'s lifetime. The pinned implementation only
        // reads the size of ALE's RAM; the C API is merely missing a const-qualified pointer.
        unsafe { ale_sys::getRAMSize(self.ptr) as usize }
    }

    pub(crate) fn screen_width(&self) -> usize {
        // SAFETY: `self.ptr` remains valid for `self`'s lifetime. The pinned implementation only
        // reads the screen width; the C API is merely missing a const-qualified pointer.
        unsafe { ale_sys::getScreenWidth(self.ptr) as usize }
    }

    pub(crate) fn screen_height(&self) -> usize {
        // SAFETY: `self.ptr` remains valid for `self`'s lifetime. The pinned implementation only
        // reads the screen height; the C API is merely missing a const-qualified pointer.
        unsafe { ale_sys::getScreenHeight(self.ptr) as usize }
    }

    pub(crate) fn get_screen_rgb(&self, screen_data: &mut [u8]) {
        assert!(screen_data.len() >= self.screen_width() * self.screen_height() * 3);
        // SAFETY: `self.ptr` remains valid for `self`'s lifetime. In the pinned ALE revision, this
        // reads screen/palette data and writes only to the separate, sufficiently large buffer.
        unsafe { ale_sys::getScreenRGB(self.ptr, screen_data.as_mut_ptr()) };
    }

    pub(crate) fn get_screen_grayscale(&self, screen_data: &mut [u8]) {
        assert!(screen_data.len() >= self.screen_width() * self.screen_height());
        // SAFETY: `self.ptr` remains valid for `self`'s lifetime. In the pinned ALE revision, this
        // reads screen/palette data and writes only to the separate, sufficiently large buffer.
        unsafe { ale_sys::getScreenGrayscale(self.ptr, screen_data.as_mut_ptr()) };
    }
}

impl Drop for Ale {
    fn drop(&mut self) {
        unsafe { ale_sys::ALE_del(self.ptr) };
    }
}

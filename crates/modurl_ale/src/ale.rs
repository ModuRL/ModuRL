use std::ffi::CStr;
use std::os::raw::c_int;

use crate::ale_sys;

pub(crate) struct Ale {
    ptr: *mut ale_sys::AleInterface,
    minimal_actions: Vec<i32>,
}

impl Ale {
    pub(crate) fn new() -> Self {
        // SAFETY: ALE_new takes no borrowed pointers and returns an owned interface, checked below.
        let ptr = unsafe { ale_sys::ale_new() };
        assert!(!ptr.is_null());
        Self {
            ptr,
            minimal_actions: vec![],
        }
    }

    pub(crate) fn set_int(&mut self, key: &CStr, value: i32) {
        // SAFETY: The interface is live and uniquely owned. The key is a valid C string for this call.
        unsafe { ale_sys::set_int(self.ptr, key.as_ptr(), value as c_int) };
    }

    pub(crate) fn set_float(&mut self, key: &CStr, value: f32) {
        // SAFETY: The interface is live and uniquely owned. The key is a valid C string for this call.
        unsafe { ale_sys::set_float(self.ptr, key.as_ptr(), value) };
    }

    pub(crate) fn set_bool(&mut self, key: &CStr, value: bool) {
        // SAFETY: The interface is live and uniquely owned. The key is a valid C string for this call.
        unsafe { ale_sys::set_bool(self.ptr, key.as_ptr(), value) };
    }

    pub(crate) fn load_rom_file(&mut self, rom_file: &CStr) {
        // SAFETY: The interface is live and uniquely owned. The ROM path is a valid C string for this call.
        unsafe { ale_sys::load_rom(self.ptr, rom_file.as_ptr()) };
    }

    pub(crate) fn act(&mut self, action: i32) -> i32 {
        // SAFETY: The interface is live and uniquely owned; the caller selects an action from ALE's minimal action set.
        unsafe { ale_sys::act(self.ptr, action) }
    }

    pub(crate) fn is_game_over(&mut self) -> bool {
        // SAFETY: The interface is live; this operation reads emulator status.
        unsafe { ale_sys::game_over(self.ptr) }
    }

    pub(crate) fn reset_game(&mut self) {
        // SAFETY: The interface is live and uniquely owned for the reset.
        unsafe { ale_sys::reset_game(self.ptr) };
    }

    pub(crate) fn minimal_action_set(&mut self) -> &[i32] {
        // SAFETY: The interface is live; this operation reads the minimal action count.
        let size = unsafe { ale_sys::get_minimal_action_size(self.ptr) };
        assert!(size >= 0);
        self.minimal_actions.resize(size as usize, 0);
        // SAFETY: The interface is live; minimal_actions has space for the action count returned above.
        unsafe { ale_sys::get_minimal_action_set(self.ptr, self.minimal_actions.as_mut_ptr()) };
        &self.minimal_actions
    }

    pub(crate) fn lives(&mut self) -> i32 {
        // SAFETY: The interface is live and this operation reads its remaining-life count.
        unsafe { ale_sys::lives(self.ptr) }
    }

    pub(crate) fn frame_number(&self) -> u32 {
        // SAFETY: The interface is live and this operation reads its frame counter.
        let frame_number = unsafe { ale_sys::get_frame_number(self.ptr) };
        u32::try_from(frame_number).expect("ALE returned a negative frame number")
    }

    pub(crate) fn episode_frame_number(&self) -> u32 {
        // SAFETY: The interface is live and this operation reads its frame counter.
        let frame_number = unsafe { ale_sys::get_episode_frame_number(self.ptr) };
        u32::try_from(frame_number).expect("ALE returned a negative episode frame number")
    }

    pub(crate) fn get_ram(&self, ram: &mut [u8]) {
        assert!(ram.len() >= self.ram_size());
        // SAFETY: `self.ptr` remains valid for `self`'s lifetime. In the pinned ALE revision,
        // `getRAM` only reads ALE's RAM and writes into the separate, sufficiently large buffer.
        unsafe { ale_sys::get_ram(self.ptr, ram.as_mut_ptr()) };
    }

    pub(crate) fn ram_size(&self) -> usize {
        // SAFETY: `self.ptr` remains valid for `self`'s lifetime. The pinned implementation only
        // reads the size of ALE's RAM; the C API is merely missing a const-qualified pointer.
        unsafe { ale_sys::get_ram_size(self.ptr) as usize }
    }

    pub(crate) fn screen_width(&self) -> usize {
        // SAFETY: `self.ptr` remains valid for `self`'s lifetime. The pinned implementation only
        // reads the screen width; the C API is merely missing a const-qualified pointer.
        unsafe { ale_sys::get_screen_width(self.ptr) as usize }
    }

    pub(crate) fn screen_height(&self) -> usize {
        // SAFETY: `self.ptr` remains valid for `self`'s lifetime. The pinned implementation only
        // reads the screen height; the C API is merely missing a const-qualified pointer.
        unsafe { ale_sys::get_screen_height(self.ptr) as usize }
    }

    pub(crate) fn get_screen_rgb(&self, screen_data: &mut [u8]) {
        assert!(screen_data.len() >= self.screen_width() * self.screen_height() * 3);
        // SAFETY: `self.ptr` remains valid for `self`'s lifetime. In the pinned ALE revision, this
        // reads screen/palette data and writes only to the separate, sufficiently large buffer.
        unsafe { ale_sys::get_screen_rgb(self.ptr, screen_data.as_mut_ptr()) };
    }

    pub(crate) fn get_screen_grayscale(&self, screen_data: &mut [u8]) {
        assert!(screen_data.len() >= self.screen_width() * self.screen_height());
        // SAFETY: `self.ptr` remains valid for `self`'s lifetime. In the pinned ALE revision, this
        // reads screen/palette data and writes only to the separate, sufficiently large buffer.
        unsafe { ale_sys::get_screen_grayscale(self.ptr, screen_data.as_mut_ptr()) };
    }
}

impl Drop for Ale {
    fn drop(&mut self) {
        // SAFETY: This interface was allocated by ALE_new and is freed exactly once when its owner is destroyed.
        unsafe { ale_sys::ale_del(self.ptr) };
    }
}

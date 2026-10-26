use burn_std::Bytes;
use std::ffi::CStr;
use std::os::raw::c_int;
use std::sync::Arc;

use crate::ale_sys;

pub(crate) struct Ale {
    ptr: *mut ale_sys::AleInterface,
    minimal_actions: Vec<i32>,
    rgb_palette: Arc<[u32; 256]>,
}

impl Ale {
    pub(crate) fn new() -> Self {
        // SAFETY: ALE_new takes no borrowed pointers and returns an owned interface, checked below.
        let ptr = unsafe { ale_sys::ale_new() };
        assert!(!ptr.is_null());
        Self {
            ptr,
            minimal_actions: vec![],
            rgb_palette: Arc::new([0; 256]),
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
        self.minimal_actions.clear();
        let mut palette = [0; 256];
        // SAFETY: The loaded interface is live and uniquely owned. The array has room for all 256 colors.
        unsafe { ale_sys::modurl_ale_get_rgb_palette(self.ptr, palette.as_mut_ptr()) };
        self.rgb_palette = Arc::new(palette);
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
        if self.minimal_actions.is_empty() {
            // SAFETY: The interface is live; this operation reads the minimal action count.
            let size = unsafe { ale_sys::get_minimal_action_size(self.ptr) };
            assert!(size >= 0);
            self.minimal_actions.resize(size as usize, 0);
            // SAFETY: The interface is live; minimal_actions has space for the action count returned above.
            unsafe { ale_sys::get_minimal_action_set(self.ptr, self.minimal_actions.as_mut_ptr()) };
        }
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

    /// Captures indexed pixels `[height * width]` as deferred RGB bytes `[height * width * 3]`.
    /// The snapshot owns its pixels and retains the loaded ROM's palette after the interface advances or is destroyed.
    pub(crate) fn screen_rgb_bytes(&self) -> Bytes {
        let mut pixels = vec![0; self.screen_width() * self.screen_height()];
        // SAFETY: The interface is live. The output has one byte for each screen pixel; the call retains no pointer.
        unsafe { ale_sys::getScreen(self.ptr, pixels.as_mut_ptr()) };
        crate::rgb_frame::snapshot(pixels, self.rgb_palette.clone())
    }

    #[cfg(any(feature = "rendering", test))]
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
        unsafe { ale_sys::get_screen_grayscale(self.ptr, screen_data.as_mut_ptr()) };
    }
}

impl Drop for Ale {
    fn drop(&mut self) {
        // SAFETY: This interface was allocated by ALE_new and is freed exactly once when its owner is destroyed.
        unsafe { ale_sys::ale_del(self.ptr) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_std::{AccessError, Reader};
    use std::ffi::CString;

    #[test]
    fn rgb_snapshots_match_native_after_steps_reload_and_interface_drop() {
        let Some(path) = std::env::var_os("MODURL_ALE_TEST_ROM") else {
            eprintln!("skipping: MODURL_ALE_TEST_ROM is not set");
            return;
        };
        let mut paths = vec![path];
        if let Some(second) = std::env::var_os("MODURL_ALE_TEST_SECOND_ROM") {
            paths.push(second);
        }
        let snapshots = {
            let mut ale = Ale::new();
            ale.set_int(c"random_seed", 42);
            ale.set_float(c"repeat_action_probability", 0.0);
            ale.set_bool(c"color_averaging", false);
            let mut snapshots = Vec::new();
            for path in paths {
                let path = CString::new(path.to_str().unwrap()).unwrap();
                ale.load_rom_file(&path);
                // SAFETY: The uniquely owned interface has a loaded ROM.
                let count = unsafe { ale_sys::get_minimal_action_size(ale.ptr) };
                assert!(count > 0);
                let mut actions = vec![0; count as usize];
                // SAFETY: The output has space for the action count from this same loaded ROM.
                unsafe { ale_sys::get_minimal_action_set(ale.ptr, actions.as_mut_ptr()) };
                assert_eq!(ale.minimal_action_set(), actions);
                for frame in 0..64 {
                    ale.act(actions[frame % actions.len()]);
                    let bytes = ale.screen_rgb_bytes();
                    let mut expected = vec![0; ale.screen_width() * ale.screen_height() * 3];
                    ale.get_screen_rgb(&mut expected);
                    assert!(matches!(
                        bytes.read(Reader::new().no_copy()),
                        Err(AccessError::WouldCopy)
                    ));
                    snapshots.push((bytes, expected));
                    if frame == 31 || ale.is_game_over() {
                        ale.reset_game();
                    }
                }
            }
            snapshots
        };
        for (bytes, expected) in snapshots {
            assert_eq!(bytes.read(Reader::new()).unwrap(), expected);
        }
    }
}

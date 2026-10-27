use modurl_ale::{AtariGym, Ram};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

struct TinyRom {
    directory: PathBuf,
    path: PathBuf,
}

impl TinyRom {
    fn new() -> Self {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
        let directory = std::env::temp_dir().join(format!(
            "modurl-ale-tiny-rom-{}-{}",
            std::process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&directory).expect("create ROM directory");
        let path = directory.join("pong.bin");
        fs::write(&path, [0xFF]).expect("write one-byte ROM");
        Self { directory, path }
    }
}

impl Drop for TinyRom {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

#[test]
fn one_byte_rom_builder_returns_without_crashing() {
    let rom = TinyRom::new();
    let result = AtariGym::<Ram>::builder()
        .rom_path(rom.path.clone())
        .build();
    eprintln!(
        "one-byte ROM builder returned {}",
        if result.is_ok() { "Ok" } else { "Err" }
    );
}

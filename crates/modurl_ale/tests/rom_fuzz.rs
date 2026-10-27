use burn::tensor::{Device, Tensor};
use modurl::gym::Gym;
use modurl_ale::{AtariGym, Ram};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::{
    env, fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

const DEFAULT_SEED: u64 = 0x4d4f_4455_524c;
const DEFAULT_ITERS: usize = 32;
const ROM_SIZES: &[usize] = &[
    1,
    2 * 1024,
    4 * 1024,
    4097,
    8 * 1024,
    10 * 1024,
    12 * 1024,
    16 * 1024,
    32 * 1024,
    64 * 1024,
    128 * 1024,
];

struct TempRom {
    directory: PathBuf,
    path: PathBuf,
    keep: bool,
}

impl TempRom {
    fn new(bytes: &[u8]) -> Self {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
        let directory = std::env::temp_dir().join(format!(
            "modurl-ale-rom-fuzz-{}-{}",
            std::process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&directory).expect("create ROM directory");
        let path = directory.join("pong.bin");
        fs::write(&path, bytes).expect("write random ROM");
        let keep = env::var_os("MODURL_FUZZ_KEEP_ROMS").is_some();
        Self {
            directory,
            path,
            keep,
        }
    }
}

impl Drop for TempRom {
    fn drop(&mut self) {
        if !self.keep {
            let _ = fs::remove_dir_all(&self.directory);
        }
    }
}

#[test]
fn deterministic_random_rom_fuzz() {
    let seed = env::var("MODURL_FUZZ_SEED")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_SEED);
    let iterations = env::var("MODURL_FUZZ_ITERS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_ITERS);
    let trace = env::var_os("MODURL_FUZZ_TRACE").is_some();
    let mut rng = StdRng::seed_from_u64(seed);

    for iteration in 0..iterations {
        let size = ROM_SIZES[rng.random_range(0..ROM_SIZES.len())];
        let mut bytes: Vec<u8> = (0..size).map(|_| rng.random()).collect();
        if size == 128 * 1024 {
            bytes[0x100..0x104].copy_from_slice(&[0x85, 0x3E, 0xA9, 0x00]);
        }
        let vector_index = if size >= 2 * 1024 {
            let address = (0x10000 - size.min(4096) + 0x80) as u16;
            let boot = [
                0xA9,
                0x00,
                0x85,
                0x96,
                0xA9,
                0x02,
                0x85,
                0x00,
                0xA2,
                0x28,
                0xCA,
                0xD0,
                0xFD,
                0xA9,
                0x00,
                0x85,
                0x00,
                0x4C,
                address as u8,
                (address >> 8) as u8,
            ];
            let last_bank = size - size.min(4096);
            let mut bank_starts = Vec::new();
            if size >= 8192 || size.is_multiple_of(4096) {
                bank_starts.extend((0..=size.saturating_sub(4096)).step_by(4096));
            }
            if !bank_starts.contains(&last_bank) {
                bank_starts.push(last_bank);
            }
            for bank_start in bank_starts {
                bytes[bank_start + 0x80..bank_start + 0x80 + boot.len()].copy_from_slice(&boot);
                if size >= 8192 && bank_start + 0x1000 <= size {
                    let index = bank_start + 0xFFC;
                    bytes[index..index + 2].copy_from_slice(&address.to_le_bytes());
                }
            }
            let index = size - 4;
            bytes[index..index + 2].copy_from_slice(&address.to_le_bytes());
            Some(index)
        } else {
            None
        };
        let rom = TempRom::new(&bytes);
        let repeat_action_probability = rng.random_range(0.0_f32..=1.0);
        let frame_skip = rng.random_range(1..=3);
        let emulator_seed = rng.random_range(0..=i32::MAX);
        if trace {
            let vector = vector_index.map(|index| &bytes[index..index + 2]);
            eprintln!(
                "ale seed={seed} iteration={iteration} size={size} frame_skip={frame_skip} repeat={repeat_action_probability:?} emulator_seed={emulator_seed} reset_vector={vector:?} rom={}",
                rom.path.display()
            );
        }

        if trace {
            eprintln!("ale seed={seed} iteration={iteration} stage=build");
        }
        let Ok(mut env) = AtariGym::<Ram>::builder()
            .rom_path(rom.path.clone())
            .random_seed(emulator_seed)
            .repeat_action_probability(repeat_action_probability)
            .build()
        else {
            if trace {
                eprintln!("ale seed={seed} iteration={iteration} stage=build-error");
            }
            continue;
        };
        if trace {
            eprintln!("ale seed={seed} iteration={iteration} stage=built");
        }
        env.set_frame_skip(frame_skip);
        if trace {
            eprintln!("ale seed={seed} iteration={iteration} stage=reset");
        }
        if env.reset().is_err() {
            if trace {
                eprintln!("ale seed={seed} iteration={iteration} stage=reset-error");
            }
            continue;
        }
        if trace {
            eprintln!("ale seed={seed} iteration={iteration} stage=reset-done");
        }
        let action_count = env.minimal_action_set().len();
        if action_count == 0 {
            continue;
        }
        if !step_random_actions(&mut env, 64, action_count, &mut rng) {
            continue;
        }
        if env.reset().is_err() {
            continue;
        }
        let _ = step_random_actions(&mut env, 36, action_count, &mut rng);
    }
}

fn step_random_actions(
    env: &mut AtariGym<Ram>,
    steps: usize,
    action_count: usize,
    rng: &mut StdRng,
) -> bool {
    for _ in 0..steps {
        let action = rng.random_range(0..action_count) as i32;
        if env
            .step(Tensor::from_data([action], &Device::flex()))
            .is_err()
        {
            return false;
        }
    }
    true
}

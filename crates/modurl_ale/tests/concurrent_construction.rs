use burn::tensor::{Device, Tensor};
use modurl::gym::Gym;
use modurl_ale::{AtariGym, Ram};
use std::{
    env, fs,
    path::PathBuf,
    process,
    sync::{
        Arc, Barrier,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
};

const STEPS: usize = 64;
const CONCURRENT_THREADS: usize = 16;
const BUILDS_PER_THREAD: usize = 10;
static ROM_COUNTER: AtomicUsize = AtomicUsize::new(0);

struct RomFixture {
    directory: PathBuf,
    path: PathBuf,
}

impl Drop for RomFixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

#[derive(Debug, PartialEq)]
struct StepSample {
    ram: Vec<u8>,
    reward: f32,
    done: bool,
    frame_number: u32,
}

fn synthetic_rom() -> RomFixture {
    let directory = loop {
        let id = ROM_COUNTER.fetch_add(1, Ordering::Relaxed);
        let directory =
            env::temp_dir().join(format!("modurl-ale-threading-{}-{id}", process::id()));
        match fs::create_dir(&directory) {
            Ok(()) => break directory,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => panic!("create ROM directory: {error}"),
        }
    };
    let path = directory.join("pong.bin");
    // This synthetic image has a NOP-filled ROM and is not a commercial ROM.
    let mut rom = vec![0xEA; 4096];
    rom[..17].copy_from_slice(&[
        0xA2, 0x80, // LDX #$80
        0xA9, 0x00, // LDA #$00
        0x95, 0x00, // STA $00,X
        0xE8, // INX
        0xD0, 0xFB, // BNE $F004
        0xF8, // SED
        0xA9, 0x49, // LDA #$49
        0x69, 0x51, // ADC #$51
        0x4C, 0x09, 0xF0, // JMP $F009
    ]);
    rom[0xFFC..0x1000].copy_from_slice(&[0x00, 0xF0, 0x00, 0xF0]);
    fs::write(&path, rom).expect("write synthetic ROM");
    RomFixture { directory, path }
}

fn build(rom_path: &std::path::Path, seed: i32) -> AtariGym<Ram> {
    AtariGym::<Ram>::builder()
        .rom_path(rom_path.to_path_buf())
        .random_seed(seed)
        .repeat_action_probability(0.25)
        .build()
        .expect("construct AtariGym")
}

fn trajectory(env: &mut AtariGym<Ram>) -> Vec<StepSample> {
    let action_count = env.minimal_action_set().len();
    assert!(action_count > 0);
    (0..STEPS)
        .map(|index| {
            let step = env
                .step(Tensor::from_data(
                    [(index % action_count) as i32],
                    &Device::flex(),
                ))
                .expect("step AtariGym");
            StepSample {
                ram: step.observation.into_data().try_to_vec::<u8>().unwrap(),
                reward: step.reward,
                done: step.done,
                frame_number: step.info.frame_number,
            }
        })
        .collect()
}

fn reference_trajectory(rom_path: &std::path::Path, seed: i32) -> Vec<StepSample> {
    let mut env = build(rom_path, seed);
    env.reset().expect("reset AtariGym");
    trajectory(&mut env)
}

fn assert_trajectory_matches(actual: &[StepSample], expected: &[StepSample], seed: i32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual_step, expected_step)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual_step, expected_step,
            "trajectory diverged for seed {seed} at step {index}"
        );
    }
}

#[test]
fn concurrent_construction_matches_seeded_reference_trajectories() {
    let rom = synthetic_rom();
    let references: Arc<Vec<_>> = Arc::new(
        (0..8)
            .map(|seed| reference_trajectory(&rom.path, seed))
            .collect(),
    );
    let rom_path = Arc::new(rom.path.clone());
    let barrier = Arc::new(Barrier::new(CONCURRENT_THREADS));
    let workers: Vec<_> = (0..CONCURRENT_THREADS)
        .map(|thread_index| {
            let references = Arc::clone(&references);
            let rom_path = Arc::clone(&rom_path);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                let seed = thread_index % references.len();
                barrier.wait();
                for _ in 0..BUILDS_PER_THREAD {
                    let mut env = build(&rom_path, seed as i32);
                    env.reset().expect("reset AtariGym");
                    assert_trajectory_matches(
                        &trajectory(&mut env),
                        &references[seed],
                        seed as i32,
                    );
                }
            })
        })
        .collect();
    let mut panicked_workers = 0;
    for worker in workers {
        panicked_workers += usize::from(worker.join().is_err());
    }
    assert_eq!(panicked_workers, 0, "construction workers panicked");
}

#[test]
fn construction_does_not_disturb_stepping_on_a_live_environment() {
    const CONSTRUCTOR_THREADS: usize = CONCURRENT_THREADS - 1;

    let rom = synthetic_rom();
    let reference = Arc::new(reference_trajectory(&rom.path, 0));
    let rom_path = Arc::new(rom.path.clone());
    let barrier = Arc::new(Barrier::new(CONCURRENT_THREADS));
    let step_rom_path = Arc::clone(&rom_path);
    let step_reference = Arc::clone(&reference);
    let step_barrier = Arc::clone(&barrier);
    let stepping_worker = thread::spawn(move || {
        let mut env = build(&step_rom_path, 0);
        env.reset().expect("reset AtariGym");
        step_barrier.wait();
        assert_trajectory_matches(&trajectory(&mut env), &step_reference, 0);
    });
    let constructors: Vec<_> = (0..CONSTRUCTOR_THREADS)
        .map(|thread_index| {
            let rom_path = Arc::clone(&rom_path);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                for build_index in 0..BUILDS_PER_THREAD {
                    let seed = (thread_index + build_index) as i32;
                    let mut env = build(&rom_path, seed);
                    env.reset().expect("reset AtariGym");
                    drop(env);
                }
            })
        })
        .collect();
    let mut panicked_workers = 0;
    for constructor in constructors {
        panicked_workers += usize::from(constructor.join().is_err());
    }
    let stepping_panicked = stepping_worker.join().is_err();
    assert_eq!(panicked_workers, 0, "constructor workers panicked");
    assert!(!stepping_panicked, "stepping worker panicked");
}

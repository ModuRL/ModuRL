use burn::tensor::{Device, Tensor, TensorData};
use modurl::{gym::Gym, prelude::TensorBatchTypes};
use modurl_mujoco::{CustomMujoco, HalfCheetahV5, HopperV5, MujocoState, MujocoTask, TaskStep};
use std::{
    fmt::Debug,
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc, Barrier,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
};

const SLIDE_XML: &str = r#"
<mujoco model="slide">
  <option timestep="0.01" gravity="0 0 0"/>
  <worldbody>
    <body name="slider">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slide" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"#;

const STEPS: usize = 64;

struct TempModel {
    directory: PathBuf,
    path: PathBuf,
}

impl TempModel {
    fn new() -> Self {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
        let directory = std::env::temp_dir().join(format!(
            "modurl-mujoco-concurrency-{}-{}",
            std::process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("model.xml");
        fs::write(&path, SLIDE_XML).unwrap();
        Self { directory, path }
    }
}

impl Drop for TempModel {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

struct SlideTask;

impl MujocoTask for SlideTask {
    fn observation(&self, state: &MujocoState) -> Vec<f64> {
        [state.qpos.as_slice(), state.qvel.as_slice()].concat()
    }

    fn transition(&mut self, previous: &MujocoState, next: &MujocoState, _: &[f64]) -> TaskStep {
        TaskStep {
            reward: (next.qpos[0] - previous.qpos[0]) as f32,
            done: false,
            truncated: false,
        }
    }
}

#[derive(Debug, PartialEq)]
struct Transition {
    observation: Vec<f32>,
    reward: f32,
    done: bool,
    truncated: bool,
}

fn trajectory<G: Gym>(env: &mut G, seed: u64, action_dim: usize) -> Vec<Transition>
where
    G::Error: Debug,
    G::ActionSpace: TensorBatchTypes<Item = Tensor<1>>,
    G::ObservationSpace: TensorBatchTypes<Item = Tensor<1>>,
{
    let mut result = Vec::with_capacity(STEPS + 1);
    let reset = env.reset().unwrap();
    result.push(Transition {
        observation: reset.observation.into_data().try_to_vec::<f32>().unwrap(),
        reward: 0.0,
        done: false,
        truncated: false,
    });
    for step in 0..STEPS {
        let actions = (0..action_dim)
            .map(|index| ((seed as usize * 31 + step * 17 + index * 13) % 101) as f32 / 50.0 - 1.0)
            .collect::<Vec<_>>();
        let action = Tensor::from_data(TensorData::new(actions, [action_dim]), &Device::flex());
        let outcome = env.step(action).unwrap();
        result.push(Transition {
            observation: outcome.observation.into_data().try_to_vec::<f32>().unwrap(),
            reward: outcome.reward,
            done: outcome.done,
            truncated: outcome.truncated,
        });
    }
    result
}

fn custom(path: &Path) -> CustomMujoco<SlideTask> {
    CustomMujoco::builder()
        .path(path)
        .task(SlideTask)
        .observation_dim(2)
        .build()
        .unwrap()
}

fn half_cheetah_trajectory(seed: u64) -> Vec<Transition> {
    let mut env = HalfCheetahV5::builder().build().unwrap();
    env.seed(seed);
    trajectory(&mut env, seed, 6)
}

fn hopper_trajectory(seed: u64) -> Vec<Transition> {
    let mut env = HopperV5::builder().build().unwrap();
    env.seed(seed);
    trajectory(&mut env, seed, 3)
}

fn custom_trajectory(path: &Path, seed: u64) -> Vec<Transition> {
    let mut env = custom(path);
    env.seed(seed);
    trajectory(&mut env, seed, 1)
}

struct References {
    half_cheetah: Vec<Transition>,
    custom: Vec<Transition>,
    hopper: Vec<Transition>,
}

#[test]
fn concurrent_construction_matches_seeded_sequential_trajectories() {
    let model = TempModel::new();
    let references = Arc::new(
        (0..8)
            .map(|seed| References {
                half_cheetah: half_cheetah_trajectory(seed),
                custom: custom_trajectory(&model.path, seed),
                hopper: hopper_trajectory(seed),
            })
            .collect::<Vec<_>>(),
    );
    let barrier = Arc::new(Barrier::new(16));
    let mut threads = Vec::new();

    for worker in 0..16 {
        let barrier = Arc::clone(&barrier);
        let references = Arc::clone(&references);
        let path = model.path.clone();
        threads.push(thread::spawn(move || {
            barrier.wait();
            let seed_index = worker % references.len();
            let seed = seed_index as u64;
            let reference = &references[seed_index];
            for _ in 0..8 {
                let mut half = HalfCheetahV5::builder().build().unwrap();
                half.seed(seed);
                assert_eq!(trajectory(&mut half, seed, 6), reference.half_cheetah);

                let mut custom = custom(&path);
                custom.seed(seed);
                assert_eq!(trajectory(&mut custom, seed, 1), reference.custom);

                let mut hopper = HopperV5::builder().build().unwrap();
                hopper.seed(seed);
                assert_eq!(trajectory(&mut hopper, seed, 3), reference.hopper);
            }
        }));
    }
    for thread in threads {
        thread.join().unwrap();
    }
}

#[test]
fn construction_and_dropping_do_not_change_a_long_lived_trajectory() {
    let model = TempModel::new();
    let seed = 41;
    let expected = half_cheetah_trajectory(seed);
    let mut long_lived = HalfCheetahV5::builder().build().unwrap();
    long_lived.seed(seed);

    let barrier = Arc::new(Barrier::new(16));
    let mut workers = Vec::new();
    let step_barrier = Arc::clone(&barrier);
    let trajectory_thread = thread::spawn(move || {
        step_barrier.wait();
        trajectory(&mut long_lived, seed, 6)
    });

    for worker in 0..15 {
        let barrier = Arc::clone(&barrier);
        let path = model.path.clone();
        workers.push(thread::spawn(move || {
            barrier.wait();
            for repeat in 0..8 {
                let mut half = HalfCheetahV5::builder().build().unwrap();
                half.seed((worker + repeat) as u64);
                half.reset().unwrap();

                let mut custom = custom(&path);
                custom.seed((worker + repeat) as u64);
                custom.reset().unwrap();

                let mut hopper = HopperV5::builder().build().unwrap();
                hopper.seed((worker + repeat) as u64);
                hopper.reset().unwrap();
            }
        }));
    }

    assert_eq!(trajectory_thread.join().unwrap(), expected);
    for thread in workers {
        thread.join().unwrap();
    }
}

#[test]
fn editing_one_shared_model_does_not_affect_peer_trajectories() {
    let model = TempModel::new();
    let seed = 73;
    let expected = Arc::new(custom_trajectory(&model.path, seed));
    let mut editor = custom(&model.path);
    editor.seed(seed);
    let peers = (0..5)
        .map(|_| {
            let mut env = custom(&model.path);
            env.seed(seed);
            env
        })
        .collect::<Vec<_>>();
    let first_model = peers[0].model() as *const _;
    assert!(
        std::ptr::eq(first_model, editor.model() as *const _),
        "the editing environment should share the compiled model before editing"
    );
    assert!(
        peers
            .iter()
            .all(|env| std::ptr::eq(first_model, env.model() as *const _)),
        "path-loaded environments should share their compiled model"
    );

    let barrier = Arc::new(Barrier::new(6));
    let editor_barrier = Arc::clone(&barrier);
    let editor_thread = thread::spawn(move || {
        editor_barrier.wait();
        for iteration in 0..32 {
            let gravity = if iteration % 2 == 0 { -0.1 } else { 0.0 };
            let friction = 0.2 + iteration as f64 * 0.001;
            editor
                .edit_model(|model| {
                    model.opt_mut().gravity = [0.0, 0.0, gravity];
                    model.geom_friction_mut()[0][0] = friction;
                    Ok(())
                })
                .unwrap();
        }
    });

    let peer_threads = peers
        .into_iter()
        .map(|mut env| {
            let barrier = Arc::clone(&barrier);
            let expected = Arc::clone(&expected);
            thread::spawn(move || {
                barrier.wait();
                assert_eq!(trajectory(&mut env, seed, 1), *expected);
            })
        })
        .collect::<Vec<_>>();

    editor_thread.join().unwrap();
    for thread in peer_threads {
        thread.join().unwrap();
    }
}

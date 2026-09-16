use candle_core::Device;
use modurl::gym::Gym;
use modurl_mujoco::{CustomMujoco, MujocoState, MujocoTask, TaskStep};

struct SlideTask;

impl MujocoTask for SlideTask {
    fn observation(&self, state: &MujocoState) -> Vec<f64> {
        [state.qpos.as_slice(), state.qvel.as_slice()].concat()
    }

    fn transition(&mut self, previous: &MujocoState, next: &MujocoState, _: &[f64]) -> TaskStep {
        TaskStep {
            reward: (next.qpos[0] - previous.qpos[0]) as f32,
            done: next.qpos[0] > 0.1,
            truncated: false,
        }
    }
}

#[test]
fn xml_path_and_task_reward_work() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/assets/slide.xml");
    let mut env = CustomMujoco::from_xml_path(path, 2, &Device::Cpu, SlideTask, 2).unwrap();
    assert_eq!(env.actuator_count(), 1);
    assert_eq!(env.reset().unwrap().state.dims(), &[2]);
    let action = candle_core::Tensor::new(&[1.0_f32], &Device::Cpu).unwrap();
    let step = env.step(action).unwrap();
    assert!(step.reward > 0.0);
    assert_eq!(step.state.dims(), &[2]);
}

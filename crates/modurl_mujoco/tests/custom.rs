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

struct ContactTask;

impl MujocoTask for ContactTask {
    fn observation(&self, state: &MujocoState) -> Vec<f64> {
        let pair_found = state.contact_body_pairs.iter().any(|pair| {
            (pair[0] == 2 && pair[1] == 3) || (pair[0] == 3 && pair[1] == 2)
        });
        assert_eq!(state.body_parent_ids[2], 1);
        assert_eq!(state.body_parent_ids[3], 1);
        vec![f64::from(pair_found)]
    }

    fn transition(&mut self, _: &MujocoState, _: &MujocoState, _: &[f64]) -> TaskStep {
        TaskStep { reward: 0.0, done: false, truncated: false }
    }
}

#[test]
fn exposes_robot_self_contacts() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/assets/self_contact.xml");
    let mut env = CustomMujoco::from_xml_path(path, 1, &Device::Cpu, ContactTask, 1).unwrap();
    let obs = env.reset().unwrap().state.to_vec1::<f32>().unwrap();
    assert_eq!(obs, vec![1.0]);
}

struct GroundForceTask;

impl MujocoTask for GroundForceTask {
    fn observation(&self, state: &MujocoState) -> Vec<f64> {
        state.world_contact_forces[1].to_vec()
    }

    fn transition(&mut self, _: &MujocoState, _: &MujocoState, _: &[f64]) -> TaskStep {
        TaskStep { reward: 0.0, done: false, truncated: false }
    }
}

#[test]
fn exposes_ground_contact_force_vector() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/assets/ground_contact.xml");
    let mut env = CustomMujoco::from_xml_path(path, 10, &Device::Cpu, GroundForceTask, 3).unwrap();
    env.reset().unwrap();
    let action = candle_core::Tensor::new(&[0.0_f32], &Device::Cpu).unwrap();
    let force = env.step(action).unwrap().state.to_vec1::<f32>().unwrap();
    assert!(force[2] > 0.0, "floor reaction should point upward: {force:?}");
    assert!(force[0].abs() < 1e-5 && force[1].abs() < 1e-5);
}

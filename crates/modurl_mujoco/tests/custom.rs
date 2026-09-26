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
    let mut env = CustomMujoco::builder()
        .path(path)
        .frame_skip(2)
        .device(&Device::Cpu)
        .task(SlideTask)
        .observation_dim(2)
        .build()
        .unwrap();
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
        let pair_found = state
            .contact_body_pairs
            .iter()
            .any(|pair| (pair[0] == 2 && pair[1] == 3) || (pair[0] == 3 && pair[1] == 2));
        assert_eq!(state.body_parent_ids[2], 1);
        assert_eq!(state.body_parent_ids[3], 1);
        vec![f64::from(pair_found)]
    }

    fn transition(&mut self, _: &MujocoState, _: &MujocoState, _: &[f64]) -> TaskStep {
        TaskStep {
            reward: 0.0,
            done: false,
            truncated: false,
        }
    }
}

#[test]
fn exposes_robot_self_contacts() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/assets/self_contact.xml");
    let mut env = CustomMujoco::builder()
        .path(path)
        .frame_skip(1)
        .device(&Device::Cpu)
        .task(ContactTask)
        .observation_dim(1)
        .build()
        .unwrap();
    let obs = env.reset().unwrap().state.to_vec1::<f32>().unwrap();
    assert_eq!(obs, vec![1.0]);
}

struct GroundForceTask;

impl MujocoTask for GroundForceTask {
    fn observation(&self, state: &MujocoState) -> Vec<f64> {
        state.world_contact_forces[1].to_vec()
    }

    fn transition(&mut self, _: &MujocoState, _: &MujocoState, _: &[f64]) -> TaskStep {
        TaskStep {
            reward: 0.0,
            done: false,
            truncated: false,
        }
    }
}

#[test]
fn exposes_ground_contact_force_vector() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/assets/ground_contact.xml"
    );
    let mut env = CustomMujoco::builder()
        .path(path)
        .frame_skip(10)
        .device(&Device::Cpu)
        .task(GroundForceTask)
        .observation_dim(3)
        .build()
        .unwrap();
    env.reset().unwrap();
    let action = candle_core::Tensor::new(&[0.0_f32], &Device::Cpu).unwrap();
    let force = env.step(action).unwrap().state.to_vec1::<f32>().unwrap();
    assert!(
        force[2] > 0.0,
        "floor reaction should point upward: {force:?}"
    );
    assert!(force[0].abs() < 1e-5 && force[1].abs() < 1e-5);
}

#[test]
fn builder_rejects_invalid_configuration() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/assets/slide.xml");
    for (frame_skip, observation_dim) in [(0, 2), (1, 0), (1, 3)] {
        let result = CustomMujoco::builder()
            .path(path)
            .task(SlideTask)
            .frame_skip(frame_skip)
            .observation_dim(observation_dim)
            .build();
        assert!(matches!(
            result,
            Err(modurl_mujoco::MujocoError::InvalidInput(_))
        ));
    }
}

#[test]
fn builder_defaults_work() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/assets/slide.xml");
    let mut env = CustomMujoco::builder()
        .path(path)
        .task(SlideTask)
        .observation_dim(2)
        .build()
        .unwrap();
    env.reset().unwrap();
    assert!(
        env.step(candle_core::Tensor::new(&[1.0_f32], &Device::Cpu).unwrap())
            .unwrap()
            .reward
            > 0.0
    );
}

struct DisturbTask(std::rc::Rc<std::cell::Cell<usize>>);

impl MujocoTask for DisturbTask {
    fn observation(&self, state: &MujocoState) -> Vec<f64> {
        [state.qpos.as_slice(), state.qvel.as_slice()].concat()
    }

    fn before_step(&mut self, state: &mut MujocoState) {
        self.0.set(self.0.get() + 1);
        state.qvel[0] += 1.0;
    }

    fn transition(&mut self, _: &MujocoState, _: &MujocoState, _: &[f64]) -> TaskStep {
        TaskStep {
            reward: 0.0,
            done: false,
            truncated: false,
        }
    }
}

#[test]
fn invalid_action_does_not_apply_pre_step_disturbance() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/assets/slide.xml");
    let calls = std::rc::Rc::new(std::cell::Cell::new(0));
    let mut env = CustomMujoco::builder()
        .path(path)
        .frame_skip(1)
        .device(&Device::Cpu)
        .task(DisturbTask(calls.clone()))
        .observation_dim(2)
        .build()
        .unwrap();
    env.reset().unwrap();
    let invalid = candle_core::Tensor::new(&[f32::NAN], &Device::Cpu).unwrap();
    assert!(env.step(invalid).is_err());
    assert_eq!(calls.get(), 0);
}

#[test]
fn runtime_model_edits_change_dynamics_and_persist_across_reset() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/assets/slide.xml");
    let mut env = CustomMujoco::builder()
        .path(path)
        .task(SlideTask)
        .observation_dim(2)
        .build()
        .unwrap();
    env.reset().unwrap();
    let action = candle_core::Tensor::new(&[1.0_f32], &Device::Cpu).unwrap();
    let normal = env.step(action.clone()).unwrap();
    env.edit_model(|model| {
        model.body_mass_mut()[1] *= 2.0;
        model.body_inertia_mut()[1] = model.body_inertia()[1].map(|v| v * 2.0);
        Ok(())
    })
    .unwrap();
    env.reset().unwrap();
    let heavier = env.step(action).unwrap();
    assert!(heavier.reward > 0.0 && heavier.reward < normal.reward);
    assert_eq!(env.model().body_mass()[1], 2.0);
}

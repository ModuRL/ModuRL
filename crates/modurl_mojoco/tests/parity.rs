use candle_core::{Device, Tensor};
use modurl::gym::Gym;
use modurl_mojoco::{
    AntV5, AntV5Info, HalfCheetahV5, HopperV5, HumanoidV5, HumanoidV5Info, MujocoError, Walker2dV5,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    gymnasium_version: String,
    mujoco_version: String,
    environment_id: String,
    qpos: Vec<f64>,
    qvel: Vec<f64>,
    actions: Vec<Vec<f32>>,
    states: Vec<ExactState>,
    outputs: Vec<ExpectedStep>,
}

#[derive(Deserialize)]
struct ExactState {
    qpos: Vec<f64>,
    qvel: Vec<f64>,
}

#[derive(Deserialize)]
struct ExpectedStep {
    observation: Vec<f64>,
    reward: f64,
    terminated: bool,
    truncated: bool,
}

trait ParityEnvironment {
    /// Sets flat MuJoCo position and velocity arrays and returns one flat
    /// observation tensor shaped `[observation_size]`.
    fn set_exact_state(&mut self, qpos: &[f64], qvel: &[f64]) -> Result<Tensor, MujocoError>;
    /// Steps with one actuator tensor shaped `[action_size]` and returns the
    /// flat observation tensor shaped `[observation_size]` plus transition
    /// scalars.
    fn parity_step(&mut self, action: Tensor) -> Result<(Tensor, f32, bool, bool), MujocoError>;
}

macro_rules! impl_parity_environment {
    ($environment:ty, $info:ty) => {
        impl ParityEnvironment for $environment {
            /// Sets flat MuJoCo state arrays and returns an observation shaped
            /// `[observation_size]`.
            fn set_exact_state(
                &mut self,
                qpos: &[f64],
                qvel: &[f64],
            ) -> Result<Tensor, MujocoError> {
                self.set_state(qpos, qvel)
            }

            /// Steps with `action` shaped `[action_size]` and returns an
            /// observation shaped `[observation_size]` plus transition data.
            fn parity_step(
                &mut self,
                action: Tensor,
            ) -> Result<(Tensor, f32, bool, bool), MujocoError> {
                let step = <Self as Gym<$info>>::step(self, action)?;
                Ok((step.state, step.reward, step.done, step.truncated))
            }
        }
    };
}

impl_parity_environment!(HalfCheetahV5, ());
impl_parity_environment!(AntV5, AntV5Info);
impl_parity_environment!(HopperV5, ());
impl_parity_environment!(HumanoidV5, HumanoidV5Info);
impl_parity_environment!(Walker2dV5, ());

fn check_parity<E: ParityEnvironment>(
    fixture_json: &str,
    mut environment: E,
    observation_tolerance: f64,
    reward_tolerance: f64,
    contact_observation_start: Option<usize>,
) {
    let fixture: Fixture = serde_json::from_str(fixture_json).unwrap();
    assert_eq!(fixture.gymnasium_version, "1.2.1");
    assert_eq!(fixture.mujoco_version, "3.9.0");
    assert!(!fixture.environment_id.is_empty());
    assert!(!fixture.actions.is_empty());
    assert_eq!(fixture.actions.len(), fixture.states.len());
    assert_eq!(fixture.actions.len(), fixture.outputs.len());
    assert_eq!(fixture.qpos, fixture.states[0].qpos);
    assert_eq!(fixture.qvel, fixture.states[0].qvel);
    if let Some(contact_start) = contact_observation_start {
        assert!(fixture.outputs.iter().any(|output| {
            output.observation[contact_start..]
                .iter()
                .any(|value| value.abs() > 1e-9)
        }));
    }

    for index in 0..fixture.actions.len() {
        let action = &fixture.actions[index];
        let state = &fixture.states[index];
        let expected = &fixture.outputs[index];
        environment
            .set_exact_state(&state.qpos, &state.qvel)
            .unwrap();
        let action = Tensor::from_vec(action.clone(), action.len(), &Device::Cpu).unwrap();
        let (state, reward, done, truncated) = environment.parity_step(action).unwrap();
        let observation = state.to_vec1::<f32>().unwrap();

        assert_eq!(observation.len(), expected.observation.len());
        for (component, (actual, expected)) in
            observation.iter().zip(&expected.observation).enumerate()
        {
            // The environments expose f32 tensors while Gymnasium fixtures
            // retain MuJoCo's f64 values. Scale the bound by one f32 epsilon
            // so large contact forces are not rejected solely by conversion.
            let tolerance = observation_tolerance + f64::from(f32::EPSILON) * expected.abs();
            assert!(
                (f64::from(*actual) - expected).abs() <= tolerance,
                "step {index}, observation {component}: Rust {actual}, Gymnasium {expected}"
            );
        }
        assert!(
            (f64::from(reward) - expected.reward).abs() <= reward_tolerance,
            "step {index}, reward: Rust {}, Gymnasium {}",
            reward,
            expected.reward
        );
        assert_eq!(done, expected.terminated, "step {index}");
        assert_eq!(truncated, expected.truncated, "step {index}");
    }
}

#[test]
fn half_cheetah_matches_gymnasium_v5() {
    check_parity(
        include_str!("../python_tests/half_cheetah/trajectory.json"),
        HalfCheetahV5::builder().build().unwrap(),
        1e-5,
        1e-5,
        None,
    );
}

#[test]
fn ant_matches_gymnasium_v5() {
    check_parity(
        include_str!("../python_tests/ant/trajectory.json"),
        AntV5::builder().build().unwrap(),
        1e-5,
        1e-5,
        Some(27),
    );
}

#[test]
fn hopper_matches_gymnasium_v5() {
    check_parity(
        include_str!("../python_tests/hopper/trajectory.json"),
        HopperV5::builder().build().unwrap(),
        1e-5,
        1e-5,
        None,
    );
}

#[test]
fn humanoid_matches_gymnasium_v5() {
    check_parity(
        include_str!("../python_tests/humanoid/trajectory.json"),
        HumanoidV5::builder().build().unwrap(),
        1e-5,
        1e-5,
        Some(270),
    );
}

#[test]
fn walker2d_matches_gymnasium_v5() {
    check_parity(
        include_str!("../python_tests/walker2d/trajectory.json"),
        Walker2dV5::builder().build().unwrap(),
        1e-5,
        1e-5,
        None,
    );
}

use candle_core::{Device, Tensor};
use modurl::gym::Gym;
use modurl_mojoco::{HalfCheetahV5, HopperV5, MujocoError, Walker2dV5};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
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

trait ParityEnvironment: Gym<Error = MujocoError> {
    fn set_exact_state(&mut self, qpos: &[f64], qvel: &[f64]) -> Result<Tensor, MujocoError>;
}

impl ParityEnvironment for HalfCheetahV5 {
    fn set_exact_state(&mut self, qpos: &[f64], qvel: &[f64]) -> Result<Tensor, MujocoError> {
        self.set_state(qpos, qvel)
    }
}

impl ParityEnvironment for HopperV5 {
    fn set_exact_state(&mut self, qpos: &[f64], qvel: &[f64]) -> Result<Tensor, MujocoError> {
        self.set_state(qpos, qvel)
    }
}

impl ParityEnvironment for Walker2dV5 {
    fn set_exact_state(&mut self, qpos: &[f64], qvel: &[f64]) -> Result<Tensor, MujocoError> {
        self.set_state(qpos, qvel)
    }
}

fn check_parity<E: ParityEnvironment>(
    fixture_json: &str,
    mut environment: E,
    observation_tolerance: f64,
    reward_tolerance: f64,
) {
    let fixture: Fixture = serde_json::from_str(fixture_json).unwrap();
    environment
        .set_exact_state(&fixture.qpos, &fixture.qvel)
        .unwrap();

    for (index, ((action, state), expected)) in fixture
        .actions
        .iter()
        .zip(&fixture.states)
        .zip(&fixture.outputs)
        .enumerate()
    {
        environment
            .set_exact_state(&state.qpos, &state.qvel)
            .unwrap();
        let action = Tensor::from_vec(action.clone(), action.len(), &Device::Cpu).unwrap();
        let actual = environment.step(action).unwrap();
        let observation = actual.state.to_vec1::<f32>().unwrap();

        assert_eq!(observation.len(), expected.observation.len());
        for (component, (actual, expected)) in
            observation.iter().zip(&expected.observation).enumerate()
        {
            assert!(
                (f64::from(*actual) - expected).abs() <= observation_tolerance,
                "step {index}, observation {component}: Rust {actual}, Gymnasium {expected}"
            );
        }
        assert!(
            (f64::from(actual.reward) - expected.reward).abs() <= reward_tolerance,
            "step {index}, reward: Rust {}, Gymnasium {}",
            actual.reward,
            expected.reward
        );
        assert_eq!(actual.done, expected.terminated, "step {index}");
        assert_eq!(actual.truncated, expected.truncated, "step {index}");
    }
}

#[test]
fn half_cheetah_matches_gymnasium_v5() {
    check_parity(
        include_str!("../python_tests/half_cheetah/trajectory.json"),
        HalfCheetahV5::builder().build().unwrap(),
        1e-5,
        1e-5,
    );
}

#[test]
fn hopper_matches_gymnasium_v5() {
    check_parity(
        include_str!("../python_tests/hopper/trajectory.json"),
        HopperV5::builder().build().unwrap(),
        1e-5,
        1e-5,
    );
}

#[test]
fn walker2d_matches_gymnasium_v5() {
    check_parity(
        include_str!("../python_tests/walker2d/trajectory.json"),
        Walker2dV5::builder().build().unwrap(),
        1e-5,
        1e-5,
    );
}

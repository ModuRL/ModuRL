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

fn check_parity<I, E>(
    fixture_json: &str,
    expected_environment_id: &str,
    mut environment: E,
    set_state: fn(&mut E, &[f64], &[f64]) -> Result<Tensor, MujocoError>,
    observation_tolerance: f64,
    reward_tolerance: f64,
    contact_observation_start: Option<usize>,
) where
    E: Gym<I, Error = MujocoError>,
{
    let fixture: Fixture = serde_json::from_str(fixture_json).unwrap();
    assert_eq!(fixture.gymnasium_version, "1.2.1");
    assert_eq!(fixture.mujoco_version, "3.9.0");
    assert_eq!(fixture.environment_id, expected_environment_id);
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
        set_state(&mut environment, &state.qpos, &state.qvel).unwrap();
        let action = Tensor::from_vec(action.clone(), action.len(), &Device::Cpu).unwrap();
        let step = <E as Gym<I>>::step(&mut environment, action).unwrap();
        let observation = step.state.to_vec1::<f32>().unwrap();

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
            (f64::from(step.reward) - expected.reward).abs() <= reward_tolerance,
            "step {index}, reward: Rust {}, Gymnasium {}",
            step.reward,
            expected.reward
        );
        assert_eq!(step.done, expected.terminated, "step {index}");
        assert_eq!(step.truncated, expected.truncated, "step {index}");
    }
}

#[test]
fn ant() {
    check_parity::<AntV5Info, _>(
        include_str!("../python_tests/ant/trajectory.json"),
        "Ant-v5",
        AntV5::builder().build().unwrap(),
        AntV5::set_state,
        1e-5,
        1e-5,
        Some(27),
    );
}

#[test]
fn half_cheetah() {
    check_parity::<(), _>(
        include_str!("../python_tests/half_cheetah/trajectory.json"),
        "HalfCheetah-v5",
        HalfCheetahV5::builder().build().unwrap(),
        HalfCheetahV5::set_state,
        1e-5,
        1e-5,
        None,
    );
}

#[test]
fn hopper() {
    check_parity::<(), _>(
        include_str!("../python_tests/hopper/trajectory.json"),
        "Hopper-v5",
        HopperV5::builder().build().unwrap(),
        HopperV5::set_state,
        1e-5,
        1e-5,
        None,
    );
}

#[test]
fn humanoid() {
    check_parity::<HumanoidV5Info, _>(
        include_str!("../python_tests/humanoid/trajectory.json"),
        "Humanoid-v5",
        HumanoidV5::builder().build().unwrap(),
        HumanoidV5::set_state,
        1e-5,
        1e-5,
        Some(270),
    );
}

#[test]
fn walker2d() {
    check_parity::<(), _>(
        include_str!("../python_tests/walker2d/trajectory.json"),
        "Walker2d-v5",
        Walker2dV5::builder().build().unwrap(),
        Walker2dV5::set_state,
        1e-5,
        1e-5,
        None,
    );
}

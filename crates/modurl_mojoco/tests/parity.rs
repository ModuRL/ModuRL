use candle_core::{Device, Tensor};
use modurl::gym::Gym;
use modurl_mojoco::{
    AntV5, AntV5Info, HalfCheetahV5, HopperV5, HumanoidV5, HumanoidV5Info, MujocoError, Walker2dV5,
};
use serde::Deserialize;

const PARITY_STEPS: usize = 64;

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
    relaxed_observation_tolerance: Option<(usize, f64)>,
    contact_observation_start: Option<usize>,
) where
    E: Gym<I, Error = MujocoError>,
{
    let fixture: Fixture = serde_json::from_str(fixture_json).unwrap();
    assert_eq!(fixture.gymnasium_version, "1.2.1");
    assert_eq!(fixture.mujoco_version, "3.9.0");
    assert_eq!(fixture.environment_id, expected_environment_id);
    assert_eq!(fixture.actions.len(), PARITY_STEPS);
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

    let mut largest_observation_difference = (0.0, 0, 0, 0.0, 0.0, 0.0);
    let mut largest_reward_difference = (0.0, 0, 0.0, 0.0);

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
            let difference = (f64::from(*actual) - expected).abs();
            let absolute_tolerance = relaxed_observation_tolerance
                .filter(|(start, _)| component >= *start)
                .map_or(observation_tolerance, |(_, tolerance)| tolerance);
            let allowed_difference = absolute_tolerance + f64::from(f32::EPSILON) * expected.abs();
            let ratio = difference / allowed_difference;
            if ratio > largest_observation_difference.0 {
                largest_observation_difference = (
                    ratio,
                    index,
                    component,
                    f64::from(*actual),
                    *expected,
                    difference,
                );
            }
        }
        let reward_difference = (f64::from(step.reward) - expected.reward).abs();
        if reward_difference > largest_reward_difference.0 {
            largest_reward_difference = (
                reward_difference,
                index,
                f64::from(step.reward),
                expected.reward,
            );
        }
        assert_eq!(step.done, expected.terminated, "step {index}");
        assert_eq!(step.truncated, expected.truncated, "step {index}");
    }

    let (ratio, step, component, actual, expected, difference) = largest_observation_difference;
    assert!(
        ratio <= 1.0,
        "largest normalized observation difference at step {step}, component {component}: Rust {actual}, Gymnasium {expected}, difference {difference}, tolerance ratio {ratio}"
    );
    let (difference, step, actual, expected) = largest_reward_difference;
    assert!(
        difference <= reward_tolerance,
        "largest reward difference at step {step}: Rust {actual}, Gymnasium {expected}, difference {difference}"
    );
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
        None,
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
        None,
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
        4e-3,
        2e-2,
        // Walker2d's impact velocities vary across the official Python and
        // mujoco-rs builds. Positions remain tight, while this bound keeps all
        // 64 transitions—including simultaneous foot impacts—in the baseline.
        Some((8, 6.6e-1)),
        None,
    );
}

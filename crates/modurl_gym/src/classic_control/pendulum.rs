use std::f64::consts::PI;

use crate::EnvironmentError;
use bon::bon;
use burn::tensor::{DType, Device, Tensor, TensorData, TensorReadError};
use modurl::{
    gym::{Gym, PerfectStateGym, ResetInfo, StepInfo},
    spaces::BoxSpace,
};
use rand::{RngExt, SeedableRng, rngs::ChaCha12Rng};

/// Gymnasium-compatible `Pendulum-v1` continuous-control environment.
pub struct PendulumV1 {
    state: [f64; 2],
    gravity: f64,
    rng: ChaCha12Rng,
    action_space: BoxSpace,
    observation_space: BoxSpace,
    #[cfg(feature = "rendering")]
    renderer: Option<crate::rendering::Renderer>,
}

/// Saved mutable state for [`PendulumV1`].
#[derive(Clone)]
pub struct PendulumV1State {
    rng_state: [u8; 49],
    state: [f64; 2],
}

#[bon]
impl PendulumV1 {
    /// Creates a pendulum with Gymnasium's default dynamics.
    #[builder]
    pub fn new(
        /// Seeds the environment's RNG; when omitted the RNG is seeded from OS entropy.
        seed: Option<u64>,
        #[builder(default = 10.0)] gravity: f64,
        #[cfg(feature = "rendering")]
        #[builder(default = false)]
        render: bool,
    ) -> Result<Self, EnvironmentError> {
        if !gravity.is_finite() {
            return Err(EnvironmentError::InvalidConfiguration(
                "gravity must be finite",
            ));
        }
        Self::from_config(
            seed,
            gravity,
            #[cfg(feature = "rendering")]
            render,
        )
    }

    fn from_config(
        seed: Option<u64>,
        gravity: f64,
        #[cfg(feature = "rendering")] render: bool,
    ) -> Result<Self, EnvironmentError> {
        let action_space = BoxSpace::new_with_universal_bounds([1, 1], -2.0, 2.0, &Device::flex());
        let observation_space = BoxSpace::new(
            Tensor::from_data(
                TensorData::new(vec![-1.0_f32, -1.0, -8.0], [1, 3]),
                &Device::flex(),
            ),
            Tensor::from_data(
                TensorData::new(vec![1.0_f32, 1.0, 8.0], [1, 3]),
                &Device::flex(),
            ),
        );
        Ok(Self {
            state: [0.0; 2],
            gravity,
            rng: seed.map_or_else(rand::make_rng, ChaCha12Rng::seed_from_u64),
            action_space,
            observation_space,
            #[cfg(feature = "rendering")]
            renderer: render
                .then(|| crate::rendering::Renderer::new(500, 500, "Pendulum"))
                .transpose()?,
        })
    }

    /// Returns one F32 observation vector on CPU, without a batch axis.
    fn observation(&self) -> Tensor<1> {
        Tensor::from_data(
            TensorData::new(
                vec![
                    self.state[0].cos() as f32,
                    self.state[0].sin() as f32,
                    self.state[1] as f32,
                ],
                [3],
            ),
            &Device::flex(),
        )
    }

    #[cfg(feature = "rendering")]
    fn render(&mut self) -> Result<(), EnvironmentError> {
        let Some(renderer) = &mut self.renderer else {
            return Ok(());
        };
        if !renderer.is_open() {
            return Ok(());
        }
        renderer.clear(0xFFFFFF);
        let center = 250.0_f32;
        let length = 113.6_f32;
        let half_width = 11.4_f32;
        let angle = self.state[0] as f32 + std::f32::consts::FRAC_PI_2;
        let direction = (angle.cos(), -angle.sin());
        let normal = (-direction.1, direction.0);
        let end = (center + length * direction.0, center + length * direction.1);
        renderer.quad(
            (
                center - half_width * normal.0,
                center - half_width * normal.1,
            ),
            (
                center + half_width * normal.0,
                center + half_width * normal.1,
            ),
            (end.0 + half_width * normal.0, end.1 + half_width * normal.1),
            (end.0 - half_width * normal.0, end.1 - half_width * normal.1),
            0xCC4D4D,
        );
        renderer.draw_circle(center as usize, center as usize, 6, 0x000000);
        renderer.draw_circle(end.0 as usize, end.1 as usize, 11, 0xCC4D4D);
        renderer.present()?;
        Ok(())
    }

    #[cfg(test)]
    fn set_raw_state(&mut self, state: [f64; 2]) {
        self.state = state;
    }
}

impl Gym for PendulumV1 {
    type Error = EnvironmentError;
    type ObservationSpace = BoxSpace;
    type ActionSpace = BoxSpace;

    fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
        self.state = [
            -PI + f64::from(self.rng.random_range(0.0_f32..1.0_f32)) * 2.0 * PI,
            -1.0 + f64::from(self.rng.random_range(0.0_f32..1.0_f32)) * 2.0,
        ];
        #[cfg(feature = "rendering")]
        self.render()?;
        Ok(ResetInfo {
            observation: self.observation(),
            info: (),
        })
    }

    /// Steps with one continuous torque vector `action` shaped `[1]`.
    fn step(&mut self, action: Tensor<1>) -> Result<StepInfo, Self::Error> {
        if action.dims() != [1] || !action.dtype().is_float() {
            return Err(EnvironmentError::InvalidAction(
                "Pendulum actions must be a floating-point tensor shaped [1]",
            ));
        }
        let action = action
            .cast(DType::F64)
            .clone()
            .try_into_data_as::<f64>()
            .and_then(|data| data.try_to_vec::<f64>().map_err(TensorReadError::from))?;
        if !action[0].is_finite() {
            return Err(EnvironmentError::InvalidAction(
                "Pendulum actions must contain only finite values",
            ));
        }
        let torque = action[0].clamp(-2.0, 2.0);
        let [theta, theta_dot] = self.state;
        let normalized = ((theta + PI).rem_euclid(2.0 * PI)) - PI;
        let cost = normalized * normalized + 0.1 * theta_dot * theta_dot + 0.001 * torque * torque;
        let mut new_theta_dot =
            theta_dot + (3.0 * self.gravity / 2.0 * theta.sin() + 3.0 * torque) * 0.05;
        new_theta_dot = new_theta_dot.clamp(-8.0, 8.0);
        self.state = [theta + new_theta_dot * 0.05, new_theta_dot];
        #[cfg(feature = "rendering")]
        self.render()?;
        Ok(StepInfo {
            observation: self.observation(),
            reward: -cost as f32,
            done: false,
            truncated: false,
            info: (),
        })
    }

    fn observation_space(&self) -> Self::ObservationSpace {
        self.observation_space.clone()
    }

    fn action_space(&self) -> Self::ActionSpace {
        self.action_space.clone()
    }
}

impl PerfectStateGym for PendulumV1 {
    type State = PendulumV1State;

    fn save_state(&self) -> Result<Self::State, Self::Error> {
        Ok(PendulumV1State {
            rng_state: self.rng.serialize_state(),
            state: self.state,
        })
    }

    fn set_state(&mut self, state: &Self::State) -> Result<(), Self::Error> {
        self.rng = ChaCha12Rng::deserialize_state(&state.rng_state);
        self.state = state.state;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{PARITY_STEPS, check_perfect_state_replay};

    #[derive(serde::Deserialize)]
    struct Transition {
        state: [f64; 2],
        action: f32,
        observation: [f64; 3],
        reward: f64,
    }

    #[test]
    fn parity() {
        let transitions: Vec<Transition> =
            serde_json::from_str(include_str!("../../python_tests/pendulum/trajectory.json"))
                .unwrap();
        assert_eq!(transitions.len(), PARITY_STEPS);
        let mut environment = PendulumV1::builder().build().unwrap();
        for (index, transition) in transitions.iter().enumerate() {
            environment.set_raw_state(transition.state);
            let action = Tensor::from_data(
                TensorData::new(vec![transition.action], [1]),
                &Device::flex(),
            );
            let actual = environment.step(action).unwrap();
            let observation = actual
                .observation
                .clone()
                .try_into_data_as::<f32>()
                .and_then(|data| data.try_to_vec::<f32>().map_err(TensorReadError::from))
                .unwrap();
            for (component, expected) in transition.observation.iter().enumerate() {
                assert!(
                    (f64::from(observation[component]) - expected).abs() <= 1e-6,
                    "transition {index}, observation {component}"
                );
            }
            assert!((f64::from(actual.reward) - transition.reward).abs() <= 1e-5);
            assert!(!actual.done);
            assert!(!actual.truncated);
        }
    }

    #[test]
    fn perfect_state_replays_resets() {
        let prefix = [
            Tensor::from_data([0.25_f32], &Device::flex()),
            Tensor::from_data([-0.5_f32], &Device::flex()),
        ];
        let actions = [
            0.2_f32, -0.4, 1.0, 0.0, -1.5, 0.75, 0.3, -0.2, 1.8, 0.1, -0.7, 0.6,
        ]
        .map(|action| Tensor::from_data([action], &Device::flex()));
        check_perfect_state_replay(
            PendulumV1::builder().seed(7).build().unwrap(),
            &prefix,
            &actions,
        );
    }

    #[test]
    fn default_spaces_match_gymnasium() {
        let mut environment = PendulumV1::builder().build().unwrap();
        assert_eq!(environment.reset().unwrap().observation.dims(), [3]);
        assert_eq!(environment.action_space().shape(), vec![1]);
        assert_eq!(environment.observation_space().shape(), vec![3]);
    }

    #[test]
    fn rejects_non_finite_actions_without_mutating_state() {
        let mut environment = PendulumV1::builder().build().unwrap();
        environment.set_raw_state([0.25, -0.5]);

        let error = environment
            .step(Tensor::from_data([f32::NAN], &Device::flex()))
            .unwrap_err();

        assert!(matches!(error, EnvironmentError::InvalidAction(_)));
        assert_eq!(environment.state, [0.25, -0.5]);
    }

    #[cfg(feature = "rendering")]
    #[test]
    fn rendering_can_be_enabled() {
        PendulumV1::builder()
            .render(true)
            .build()
            .unwrap()
            .reset()
            .unwrap();
    }
}

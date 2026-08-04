use std::f64::consts::PI;

use crate::EnvironmentError;
use bon::bon;
use candle_core::{DType, Device, Tensor};
use modurl::{
    gym::{Gym, ResetInfo, StepInfo},
    spaces::{BoxSpace, Space},
};

/// Gymnasium-compatible `Pendulum-v1` continuous-control environment.
pub struct PendulumV1 {
    state: [f64; 2],
    gravity: f64,
    device: Device,
    action_space: BoxSpace,
    observation_space: BoxSpace,
    #[cfg(feature = "rendering")]
    renderer: Option<crate::rendering::Renderer>,
}

#[bon]
impl PendulumV1 {
    /// Creates a pendulum with Gymnasium's default dynamics.
    #[builder]
    pub fn new(
        #[builder(default = &Device::Cpu)] device: &Device,
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
            device,
            gravity,
            #[cfg(feature = "rendering")]
            render,
        )
    }

    fn from_config(
        device: &Device,
        gravity: f64,
        #[cfg(feature = "rendering")] render: bool,
    ) -> Result<Self, EnvironmentError> {
        let action_space = BoxSpace::new_with_universal_bounds(vec![1], -2.0, 2.0, device);
        let observation_space = BoxSpace::new(
            Tensor::from_vec(vec![-1.0_f32, -1.0, -8.0], 3, device)?,
            Tensor::from_vec(vec![1.0_f32, 1.0, 8.0], 3, device)?,
        );
        Ok(Self {
            state: [0.0; 2],
            gravity,
            device: device.clone(),
            action_space,
            observation_space,
            #[cfg(feature = "rendering")]
            renderer: render
                .then(|| crate::rendering::Renderer::new(500, 500, "Pendulum"))
                .transpose()?,
        })
    }

    fn observation(&self) -> Result<Tensor, candle_core::Error> {
        Tensor::from_vec(
            vec![
                self.state[0].cos() as f32,
                self.state[0].sin() as f32,
                self.state[1] as f32,
            ],
            3,
            &self.device,
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
    type SpaceError = candle_core::Error;

    fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
        let random = Tensor::rand(0.0_f32, 1.0, 2, &self.device)?.to_vec1::<f32>()?;
        self.state = [
            -PI + f64::from(random[0]) * 2.0 * PI,
            -1.0 + f64::from(random[1]) * 2.0,
        ];
        #[cfg(feature = "rendering")]
        self.render()?;
        Ok(ResetInfo {
            state: self.observation()?,
            info: (),
        })
    }

    /// Steps with one continuous torque vector `action` shaped `[1]`.
    fn step(&mut self, action: Tensor) -> Result<StepInfo, Self::Error> {
        if action.dims() != [1] || !action.dtype().is_float() {
            return Err(EnvironmentError::InvalidAction(
                "Pendulum actions must be a floating-point tensor shaped [1]",
            ));
        }
        let action = action.to_dtype(DType::F64)?.to_vec1::<f64>()?;
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
            state: self.observation()?,
            reward: -cost as f32,
            done: false,
            truncated: false,
            info: (),
        })
    }

    fn observation_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
        Box::new(self.observation_space.clone())
    }

    fn action_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
        Box::new(self.action_space.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(serde::Deserialize)]
    struct Transition {
        state: [f64; 2],
        action: f32,
        observation: [f64; 3],
        reward: f64,
    }

    #[test]
    fn matches_gymnasium_v1() {
        let transitions: Vec<Transition> =
            serde_json::from_str(include_str!("../../python_tests/pendulum/trajectory.json"))
                .unwrap();
        let mut environment = PendulumV1::builder().build().unwrap();
        for (index, transition) in transitions.iter().enumerate() {
            environment.set_raw_state(transition.state);
            let action = Tensor::from_vec(vec![transition.action], 1, &Device::Cpu).unwrap();
            let actual = environment.step(action).unwrap();
            let observation = actual.state.to_vec1::<f32>().unwrap();
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
    fn default_spaces_match_gymnasium() {
        let mut environment = PendulumV1::builder().build().unwrap();
        assert_eq!(environment.reset().unwrap().state.dims(), &[3]);
        assert_eq!(environment.action_space().shape(), vec![1]);
        assert_eq!(environment.observation_space().shape(), vec![3]);
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

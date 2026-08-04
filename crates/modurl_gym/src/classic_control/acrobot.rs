use std::f64::consts::PI;

use crate::EnvironmentError;
use bon::bon;
use candle_core::{Device, Tensor};
use modurl::{
    gym::{Gym, ResetInfo, StepInfo},
    spaces::{BoxSpace, Discrete, Space},
};

const MAX_VELOCITY_1: f64 = 4.0 * PI;
const MAX_VELOCITY_2: f64 = 9.0 * PI;
const LINK_MASS_1: f64 = 1.0;
const LINK_MASS_2: f64 = 1.0;
const LINK_LENGTH_1: f64 = 1.0;
const LINK_COM_1: f64 = 0.5;
const LINK_COM_2: f64 = 0.5;
const LINK_MOI: f64 = 1.0;
const GRAVITY: f64 = 9.8;
const DT: f64 = 0.2;

/// Gymnasium-compatible `Acrobot-v1` underactuated control environment.
pub struct AcrobotV1 {
    state: [f64; 4],
    use_nips_dynamics: bool,
    torque_noise_max: f64,
    device: Device,
    action_space: Discrete,
    observation_space: BoxSpace,
    #[cfg(feature = "rendering")]
    renderer: Option<crate::rendering::Renderer>,
}

#[bon]
impl AcrobotV1 {
    /// Creates an Acrobot. The default equations match Sutton and Barto's book.
    #[builder]
    pub fn new(
        #[builder(default = &Device::Cpu)] device: &Device,
        #[builder(default = false)] use_nips_dynamics: bool,
        #[builder(default = 0.0)] torque_noise_max: f64,
        #[cfg(feature = "rendering")]
        #[builder(default = false)]
        render: bool,
    ) -> Result<Self, EnvironmentError> {
        if !torque_noise_max.is_finite() || torque_noise_max < 0.0 {
            return Err(EnvironmentError::InvalidConfiguration(
                "torque_noise_max must be finite and non-negative",
            ));
        }
        Self::from_config(
            device,
            use_nips_dynamics,
            torque_noise_max,
            #[cfg(feature = "rendering")]
            render,
        )
    }

    fn from_config(
        device: &Device,
        use_nips_dynamics: bool,
        torque_noise_max: f64,
        #[cfg(feature = "rendering")] render: bool,
    ) -> Result<Self, EnvironmentError> {
        let high = vec![
            1.0_f32,
            1.0,
            1.0,
            1.0,
            MAX_VELOCITY_1 as f32,
            MAX_VELOCITY_2 as f32,
        ];
        let low = high.iter().map(|value| -*value).collect::<Vec<_>>();
        Ok(Self {
            state: [0.0; 4],
            use_nips_dynamics,
            torque_noise_max,
            device: device.clone(),
            action_space: Discrete::new(3),
            observation_space: BoxSpace::new(
                Tensor::from_vec(low, 6, device)?,
                Tensor::from_vec(high, 6, device)?,
            ),
            #[cfg(feature = "rendering")]
            renderer: render
                .then(|| crate::rendering::Renderer::new(500, 500, "Acrobot"))
                .transpose()?,
        })
    }

    fn observation(&self) -> Result<Tensor, candle_core::Error> {
        Tensor::from_vec(
            vec![
                self.state[0].cos() as f32,
                self.state[0].sin() as f32,
                self.state[1].cos() as f32,
                self.state[1].sin() as f32,
                self.state[2] as f32,
                self.state[3] as f32,
            ],
            6,
            &self.device,
        )
    }

    fn derivatives(&self, state: [f64; 5]) -> [f64; 5] {
        let [theta1, theta2, dtheta1, dtheta2, torque] = state;
        let d1 = LINK_MASS_1 * LINK_COM_1.powi(2)
            + LINK_MASS_2
                * (LINK_LENGTH_1.powi(2)
                    + LINK_COM_2.powi(2)
                    + 2.0 * LINK_LENGTH_1 * LINK_COM_2 * theta2.cos())
            + 2.0 * LINK_MOI;
        let d2 = LINK_MASS_2 * (LINK_COM_2.powi(2) + LINK_LENGTH_1 * LINK_COM_2 * theta2.cos())
            + LINK_MOI;
        let phi2 = LINK_MASS_2 * LINK_COM_2 * GRAVITY * (theta1 + theta2 - PI / 2.0).cos();
        let phi1 = -LINK_MASS_2 * LINK_LENGTH_1 * LINK_COM_2 * dtheta2.powi(2) * theta2.sin()
            - 2.0 * LINK_MASS_2 * LINK_LENGTH_1 * LINK_COM_2 * dtheta2 * dtheta1 * theta2.sin()
            + (LINK_MASS_1 * LINK_COM_1 + LINK_MASS_2 * LINK_LENGTH_1)
                * GRAVITY
                * (theta1 - PI / 2.0).cos()
            + phi2;
        let extra = if self.use_nips_dynamics {
            0.0
        } else {
            LINK_MASS_2 * LINK_LENGTH_1 * LINK_COM_2 * dtheta1.powi(2) * theta2.sin()
        };
        let ddtheta2 = (torque + d2 / d1 * phi1 - extra - phi2)
            / (LINK_MASS_2 * LINK_COM_2.powi(2) + LINK_MOI - d2.powi(2) / d1);
        let ddtheta1 = -(d2 * ddtheta2 + phi1) / d1;
        [dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0]
    }

    fn integrate(&self, state: [f64; 5]) -> [f64; 4] {
        let scale_add = |base: [f64; 5], delta: [f64; 5], scale: f64| {
            std::array::from_fn(|index| base[index] + scale * delta[index])
        };
        let k1 = self.derivatives(state);
        let k2 = self.derivatives(scale_add(state, k1, DT / 2.0));
        let k3 = self.derivatives(scale_add(state, k2, DT / 2.0));
        let k4 = self.derivatives(scale_add(state, k3, DT));
        std::array::from_fn(|index| {
            state[index] + DT / 6.0 * (k1[index] + 2.0 * k2[index] + 2.0 * k3[index] + k4[index])
        })
    }

    fn terminal(&self) -> bool {
        -self.state[0].cos() - (self.state[0] + self.state[1]).cos() > 1.0
    }

    #[cfg(any(feature = "rendering", test))]
    fn link_endpoints(&self, center: (f32, f32), scale: f32) -> [(f32, f32); 2] {
        let first = (
            center.0 + self.state[0].sin() as f32 * scale,
            center.1 + self.state[0].cos() as f32 * scale,
        );
        let second = (
            first.0 + (self.state[0] + self.state[1]).sin() as f32 * scale,
            first.1 + (self.state[0] + self.state[1]).cos() as f32 * scale,
        );
        [first, second]
    }

    #[cfg(feature = "rendering")]
    fn render(&mut self) -> Result<(), EnvironmentError> {
        let center = (250.0_f32, 250.0_f32);
        let scale = 113.6_f32;
        let [first, second] = self.link_endpoints(center, scale);
        let Some(renderer) = &mut self.renderer else {
            return Ok(());
        };
        if !renderer.is_open() {
            return Ok(());
        }
        renderer.clear(0xFFFFFF);
        Self::draw_link(renderer, center, first, 0x00CCCC);
        Self::draw_link(renderer, first, second, 0x00CCCC);
        renderer.draw_circle(center.0 as usize, center.1 as usize, 11, 0xCCCC00);
        renderer.draw_circle(first.0 as usize, first.1 as usize, 11, 0xCCCC00);
        renderer.rect(0, (center.1 - scale) as usize, 500, 2, 0x000000);
        renderer.present()?;
        Ok(())
    }

    #[cfg(feature = "rendering")]
    fn draw_link(
        renderer: &mut crate::rendering::Renderer,
        start: (f32, f32),
        end: (f32, f32),
        color: u32,
    ) {
        let dx = end.0 - start.0;
        let dy = end.1 - start.1;
        let length = (dx * dx + dy * dy).sqrt();
        let normal = (-dy / length * 11.0, dx / length * 11.0);
        renderer.quad(
            (start.0 - normal.0, start.1 - normal.1),
            (start.0 + normal.0, start.1 + normal.1),
            (end.0 + normal.0, end.1 + normal.1),
            (end.0 - normal.0, end.1 - normal.1),
            color,
        );
    }

    #[cfg(test)]
    fn set_raw_state(&mut self, state: [f64; 4]) {
        self.state = state;
    }
}

impl Gym for AcrobotV1 {
    type Error = EnvironmentError;
    type SpaceError = candle_core::Error;

    fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
        let random = Tensor::rand(-0.1_f32, 0.1, 4, &self.device)?.to_vec1::<f32>()?;
        self.state = std::array::from_fn(|index| f64::from(random[index]));
        #[cfg(feature = "rendering")]
        self.render()?;
        Ok(ResetInfo {
            state: self.observation()?,
            info: (),
        })
    }

    /// Steps with one scalar discrete action shaped `[]`.
    fn step(&mut self, action: Tensor) -> Result<StepInfo, Self::Error> {
        if !action.dims().is_empty() || action.dtype() != candle_core::DType::U32 {
            return Err(EnvironmentError::InvalidAction(
                "Acrobot actions must be scalar u32 values in 0..3",
            ));
        }
        let mut torque = f64::from(action.to_vec0::<u32>()?) - 1.0;
        if !(-1.0..=1.0).contains(&torque) {
            return Err(EnvironmentError::InvalidAction(
                "Acrobot actions must be scalar u32 values in 0..3",
            ));
        }
        if self.torque_noise_max > 0.0 {
            let noise = Tensor::rand(
                -self.torque_noise_max as f32,
                self.torque_noise_max as f32,
                (),
                &self.device,
            )?
            .to_vec0::<f32>()?;
            torque += f64::from(noise);
        }
        let integrated = self.integrate([
            self.state[0],
            self.state[1],
            self.state[2],
            self.state[3],
            torque,
        ]);
        self.state = [
            wrap(integrated[0], -PI, PI),
            wrap(integrated[1], -PI, PI),
            integrated[2].clamp(-MAX_VELOCITY_1, MAX_VELOCITY_1),
            integrated[3].clamp(-MAX_VELOCITY_2, MAX_VELOCITY_2),
        ];
        let done = self.terminal();
        #[cfg(feature = "rendering")]
        self.render()?;
        Ok(StepInfo {
            state: self.observation()?,
            reward: if done { 0.0 } else { -1.0 },
            done,
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

fn wrap(mut value: f64, minimum: f64, maximum: f64) -> f64 {
    let width = maximum - minimum;
    while value > maximum {
        value -= width;
    }
    while value < minimum {
        value += width;
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(serde::Deserialize)]
    struct Transition {
        state: [f64; 4],
        action: u32,
        observation: [f64; 6],
        reward: f64,
        terminated: bool,
    }

    #[test]
    fn matches_gymnasium_v1() {
        let transitions: Vec<Transition> =
            serde_json::from_str(include_str!("../../python_tests/acrobot/trajectory.json"))
                .unwrap();
        let mut environment = AcrobotV1::builder().build().unwrap();
        for (index, transition) in transitions.iter().enumerate() {
            environment.set_raw_state(transition.state);
            let action = Tensor::from_vec(vec![transition.action], (), &Device::Cpu).unwrap();
            let actual = environment.step(action).unwrap();
            let observation = actual.state.to_vec1::<f32>().unwrap();
            for (component, expected) in transition.observation.iter().enumerate() {
                assert!(
                    (f64::from(observation[component]) - expected).abs() <= 2e-6,
                    "transition {index}, observation {component}: {} != {expected}",
                    observation[component]
                );
            }
            assert_eq!(f64::from(actual.reward), transition.reward);
            assert_eq!(actual.done, transition.terminated);
            assert!(!actual.truncated);
        }
    }

    #[test]
    fn default_spaces_match_gymnasium() {
        let mut environment = AcrobotV1::builder().build().unwrap();
        assert_eq!(environment.reset().unwrap().state.dims(), &[6]);
        assert_eq!(environment.action_space().shape(), vec![3]);
        assert_eq!(environment.observation_space().shape(), vec![6]);
    }

    #[test]
    fn zero_angles_render_links_vertically_downward() {
        let environment = AcrobotV1::builder().build().unwrap();
        let [first, second] = environment.link_endpoints((250.0, 250.0), 100.0);
        assert_eq!(first, (250.0, 350.0));
        assert_eq!(second, (250.0, 450.0));
    }

    #[cfg(feature = "rendering")]
    #[test]
    fn rendering_can_be_enabled() {
        AcrobotV1::builder()
            .render(true)
            .build()
            .unwrap()
            .reset()
            .unwrap();
    }
}

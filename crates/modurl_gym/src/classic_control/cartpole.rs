use bon::bon;
use candle_core::{Device, Tensor};
use modurl::{
    gym::{Gym, ResetInfo, StepInfo},
    spaces::{self, Space},
};

#[cfg(feature = "rendering")]
use crate::rendering::Renderer;

/// The classic CartPole environment.
/// Converted from the OpenAI Gym CartPole environment.
pub struct CartPoleV1 {
    gravity: f32,
    masspole: f32,
    total_mass: f32,
    length: f32,
    polemass_length: f32,
    force_mag: f32,
    tau: f32,
    x_threshold: f32,
    theta_threshold_radians: f32,
    is_euler: bool,
    steps_beyond_terminated: Option<usize>,
    action_space: spaces::Discrete,
    observation_space: spaces::BoxSpace,
    state: Tensor,
    steps_since_reset: usize,
    sutton_barto_reward: bool,
    #[cfg(feature = "rendering")]
    renderer: Option<Renderer>,
}

#[bon]
impl CartPoleV1 {
    #[builder]
    pub fn new(
        #[builder(default = &Device::Cpu)] device: &Device,
        #[builder(default = false)] sutton_barto_reward: bool,
        #[builder(default = true)] is_euler: bool,
        #[cfg(feature = "rendering")]
        #[builder(default = false)]
        render: bool,
    ) -> Self {
        let gravity = 9.8;
        let masscart = 1.0;
        let masspole = 0.1;
        let total_mass = masspole + masscart;
        let length = 0.5; // actually half the pole's length
        let polemass_length = masspole * length;
        let force_mag = 10.0;
        let tau = 0.02;

        // Angle at which to fail the episode
        let theta_threshold_radians = 12.0 * 2.0 * std::f32::consts::PI / 360.0;
        let x_threshold = 2.4;

        let high = vec![
            x_threshold * 2.0,
            std::f32::INFINITY,
            theta_threshold_radians * 2.0,
            std::f32::INFINITY,
        ];
        let low = high.iter().map(|x| -x).collect::<Vec<_>>();
        let high = Tensor::from_vec(high, vec![4], device).expect("Failed to create tensor.");
        let low = Tensor::from_vec(low, vec![4], device).expect("Failed to create tensor.");

        let action_space = spaces::Discrete::new(2);
        let observation_space = spaces::BoxSpace::new(low, high);

        Self {
            gravity,
            masspole,
            total_mass,
            length,
            polemass_length,
            force_mag,
            tau,
            x_threshold,
            theta_threshold_radians,
            steps_beyond_terminated: Some(0),
            is_euler,
            action_space,
            observation_space,
            state: Tensor::zeros(vec![4], candle_core::DType::F32, device)
                .expect("Failed to create tensor."),
            steps_since_reset: 0,
            sutton_barto_reward,
            #[cfg(feature = "rendering")]
            renderer: if render {
                Some(Renderer::new(600, 400, "CartPole"))
            } else {
                None
            },
        }
    }

    #[cfg(feature = "rendering")]
    fn render(&mut self) {
        if let Some(renderer) = &mut self.renderer {
            let screen_width = renderer.get_width() as f32;

            renderer.clear(0xFFFFFF);

            let state_vec = self.state.to_vec1::<f32>().unwrap();
            let cart_position = state_vec[0];
            let pole_angle = state_vec[2];

            let world_width = self.x_threshold * 2.0;
            let scale = screen_width / world_width / 4.0;
            let pole_width = 6.0;
            let pole_length = scale * (2.0 * self.length) * 3.0;
            let cart_width = 30.0;
            let cart_height = 20.0;
            let axle_offset = cart_height / 4.0;

            let cart_x = cart_position * scale + screen_width / 2.0;
            let cart_y = 300.0;

            Self::draw_track(renderer, screen_width, cart_y);

            Self::draw_cart(renderer, cart_x, cart_y, cart_width, cart_height);

            // Draw the pole
            Self::draw_pole(
                renderer,
                cart_x,
                cart_y + axle_offset,
                pole_angle,
                pole_width,
                pole_length,
            );

            // Draw the axle (connection point)
            Self::draw_axle(renderer, cart_x, cart_y + axle_offset, pole_width);

            renderer.present();
        }
    }

    #[cfg(feature = "rendering")]
    #[cfg(feature = "rendering")]
    fn draw_track(renderer: &mut crate::rendering::Renderer, screen_width: f32, cart_y: f32) {
        for x in 0..(screen_width as usize) {
            renderer.rect(x, cart_y as usize, 1, 2, 0x000000);
        }
    }
    #[cfg(feature = "rendering")]
    #[cfg(feature = "rendering")]
    fn draw_cart(
        renderer: &mut crate::rendering::Renderer,
        cart_x: f32,
        cart_y: f32,
        cart_width: f32,
        cart_height: f32,
    ) {
        let l = -cart_width / 2.0;
        let r = cart_width / 2.0;
        let t = cart_height / 2.0;
        let b = -cart_height / 2.0;

        let cart_corners = [
            (cart_x + l, cart_y + b),
            (cart_x + l, cart_y + t),
            (cart_x + r, cart_y + t),
            (cart_x + r, cart_y + b),
        ];
        renderer.quad(
            cart_corners[0],
            cart_corners[1],
            cart_corners[2],
            cart_corners[3],
            0x000000,
        );
    }
    #[cfg(feature = "rendering")]
    #[cfg(feature = "rendering")]
    fn draw_pole(
        renderer: &mut crate::rendering::Renderer,
        axle_x: f32,
        axle_y: f32,
        angle: f32,
        pole_width: f32,
        pole_length: f32,
    ) {
        let l = -pole_width / 2.0;
        let r = pole_width / 2.0;
        let t = pole_length - pole_width / 2.0;
        let b = -pole_width / 2.0;

        let cos_a = (-angle).cos();
        let sin_a = (-angle).sin();
        let pole_corners = [
            (l * cos_a - b * sin_a, l * sin_a + b * cos_a),
            (l * cos_a - t * sin_a, l * sin_a + t * cos_a),
            (r * cos_a - t * sin_a, r * sin_a + t * cos_a),
            (r * cos_a - b * sin_a, r * sin_a + b * cos_a),
        ];

        let pole_screen_coords = [
            (axle_x + pole_corners[0].0, axle_y - pole_corners[0].1),
            (axle_x + pole_corners[1].0, axle_y - pole_corners[1].1),
            (axle_x + pole_corners[2].0, axle_y - pole_corners[2].1),
            (axle_x + pole_corners[3].0, axle_y - pole_corners[3].1),
        ];

        renderer.quad(
            pole_screen_coords[0],
            pole_screen_coords[1],
            pole_screen_coords[2],
            pole_screen_coords[3],
            0xCA9865,
        );
    }

    #[cfg(feature = "rendering")]
    fn draw_axle(
        renderer: &mut crate::rendering::Renderer,
        axle_x: f32,
        axle_y: f32,
        pole_width: f32,
    ) {
        let axle_radius = (pole_width / 2.0) as usize;
        renderer.draw_circle(axle_x as usize, axle_y as usize, axle_radius, 0x8184CB);
    }
}

impl Default for CartPoleV1 {
    fn default() -> Self {
        CartPoleV1::builder().build()
    }
}

impl Gym for CartPoleV1 {
    type Error = candle_core::Error;
    type SpaceError = candle_core::Error;

    fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
        self.steps_beyond_terminated = None;
        let state = Tensor::rand(-0.05, 0.05, vec![4], self.state.device())?
            .to_dtype(candle_core::DType::F32)?;
        self.state = state;
        self.steps_since_reset = 0;

        #[cfg(feature = "rendering")]
        self.render();

        Ok(ResetInfo {
            state: self.state.clone(),
            info: (),
        })
    }

    /// Steps with one scalar discrete action shaped `[]`.
    fn step(&mut self, action: Tensor) -> Result<StepInfo, Self::Error> {
        assert!(self.action_space.contains(&action));
        let state_vec = self.state.to_vec1::<f32>()?;
        let (mut x, mut x_dot, mut theta, mut theta_dot) =
            (state_vec[0], state_vec[1], state_vec[2], state_vec[3]);

        let action_vec = action.to_vec0::<u32>()?;
        let force = if action_vec == 0 {
            -self.force_mag
        } else {
            self.force_mag
        };

        let costheta = theta.cos();
        let sintheta = theta.sin();

        let temp =
            (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass;
        let thetaacc = (self.gravity * sintheta - costheta * temp)
            / (self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass));
        let xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass;

        if self.is_euler {
            x += self.tau * x_dot;
            x_dot += self.tau * xacc;
            theta += self.tau * theta_dot;
            theta_dot += self.tau * thetaacc;
        } else {
            x_dot += 0.5 * self.tau * (xacc + temp);
            theta_dot += 0.5 * self.tau * (thetaacc + temp);
            theta += self.tau * theta_dot + 0.5 * self.tau * self.tau * thetaacc;
            theta_dot += 0.5 * self.tau * (thetaacc + temp);
        }

        self.state = Tensor::from_vec(
            vec![x, x_dot, theta, theta_dot],
            vec![4],
            self.state.device(),
        )
        .expect("Failed to create tensor.");
        let terminated = x < -self.x_threshold
            || x > self.x_threshold
            || theta < -self.theta_threshold_radians
            || theta > self.theta_threshold_radians;

        self.steps_since_reset += 1;
        if self.steps_since_reset >= 500 {
            // Consider it done if it has lasted 500 steps.
            self.steps_beyond_terminated = Some(0);
            return Ok(StepInfo {
                state: self.state.clone(),
                reward: 1.0,
                done: false,
                truncated: true,
                info: (),
            });
        }

        #[cfg(feature = "rendering")]
        self.render();
        if !terminated {
            let reward = if self.sutton_barto_reward { 0.0 } else { 1.0 };

            Ok(StepInfo {
                state: self.state.clone(),
                reward,
                done: false,
                truncated: false,
                info: (),
            })
        } else if self.steps_beyond_terminated.is_none() {
            // Pole just fell!
            self.steps_beyond_terminated = Some(0);
            let reward = if self.sutton_barto_reward { -1.0 } else { 1.0 };

            Ok(StepInfo {
                state: self.state.clone(),
                reward,
                done: true,
                truncated: false,
                info: (),
            })
        } else {
            #[cfg(feature = "logging")]
            if self.steps_beyond_terminated == Some(0) {
                log::warn!(
                    "You are calling 'step()' even though this environment has already returned terminated = True. You should always call 'reset()' once you receive 'terminated = True'"
                );
            }
            let reward = if self.sutton_barto_reward { -1.0 } else { 0.0 };
            // We already checked this is Some above, so this is safe.
            self.steps_beyond_terminated = Some(self.steps_beyond_terminated.unwrap() + 1);

            Ok(StepInfo {
                state: self.state.clone(),
                reward,
                done: true,
                truncated: false,
                info: (),
            })
        }
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
    use crate::testing::{Testable, test_gym_against_python};
    use Gym;

    #[test]
    fn test_cartpole() {
        let mut env = CartPoleV1::builder().build();
        let state = env.reset().expect("Failed to reset environment.");
        assert_eq!(
            state
                .state
                .shape()
                .dim(0)
                .expect("Failed to get state dim."),
            4
        );
        let StepInfo {
            state: next_state,
            reward,
            done,
            truncated: _truncated,
            ..
        } = env
            .step(
                Tensor::from_vec(vec![0 as u32], vec![], &Device::Cpu)
                    .expect("Failed to create tensor."),
            )
            .expect("Failed to step environment.");
        assert_eq!(
            next_state
                .shape()
                .dim(0)
                .expect("Failed to get next state dim."),
            4
        );
        assert!(reward == 1.0);
        assert!(!done);
    }

    #[test]
    #[should_panic]
    fn test_cartpole_invalid_action() {
        let mut env = CartPoleV1::builder().build();
        let _state = env.reset();
        let _info = env
            .step(
                Tensor::from_vec(vec![1 as u32], vec![1], &Device::Cpu)
                    .expect("Failed to create tensor."),
            )
            .expect("Failed to step environment.");
    }

    #[test]
    fn reward_is_one_when_not_terminated() {
        let mut env = CartPoleV1::builder().build();
        env.reset().unwrap();
        let action = Tensor::from_vec(vec![1u32], vec![], &Device::Cpu).unwrap();
        let StepInfo {
            state: _state,
            reward,
            done,
            truncated: _truncated,
            ..
        } = env.step(action).unwrap();
        assert_eq!(reward, 1.0);
        assert!(!done);

        let mut done = false;
        for _ in 0..50 {
            let action = Tensor::from_vec(vec![1u32], vec![], &Device::Cpu).unwrap();
            let StepInfo {
                state: _,
                reward: _,
                done: d,
                truncated: _,
                ..
            } = env.step(action).unwrap();
            done = d;
            if done {
                break;
            }
        }
        assert!(done);
    }

    impl Testable for CartPoleV1 {
        fn reset_deterministic(&mut self) -> Result<Tensor, candle_core::Error> {
            self.reset()?;
            self.state = Tensor::from_vec(vec![0.0f32, 0.0, 0.0, 0.0], vec![4], &Device::Cpu)
                .expect("Failed to create tensor.");
            Ok(self.state.clone())
        }

        /// Restores one CartPole state shaped `[4]`.
        fn set_state(&mut self, state: Tensor, _: Option<serde_json::Value>) {
            self.state = state;
        }
    }

    #[test]
    fn test_cartpole_against_python() {
        test_gym_against_python("cartpole", CartPoleV1::builder().build(), None);
    }

    #[cfg(feature = "rendering")]
    #[test]
    fn test_cartpole_rendering() {
        let mut env = CartPoleV1::builder().render(true).build();
        let _state = env.reset().expect("Failed to reset environment.");
        let action_space = env.action_space();
        for _ in 0..200 {
            let action = action_space.sample(&Device::Cpu).unwrap();
            let StepInfo {
                state: _,
                reward: _,
                done,
                truncated: _,
                ..
            } = env.step(action).unwrap();
            if done {
                env.reset().unwrap();
            }
        }
    }

    #[cfg(any(feature = "cuda", feature = "metal"))]
    #[test]
    fn test_cartpole_tensor_determinism() {
        #[cfg(feature = "cuda")]
        let device = Device::new_cuda(0).unwrap();
        #[cfg(feature = "metal")]
        let device = Device::Metal(0);

        let mut last_state: Option<Tensor> = None;
        for i in 0..10 {
            device.set_seed(42).unwrap();
            let mut env = CartPoleV1::builder().device(&device).build();
            env.reset().unwrap();
            let action_space = env.action_space();
            let mut state = env.state.clone();
            for _ in 0..10 {
                let action = action_space.sample(&device).unwrap();
                let StepInfo {
                    state: next_state,
                    reward: _,
                    done,
                    truncated: _,
                } = env.step(action).unwrap();
                state = next_state;
                if done {
                    break;
                }
            }
            if let Some(last) = &last_state {
                let state_vec = state.to_vec1::<f32>().unwrap();
                let last_vec = last.to_vec1::<f32>().unwrap();
                for (a, b) in state_vec.iter().zip(last_vec.iter()) {
                    assert!(
                        (a - b) == 0.0,
                        "States do not match on iteration {}: {} vs {}",
                        i,
                        a,
                        b
                    );
                }
            }
            last_state = Some(state);
        }
    }
}

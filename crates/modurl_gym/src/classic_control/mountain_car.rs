use bon::bon;
use candle_core::{Device, Tensor};
use modurl::{
    gym::{Gym, ResetInfo, StepInfo},
    spaces::{self, Space},
};

/// The classic Mountain Car environment.
/// Converted from the OpenAI Gym Mountain Car environment.
pub struct MountainCarV0 {
    state: Tensor,
    action_space: spaces::Discrete,
    observation_space: spaces::BoxSpace,
    min_position: f32,
    max_position: f32,
    max_speed: f32,
    goal_position: f32,
    goal_velocity: f32,
    force: f32,
    gravity: f32,
    #[cfg(feature = "rendering")]
    renderer: Option<crate::rendering::Renderer>,
}

#[bon]
impl MountainCarV0 {
    #[builder]
    pub fn new(
        #[builder(default = &Device::Cpu)] device: &Device,
        #[cfg(feature = "rendering")]
        #[builder(default = false)]
        render: bool,
        #[builder(default = 0.0)] goal_velocity: f32,
    ) -> Self {
        let min_position = -1.2;
        let max_position = 0.6;
        let max_speed = 0.07;
        let goal_position = 0.5;
        let force = 0.001;
        let gravity = 0.0025;

        let low = vec![min_position, -max_speed];
        let high = vec![max_position, max_speed];
        let low = Tensor::from_vec(low, vec![2], device).expect("Failed to create tensor.");
        let high = Tensor::from_vec(high, vec![2], device).expect("Failed to create tensor.");

        let action_space = spaces::Discrete::new(3);
        let observation_space = spaces::BoxSpace::new(low, high);

        Self {
            state: Tensor::zeros(vec![2], candle_core::DType::F32, device)
                .expect("Failed to create tensor."),
            action_space,
            observation_space,
            min_position,
            max_position,
            max_speed,
            goal_position,
            goal_velocity,
            force,
            gravity,
            #[cfg(feature = "rendering")]
            renderer: if render {
                Some(crate::rendering::Renderer::new(600, 400, "Mountain Car"))
            } else {
                None
            },
        }
    }

    #[cfg(feature = "rendering")]
    fn render(&mut self) {
        if let Some(renderer) = &mut self.renderer {
            let screen_width = renderer.get_width() as f32;
            let screen_height = renderer.get_height() as f32;

            // Clear screen with white background
            renderer.clear(0xFFFFFF);

            let world_width = self.max_position - self.min_position;
            let scale = screen_width / world_width;
            let car_width = 40.0;
            let car_height = 20.0;
            let clearance = 10.0;

            let state_vec = self.state.to_vec1::<f32>().unwrap();
            let pos = state_vec[0];

            // Collect all the parameters we need
            let min_position = self.min_position;
            let max_position = self.max_position;
            let goal_position = self.goal_position;

            // Draw the mountain contour
            Self::draw_mountain_contour(renderer, scale, screen_height, min_position, max_position);

            // Draw the car
            Self::draw_car(
                renderer,
                pos,
                scale,
                car_width,
                car_height,
                clearance,
                screen_height,
                min_position,
                max_position,
            );

            // Draw the goal flag
            Self::draw_goal_flag(renderer, scale, screen_height, goal_position, min_position);

            renderer.present();
        }
    }

    #[cfg(feature = "rendering")]
    fn height(x: f32) -> f32 {
        (3.0 * x).sin() * 0.45 + 0.55
    }

    #[cfg(feature = "rendering")]
    fn draw_mountain_contour(
        renderer: &mut crate::rendering::Renderer,
        scale: f32,
        screen_height: f32,
        min_position: f32,
        max_position: f32,
    ) {
        // Draw the mountain contour as a series of line segments
        let num_points = 100;
        let mut points = Vec::new();

        for i in 0..=num_points {
            let x = min_position + (max_position - min_position) * (i as f32 / num_points as f32);
            let y = Self::height(x);
            let screen_x = (x - min_position) * scale;
            let screen_y = screen_height - y * scale;
            points.push((screen_x as usize, screen_y as usize));
        }

        // Draw mountain as connected line segments (approximated with small rectangles)
        for i in 0..points.len() - 1 {
            let (x1, y1) = points[i];
            let (x2, y2) = points[i + 1];

            // Draw a small line segment as a filled rectangle
            let dx = x2 as i32 - x1 as i32;
            let dy = y2 as i32 - y1 as i32;
            let steps = dx.abs().max(dy.abs()).max(1);

            for step in 0..steps {
                let x = x1 as i32 + dx * step / steps;
                let y = y1 as i32 + dy * step / steps;
                if x >= 0
                    && y >= 0
                    && x < renderer.get_width() as i32
                    && y < renderer.get_height() as i32
                {
                    renderer.rect(x as usize, y as usize, 2, 2, 0x000000); // Black line
                }
            }
        }
    }

    #[cfg(feature = "rendering")]
    fn draw_car(
        renderer: &mut crate::rendering::Renderer,
        pos: f32,
        scale: f32,
        car_width: f32,
        car_height: f32,
        clearance: f32,
        screen_height: f32,
        min_position: f32,
        max_position: f32,
    ) {
        let car_x = (pos - min_position) * scale;
        let car_y = screen_height - (clearance + Self::height(pos) * scale);

        // Calculate car rotation angle based on slope at current position
        let angle = (3.0 * pos).cos();

        // Draw car body as a rotated rectangle (approximated as quad)
        let l = -car_width / 2.0;
        let r = car_width / 2.0;
        let t = car_height;
        let b = 0.0;

        // Calculate rotated car corners
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        let corners = [
            (l * cos_a - b * sin_a, l * sin_a + b * cos_a),
            (l * cos_a - t * sin_a, l * sin_a + t * cos_a),
            (r * cos_a - t * sin_a, r * sin_a + t * cos_a),
            (r * cos_a - b * sin_a, r * sin_a + b * cos_a),
        ];

        let quad_points = [
            (car_x + corners[0].0, car_y - corners[0].1),
            (car_x + corners[1].0, car_y - corners[1].1),
            (car_x + corners[2].0, car_y - corners[2].1),
            (car_x + corners[3].0, car_y - corners[3].1),
        ];

        // Draw car body using quad function
        renderer.quad(
            quad_points[0],
            quad_points[1],
            quad_points[2],
            quad_points[3],
            0x000000, // Black car
        );

        // Draw wheels - positioned on the mountain surface under the car
        let wheel_offset_x = car_width / 4.0;
        let wheel_radius = (car_height / 2.5) as usize;

        // Calculate wheel positions on the mountain surface
        let wheel_positions_x = [pos - wheel_offset_x / scale, pos + wheel_offset_x / scale];

        for wheel_pos_x in wheel_positions_x.iter() {
            // Make sure wheel position is within bounds
            let clamped_wheel_pos = wheel_pos_x.clamp(min_position, max_position);
            let wheel_screen_x = (clamped_wheel_pos - min_position) * scale;
            let wheel_screen_y =
                screen_height - (clearance + Self::height(clamped_wheel_pos) * scale);

            // Draw wheel as a filled circle (approximated with filled rectangles)
            for dy in -(wheel_radius as i32)..=(wheel_radius as i32) {
                for dx in -(wheel_radius as i32)..=(wheel_radius as i32) {
                    if dx * dx + dy * dy <= (wheel_radius as i32 * wheel_radius as i32) {
                        let px = (wheel_screen_x as i32 + dx) as usize;
                        let py = (wheel_screen_y as i32 + dy) as usize;
                        if px < renderer.get_width() && py < renderer.get_height() {
                            renderer.rect(px, py, 1, 1, 0x808080); // Gray wheels
                        }
                    }
                }
            }
        }
    }

    #[cfg(feature = "rendering")]
    fn draw_goal_flag(
        renderer: &mut crate::rendering::Renderer,
        scale: f32,
        screen_height: f32,
        goal_position: f32,
        min_position: f32,
    ) {
        let flag_x = ((goal_position - min_position) * scale) as usize;
        let flag_y_bottom = (screen_height - Self::height(goal_position) * scale) as usize;

        renderer.draw_flag(
            flag_x,
            flag_y_bottom,
            50,       // pole height
            25,       // flag width
            10,       // flag height
            0x000000, // black pole
            0xCCCC00, // yellow flag
        );
    }
}

impl Default for MountainCarV0 {
    fn default() -> Self {
        MountainCarV0::builder().build()
    }
}

impl Gym for MountainCarV0 {
    type Error = candle_core::Error;
    type SpaceError = candle_core::Error;

    fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
        // Initialize position uniformly between -0.6 and -0.4
        let position = Tensor::rand(-0.6, -0.4, vec![1], self.state.device())?
            .to_dtype(candle_core::DType::F32)?;
        let velocity = Tensor::zeros(vec![1], candle_core::DType::F32, self.state.device())?;

        self.state = Tensor::cat(&[position, velocity], 0)?;

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
        let (mut position, mut velocity) = (state_vec[0], state_vec[1]);

        let action_vec = action.to_vec0::<u32>()?;

        velocity +=
            (action_vec as f32 - 1.0) * self.force + (3.0 * position).cos() * (-self.gravity);

        velocity = velocity.clamp(-self.max_speed, self.max_speed);

        position += velocity;

        position = position.clamp(self.min_position, self.max_position);

        // Handle collision with left wall
        if position == self.min_position && velocity < 0.0 {
            velocity = 0.0;
        }

        self.state = Tensor::from_vec(vec![position, velocity], vec![2], self.state.device())?;

        // Check if goal is reached
        let terminated = position >= self.goal_position && velocity >= self.goal_velocity;
        let reward = -1.0;

        #[cfg(feature = "rendering")]
        self.render();

        Ok(StepInfo {
            state: self.state.clone(),
            reward,
            done: terminated,
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
    use crate::testing::{Testable, test_gym_against_python};
    use Gym;

    #[test]
    fn test_mountain_car() {
        let mut env = MountainCarV0::builder().build();
        let state = env.reset().expect("Failed to reset environment.");
        assert_eq!(
            state
                .state
                .shape()
                .dim(0)
                .expect("Failed to get state dim."),
            2
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
            2
        );
        assert!(reward == -1.0);
        assert!(!done);
    }

    #[test]
    #[should_panic]
    fn test_mountain_car_invalid_action() {
        let mut env = MountainCarV0::builder().build();
        let _state = env.reset();
        let _info = env
            .step(
                Tensor::from_vec(vec![3 as u32], vec![], &Device::Cpu)
                    .expect("Failed to create tensor."),
            )
            .expect("Failed to step environment.");
    }

    #[test]
    fn reward_is_negative_one_when_not_terminated() {
        let mut env = MountainCarV0::builder().build();
        env.reset().unwrap();
        let action = Tensor::from_vec(vec![1u32], vec![], &Device::Cpu).unwrap();
        let StepInfo {
            state: _state,
            reward,
            done,
            truncated: _truncated,
            ..
        } = env.step(action).unwrap();
        assert_eq!(reward, -1.0);
        assert!(!done);
    }

    impl Testable for MountainCarV0 {
        fn reset_deterministic(&mut self) -> Result<Tensor, candle_core::Error> {
            // Set deterministic initial state for testing (matches Python test with options={"low": 0.0, "high": 0.0})
            self.state = Tensor::from_vec(vec![0.0f32, 0.0], vec![2], &Device::Cpu)
                .expect("Failed to create tensor.");
            Ok(self.state.clone())
        }

        /// Restores one MountainCar state shaped `[2]`.
        fn set_state(&mut self, state: Tensor, _: Option<serde_json::Value>) {
            self.state = state;
        }
    }

    #[test]
    fn test_mountain_car_against_python() {
        test_gym_against_python("mountain_car", MountainCarV0::builder().build(), None);
    }

    #[cfg(feature = "rendering")]
    #[test]
    fn test_mountain_car_rendering() {
        let mut env = MountainCarV0::builder().render(true).build();
        env.reset().unwrap();
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
            env.render();
            if done {
                break;
            }
        }
        assert!(env.renderer.is_some());
    }
}

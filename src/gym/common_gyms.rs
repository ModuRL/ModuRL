use crate::{
    gym::{Gym, StepInfo},
    spaces::{self, Space},
};
use candle_core::{Device, Tensor};

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
}

impl CartPoleV1 {
    pub fn new(device: &Device) -> Self {
        let gravity = 9.8;
        let masscart = 1.0;
        let masspole = 0.1;
        let total_mass = masspole + masscart;
        let length = 0.5; // actually half the pole's length
        let polemass_length = masspole * length;
        let force_mag = 10.0;
        let tau = 0.02;
        let is_euler = true;

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

        let action_space = spaces::Discrete::new(2, 0);
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
        }
    }
}

impl Default for CartPoleV1 {
    fn default() -> Self {
        Self::new(&Device::Cpu)
    }
}

impl Gym for CartPoleV1 {
    type Error = candle_core::Error;

    fn get_name(&self) -> &str {
        "CartPoleV1"
    }

    fn reset(&mut self) -> Result<Tensor, Self::Error> {
        self.steps_beyond_terminated = None;
        let state = Tensor::rand(-0.05, 0.05, vec![4], self.state.device())?
            .to_dtype(candle_core::DType::F32)?;
        self.state = state;
        self.steps_since_reset = 0;
        Ok(self.state.clone())
    }

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
            });
        }

        if !terminated {
            Ok(StepInfo {
                state: self.state.clone(),
                reward: 1.0,
                done: false,
                truncated: false,
            })
        } else if self.steps_beyond_terminated.is_none() {
            // Pole just fell!
            self.steps_beyond_terminated = Some(0);
            Ok(StepInfo {
                state: self.state.clone(),
                reward: 0.0,
                done: true,
                truncated: false,
            })
        } else {
            #[cfg(feature = "logging")]
            if self.steps_beyond_terminated == Some(0) {
                log::warn!("You are calling 'step()' even though this environment has already returned terminated = True. You should always call 'reset()' once you receive 'terminated = True'");
            }
            // We already checked this is Some above, so this is safe.
            self.steps_beyond_terminated = Some(self.steps_beyond_terminated.unwrap() + 1);
            Ok(StepInfo {
                state: self.state.clone(),
                reward: 0.0,
                done: true,
                truncated: false,
            })
        }
    }

    fn observation_space(&self) -> Box<dyn crate::spaces::Space> {
        Box::new(self.observation_space.clone())
    }

    fn action_space(&self) -> Box<dyn crate::spaces::Space> {
        Box::new(self.action_space.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gym::Gym;

    #[test]
    fn test_cartpole() {
        let mut env = CartPoleV1::new(&Device::Cpu);
        let state = env.reset().expect("Failed to reset environment.");
        assert_eq!(state.shape().dim(0).expect("Failed to get state dim."), 4);
        let StepInfo {
            state: next_state,
            reward,
            done,
            truncated: _truncated,
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
        let mut env = CartPoleV1::new(&Device::Cpu);
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
        let mut env = CartPoleV1::new(&Device::Cpu);
        env.reset().unwrap();
        let action = Tensor::from_vec(vec![1u32], vec![], &Device::Cpu).unwrap();
        let StepInfo {
            state: _state,
            reward,
            done,
            truncated: _truncated,
        } = env.step(action).unwrap();
        assert_eq!(reward, 1.0);
        assert!(!done);

        let mut reward = 0.0;
        let mut done = false;
        for _ in 0..50 {
            let action = Tensor::from_vec(vec![1u32], vec![], &Device::Cpu).unwrap();
            let StepInfo {
                state: _,
                reward: r,
                done: d,
                truncated: _,
            } = env.step(action).unwrap();
            reward = r;
            done = d;
            if done {
                break;
            }
        }
        assert_eq!(reward, 0.0);
        assert!(done);
    }

    #[test]
    fn reward_is_zero_when_terminated() {
        let mut env = CartPoleV1::new(&Device::Cpu);
        env.reset().unwrap();

        // Force termination by moving x beyond threshold
        env.state = Tensor::from_vec(
            vec![env.x_threshold + 0.1, 0.0, 0.0, 0.0],
            vec![4],
            &Device::Cpu,
        )
        .unwrap();

        let action = Tensor::from_vec(vec![0u32], vec![], &Device::Cpu).unwrap();
        let StepInfo {
            state: _,
            reward,
            done,
            truncated: _,
        } = env.step(action).unwrap();

        assert_eq!(reward, 0.0);
        assert!(done);
    }
}

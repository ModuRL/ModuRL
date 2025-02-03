use crate::{spaces, Gym, Space};
use candle_core::{Device, Tensor};
use log;

/// The classic CartPole environment.
/// Converted from the OpenAI Gym CartPole environment.
pub struct CartPole {
    pub(crate) gravity: f32,
    pub(crate) masscart: f32,
    pub(crate) masspole: f32,
    pub(crate) total_mass: f32,
    pub(crate) length: f32,
    pub(crate) polemass_length: f32,
    pub(crate) force_mag: f32,
    pub(crate) tau: f32,
    pub(crate) x_threshold: f32,
    pub(crate) theta_threshold_radians: f32,
    pub(crate) is_euler: bool,
    pub(crate) steps_beyond_terminated: Option<usize>,
    pub(crate) action_space: spaces::Discrete,
    pub(crate) observation_space: spaces::BoxSpace,
    pub(crate) state: Tensor,
}

impl CartPole {
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
            masscart,
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
        }
    }
}

impl Default for CartPole {
    fn default() -> Self {
        Self::new(&Device::Cpu)
    }
}

impl Gym for CartPole {
    fn get_name(&self) -> &str {
        "CartPole"
    }

    fn reset(&mut self) -> Tensor {
        self.steps_beyond_terminated = None;
        // TODO: make this a little bit random
        self.state = Tensor::zeros(vec![4], candle_core::DType::F32, self.state.device())
            .expect("Failed to create tensor.");
        self.state.clone()
    }

    fn step(&mut self, action: Tensor) -> (Tensor, f32, bool) {
        assert!(self.action_space.contains(&action));
        let state_vec = self.state.to_vec1::<f32>().unwrap();
        let (mut x, mut x_dot, mut theta, mut theta_dot) =
            (state_vec[0], state_vec[1], state_vec[2], state_vec[3]);

        let action_vec = action.to_vec0::<u32>().unwrap();
        let force = if action_vec == 0 {
            self.force_mag
        } else {
            -self.force_mag
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
            x += self.tau * x_dot + 0.5 * self.tau * self.tau * xacc;
            x_dot += 0.5 * self.tau * (xacc + temp);
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

        if !terminated {
            (self.state.clone(), 1.0, false)
        } else if self.steps_beyond_terminated.is_none() {
            // Pole just fell!
            self.steps_beyond_terminated = Some(0);
            (self.state.clone(), 1.0, false)
        } else {
            if self.steps_beyond_terminated == Some(0) {
                log::warn!("You are calling 'step()' even though this environment has already returned terminated = True. You should always call 'reset()' once you receive 'terminated = True' -- any further steps are undefined behavior.");
            }
            self.steps_beyond_terminated = Some(self.steps_beyond_terminated.unwrap() + 1);
            (self.state.clone(), 0.0, true)
        }
    }

    fn observation_space(&self) -> Box<dyn crate::Space> {
        Box::new(self.observation_space.clone())
    }

    fn action_space(&self) -> Box<dyn crate::Space> {
        Box::new(self.action_space.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Gym;

    #[test]
    fn test_cartpole() {
        let mut env = CartPole::new(&Device::Cpu);
        let state = env.reset();
        assert_eq!(state.shape().dim(0).unwrap(), 4);
        let (next_state, reward, done) =
            env.step(Tensor::from_vec(vec![0 as u32], vec![], &Device::Cpu).unwrap());
        assert_eq!(next_state.shape().dim(0).unwrap(), 4);
        assert!(reward == 1.0);
        assert!(!done);
    }

    #[test]
    #[should_panic]
    fn test_cartpole_invalid_action() {
        let mut env = CartPole::new(&Device::Cpu);
        let _state = env.reset();
        let (_next_state, _reward, _done) =
            env.step(Tensor::from_vec(vec![1 as u32], vec![1], &Device::Cpu).unwrap());
    }
}

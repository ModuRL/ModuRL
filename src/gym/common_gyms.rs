use crate::{spaces, Gym};
use log;
use ndarray::ArrayD;

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
    pub(crate) observation_space: spaces::Box,
    pub(crate) state: ArrayD<f32>,
}

impl CartPole {
    pub fn new() -> Self {
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
        let high = ArrayD::from_shape_vec(vec![4], high).unwrap();
        let low = ArrayD::from_shape_vec(vec![4], low).unwrap();

        let action_space = spaces::Discrete::new(vec![2]);
        let observation_space = spaces::Box::new(low, high);

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
            state: ArrayD::zeros(vec![4]),
        }
    }
}

impl Default for CartPole {
    fn default() -> Self {
        Self::new()
    }
}

impl Gym<bool, f32> for CartPole {
    fn get_name(&self) -> &str {
        "CartPole"
    }
    fn reset(&mut self) -> ArrayD<f32> {
        self.steps_beyond_terminated = None;
        // TODO: make this a little bit random
        self.state = ArrayD::zeros(vec![4]);
        self.state.clone()
    }
    fn step(&mut self, action: ArrayD<bool>) -> (ArrayD<f32>, f32, bool) {
        let (mut x, mut x_dot, mut theta, mut theta_dot) =
            (self.state[0], self.state[1], self.state[2], self.state[3]);

        let force = if action[0] {
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

        if (self.is_euler) {
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

        (self.state[0], self.state[1], self.state[2], self.state[3]) = (x, x_dot, theta, theta_dot);
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spaces::Space;
    use crate::Gym;
    use ndarray::ArrayD;

    #[test]
    fn test_cartpole() {
        let mut env = CartPole::new();
        let state = env.reset();
        assert_eq!(state.len(), 4);
        // This is really just to test the action space is discrete and with shape [2]
        assert!(env
            .action_space
            .contains(&ArrayD::from_shape_vec(vec![2], vec![true, true]).unwrap()));
        let (next_state, reward, done) =
            env.step(ArrayD::from_shape_vec(vec![2], vec![true, true]).unwrap());
        assert_eq!(next_state.len(), 4);
        assert!(reward == 1.0);
        assert!(!done);
    }
}

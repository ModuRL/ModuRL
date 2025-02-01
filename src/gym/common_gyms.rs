use crate::Gym;

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
    pub(crate) is_euler: bool,
}

impl CartPole {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for CartPole {
    fn default() -> Self {
        Self {
            gravity: 9.8,
            masscart: 1.0,
            masspole: 0.1,
            total_mass: 1.1,       // masspole + masscart
            length: 0.5,           // actually half the pole's length
            polemass_length: 0.05, // polemass * length
            force_mag: 10.0,
            tau: 0.02,
            is_euler: true,
        }
    }
}

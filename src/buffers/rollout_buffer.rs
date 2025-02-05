use std::vec;

use candle_core::Tensor;

use super::{Experience, ExperienceSample};

pub struct RolloutBuffer {
    buffer: Vec<Experience>,
    batch_size: usize,
}

impl RolloutBuffer {
    pub fn new(batch_size: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(batch_size),
            batch_size,
        }
    }

    pub fn add(&mut self, experience: Experience) {
        self.buffer.push(experience);
    }

    pub fn get_all(&mut self) -> Vec<ExperienceSample> {
        let mut samples = vec![];
        for i in 0..(self.buffer.len() / self.batch_size) {
            let start = i * self.batch_size;
            let end = start + self.batch_size;
            let experiences = &self.buffer[start..end];
            let mut actions = vec![];
            let mut rewards = vec![];
            let mut dones = vec![];
            let mut states = vec![];
            let mut next_states = vec![];

            for experience in experiences {
                actions.push(experience.action.clone());
                rewards.push(experience.reward);
                dones.push(if experience.done { 1.0 } else { 0.0 });
                states.push(experience.state.clone());
                next_states.push(experience.next_state.clone());
            }

            let device = states[0].device();
            let actions = Tensor::stack(&actions, 0).unwrap();
            let rewards = Tensor::from_vec(rewards, vec![self.batch_size], device).unwrap();
            let dones = Tensor::from_vec(dones, vec![self.batch_size], device).unwrap();
            let states = Tensor::stack(&states, 0).unwrap();
            let next_states = Tensor::stack(&next_states, 0).unwrap();

            samples.push(ExperienceSample::new(
                states,
                actions,
                rewards,
                next_states,
                dones,
            ));
        }

        // Clear the buffer
        self.buffer.clear();

        samples
    }
}

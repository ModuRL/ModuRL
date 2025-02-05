use std::collections::VecDeque;

use candle_core::{Error, Tensor};
use rand::Rng;

use super::{Experience, ExperienceSample};

pub(crate) struct ExperienceReplay {
    buffer: VecDeque<Experience>,
    capacity: usize,
    batch_size: usize,
}

impl ExperienceReplay {
    pub fn new(capacity: usize, batch_size: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            batch_size,
        }
    }

    pub fn add(&mut self, experience: Experience) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_back(); // Remove the oldest experience
        }
        self.buffer.push_front(experience);
    }

    pub fn sample(&self) -> Result<ExperienceSample, Error> {
        let mut rng = rand::rng();
        let mut states = vec![];
        let mut actions = vec![];
        let mut rewards = vec![];
        let mut next_states = vec![];
        let mut dones = vec![];

        for _ in 0..self.batch_size {
            let i = rng.random_range(0..self.buffer.len());
            states.push(self.buffer[i].state.clone());
            actions.push(self.buffer[i].action.clone());
            rewards.push(self.buffer[i].reward);
            next_states.push(self.buffer[i].next_state.clone());
            dones.push(if self.buffer[i].done { 1.0 } else { 0.0 });
        }

        let states = Tensor::stack(&states, 0)?;
        let actions = Tensor::stack(&actions, 0)?;
        let rewards = Tensor::from_vec(rewards, vec![self.batch_size], states.device())?;
        let next_states = Tensor::stack(&next_states, 0)?;
        let dones = Tensor::from_vec(dones, vec![self.batch_size], states.device())?;
        Ok(ExperienceSample::new(
            states,
            actions,
            rewards,
            next_states,
            dones,
        ))
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn get_batch_size(&self) -> usize {
        self.batch_size
    }
}

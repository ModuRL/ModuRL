use std::vec;

use candle_core::Tensor;
use rand::seq::SliceRandom;

use super::{experience, ExperienceBatch};

pub struct RolloutBuffer<T> {
    buffer: Vec<T>,
    batch_size: usize,
}

impl<T> RolloutBuffer<T>
where
    T: experience::Experience + Clone,
{
    pub fn new(batch_size: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(batch_size),
            batch_size,
        }
    }

    pub fn add(&mut self, experience: T) {
        self.buffer.push(experience);
    }

    pub fn clear_and_get_all(&mut self) -> Vec<ExperienceBatch<T>> {
        let mut samples = vec![];
        for i in 0..(self.buffer.len() / self.batch_size) {
            let start = i * self.batch_size;
            let end = start + self.batch_size;
            let experiences = &self.buffer[start..end];

            let experience_sample = ExperienceBatch::new(experiences.to_vec());

            samples.push(experience_sample);
        }

        self.buffer.clear();

        samples
    }

    pub fn get_all_shuffled(&mut self) -> Vec<ExperienceBatch<T>> {
        // Shuffle the buffer
        self.buffer.shuffle(&mut rand::rng());
        let samples = self.clear_and_get_all();
        samples
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn get_batch_size(&self) -> usize {
        self.batch_size
    }
}

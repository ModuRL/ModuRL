use super::{experience, ExperienceBatch};
use rand::seq::SliceRandom;
use std::vec;

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

    pub fn get_raw(&self) -> &Vec<T> {
        &self.buffer
    }

    pub fn get_raw_mut(&mut self) -> &mut Vec<T> {
        &mut self.buffer
    }

    pub fn clear_and_get_all(&mut self) -> Vec<ExperienceBatch<T>> {
        let mut samples = vec![];
        for i in 0..(self.buffer.len() / self.batch_size) {
            let start = i * self.batch_size;
            let mut end = start + self.batch_size;
            end = end.max(self.buffer.len());
            let experiences = &self.buffer[start..end];

            let experience_sample = ExperienceBatch::new(experiences.to_vec());

            samples.push(experience_sample);
        }

        self.buffer.clear();

        samples
    }

    /// Shuffles the buffer and returns all samples.
    /// Clears the buffer after returning the samples.
    pub fn clear_and_get_all_shuffled(&mut self) -> Vec<ExperienceBatch<T>> {
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

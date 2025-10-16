use crate::tensor_operations::resevoir_sample;

use super::{ExperienceBatch, experience};
use std::collections::VecDeque;

pub enum ExperienceReplayError<E> {
    TensorError(candle_core::Error),
    ExperienceError(E),
}

impl<E> From<candle_core::Error> for ExperienceReplayError<E> {
    fn from(err: candle_core::Error) -> Self {
        ExperienceReplayError::TensorError(err)
    }
}

pub struct ExperienceReplay<T>
where
    T: experience::Experience,
{
    buffer: VecDeque<T>,
    capacity: usize,
    batch_size: usize,
    device: candle_core::Device,
}

impl<T> ExperienceReplay<T>
where
    T: experience::Experience + Clone,
{
    pub fn new(capacity: usize, batch_size: usize, device: candle_core::Device) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            batch_size,
            device,
        }
    }

    pub fn add(&mut self, experience: T) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_back(); // Remove the oldest experience
        }
        self.buffer.push_front(experience);
    }

    pub fn sample(&self) -> Result<ExperienceBatch<T>, ExperienceReplayError<T::Error>> {
        // shuffle the buffer
        let buffer = self.buffer.iter().collect::<Vec<_>>();

        // Slice the buffer to get the batch
        let total_samples = self.buffer.len();
        let size_to_sample = self.batch_size.min(total_samples);
        let batch = resevoir_sample(&buffer, size_to_sample, &self.device);

        let batch = batch.iter().cloned().cloned().collect();

        let experience_sample =
            ExperienceBatch::new(batch).map_err(ExperienceReplayError::ExperienceError)?;

        Ok(experience_sample)
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn get_batch_size(&self) -> usize {
        self.batch_size
    }
}

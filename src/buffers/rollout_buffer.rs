use crate::tensor_operations;

use super::{ExperienceBatch, experience};
use std::vec;

pub enum RolloutBufferError<E> {
    TensorError(candle_core::Error),
    ExperienceError(E),
}

impl<E> From<candle_core::Error> for RolloutBufferError<E> {
    fn from(err: candle_core::Error) -> Self {
        RolloutBufferError::TensorError(err)
    }
}

pub struct RolloutBuffer<T> {
    buffer: Vec<T>,
    batch_size: usize,
    device: candle_core::Device,
}

impl<T> RolloutBuffer<T>
where
    T: experience::Experience + Clone,
{
    pub fn new(batch_size: usize, device: candle_core::Device) -> Self {
        Self {
            buffer: Vec::with_capacity(batch_size),
            batch_size,
            device,
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

    pub fn get_all(&mut self) -> Result<Vec<ExperienceBatch<T>>, T::Error> {
        let mut samples = vec![];
        for i in 0..(self.buffer.len() / self.batch_size) {
            let start = i * self.batch_size;
            let mut end = start + self.batch_size;
            end = end.min(self.buffer.len());
            let experiences = &self.buffer[start..end];

            let experience_sample = ExperienceBatch::new(experiences.to_vec())?;

            samples.push(experience_sample);
        }

        Ok(samples)
    }

    /// Shuffles the buffer and returns all samples.
    /// Clears the buffer after returning the samples.
    pub fn get_all_shuffled(
        &mut self,
    ) -> Result<Vec<ExperienceBatch<T>>, RolloutBufferError<T::Error>> {
        tensor_operations::fisher_yates_shuffle(&mut self.buffer, &self.device);

        let samples = self
            .get_all()
            .map_err(RolloutBufferError::ExperienceError)?;
        Ok(samples)
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn get_batch_size(&self) -> usize {
        self.batch_size
    }
}

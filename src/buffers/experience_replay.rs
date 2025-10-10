use crate::tensor_operations::gen_range_int_tensor;

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
        let mut buffer = self.buffer.iter().collect::<Vec<_>>();

        // Shuffle using Fisher-Yates algorithm
        for i in (1..buffer.len()).rev() {
            let j = gen_range_int_tensor(0, i as u32, &self.device)? as usize;
            buffer.swap(i, j);
        }

        // Slice the buffer to get the batch
        let total_samples = self.buffer.len();
        let batch = &buffer[0..self.batch_size.min(total_samples)];

        let experience_sample =
            ExperienceBatch::new(batch.to_vec().iter().cloned().cloned().collect())
                .map_err(ExperienceReplayError::ExperienceError)?;

        Ok(experience_sample)
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn get_batch_size(&self) -> usize {
        self.batch_size
    }
}

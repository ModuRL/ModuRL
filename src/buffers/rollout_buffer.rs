use super::{ExperienceBatch, experience};
use crate::sampling::shuffle_with_device_rng;
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
    T: experience::Experience,
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

    pub fn get_all(&self) -> Result<Vec<T::Batch>, T::Error> {
        if self.batch_size == 0 {
            return Ok(Vec::new());
        }

        self.buffer.chunks(self.batch_size).map(T::batch).collect()
    }

    /// Shuffles the buffer and returns all samples.
    pub fn get_all_shuffled(
        &mut self,
    ) -> Result<Vec<ExperienceBatch<T>>, RolloutBufferError<T::Error>> {
        shuffle_with_device_rng(&mut self.buffer, &self.device)?;

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

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn get_batch_size(&self) -> usize {
        self.batch_size
    }
}

use super::{experience, ExperienceBatch};
use rand::seq::SliceRandom;
use std::collections::VecDeque;

pub(crate) struct ExperienceReplay<T>
where
    T: experience::Experience,
{
    buffer: VecDeque<T>,
    capacity: usize,
    batch_size: usize,
}

impl<T> ExperienceReplay<T>
where
    T: experience::Experience + Clone,
{
    pub fn new(capacity: usize, batch_size: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            batch_size,
        }
    }

    pub fn add(&mut self, experience: T) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_back(); // Remove the oldest experience
        }
        self.buffer.push_front(experience);
    }

    pub fn sample(&self) -> Result<ExperienceBatch<T>, T::Error> {
        let mut rng = rand::rng();

        // shuffle the buffer
        let mut buffer = self.buffer.iter().collect::<Vec<_>>();
        buffer.shuffle(&mut rng);

        // Slice the buffer to get the batch
        let total_samples = self.buffer.len();
        let batch = &buffer[0..self.batch_size.min(total_samples)];

        let experience_sample =
            ExperienceBatch::new(batch.to_vec().iter().cloned().cloned().collect())?;

        Ok(experience_sample)
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn get_batch_size(&self) -> usize {
        self.batch_size
    }
}

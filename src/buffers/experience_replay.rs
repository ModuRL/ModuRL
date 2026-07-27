use super::{ExperienceBatch, experience};
use rand::Rng;
use std::collections::{HashMap, VecDeque};

#[derive(Debug)]
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
}

impl<T> ExperienceReplay<T>
where
    T: experience::Experience + Clone,
{
    pub fn new(capacity: usize, batch_size: usize, _device: candle_core::Device) -> Self {
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

    pub fn sample(&self) -> Result<ExperienceBatch<T>, ExperienceReplayError<T::Error>> {
        let total_samples = self.buffer.len();
        let size_to_sample = self.batch_size.min(total_samples);
        let indices = sample_indices_without_replacement(total_samples, size_to_sample);
        let batch = indices
            .into_iter()
            .map(|index| self.buffer[index].clone())
            .collect();

        let experience_sample =
            ExperienceBatch::new(batch).map_err(ExperienceReplayError::ExperienceError)?;

        Ok(experience_sample)
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

/// Performs the first `sample_size` steps of a Fisher-Yates shuffle without
/// allocating or scanning a full `population_size` index vector.
fn sample_indices_without_replacement(population_size: usize, sample_size: usize) -> Vec<usize> {
    if sample_size == 0 {
        return Vec::new();
    }
    let mut rng = rand::rng();
    let mut swaps = HashMap::with_capacity(sample_size * 2);
    let mut indices = Vec::with_capacity(sample_size);
    for i in 0..sample_size {
        let j = rng.random_range(i..population_size);
        let at_i = swaps.get(&i).copied().unwrap_or(i);
        let at_j = swaps.get(&j).copied().unwrap_or(j);
        indices.push(at_j);
        swaps.insert(j, at_i);
    }
    indices
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffers::experience::Experience;
    use candle_core::{Device, Tensor};

    #[derive(Clone)]
    struct Number(u32);

    impl Experience for Number {
        type Error = candle_core::Error;

        fn get_elements(&self) -> Result<Vec<Tensor>, Self::Error> {
            Ok(vec![Tensor::new(self.0, &Device::Cpu)?])
        }
    }

    #[test]
    fn sampling_is_unique_and_uses_requested_batch_size() {
        let mut replay = ExperienceReplay::new(100, 32, Device::Cpu);
        for value in 0..100 {
            replay.add(Number(value));
        }
        let values = replay.sample().unwrap().get_elements()[0]
            .to_vec1::<u32>()
            .unwrap();
        let unique = values
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(values.len(), 32);
        assert_eq!(unique.len(), 32);
    }

    #[test]
    fn undersized_replay_returns_every_available_item_once() {
        let mut replay = ExperienceReplay::new(100, 32, Device::Cpu);
        for value in 0..7 {
            replay.add(Number(value));
        }
        let values = replay.sample().unwrap().get_elements()[0]
            .to_vec1::<u32>()
            .unwrap();
        let unique = values
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(values.len(), 7);
        assert_eq!(unique.len(), 7);
    }
}

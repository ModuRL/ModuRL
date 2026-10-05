use burn::tensor::Device;

use super::experience;
use crate::sampling::{SamplingError, shuffle_with_device_rng};

#[derive(Debug, thiserror::Error)]
pub enum RolloutBufferError<E> {
    #[error("rollout shuffling failed: {0}")]
    SamplingError(#[from] SamplingError),
    #[error("rollout experience failed: {0}")]
    ExperienceError(#[source] E),
}

/// Stores experiences and delegates tensor layout to [`experience::Experience`].
pub struct RolloutBuffer<T> {
    buffer: Vec<T>,
    batch_size: usize,
    device: Device,
}

impl<T> RolloutBuffer<T>
where
    T: experience::Experience,
{
    /// Creates a buffer whose device supplies shuffle randomness. A zero batch
    /// size disables batch production until the buffer is replaced.
    pub fn new(batch_size: usize, device: Device) -> Self {
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
    pub fn get_all_shuffled(&mut self) -> Result<Vec<T::Batch>, RolloutBufferError<T::Error>> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Tensor};

    #[derive(Clone)]
    struct TestExperience {
        id: i32,
    }

    #[derive(Debug, PartialEq, Eq, thiserror::Error)]
    #[error("negative experience id")]
    struct InvalidExperience;

    impl experience::Experience for TestExperience {
        type Batch = Tensor<1, Int>;
        type Error = InvalidExperience;

        fn batch(experiences: &[Self]) -> Result<Self::Batch, Self::Error> {
            if experiences.iter().any(|experience| experience.id < 0) {
                return Err(InvalidExperience);
            }
            let ids: Vec<_> = experiences.iter().map(|experience| experience.id).collect();
            Ok(Tensor::from_data(ids.as_slice(), &Device::flex()))
        }
    }

    #[test]
    fn batches_native_tensors_without_dropping_the_remainder() {
        let mut buffer = RolloutBuffer::new(2, Device::flex());
        for id in 0..5 {
            buffer.add(TestExperience { id });
        }
        let batches = buffer.get_all().unwrap();
        let values: Vec<_> = batches
            .into_iter()
            .map(|batch| batch.into_data().try_to_vec::<i32>().unwrap())
            .collect();
        assert_eq!(values, vec![vec![0, 1], vec![2, 3], vec![4]]);
    }

    #[test]
    fn shuffled_batches_preserve_experiences() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let mut buffer = RolloutBuffer::new(3, Device::flex());
        for id in 0..10 {
            buffer.add(TestExperience { id });
        }
        let batches = buffer.get_all_shuffled().unwrap();
        assert_eq!(
            batches
                .iter()
                .map(|batch| batch.dims()[0])
                .collect::<Vec<_>>(),
            vec![3, 3, 3, 1]
        );
        let mut values: Vec<_> = batches
            .into_iter()
            .flat_map(|batch| batch.into_data().try_to_vec::<i32>().unwrap())
            .collect();
        values.sort_unstable();
        assert_eq!(values, (0..10).collect::<Vec<_>>());
        assert_eq!(buffer.len(), 10);
    }

    #[test]
    fn propagates_experience_failures() {
        let mut buffer = RolloutBuffer::new(1, Device::flex());
        buffer.add(TestExperience { id: -1 });
        assert!(matches!(buffer.get_all(), Err(InvalidExperience)));
        assert!(matches!(
            buffer.get_all_shuffled(),
            Err(RolloutBufferError::ExperienceError(InvalidExperience))
        ));
    }

    #[test]
    fn disabled_batching_keeps_raw_experiences() {
        let mut buffer = RolloutBuffer::new(0, Device::flex());
        buffer.add(TestExperience { id: 7 });
        assert!(buffer.get_all().unwrap().is_empty());
        assert_eq!(buffer.get_raw()[0].id, 7);
        buffer.clear();
        assert!(buffer.is_empty());
    }
}

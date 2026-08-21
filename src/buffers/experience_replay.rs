use candle_core::{DType, Device, Tensor};
use rand::Rng;
use std::collections::HashMap;
use std::marker::PhantomData;

#[derive(Debug)]
pub enum ExperienceReplayError<E> {
    TensorError(candle_core::Error),
    ExperienceError(E),
    InsertionExceedsCapacity { capacity: usize, inserted: usize },
}

impl<E> From<candle_core::Error> for ExperienceReplayError<E> {
    fn from(err: candle_core::Error) -> Self {
        ExperienceReplayError::TensorError(err)
    }
}

#[derive(Debug)]
pub enum ReplayStorageError {
    TensorError(candle_core::Error),
    MissingBatchDimension,
    ItemShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    BatchLengthMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    InsertionExceedsCapacity {
        capacity: usize,
        inserted: usize,
    },
}

impl From<candle_core::Error> for ReplayStorageError {
    fn from(error: candle_core::Error) -> Self {
        Self::TensorError(error)
    }
}

impl From<ExperienceReplayError<ReplayStorageError>> for ReplayStorageError {
    fn from(error: ExperienceReplayError<ReplayStorageError>) -> Self {
        match error {
            ExperienceReplayError::TensorError(error) => Self::TensorError(error),
            ExperienceReplayError::ExperienceError(error) => error,
            ExperienceReplayError::InsertionExceedsCapacity { capacity, inserted } => {
                Self::InsertionExceedsCapacity { capacity, inserted }
            }
        }
    }
}

pub trait ReplayStorage {
    type Insert;
    type Batch;
    type Error;

    fn capacity(&self) -> usize;
    fn insert(&mut self, start: usize, transitions: Self::Insert) -> Result<usize, Self::Error>;
    fn gather(&self, indices: &[usize]) -> Result<Self::Batch, Self::Error>;
}

pub(crate) struct TensorReplayColumn {
    tensor: Tensor,
}

impl TensorReplayColumn {
    pub(crate) fn new(
        capacity: usize,
        item_shape: &[usize],
        dtype: DType,
        device: &Device,
    ) -> Result<Self, ReplayStorageError> {
        let mut shape = Vec::with_capacity(item_shape.len() + 1);
        shape.push(capacity);
        shape.extend_from_slice(item_shape);
        Ok(Self {
            tensor: Tensor::zeros(shape.as_slice(), dtype, device)?,
        })
    }

    /// Converts a `[batch, ...]` tensor to this column's dtype and device.
    pub(crate) fn prepare(&self, source: &Tensor) -> Result<Tensor, ReplayStorageError> {
        if source.rank() == 0 {
            return Err(ReplayStorageError::MissingBatchDimension);
        }
        if self.tensor.dims()[1..] != source.dims()[1..] {
            return Err(ReplayStorageError::ItemShapeMismatch {
                expected: self.tensor.dims()[1..].to_vec(),
                actual: source.dims()[1..].to_vec(),
            });
        }
        Ok(source
            .to_dtype(self.tensor.dtype())?
            .to_device(self.tensor.device())?
            .contiguous()?)
    }

    /// Copies `[batch, ...]` into the capacity-sized column with ring wrapping.
    pub(crate) fn write(&self, start: usize, prepared: &Tensor) -> Result<(), ReplayStorageError> {
        let capacity = self.tensor.dim(0)?;
        let count = prepared.dim(0)?;
        if count > capacity {
            return Err(ReplayStorageError::InsertionExceedsCapacity {
                capacity,
                inserted: count,
            });
        }
        let first_count = count.min(capacity - start);
        if first_count != 0 {
            self.tensor
                .slice_set(&prepared.narrow(0, 0, first_count)?.contiguous()?, 0, start)?;
        }
        let second_count = count - first_count;
        if second_count != 0 {
            self.tensor.slice_set(
                &prepared
                    .narrow(0, first_count, second_count)?
                    .contiguous()?,
                0,
                0,
            )?;
        }
        Ok(())
    }

    /// Gathers owning rows using indices shaped `[sample_count]`.
    pub(crate) fn gather(&self, indices: &Tensor) -> Result<Tensor, ReplayStorageError> {
        Ok(self.tensor.index_select(indices, 0)?.detach())
    }
}

pub(crate) fn replay_index_tensor(
    indices: &[usize],
    device: &Device,
) -> Result<Tensor, ReplayStorageError> {
    let indices = indices
        .iter()
        .map(|&index| i64::try_from(index).expect("replay index must fit in i64"))
        .collect::<Vec<_>>();
    let count = indices.len();
    Ok(Tensor::from_vec(indices, count, device)?)
}

pub struct ExperienceReplay<T, S>
where
    S: ReplayStorage<Insert = T>,
{
    storage: S,
    position: usize,
    len: usize,
    batch_size: usize,
    _insert: PhantomData<fn(T)>,
}

impl<T, S> ExperienceReplay<T, S>
where
    S: ReplayStorage<Insert = T>,
{
    pub fn with_storage(storage: S, batch_size: usize) -> Self {
        Self {
            storage,
            position: 0,
            len: 0,
            batch_size,
            _insert: PhantomData,
        }
    }

    pub fn add(&mut self, transitions: T) -> Result<(), ExperienceReplayError<S::Error>> {
        let capacity = self.storage.capacity();
        let inserted = self
            .storage
            .insert(self.position, transitions)
            .map_err(ExperienceReplayError::ExperienceError)?;
        if inserted > capacity {
            return Err(ExperienceReplayError::InsertionExceedsCapacity { capacity, inserted });
        }
        self.position = (self.position + inserted) % capacity;
        self.len = (self.len + inserted).min(capacity);
        Ok(())
    }

    pub fn sample(&self) -> Result<S::Batch, ExperienceReplayError<S::Error>> {
        let total_samples = self.len;
        let size_to_sample = self.batch_size.min(total_samples);
        let indices = sample_indices_without_replacement(total_samples, size_to_sample);
        self.storage
            .gather(&indices)
            .map_err(ExperienceReplayError::ExperienceError)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
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
    use candle_core::{DType, Device, Tensor};
    use std::convert::Infallible;

    struct IndexStorage {
        capacity: usize,
    }

    impl ReplayStorage for IndexStorage {
        type Insert = usize;
        type Batch = Vec<usize>;
        type Error = Infallible;

        fn capacity(&self) -> usize {
            self.capacity
        }

        fn insert(&mut self, _start: usize, count: usize) -> Result<usize, Self::Error> {
            Ok(count)
        }

        fn gather(&self, indices: &[usize]) -> Result<Self::Batch, Self::Error> {
            Ok(indices.to_vec())
        }
    }

    #[test]
    fn tensor_column_wraps_without_mutating_an_already_gathered_batch() {
        let device = Device::Cpu;
        let column = TensorReplayColumn::new(3, &[1], DType::F32, &device).unwrap();
        let first = Tensor::from_vec(vec![1.0f32, 2.0], (2, 1), &device).unwrap();
        let first = column.prepare(&first).unwrap();
        column.write(0, &first).unwrap();
        let gathered = column
            .gather(&Tensor::new(&[0u32], &device).unwrap())
            .unwrap();

        let wrapped = Tensor::from_vec(vec![3.0f32, 4.0], (2, 1), &device).unwrap();
        let wrapped = column.prepare(&wrapped).unwrap();
        column.write(2, &wrapped).unwrap();

        assert_eq!(gathered.to_vec2::<f32>().unwrap(), vec![vec![1.0]]);
        assert_eq!(
            column
                .gather(&Tensor::new(&[0u32, 1, 2], &device).unwrap())
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            vec![vec![4.0], vec![2.0], vec![3.0]]
        );
    }

    #[test]
    fn tensor_column_reports_invalid_insert_shapes_and_capacity() {
        let device = Device::Cpu;
        let column = TensorReplayColumn::new(3, &[2], DType::F32, &device).unwrap();

        assert!(matches!(
            column.prepare(&Tensor::new(1.0f32, &device).unwrap()),
            Err(ReplayStorageError::MissingBatchDimension)
        ));
        assert!(matches!(
            column.prepare(&Tensor::zeros((1, 3), DType::F32, &device).unwrap()),
            Err(ReplayStorageError::ItemShapeMismatch {
                expected,
                actual
            }) if expected == [2] && actual == [3]
        ));
        assert!(matches!(
            column.write(0, &Tensor::zeros((4, 2), DType::F32, &device).unwrap()),
            Err(ReplayStorageError::InsertionExceedsCapacity {
                capacity: 3,
                inserted: 4
            })
        ));
    }

    #[test]
    fn sampling_is_unique_and_uses_requested_batch_size() {
        let mut replay = ExperienceReplay::with_storage(IndexStorage { capacity: 100 }, 32);
        replay.add(100).unwrap();
        let values = replay.sample().unwrap();
        let unique = values
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(values.len(), 32);
        assert_eq!(unique.len(), 32);
    }

    #[test]
    fn undersized_replay_returns_every_available_item_once() {
        let mut replay = ExperienceReplay::with_storage(IndexStorage { capacity: 100 }, 32);
        replay.add(7).unwrap();
        let values = replay.sample().unwrap();
        let unique = values
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(values.len(), 7);
        assert_eq!(unique.len(), 7);
    }

    #[test]
    fn replay_reports_storage_insertions_larger_than_capacity() {
        let mut replay = ExperienceReplay::with_storage(IndexStorage { capacity: 3 }, 2);
        assert!(matches!(
            replay.add(4),
            Err(ExperienceReplayError::InsertionExceedsCapacity {
                capacity: 3,
                inserted: 4
            })
        ));
    }
}

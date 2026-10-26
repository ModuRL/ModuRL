use burn::tensor::{
    DType, Device, Float, IndexingUpdateOp, Int, Slice, Tensor, TensorCreationOptions, TensorData,
    kind::{Autodiff, Kind},
};
use rand::RngExt;
use std::collections::HashMap;
use std::marker::PhantomData;

#[derive(Debug, thiserror::Error)]
pub enum ExperienceReplayError<E> {
    #[error("replay experience failed: {0}")]
    ExperienceError(#[source] E),
    #[error("cannot insert {inserted} items into replay capacity {capacity}")]
    InsertionExceedsCapacity { capacity: usize, inserted: usize },
}

#[derive(Debug, thiserror::Error)]
pub enum ReplayStorageError {
    #[error("replay field {field} has length {actual}, expected {expected}")]
    BatchLengthMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("replay field {field} has shape {actual:?}, expected {expected:?}")]
    ShapeMismatch {
        field: &'static str,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("replay field {field} has dtype {actual:?}, expected {expected:?}")]
    DTypeMismatch {
        field: &'static str,
        expected: DType,
        actual: DType,
    },
    #[error("cannot insert {inserted} items into replay capacity {capacity}")]
    InsertionExceedsCapacity { capacity: usize, inserted: usize },
    #[error("replay environment count {actual} does not match expected count {expected}")]
    EnvironmentCountMismatch { expected: usize, actual: usize },
    #[error("replay capacity {capacity} is not aligned to environment count {environment_count}")]
    InvalidReplayAlignment {
        capacity: usize,
        environment_count: usize,
    },
    #[error("replay capacity {capacity} overflows when adding {additional} items")]
    CapacityOverflow { capacity: usize, additional: usize },
    #[error("the replay environment count is not set")]
    EnvironmentCountNotSet,
    #[error("replay index {0} exceeds the supported integer range")]
    IndexTooLarge(usize),
}

impl From<ExperienceReplayError<ReplayStorageError>> for ReplayStorageError {
    fn from(error: ExperienceReplayError<ReplayStorageError>) -> Self {
        match error {
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

/// Stores one replay field as a native tensor `[capacity, ...item_shape]` of rank `R`.
/// Tensor kind is fixed by `K`; callers convert inserted values to the storage dtype and device.
pub(crate) struct TensorReplayColumn<const R: usize, K: Autodiff = Float> {
    tensor: Tensor<R, K>,
}

impl<const R: usize, K: Autodiff> TensorReplayColumn<R, K> {
    /// Returns storage `[capacity, ...item_shape]` of rank `R` without copying tensor data.
    pub(crate) fn tensor(&self) -> &Tensor<R, K> {
        &self.tensor
    }

    /// Allocates `[capacity, ...item_shape]` of rank `R >= 1` without autodiff.
    /// The first axis is the ring capacity. The selected dtype must belong to kind `K`.
    pub(crate) fn new(shape: [usize; R], options: impl Into<TensorCreationOptions>) -> Self {
        const {
            assert!(R >= 1, "replay columns require a capacity axis");
        }
        let mut options = options.into();
        options.device = options.device.without_autodiff();
        Self {
            tensor: Tensor::zeros(shape, options),
        }
    }

    /// Stores `[batch_size, ...item_shape]` of rank `R`, wrapping at the capacity axis.
    /// Item shape, dtype, and device must match the column. Start must be less than capacity.
    /// Removes the input's autodiff association before storage.
    pub(crate) fn write(
        &mut self,
        start: usize,
        source: Tensor<R, K>,
    ) -> Result<(), ReplayStorageError> {
        let capacity = self.tensor.dims()[0];
        let count = source.dims()[0];
        if count > capacity {
            return Err(ReplayStorageError::InsertionExceedsCapacity {
                capacity,
                inserted: count,
            });
        }
        let source = source.without_autodiff();
        let first_count = count.min(capacity - start);
        if first_count != 0 {
            // Copy the rows that fit between start and the end of the ring, including every item axis.
            // slice_assign returns the updated tensor; inplace stores that tensor back in the column.
            let first = source.clone().slice([Slice::from(0..first_count)]);
            self.tensor.inplace(|tensor| {
                tensor.slice_assign([Slice::from(start..start + first_count)], first)
            });
        }
        if first_count < count {
            // Copy the remaining source rows to the beginning of the ring after wrapping past capacity.
            let second = source.slice([Slice::from(first_count..count)]);
            self.tensor.inplace(|tensor| {
                tensor.slice_assign([Slice::from(0..count - first_count)], second)
            });
        }
        Ok(())
    }

    /// Selects rows with `Int` indices `[sample_count]`, returning `[sample_count, ...item_shape]` of rank `R`.
    /// Indices must address rows containing stored transitions and use the storage device.
    /// The result preserves dtype and kind, has no autodiff graph, and remains unchanged after later writes.
    pub(crate) fn gather(&self, indices: Tensor<1, Int>) -> Tensor<R, K> {
        self.tensor.clone().select(0, indices)
    }
}

/// Stores observations in one `[capacity + environment_count, ...item_shape]` tensor of rank `R`.
/// A transition at row `i` uses observations at `i` and `i + environment_count`.
/// Wrapped frontier states and truncated terminal observations are retained separately.
pub(crate) struct AlignedObservationReplay<const R: usize, K: Autodiff = Float> {
    observations: Tensor<R, K>,
    truncated_next_states: HashMap<usize, Tensor<R, K>>,
    frontier_states: Option<(usize, Tensor<R, K>)>,
    environment_count: usize,
    inserted: usize,
}

impl<const R: usize, K: Autodiff> AlignedObservationReplay<R, K> {
    /// Returns the logical shape `[capacity, ...item_shape]`, excluding the extra rows for next observations.
    pub(crate) fn shape(&self) -> [usize; R] {
        let mut shape = self.observations.dims();
        shape[0] -= self.environment_count;
        shape
    }

    /// Returns storage `[capacity + environment_count, ...item_shape]` of rank `R` without copying tensor data.
    #[cfg(test)]
    pub(crate) fn observations(&self) -> &Tensor<R, K> {
        &self.observations
    }

    /// Allocates rank-`R >= 2` storage from logical shape `[capacity, ...item_shape]`.
    /// Adds `environment_count` rows for next observations. Capacity must exceed and be divisible by the environment count.
    /// The selected dtype must belong to kind `K`. Storage has no autodiff association.
    pub(crate) fn new(
        mut shape: [usize; R],
        environment_count: usize,
        options: impl Into<TensorCreationOptions>,
    ) -> Result<Self, ReplayStorageError> {
        const {
            assert!(R >= 2, "replay observations require capacity and item axes");
        }
        shape[0] = validate_replay_alignment(shape[0], environment_count)?;
        let mut options = options.into();
        options.device = options.device.without_autodiff();
        Ok(Self {
            observations: Tensor::zeros(shape, options),
            truncated_next_states: HashMap::new(),
            frontier_states: None,
            environment_count,
            inserted: 0,
        })
    }

    pub(crate) fn validate_environment_count(
        &self,
        environment_count: usize,
    ) -> Result<(), ReplayStorageError> {
        if self.environment_count != environment_count {
            return Err(ReplayStorageError::EnvironmentCountMismatch {
                expected: self.environment_count,
                actual: environment_count,
            });
        }
        Ok(())
    }

    /// Validates and stores `states` and `next_states` shaped `[environment_count, ...item_shape]` of rank `R`.
    /// Both tensors must match storage shape, kind, and dtype; they are transferred to the storage device.
    /// Start must be a multiple of `environment_count` below capacity; `truncateds` has one flag per environment.
    /// Removes autodiff associations. Later state batches overwrite shared next-state rows.
    pub(crate) fn insert(
        &mut self,
        start: usize,
        states: Tensor<R, K>,
        next_states: Tensor<R, K>,
        truncateds: &[bool],
    ) -> Result<usize, ReplayStorageError> {
        let capacity = self.shape()[0];
        let count = states.dims()[0];
        let mut expected_shape = self.shape();
        expected_shape[0] = count;
        validate_replay_shape("observations", &states, expected_shape)?;
        validate_replay_shape("next observations", &next_states, expected_shape)?;

        let expected_dtype = self.observations.dtype();
        for (field, actual) in [
            ("observations", states.dtype()),
            ("next observations", next_states.dtype()),
        ] {
            if actual != expected_dtype {
                return Err(ReplayStorageError::DTypeMismatch {
                    field,
                    expected: expected_dtype,
                    actual,
                });
            }
        }
        if truncateds.len() != count {
            return Err(ReplayStorageError::BatchLengthMismatch {
                field: "truncateds",
                expected: count,
                actual: truncateds.len(),
            });
        }
        self.validate_environment_count(count)?;
        let device = self.observations.device();
        let states = states.to_device(&device).without_autodiff();
        let next_states = next_states.to_device(&device).without_autodiff();
        let offsets: Vec<_> = truncateds
            .iter()
            .enumerate()
            .filter_map(|(offset, &truncated)| truncated.then_some(offset))
            .collect();
        let terminal_states = if offsets.is_empty() {
            None
        } else {
            Some(next_states.clone().select(
                0,
                replay_index_tensor(&offsets, &self.observations.device())?,
            ))
        };
        self.frontier_states = if start + count < capacity && self.inserted >= capacity {
            let frontier_start = start + count;
            let frontier_rows = (frontier_start..start + 2 * count).collect::<Vec<_>>();
            Some((
                frontier_start,
                self.observations
                    .clone()
                    .select(0, replay_index_tensor(&frontier_rows, &device)?),
            ))
        } else {
            None
        };
        let observations = &mut self.observations;
        // Reuse the previous batch's next-observation rows for current observations to avoid storing each observation twice.
        observations
            .inplace(|tensor| tensor.slice_assign([Slice::from(start..start + count)], states));
        // Place next observations one environment batch ahead so transition i reads rows i and i + count.
        observations.inplace(|tensor| {
            tensor.slice_assign([Slice::from(start + count..start + 2 * count)], next_states)
        });
        for offset in 0..count {
            self.truncated_next_states
                .remove(&((start + offset) % capacity));
        }
        if let Some(terminals) = terminal_states {
            for (row, offset) in offsets.into_iter().enumerate() {
                // Each saved row references the compact terminal batch rather than the replay buffer.
                self.truncated_next_states.insert(
                    (start + offset) % capacity,
                    terminals.clone().slice([Slice::from(row..row + 1)]),
                );
            }
        }
        self.inserted = self.inserted.saturating_add(count);
        Ok(count)
    }

    /// Returns states and next states `[sample_count, ...item_shape]` of rank `R` for host indices `[sample_count]`.
    /// Indices must address rows containing stored transitions. Repeated indices are supported.
    /// Both results preserve storage kind, dtype, and device without gradients.
    /// Terminal observations replace next states for truncated transitions; later writes do not change the returned batches.
    pub(crate) fn gather(
        &self,
        indices: &[usize],
    ) -> Result<(Tensor<R, K>, Tensor<R, K>), ReplayStorageError> {
        let observations = &self.observations;
        let environment_count = self.environment_count;
        let rows = replay_index_tensor(indices, &self.observations.device())?;
        let mut states = observations.clone().select(0, rows.clone());
        let mut next_states = observations
            .clone()
            .select(0, rows + environment_count as i64);
        let mut frontier_positions = Vec::new();
        let mut saved_states = Vec::new();
        if let Some((frontier_start, frontier_states)) = &self.frontier_states {
            let frontier_end = *frontier_start + environment_count;
            for (batch_index, replay_index) in indices.iter().copied().enumerate() {
                if (*frontier_start..frontier_end).contains(&replay_index) {
                    frontier_positions.push(batch_index);
                    let offset = replay_index - *frontier_start;
                    saved_states.push(
                        frontier_states
                            .clone()
                            .slice([Slice::from(offset..offset + 1)]),
                    );
                }
            }
        }
        if !frontier_positions.is_empty() {
            if const { matches!(K::KIND, Kind::Bool) } {
                for (position, state) in frontier_positions.into_iter().zip(saved_states) {
                    states = states.slice_assign([Slice::from(position..position + 1)], state);
                }
            } else {
                let saved_states = Tensor::cat(saved_states, 0);
                let rows = replay_index_tensor(&frontier_positions, &self.observations.device())?;
                states = states.select_assign(0, rows, saved_states, IndexingUpdateOp::Assign);
            }
        }
        let mut positions = Vec::new();
        let mut terminals = Vec::new();
        for (batch_index, replay_index) in indices.iter().copied().enumerate() {
            if let Some(terminal_state) = self.truncated_next_states.get(&replay_index) {
                positions.push(batch_index);
                terminals.push(terminal_state.clone());
            }
        }
        if !positions.is_empty() {
            if const { matches!(K::KIND, Kind::Bool) } {
                // Boolean indexed updates support OR, so replace each terminal row with a slice assignment.
                for (position, terminal) in positions.into_iter().zip(terminals) {
                    next_states =
                        next_states.slice_assign([Slice::from(position..position + 1)], terminal);
                }
            } else {
                let terminal_states = Tensor::cat(terminals, 0);
                let rows = replay_index_tensor(&positions, &self.observations.device())?;
                next_states =
                    next_states.select_assign(0, rows, terminal_states, IndexingUpdateOp::Assign);
            }
        }
        Ok((states, next_states))
    }
}

/// Validates a replay tensor's shape against the expected shape.
fn validate_replay_shape<const R: usize, K: Autodiff>(
    field: &'static str,
    tensor: &Tensor<R, K>,
    expected: [usize; R],
) -> Result<(), ReplayStorageError> {
    if tensor.dims() != expected {
        return Err(ReplayStorageError::ShapeMismatch {
            field,
            expected: expected.to_vec(),
            actual: tensor.dims().to_vec(),
        });
    }
    Ok(())
}

/// Validates the replay capacity and returns the allocated row count, including one next observation per environment.
pub(crate) fn validate_replay_alignment(
    capacity: usize,
    environment_count: usize,
) -> Result<usize, ReplayStorageError> {
    if environment_count == 0
        || capacity <= environment_count
        || !capacity.is_multiple_of(environment_count)
    {
        return Err(ReplayStorageError::InvalidReplayAlignment {
            capacity,
            environment_count,
        });
    }
    capacity
        .checked_add(environment_count)
        .ok_or(ReplayStorageError::CapacityOverflow {
            capacity,
            additional: environment_count,
        })
}

/// Converts host indices `[sample_count]` to an `I64` `Int` tensor `[sample_count]` on `device`.
/// Axis order is unchanged; indices must fit the signed 64-bit range.
pub(crate) fn replay_index_tensor(
    indices: &[usize],
    device: &Device,
) -> Result<Tensor<1, Int>, ReplayStorageError> {
    let indices = indices
        .iter()
        .map(|&index| i64::try_from(index).map_err(|_| ReplayStorageError::IndexTooLarge(index)))
        .collect::<Result<Vec<_>, _>>()?;
    let count = indices.len();
    Ok(Tensor::from_data(
        TensorData::new(indices, [count]),
        (device, DType::I64),
    ))
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

    pub(crate) fn storage_mut(&mut self) -> &mut S {
        &mut self.storage
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
    use burn::tensor::Bool;
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
    fn terminal_patches_preserve_bytes_duplicates_and_overwrites() {
        let device = Device::flex();
        let mut replay =
            AlignedObservationReplay::<3, Int>::new([6, 2, 2], 3, (&device, DType::U8)).unwrap();
        let batch = |value: u8| Tensor::<3, Int>::full([3, 2, 2], value, (&device, DType::U8));
        let terminals = Tensor::from_data(
            TensorData::new((20u8..32).collect(), [3, 2, 2]),
            (&device, DType::U8),
        );
        replay
            .insert(0, batch(1), terminals, &[true, false, true])
            .unwrap();
        replay.insert(3, batch(2), batch(3), &[false; 3]).unwrap();
        let (_, next) = replay.gather(&[2, 0, 2, 1]).unwrap();
        assert_eq!(next.dtype(), DType::U8);
        assert_eq!(
            next.clone().into_data().try_to_vec::<u8>().unwrap(),
            vec![28, 29, 30, 31, 20, 21, 22, 23, 28, 29, 30, 31, 2, 2, 2, 2]
        );
        replay.insert(0, batch(4), batch(5), &[false; 3]).unwrap();
        let (_, overwritten) = replay.gather(&[0, 2]).unwrap();
        assert_eq!(
            overwritten.into_data().try_to_vec::<u8>().unwrap(),
            vec![5; 8]
        );
        assert_eq!(
            next.slice([Slice::from(0..1)])
                .into_data()
                .try_to_vec::<u8>()
                .unwrap(),
            vec![28, 29, 30, 31]
        );
    }

    #[test]
    fn column_wraps_without_changing_previously_gathered_values() {
        let device = Device::flex();
        for dtype in [DType::F32, DType::F64] {
            let mut column = TensorReplayColumn::<2>::new([3, 1], (&device, dtype));
            column
                .write(0, Tensor::from_data([[1.0], [2.0]], (&device, dtype)))
                .unwrap();
            let gathered = column.gather(replay_index_tensor(&[0], &device).unwrap());
            column
                .write(2, Tensor::from_data([[3.0], [4.0]], (&device, dtype)))
                .unwrap();
            assert_eq!(
                gathered.into_data().iter::<f64>().collect::<Vec<_>>(),
                vec![1.0]
            );
            let all = column.gather(replay_index_tensor(&[0, 1, 2], &device).unwrap());
            assert_eq!(all.dtype(), dtype);
            assert_eq!(
                all.into_data().iter::<f64>().collect::<Vec<_>>(),
                vec![4.0, 2.0, 3.0]
            );
        }
    }

    #[test]
    fn column_preserves_unsigned_integer_precision() {
        let device = Device::flex();
        let mut column = TensorReplayColumn::<1, Int>::new([2], (&device, DType::U64));
        let values = vec![u64::from(u32::MAX) + 3, u64::MAX - 2];
        column
            .write(
                0,
                Tensor::from_data(TensorData::new(values.clone(), [2]), (&device, DType::U64)),
            )
            .unwrap();
        let gathered = column.gather(replay_index_tensor(&[1, 0], &device).unwrap());
        assert_eq!(gathered.dtype(), DType::U64);
        assert_eq!(
            gathered.into_data().try_to_vec::<u64>().unwrap(),
            vec![values[1], values[0]]
        );
    }

    #[test]
    fn boolean_observations_preserve_terminal_overrides() {
        let device = Device::flex();
        let mut replay = AlignedObservationReplay::<2, Bool>::new([2, 2], 1, &device).unwrap();
        replay
            .insert(
                0,
                Tensor::from_data([[false, false]], &device),
                Tensor::from_data([[false, false]], &device),
                &[true],
            )
            .unwrap();
        replay
            .insert(
                1,
                Tensor::from_data([[true, true]], &device),
                Tensor::from_data([[true, true]], &device),
                &[false],
            )
            .unwrap();
        let (states, next) = replay.gather(&[0, 0, 1]).unwrap();
        assert_eq!(
            states.into_data().iter::<bool>().collect::<Vec<_>>(),
            vec![false, false, false, false, true, true]
        );
        assert_eq!(
            next.into_data().iter::<bool>().collect::<Vec<_>>(),
            vec![false, false, false, false, true, true]
        );
    }

    #[test]
    fn replay_drops_autodiff_without_freezing_source_tensors() {
        let device = Device::flex().autodiff();
        let input = Tensor::<2>::ones([1, 2], &device).require_grad();
        let mut column = TensorReplayColumn::<2>::new([2, 2], &device);
        column.write(0, input.clone()).unwrap();
        let storage_device = device.clone().without_autodiff();
        assert!(
            !column
                .gather(replay_index_tensor(&[0], &storage_device).unwrap())
                .is_autodiff()
        );
        let mut replay = AlignedObservationReplay::<2>::new([2, 2], 1, &device).unwrap();
        replay
            .insert(0, input.clone(), input.clone() * 2.0, &[true])
            .unwrap();
        let (states, next) = replay.gather(&[0]).unwrap();
        assert!(!states.is_autodiff());
        assert!(!next.is_autodiff());
        assert_eq!(
            next.into_data().try_to_vec::<f32>().unwrap(),
            vec![2.0, 2.0]
        );
        let gradients = input.clone().sum().backward();
        assert_eq!(
            input
                .grad(&gradients)
                .unwrap()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            vec![1.0, 1.0]
        );
    }

    #[test]
    fn column_reports_insertions_larger_than_capacity() {
        let device = Device::flex();
        let mut column = TensorReplayColumn::<2>::new([3, 2], &device);
        assert!(matches!(
            column.write(0, Tensor::zeros([4, 2], &device)),
            Err(ReplayStorageError::InsertionExceedsCapacity {
                capacity: 3,
                inserted: 4
            })
        ));
    }

    #[test]
    fn aligned_observations_preserve_frontier_and_truncations() {
        let device = Device::flex();
        let mut replay = AlignedObservationReplay::<2>::new([4, 1], 2, &device).unwrap();
        let batch = |values: [f32; 2]| Tensor::<1>::from_data(values, &device).reshape([2, 1]);
        replay
            .insert(0, batch([0.0, 10.0]), batch([1.0, 11.0]), &[false; 2])
            .unwrap();
        replay
            .insert(2, batch([1.0, 11.0]), batch([99.0, 12.0]), &[true, false])
            .unwrap();
        let (states, next_states) = replay.gather(&[0, 1, 2, 3]).unwrap();
        assert_eq!(
            states.into_data().try_to_vec::<f32>().unwrap(),
            vec![0.0, 10.0, 1.0, 11.0]
        );
        assert_eq!(
            next_states.into_data().try_to_vec::<f32>().unwrap(),
            vec![1.0, 11.0, 99.0, 12.0]
        );
        replay
            .insert(0, batch([2.0, 12.0]), batch([3.0, 13.0]), &[false; 2])
            .unwrap();
        let (states, next_states) = replay.gather(&[0, 1, 2, 3]).unwrap();
        assert_eq!(
            states.into_data().try_to_vec::<f32>().unwrap(),
            vec![2.0, 12.0, 1.0, 11.0]
        );
        assert_eq!(
            next_states.into_data().try_to_vec::<f32>().unwrap(),
            vec![3.0, 13.0, 99.0, 12.0]
        );
        replay
            .insert(2, batch([3.0, 13.0]), batch([4.0, 14.0]), &[false; 2])
            .unwrap();
        let (states, next_states) = replay.gather(&[0, 1, 2, 3]).unwrap();
        assert_eq!(
            states.into_data().try_to_vec::<f32>().unwrap(),
            vec![2.0, 12.0, 3.0, 13.0]
        );
        assert_eq!(
            next_states.into_data().try_to_vec::<f32>().unwrap(),
            vec![3.0, 13.0, 4.0, 14.0]
        );
    }

    #[test]
    fn boolean_frontier_states_survive_multiple_count_two_wraps() {
        let device = Device::flex();
        let mut replay = AlignedObservationReplay::<2, Bool>::new([4, 2], 2, &device).unwrap();
        let batch = |values: [[bool; 2]; 2]| Tensor::from_data(values, &device);
        replay
            .insert(
                0,
                batch([[false, false], [false, true]]),
                batch([[true, false], [true, true]]),
                &[false; 2],
            )
            .unwrap();
        replay
            .insert(
                2,
                batch([[true, false], [true, true]]),
                batch([[false, false], [false, true]]),
                &[false; 2],
            )
            .unwrap();
        replay
            .insert(
                0,
                batch([[false, true], [false, false]]),
                batch([[true, true], [true, false]]),
                &[false; 2],
            )
            .unwrap();
        replay
            .insert(
                2,
                batch([[true, true], [true, false]]),
                batch([[false, false], [false, true]]),
                &[false; 2],
            )
            .unwrap();
        replay
            .insert(
                0,
                batch([[false, false], [true, false]]),
                batch([[false, true], [true, true]]),
                &[false; 2],
            )
            .unwrap();

        let (states, next_states) = replay.gather(&[0, 1, 2, 3]).unwrap();
        assert_eq!(
            states.into_data().iter::<bool>().collect::<Vec<_>>(),
            [false, false, true, false, true, true, true, false]
        );
        assert_eq!(
            next_states.into_data().iter::<bool>().collect::<Vec<_>>(),
            [false, true, true, true, false, false, false, true]
        );
    }

    #[test]
    fn aligned_observation_insert_validates_shape_and_dtype_without_mutation() {
        let device = Device::flex();
        let mut replay =
            AlignedObservationReplay::<2>::new([4, 2], 1, (&device, DType::F32)).unwrap();
        let states = Tensor::from_data([[1.0f32, 2.0]], &device);
        let next_states = Tensor::from_data([[3.0f32, 4.0]], &device);
        replay
            .insert(0, states.clone(), next_states, &[false])
            .unwrap();

        let before = replay
            .observations()
            .clone()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let error = replay.insert(
            1,
            states.clone(),
            Tensor::from_data([[5.0f32, 6.0, 7.0]], &device),
            &[false],
        );
        assert!(matches!(
            error,
            Err(ReplayStorageError::ShapeMismatch {
                field: "next observations",
                expected,
                actual,
            }) if expected == vec![1, 2] && actual == vec![1, 3]
        ));
        assert_eq!(
            replay
                .observations()
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            before
        );

        let error = replay.insert(
            1,
            Tensor::from_data([[5.0f64, 6.0]], (&device, DType::F64)),
            Tensor::from_data([[7.0f64, 8.0]], (&device, DType::F64)),
            &[false],
        );
        assert!(matches!(
            error,
            Err(ReplayStorageError::DTypeMismatch {
                field: "observations",
                expected: DType::F32,
                actual: DType::F64,
            })
        ));
        assert_eq!(
            replay
                .observations()
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            before
        );
    }

    #[test]
    fn environment_count_validation_does_not_mutate_storage() {
        let device = Device::flex();
        let replay = AlignedObservationReplay::<2>::new([8, 1], 2, &device).unwrap();
        let before = replay
            .observations()
            .clone()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();

        replay.validate_environment_count(2).unwrap();
        assert!(matches!(
            replay.validate_environment_count(4),
            Err(ReplayStorageError::EnvironmentCountMismatch {
                expected: 2,
                actual: 4
            })
        ));
        assert_eq!(
            replay
                .observations()
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            before
        );
    }

    #[test]
    fn aligned_storage_reports_configuration_and_batch_errors() {
        let device = Device::flex();
        for capacity in [0, 2, 3] {
            assert!(matches!(
                AlignedObservationReplay::<2>::new([capacity, 1], 2, &device),
                Err(ReplayStorageError::InvalidReplayAlignment { .. })
            ));
        }
        let mut replay = AlignedObservationReplay::<2>::new([4, 1], 2, &device).unwrap();
        assert!(matches!(
            replay.validate_environment_count(1),
            Err(ReplayStorageError::EnvironmentCountMismatch {
                expected: 2,
                actual: 1
            })
        ));
        assert!(matches!(
            AlignedObservationReplay::<2>::new([usize::MAX - 1, 1], 2, &device),
            Err(ReplayStorageError::CapacityOverflow { .. })
        ));
        let before = replay
            .observations()
            .clone()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert!(matches!(
            replay.insert(
                0,
                Tensor::zeros([2, 1], &device),
                Tensor::zeros([1, 1], &device),
                &[false; 2]
            ),
            Err(ReplayStorageError::ShapeMismatch {
                field: "next observations",
                ..
            })
        ));
        let after = replay
            .observations()
            .clone()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn aligned_observations_reject_dtype_mismatches_without_mutation() {
        let device = Device::flex();
        let options = (&device, DType::F32);
        let mut replay = AlignedObservationReplay::<2>::new([4, 2], 1, options).unwrap();
        let f64_states = Tensor::zeros([1, 2], (&device, DType::F64));
        let before = replay
            .observations()
            .clone()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert!(matches!(
            replay.insert(0, f64_states.clone(), f64_states.clone(), &[false]),
            Err(ReplayStorageError::DTypeMismatch {
                field: "observations",
                expected: DType::F32,
                actual: DType::F64,
            })
        ));
        let after = replay
            .observations()
            .clone()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(before, after);

        replay
            .insert(
                0,
                Tensor::ones([1, 2], options),
                Tensor::ones([1, 2], options),
                &[false],
            )
            .unwrap();
        let before = replay
            .observations()
            .clone()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert!(matches!(
            replay.insert(1, f64_states.clone(), f64_states, &[false]),
            Err(ReplayStorageError::DTypeMismatch {
                field: "observations",
                expected: DType::F32,
                actual: DType::F64,
            })
        ));
        let after = replay
            .observations()
            .clone()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(before, after);
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

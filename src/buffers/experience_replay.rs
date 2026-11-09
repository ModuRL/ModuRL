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

    fn sampleable_len(&self, len: usize) -> usize {
        len
    }

    fn sample_index(&self, index: usize, _len: usize) -> usize {
        index
    }
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
/// Terminal observations are retained separately when an environment resets after truncation.
pub(crate) struct AlignedObservationReplay<const R: usize, K: Autodiff = Float> {
    observations: Tensor<R, K>,
    truncated_next_states: HashMap<usize, Tensor<R, K>>,
    environment_count: usize,
    frontier: usize,
    frontier_is_invalid: bool,
    inserted: usize,
    valid_indices: Vec<usize>,
    valid_positions: Vec<Option<usize>>,
}

impl<const R: usize, K: Autodiff> AlignedObservationReplay<R, K> {
    /// Returns the logical shape `[capacity, ...item_shape]`, excluding the extra rows for next observations.
    pub(crate) fn shape(&self) -> [usize; R] {
        let mut shape = self.observations.dims();
        shape[0] -= self.environment_count;
        shape
    }

    /// Returns storage `[capacity + environment_count, ...item_shape]` of rank `R` without copying tensor data.
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
            environment_count,
            frontier: 0,
            frontier_is_invalid: false,
            inserted: 0,
            valid_indices: Vec::new(),
            valid_positions: Vec::new(),
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
    /// Optional active flags exclude padding from sampling while keeping whole observation batches aligned.
    /// Removes autodiff associations. Later state batches overwrite shared next-state rows.
    pub(crate) fn insert(
        &mut self,
        start: usize,
        states: Tensor<R, K>,
        next_states: Tensor<R, K>,
        truncateds: &[bool],
        active: Option<&[bool]>,
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
        if let Some(active) = active
            && active.len() != count
        {
            return Err(ReplayStorageError::BatchLengthMismatch {
                field: "active flags",
                expected: count,
                actual: active.len(),
            });
        }
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
        self.frontier_is_invalid |= self.inserted >= capacity;
        self.inserted = self.inserted.saturating_add(count);
        self.frontier = (start + count) % capacity;
        if self.valid_positions.is_empty() && active.is_some_and(|active| active.contains(&false)) {
            self.valid_positions = vec![None; capacity];
            for row in 0..self.inserted.min(capacity) {
                if !self.frontier_is_invalid
                    || !(self.frontier..self.frontier + count).contains(&row)
                {
                    self.set_valid(row, true);
                }
            }
        }
        if !self.valid_positions.is_empty() {
            for offset in 0..count {
                self.set_valid(start + offset, active.is_none_or(|active| active[offset]));
            }
            if self.frontier_is_invalid {
                for offset in 0..count {
                    self.set_valid(self.frontier + offset, false);
                }
            }
        }
        Ok(count)
    }

    fn set_valid(&mut self, row: usize, valid: bool) {
        match (self.valid_positions[row], valid) {
            (None, true) => {
                self.valid_positions[row] = Some(self.valid_indices.len());
                self.valid_indices.push(row);
            }
            (Some(position), false) => {
                self.valid_indices.swap_remove(position);
                if let Some(&moved) = self.valid_indices.get(position) {
                    self.valid_positions[moved] = Some(position);
                }
                self.valid_positions[row] = None;
            }
            _ => {}
        }
    }

    pub(crate) fn sampleable_len(&self, len: usize) -> usize {
        if !self.valid_positions.is_empty() {
            self.valid_indices.len()
        } else if len == self.shape()[0] && self.frontier_is_invalid {
            len - self.environment_count
        } else {
            len
        }
    }

    pub(crate) fn sample_index(&self, index: usize, len: usize) -> usize {
        let capacity = self.shape()[0];
        let sampleable = if len == capacity && self.frontier_is_invalid {
            len - self.environment_count
        } else {
            len
        };
        if !self.valid_positions.is_empty() && self.valid_indices.len() != sampleable {
            return self.valid_indices[index];
        }
        if len == capacity && self.frontier_is_invalid {
            (self.frontier + self.environment_count + index) % capacity
        } else {
            index
        }
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
        let states = observations.clone().select(0, rows.clone());
        let mut next_states = observations
            .clone()
            .select(0, rows + environment_count as i64);
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
        let total_samples = self.storage.sampleable_len(self.len);
        let size_to_sample = self.batch_size.min(total_samples);
        let indices = sample_indices_without_replacement(total_samples, size_to_sample)
            .into_iter()
            .map(|index| self.storage.sample_index(index, self.len))
            .collect::<Vec<_>>();
        self.storage
            .gather(&indices)
            .map_err(ExperienceReplayError::ExperienceError)
    }

    pub fn len(&self) -> usize {
        self.storage.sampleable_len(self.len)
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
    use proptest::{collection::vec, prelude::*};
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
            .insert(0, batch(1), terminals, &[true, false, true], None)
            .unwrap();
        replay
            .insert(3, batch(2), batch(3), &[false; 3], None)
            .unwrap();
        let (_, next) = replay.gather(&[2, 0, 2, 1]).unwrap();
        assert_eq!(next.dtype(), DType::U8);
        assert_eq!(
            next.clone().into_data().try_to_vec::<u8>().unwrap(),
            vec![28, 29, 30, 31, 20, 21, 22, 23, 28, 29, 30, 31, 2, 2, 2, 2]
        );
        replay
            .insert(0, batch(4), batch(5), &[false; 3], None)
            .unwrap();
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
    fn validity_tracking_stays_unallocated_for_all_active_rows() {
        let device = Device::flex();
        let mut replay = AlignedObservationReplay::<2>::new([8, 1], 2, &device).unwrap();
        for batch in 0usize..16 {
            let start = (batch * 2) % 8;
            let states = Tensor::full([2, 1], batch as f32, &device);
            let active = batch.is_multiple_of(2).then_some(&[true; 2][..]);
            replay
                .insert(start, states.clone(), states + 1.0, &[false; 2], active)
                .unwrap();
            let len = ((batch + 1) * 2).min(8);
            let expected_count = if batch >= 4 { 6 } else { len };
            assert_eq!(replay.sampleable_len(len), expected_count);
            for index in 0..expected_count {
                let expected_row = if batch >= 4 {
                    (start + 4 + index) % 8
                } else {
                    index
                };
                assert_eq!(replay.sample_index(index, len), expected_row);
            }
        }
        assert!(replay.valid_positions.is_empty());
        assert_eq!(replay.valid_positions.capacity(), 0);
        assert!(replay.valid_indices.is_empty());
        assert_eq!(replay.valid_indices.capacity(), 0);
    }

    #[test]
    fn validity_tracking_switches_to_sparse_after_wraparound() {
        let device = Device::flex();
        let mut replay = AlignedObservationReplay::<2>::new([8, 1], 2, &device).unwrap();
        let mut valid = [false; 8];
        for batch in 0..20 {
            let start = (batch * 2) % 8;
            let active = if batch < 9 {
                [true; 2]
            } else {
                match batch % 3 {
                    0 => [false, true],
                    1 => [false; 2],
                    _ => [true, false],
                }
            };
            let states = Tensor::full([2, 1], batch as f32, &device);
            replay
                .insert(
                    start,
                    states.clone(),
                    states + 1.0,
                    &[false; 2],
                    Some(&active),
                )
                .unwrap();
            valid[start..start + 2].copy_from_slice(&active);
            if batch >= 4 {
                let frontier = (start + 2) % 8;
                valid[frontier..frontier + 2].fill(false);
            }
            let len = ((batch + 1) * 2).min(8);
            let mut rows = (0..replay.sampleable_len(len))
                .map(|index| replay.sample_index(index, len))
                .collect::<Vec<_>>();
            rows.sort_unstable();
            let expected = (0..8).filter(|&row| valid[row]).collect::<Vec<_>>();
            assert_eq!(rows, expected);
        }
    }

    #[test]
    fn sparse_valid_rows_follow_frontier_and_recover_after_overwrites() {
        let device = Device::flex();
        let mut replay = AlignedObservationReplay::<2>::new([8, 1], 2, &device).unwrap();
        let mut valid = [false; 8];
        let states = Tensor::zeros([2, 1], &device);
        assert!(matches!(
            replay.insert(0, states.clone(), states, &[false; 2], Some(&[true])),
            Err(ReplayStorageError::BatchLengthMismatch { .. })
        ));
        assert_eq!(replay.sampleable_len(0), 0);
        for batch in 0..12 {
            let start = (batch * 2) % 8;
            let active = match batch % 3 {
                0 => [true, false],
                1 => [false, false],
                _ => [true, true],
            };
            let states = Tensor::full([2, 1], batch as f32, &device);
            replay
                .insert(
                    start,
                    states.clone(),
                    states + 1.0,
                    &[false; 2],
                    Some(&active),
                )
                .unwrap();
            let len = ((batch + 1) * 2).min(8);
            let frontier = (start + 2) % 8;
            valid[start..start + 2].copy_from_slice(&active);
            if batch >= 4 {
                valid[frontier..frontier + 2].fill(false);
            }
            assert_eq!(
                replay.sampleable_len(len),
                valid.iter().filter(|&&v| v).count()
            );
            for index in 0..replay.sampleable_len(len) {
                let row = replay.sample_index(index, len);
                assert!(valid[row]);
                if batch >= 4 {
                    assert!(!(frontier..frontier + 2).contains(&row));
                }
            }
        }
        for start in [0, 2, 4, 6] {
            let states = Tensor::zeros([2, 1], &device);
            replay
                .insert(
                    start,
                    states.clone(),
                    states,
                    &[false; 2],
                    Some(&[false; 2]),
                )
                .unwrap();
        }
        assert_eq!(replay.sampleable_len(8), 0);
        let states = Tensor::zeros([2, 1], &device);
        replay
            .insert(0, states.clone(), states, &[false; 2], None)
            .unwrap();
        assert_eq!(replay.sampleable_len(8), 2);
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
                None,
            )
            .unwrap();
        replay
            .insert(
                1,
                Tensor::from_data([[true, true]], &device),
                Tensor::from_data([[true, true]], &device),
                &[false],
                None,
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
            .insert(0, input.clone(), input.clone() * 2.0, &[true], None)
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
    fn aligned_storage_rejects_invalid_configurations() {
        let device = Device::flex();
        for capacity in [0, 2, 3] {
            assert!(matches!(
                AlignedObservationReplay::<2>::new([capacity, 1], 2, &device),
                Err(ReplayStorageError::InvalidReplayAlignment { .. })
            ));
        }
        assert!(matches!(
            AlignedObservationReplay::<2>::new([usize::MAX - 1, 1], 2, &device),
            Err(ReplayStorageError::CapacityOverflow { .. })
        ));
    }

    /// Flattens `tensor`, shaped `[rows, 1]`, into one value per row.
    fn column_values(tensor: Tensor<2>) -> Vec<f64> {
        tensor
            .into_data()
            .convert::<f64>()
            .try_to_vec::<f64>()
            .unwrap()
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    enum EpisodeEnd {
        Continue,
        Terminate,
        Truncate,
    }

    /// One episode end per environment, and optionally an invalid insert tried before the valid one.
    type ReplayStep = (Vec<EpisodeEnd>, Option<usize>);

    /// `(environment_count, capacity, steps)`; invalid inserts index the cases tried in the property.
    fn aligned_replay_run() -> impl Strategy<Value = (usize, usize, Vec<ReplayStep>)> {
        (1usize..=3, 2usize..=5).prop_flat_map(|(environment_count, batches)| {
            let end = prop_oneof![
                3 => Just(EpisodeEnd::Continue),
                1 => Just(EpisodeEnd::Terminate),
                1 => Just(EpisodeEnd::Truncate),
            ];
            let step = (
                vec(end, environment_count),
                prop::option::weighted(0.25, 0usize..4),
            );
            (
                Just(environment_count),
                Just(environment_count * batches),
                vec(step, 1..=16),
            )
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn column_matches_a_ring_model(
            capacity in 1usize..=6,
            writes in vec(0usize..=7, 1..=8),
            dtype in prop_oneof![Just(DType::F32), Just(DType::F64)],
        ) {
            let device = Device::flex();
            let mut column = TensorReplayColumn::<2>::new([capacity, 1], (&device, dtype));
            let all_rows = replay_index_tensor(&(0..capacity).collect::<Vec<_>>(), &device).unwrap();
            let (mut model, mut position, mut written) = (vec![0.0; capacity], 0, 0);
            let mut first: Option<(Tensor<2>, Vec<f64>)> = None;
            for count in writes {
                let values: Vec<f64> = (written..written + count).map(|value| value as f64 + 1.0).collect();
                let source = Tensor::from_data(TensorData::new(values.clone(), [count, 1]), (&device, dtype));
                match column.write(position, source) {
                    Err(ReplayStorageError::InsertionExceedsCapacity { capacity: c, inserted }) => {
                        prop_assert_eq!((c, inserted), (capacity, count));
                        prop_assert!(count > capacity);
                    }
                    result => {
                        result.unwrap();
                        for (offset, value) in values.into_iter().enumerate() {
                            model[(position + offset) % capacity] = value;
                        }
                        (position, written) = ((position + count) % capacity, written + count);
                    }
                }
                let gathered = column.gather(all_rows.clone());
                prop_assert_eq!(gathered.dtype(), dtype);
                prop_assert_eq!(column_values(gathered.clone()), model.clone());
                first.get_or_insert((gathered, model.clone()));
            }
            // Earlier gathers are copies, so later writes leave them unchanged.
            let (gathered, values) = first.unwrap();
            prop_assert_eq!(column_values(gathered), values);
        }

        #[test]
        fn aligned_observations_match_a_transition_model(
            (environment_count, capacity, steps) in aligned_replay_run(),
        ) {
            let device = Device::flex();
            let n = environment_count;
            let mut replay = AlignedObservationReplay::<2>::new([capacity, 1], n, &device).unwrap();
            let all_rows: Vec<_> = (0..capacity).collect();
            let gather_all = |replay: &AlignedObservationReplay<2>| {
                let (states, next) = replay.gather(&all_rows).unwrap();
                (column_values(states), column_values(next))
            };
            let batch = |values: &[f32], dtype: DType| {
                Tensor::<2>::from_data(TensorData::new(values.to_vec(), [values.len(), 1]), (&device, dtype))
            };
            let mut ids = (1..).map(|id| id as f32);
            let mut observations: Vec<f32> = ids.by_ref().take(n).collect();
            // Per replay row: the state and, unless the episode terminated, the next state.
            let mut model: Vec<Option<(f32, Option<f32>)>> = vec![None; capacity];
            let (mut position, mut inserted) = (0, 0);
            for (ends, invalid) in steps {
                let next_states: Vec<f32> = ids.by_ref().take(n).collect();
                let truncateds: Vec<_> = ends.iter().map(|end| *end == EpisodeEnd::Truncate).collect();
                if let Some(invalid) = invalid {
                    let before = gather_all(&replay);
                    let (states, next) = (batch(&observations, DType::F32), batch(&next_states, DType::F32));
                    let error = match invalid {
                        0 => replay.insert(position, states, Tensor::zeros([n, 2], &device), &truncateds, None),
                        1 => replay.insert(
                            position,
                            batch(&observations, DType::F64),
                            batch(&next_states, DType::F64),
                            &truncateds,
                         None,),
                        2 => replay.insert(position, states, next, &vec![false; n + 1], None),
                        _ => replay.insert(position, Tensor::zeros([n + 1, 1], &device), Tensor::zeros([n + 1, 1], &device), &vec![false; n + 1], None),
                    }
                    .unwrap_err();
                    let expected = ["ShapeMismatch", "DTypeMismatch", "BatchLengthMismatch", "EnvironmentCountMismatch"];
                    prop_assert!(format!("{error:?}").starts_with(expected[invalid]), "{error:?}");
                    prop_assert_eq!(gather_all(&replay), before);
                }

                replay
                    .insert(position, batch(&observations, DType::F32), batch(&next_states, DType::F32), &truncateds, None)
                    .unwrap();
                for (env, end) in ends.into_iter().enumerate() {
                    let next = (end != EpisodeEnd::Terminate).then_some(next_states[env]);
                    model[position + env] = Some((observations[env], next));
                    observations[env] = if end == EpisodeEnd::Continue { next_states[env] } else { ids.next().unwrap() };
                }
                (position, inserted) = ((position + n) % capacity, inserted + n);

                let len = inserted.min(capacity);
                let sampleable = replay.sampleable_len(len);
                // Once the ring has wrapped, the oldest batch shares rows with the newest next states.
                prop_assert_eq!(sampleable, if inserted > capacity { len - n } else { len });
                let rows: Vec<_> = (0..sampleable).map(|index| replay.sample_index(index, len)).collect();
                let mut newest: Vec<_> = (1..=sampleable).map(|age| (position + capacity - age) % capacity).collect();
                let mut sorted_rows = rows.clone();
                newest.sort_unstable();
                sorted_rows.sort_unstable();
                prop_assert_eq!(sorted_rows, newest);

                // Repeating every row checks that duplicate indices gather the same transition.
                let (states, next) = replay.gather(&rows.repeat(2)).unwrap();
                let (states, next) = (column_values(states), column_values(next));
                for (index, row) in rows.iter().cycle().enumerate().take(2 * rows.len()) {
                    let (state, expected_next) = model[*row].unwrap();
                    prop_assert_eq!(states[index], f64::from(state), "state at row {}", row);
                    if let Some(expected_next) = expected_next {
                        prop_assert_eq!(next[index], f64::from(expected_next), "next at row {}", row);
                    }
                }
            }
        }

        #[test]
        fn experience_replay_tracks_length_and_samples_unique_indices(
            capacity in 1usize..=64,
            batch_size in 0usize..=80,
            adds in vec(0usize..=70, 1..=8),
        ) {
            let mut replay = ExperienceReplay::with_storage(IndexStorage { capacity }, batch_size);
            let mut len = 0;
            for count in adds {
                match replay.add(count) {
                    Err(ExperienceReplayError::InsertionExceedsCapacity { capacity: c, inserted }) => {
                        prop_assert_eq!((c, inserted), (capacity, count));
                        prop_assert!(count > capacity);
                    }
                    result => {
                        result.unwrap();
                        len = (len + count).min(capacity);
                    }
                }
                prop_assert_eq!((replay.len(), replay.is_empty()), (len, len == 0));
                let sample = replay.sample().unwrap();
                let unique: std::collections::HashSet<_> = sample.iter().collect();
                prop_assert_eq!((sample.len(), unique.len()), (batch_size.min(len), sample.len()));
                prop_assert!(sample.iter().all(|&index| index < len));
            }
        }
    }
}

use burn::tensor::{
    DType, Device, Float, Int, Slice, Tensor, TensorCreationOptions, TensorData, kind::Autodiff,
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
    #[error("replay start {start} must be below capacity {capacity}")]
    InvalidStart { start: usize, capacity: usize },
    #[error("duplicate replay slot {0}")]
    DuplicateSlot(usize),
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
    #[error("replay capacity {capacity} and environment count {environment_count} must be nonzero")]
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

/// Stores native observations `[2 * capacity, ...item_shape]` with Int transition pointers.
/// Rows are reclaimed only after their last referencing transition is evicted.
/// Two rows per transition suffice even when every transition starts a new episode.
pub(crate) struct PointerObservationReplay<const R: usize, K: Autodiff = Float> {
    observations: TensorReplayColumn<R, K>,
    obs_rows: TensorReplayColumn<1, Int>,
    next_obs_rows: TensorReplayColumn<1, Int>,
    transitions: Vec<Option<(usize, usize)>>,
    references: Vec<usize>,
    owners: Vec<Option<usize>>,
    latest: HashMap<usize, usize>,
    cursor: usize,
    environment_count: usize,
}

impl<const R: usize, K: Autodiff> PointerObservationReplay<R, K> {
    /// Returns logical observation shape `[capacity, ...item_shape]`.
    pub(crate) fn shape(&self) -> [usize; R] {
        let mut shape = self.observations.tensor().dims();
        shape[0] /= 2;
        shape
    }

    /// Allocates native observations `[2 * capacity, ...item_shape]` and Int pointers `[capacity]`.
    /// Storage is detached. Environment count validates callers using the current mask API.
    pub(crate) fn new(
        mut shape: [usize; R],
        environment_count: usize,
        options: impl Into<TensorCreationOptions>,
    ) -> Result<Self, ReplayStorageError> {
        const {
            assert!(R >= 2, "replay observations require capacity and item axes");
        }
        let capacity = shape[0];
        shape[0] = validate_replay_alignment(capacity, environment_count)?;
        let options = options.into();
        let device = options.device.clone().without_autodiff();
        Ok(Self {
            observations: TensorReplayColumn::new(shape, options),
            obs_rows: TensorReplayColumn::new([capacity], (&device, DType::I64)),
            next_obs_rows: TensorReplayColumn::new([capacity], (&device, DType::I64)),
            transitions: vec![None; capacity],
            references: vec![0; shape[0]],
            owners: vec![None; shape[0]],
            latest: HashMap::new(),
            cursor: 0,
            environment_count,
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

    pub(crate) fn clear_streams(&mut self) {
        self.latest.clear();
        self.owners.fill(None);
    }

    /// Stores real observations and next observations `[slots.len(), ...item_shape]`.
    /// Each slot continues until an episode end; absent slots write nothing.
    /// Cached next rows are reused only while their contents have not been overwritten.
    pub(crate) fn insert(
        &mut self,
        start: usize,
        states: Tensor<R, K>,
        next_states: Tensor<R, K>,
        ends: &[bool],
        slots: &[usize],
    ) -> Result<usize, ReplayStorageError> {
        let capacity = self.shape()[0];
        let count = states.dims()[0];
        if start >= capacity {
            return Err(ReplayStorageError::InvalidStart { start, capacity });
        }
        if count > capacity {
            return Err(ReplayStorageError::InsertionExceedsCapacity {
                capacity,
                inserted: count,
            });
        }
        let mut expected_shape = self.shape();
        expected_shape[0] = count;
        validate_replay_shape("observations", &states, expected_shape)?;
        validate_replay_shape("next observations", &next_states, expected_shape)?;
        let expected_dtype = self.observations.tensor().dtype();
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
        for (field, actual) in [("episode ends", ends.len()), ("slots", slots.len())] {
            if actual != count {
                return Err(ReplayStorageError::BatchLengthMismatch {
                    field,
                    expected: count,
                    actual,
                });
            }
        }
        if !slots.iter().copied().eq(0..count) {
            let mut unique = std::collections::HashSet::with_capacity(count);
            for &slot in slots {
                if !unique.insert(slot) {
                    return Err(ReplayStorageError::DuplicateSlot(slot));
                }
            }
        }
        if count == 0 {
            return Ok(0);
        }
        let device = self.observations.tensor().device();
        let next_states = next_states.to_device(&device).without_autodiff();
        for offset in 0..count {
            if let Some((obs, next)) = self.transitions[(start + offset) % capacity].take() {
                self.references[obs] -= 1;
                self.references[next] -= 1;
            }
        }
        // Pin reusable rows before allocating, including slots later in this batch.
        let mut obs_rows: Vec<_> = slots
            .iter()
            .map(|slot| self.latest.get(slot).copied())
            .collect();
        for row in obs_rows.iter().flatten() {
            self.references[*row] += 1;
        }
        let mut missing = Vec::new();
        let mut written_rows = Vec::new();
        for (offset, row) in obs_rows.iter_mut().enumerate() {
            if row.is_none() {
                let allocated = self.allocate_row();
                *row = Some(allocated);
                missing.push(offset);
                written_rows.push(allocated);
            }
        }
        if !missing.is_empty() {
            let state_device = states.device();
            let states = if missing.len() == count {
                states
            } else {
                states.select(0, replay_index_tensor(&missing, &state_device)?)
            };
            self.write_rows(&written_rows, states.to_device(&device))?;
        }
        let next_rows: Vec<_> = (0..count).map(|_| self.allocate_row()).collect();
        self.write_rows(&next_rows, next_states)?;
        let obs_rows: Vec<_> = obs_rows.into_iter().flatten().collect();
        self.obs_rows
            .write(start, replay_index_tensor(&obs_rows, &device)?)?;
        self.next_obs_rows
            .write(start, replay_index_tensor(&next_rows, &device)?)?;
        for (offset, &slot) in slots.iter().enumerate() {
            let next = next_rows[offset];
            self.transitions[(start + offset) % capacity] = Some((obs_rows[offset], next));
            if ends[offset] {
                self.latest.remove(&slot);
            } else {
                self.latest.insert(slot, next);
                self.owners[next] = Some(slot);
            }
        }
        Ok(count)
    }

    fn allocate_row(&mut self) -> usize {
        while self.references[self.cursor] != 0 {
            self.cursor = (self.cursor + 1) % self.references.len();
        }
        let row = self.cursor;
        if let Some(slot) = self.owners[row].take()
            && self.latest.get(&slot) == Some(&row)
        {
            self.latest.remove(&slot);
        }
        self.references[row] = 1;
        self.cursor = (row + 1) % self.references.len();
        row
    }

    /// Writes real observations `[rows.len(), ...item_shape]` in contiguous ring runs.
    fn write_rows(
        &mut self,
        rows: &[usize],
        source: Tensor<R, K>,
    ) -> Result<(), ReplayStorageError> {
        let mut begin = 0;
        while begin < rows.len() {
            let mut end = begin + 1;
            while end < rows.len() && rows[end] == (rows[end - 1] + 1) % self.references.len() {
                end += 1;
            }
            self.observations
                .write(rows[begin], source.clone().slice([Slice::from(begin..end)]))?;
            begin = end;
        }
        Ok(())
    }

    pub(crate) fn sampleable_len(&self, len: usize) -> usize {
        len
    }
    pub(crate) fn sample_index(&self, index: usize, _len: usize) -> usize {
        index
    }

    /// Gathers native observations and next observations `[sample_count, ...item_shape]`.
    /// Int pointer columns resolve both rows, including terminal next observations.
    pub(crate) fn gather(
        &self,
        indices: &[usize],
    ) -> Result<(Tensor<R, K>, Tensor<R, K>), ReplayStorageError> {
        let rows = replay_index_tensor(indices, &self.observations.tensor().device())?;
        Ok((
            self.observations.gather(self.obs_rows.gather(rows.clone())),
            self.observations.gather(self.next_obs_rows.gather(rows)),
        ))
    }
}

/// Selects real rows from masked batches, with no tensor selection for all-active batches.
pub(crate) struct ReplayRows {
    pub(crate) slots: Vec<usize>,
    indices: Option<Tensor<1, Int>>,
}

impl ReplayRows {
    pub(crate) fn new(
        count: usize,
        active: Option<&[bool]>,
        device: &Device,
    ) -> Result<Self, ReplayStorageError> {
        if let Some(active) = active
            && active.len() != count
        {
            return Err(ReplayStorageError::BatchLengthMismatch {
                field: "active flags",
                expected: count,
                actual: active.len(),
            });
        }
        let slots: Vec<_> = (0..count)
            .filter(|&slot| active.is_none_or(|flags| flags[slot]))
            .collect();
        let indices = if slots.len() == count {
            None
        } else {
            Some(replay_index_tensor(&slots, device)?)
        };
        Ok(Self { slots, indices })
    }

    /// Selects compact values `[slots.len(), ...]` from a tensor `[batch_size, ...]`.
    pub(crate) fn select<const R: usize, K: Autodiff>(&self, values: Tensor<R, K>) -> Tensor<R, K> {
        match &self.indices {
            Some(indices) => {
                let device = values.device();
                values.select(0, indices.clone().to_device(&device))
            }
            None => values,
        }
    }

    pub(crate) fn ends(&self, dones: &[bool], truncateds: &[bool]) -> Vec<bool> {
        self.slots
            .iter()
            .map(|&slot| dones[slot] || truncateds[slot])
            .collect()
    }
}

/// Checks native observations `[batch_size, ...item_shape]` against the storage axes.
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

/// Validates the replay capacity and returns the allocated row count, including worst-case current and next observation rows.
pub(crate) fn validate_replay_alignment(
    capacity: usize,
    environment_count: usize,
) -> Result<usize, ReplayStorageError> {
    if environment_count == 0 || capacity == 0 {
        return Err(ReplayStorageError::InvalidReplayAlignment {
            capacity,
            environment_count,
        });
    }
    capacity
        .checked_mul(2)
        .ok_or(ReplayStorageError::CapacityOverflow {
            capacity,
            additional: capacity,
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
        self.len() == 0
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
            PointerObservationReplay::<3, Int>::new([6, 2, 2], 3, (&device, DType::U8)).unwrap();
        let batch = |value: u8| Tensor::<3, Int>::full([3, 2, 2], value, (&device, DType::U8));
        let terminals = Tensor::from_data(
            TensorData::new((20u8..32).collect(), [3, 2, 2]),
            (&device, DType::U8),
        );
        replay
            .insert(0, batch(1), terminals, &[true, false, true], &[0, 1, 2])
            .unwrap();
        replay
            .insert(3, batch(2), batch(3), &[false; 3], &[0, 1, 2])
            .unwrap();
        let (_, next) = replay.gather(&[2, 0, 2, 1]).unwrap();
        assert_eq!(next.dtype(), DType::U8);
        assert_eq!(
            next.clone().into_data().try_to_vec::<u8>().unwrap(),
            vec![
                28, 29, 30, 31, 20, 21, 22, 23, 28, 29, 30, 31, 24, 25, 26, 27
            ]
        );
        replay
            .insert(0, batch(4), batch(5), &[false; 3], &[0, 1, 2])
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
    fn continuous_slots_write_only_one_next_row_per_transition() {
        let device = Device::flex();
        let mut replay = PointerObservationReplay::<2>::new([8, 1], 2, &device).unwrap();
        replay
            .insert(
                0,
                Tensor::zeros([2, 1], &device),
                Tensor::ones([2, 1], &device),
                &[false; 2],
                &[0, 1],
            )
            .unwrap();
        assert_eq!(replay.cursor, 4);
        let previous = [
            replay.transitions[0].unwrap().1,
            replay.transitions[1].unwrap().1,
        ];
        replay
            .insert(
                2,
                Tensor::ones([2, 1], &device),
                Tensor::full([2, 1], 2.0, &device),
                &[false; 2],
                &[0, 1],
            )
            .unwrap();
        assert_eq!(replay.cursor, 6);
        assert_eq!(
            [
                replay.transitions[2].unwrap().0,
                replay.transitions[3].unwrap().0
            ],
            previous
        );
        assert_eq!(replay.transitions.iter().flatten().count(), 4);
        let rows = ReplayRows::new(2, Some(&[true; 2]), &device).unwrap();
        assert!(rows.indices.is_none());
    }

    #[test]
    fn cleared_streams_preserve_history_without_reusing_reset_observations() {
        let device = Device::flex();
        let mut replay = PointerObservationReplay::<2>::new([3, 1], 1, &device).unwrap();
        replay
            .insert(
                0,
                Tensor::zeros([1, 1], &device),
                Tensor::ones([1, 1], &device),
                &[false],
                &[0],
            )
            .unwrap();
        replay.clear_streams();
        replay
            .insert(
                1,
                Tensor::full([1, 1], 10.0, &device),
                Tensor::full([1, 1], 11.0, &device),
                &[false],
                &[0],
            )
            .unwrap();
        let (states, next) = replay.gather(&[0, 1]).unwrap();
        assert_eq!(column_values(states), [0.0, 10.0]);
        assert_eq!(column_values(next), [1.0, 11.0]);
    }

    #[test]
    fn pointer_rows_reuse_continuous_slots_and_preserve_episodes() {
        let device = Device::flex();
        let mut replay = PointerObservationReplay::<2>::new([5, 1], 2, &device).unwrap();
        let batch = |values: Vec<f32>| {
            let count = values.len();
            Tensor::from_data(TensorData::new(values, [count, 1]), &device)
        };
        replay
            .insert(
                0,
                batch(vec![1.0, 10.0]),
                batch(vec![2.0, 11.0]),
                &[false; 2],
                &[7, 900],
            )
            .unwrap();
        let first_next = replay.transitions[0].unwrap().1;
        replay
            .insert(2, batch(vec![11.0]), batch(vec![12.0]), &[false], &[900])
            .unwrap();
        replay
            .insert(3, batch(vec![12.0]), batch(vec![13.0]), &[true], &[900])
            .unwrap();
        replay
            .insert(4, batch(vec![2.0]), batch(vec![3.0]), &[true], &[7])
            .unwrap();
        assert_eq!(replay.transitions[4].unwrap().0, first_next);
        replay
            .insert(0, batch(vec![100.0]), batch(vec![101.0]), &[false], &[900])
            .unwrap();
        let (states, next) = replay.gather(&[0, 2, 3, 4]).unwrap();
        assert_eq!(column_values(states), [100.0, 11.0, 12.0, 2.0]);
        assert_eq!(column_values(next), [101.0, 12.0, 13.0, 3.0]);
        let transitions = replay.transitions.clone();
        let references = replay.references.clone();
        replay
            .insert(1, batch(vec![]), batch(vec![]), &[], &[])
            .unwrap();
        assert_eq!(replay.transitions, transitions);
        assert_eq!(replay.references, references);
    }

    #[test]
    fn pointer_rows_wrap_without_overwriting_live_observations() {
        let device = Device::flex();
        let mut replay = PointerObservationReplay::<2>::new([3, 1], 1, &device).unwrap();
        let mut expected = [(0.0, 0.0); 3];
        for time in 0..100 {
            let slot = if time % 3 == 0 { usize::MAX } else { 42 };
            let state = (time * 10) as f32;
            let next = state + 1.0;
            replay
                .insert(
                    time % 3,
                    Tensor::from_data([[state]], &device),
                    Tensor::from_data([[next]], &device),
                    &[true],
                    &[slot],
                )
                .unwrap();
            expected[time % 3] = (state as f64, next as f64);
            let rows: Vec<_> = (0..(time + 1).min(3)).collect();
            let (states, next) = replay.gather(&rows).unwrap();
            assert_eq!(
                column_values(states),
                rows.iter().map(|&row| expected[row].0).collect::<Vec<_>>()
            );
            assert_eq!(
                column_values(next),
                rows.iter().map(|&row| expected[row].1).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn pointer_rows_reject_invalid_slots_without_mutating_storage() {
        let device = Device::flex();
        let mut replay = PointerObservationReplay::<2>::new([2, 1], 1, &device).unwrap();
        for slots in [vec![9], vec![9, 9]] {
            let error = replay
                .insert(
                    0,
                    Tensor::zeros([2, 1], &device),
                    Tensor::ones([2, 1], &device),
                    &[false; 2],
                    &slots,
                )
                .unwrap_err();
            assert!(matches!(
                error,
                ReplayStorageError::BatchLengthMismatch { .. }
                    | ReplayStorageError::DuplicateSlot(9)
            ));
            assert!(replay.transitions.iter().all(Option::is_none));
            assert!(replay.references.iter().all(|&references| references == 0));
        }
    }

    #[test]
    fn boolean_observations_preserve_terminal_overrides() {
        let device = Device::flex();
        let mut replay = PointerObservationReplay::<2, Bool>::new([2, 2], 1, &device).unwrap();
        replay
            .insert(
                0,
                Tensor::from_data([[false, false]], &device),
                Tensor::from_data([[false, false]], &device),
                &[true],
                &[0],
            )
            .unwrap();
        replay
            .insert(
                1,
                Tensor::from_data([[true, true]], &device),
                Tensor::from_data([[true, true]], &device),
                &[false],
                &[0],
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
        let mut replay = PointerObservationReplay::<2>::new([2, 2], 1, &device).unwrap();
        replay
            .insert(0, input.clone(), input.clone() * 2.0, &[true], &[0])
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
    fn pointer_storage_rejects_invalid_configurations() {
        let device = Device::flex();
        assert!(matches!(
            PointerObservationReplay::<2>::new([0, 1], 2, &device),
            Err(ReplayStorageError::InvalidReplayAlignment { .. })
        ));
        assert!(matches!(
            PointerObservationReplay::<2>::new([usize::MAX - 1, 1], 2, &device),
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
        fn sparse_pointer_rows_match_a_transition_model(
            capacity in 3usize..=12,
            steps in vec((any::<[bool; 3]>(), any::<[bool; 3]>(), any::<bool>()), 1..=40),
        ) {
            let device = Device::flex();
            let mut replay = PointerObservationReplay::<2>::new([capacity, 1], 3, &device).unwrap();
            let mut observations = [1.0f32, 2.0, 3.0];
            let mut model = vec![None; capacity];
            let mut position = 0;
            let mut inserted = 0;
            let mut id = 4.0;
            for (active, ends, reverse) in steps {
                let mut slots: Vec<_> = (0..3).filter(|&slot| active[slot]).collect();
                if reverse { slots.reverse(); }
                let count = slots.len();
                let states: Vec<_> = slots.iter().map(|&slot| observations[slot]).collect();
                let next: Vec<_> = (0..count).map(|_| { id += 1.0; id }).collect();
                let episode_ends: Vec<_> = slots.iter().map(|&slot| ends[slot]).collect();
                let batch = |values: Vec<f32>| Tensor::from_data(TensorData::new(values, [count, 1]), &device);
                let slot_ids: Vec<_> = slots.iter().map(|&slot| [usize::MAX, 100, 9][slot]).collect();
                replay.insert(position, batch(states.clone()), batch(next.clone()), &episode_ends, &slot_ids).unwrap();
                for (offset, &slot) in slots.iter().enumerate() {
                    model[(position + offset) % capacity] = Some((states[offset] as f64, next[offset] as f64));
                    if ends[slot] { id += 1.0; observations[slot] = id; }
                    else { observations[slot] = next[offset]; }
                }
                position = (position + count) % capacity;
                inserted += count;
                prop_assert_eq!(replay.transitions.iter().flatten().count(), inserted.min(capacity));
                let rows: Vec<_> = (0..inserted.min(capacity)).collect();
                if !rows.is_empty() {
                    let (states, next) = replay.gather(&rows).unwrap();
                    prop_assert_eq!(column_values(states), rows.iter().map(|&row| model[row].unwrap().0).collect::<Vec<_>>());
                    prop_assert_eq!(column_values(next), rows.iter().map(|&row| model[row].unwrap().1).collect::<Vec<_>>());
                }
            }
        }

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
        fn pointer_observations_match_a_transition_model(
            (environment_count, capacity, steps) in aligned_replay_run(),
        ) {
            let device = Device::flex();
            let n = environment_count;
            let mut replay = PointerObservationReplay::<2>::new([capacity, 1], n, &device).unwrap();
            let all_rows: Vec<_> = (0..capacity).collect();
            let gather_all = |replay: &PointerObservationReplay<2>| {
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
                let truncateds: Vec<_> = ends.iter().map(|end| *end != EpisodeEnd::Continue).collect();
                let slots: Vec<_> = (0..n).collect();
                if let Some(invalid) = invalid {
                    let before = gather_all(&replay);
                    let (states, next) = (batch(&observations, DType::F32), batch(&next_states, DType::F32));
                    let error = match invalid {
                        0 => replay.insert(position, states, Tensor::zeros([n, 2], &device), &truncateds, &slots),
                        1 => replay.insert(
                            position,
                            batch(&observations, DType::F64),
                            batch(&next_states, DType::F64),
                            &truncateds,
                         &slots,),
                        2 => replay.insert(position, states, next, &vec![false; n + 1], &slots),
                        _ => replay.insert(position, Tensor::zeros([n + 1, 1], &device), Tensor::zeros([n + 1, 1], &device), &vec![false; n + 1], &slots),
                    }
                    .unwrap_err();
                    let expected = ["ShapeMismatch", "DTypeMismatch", "BatchLengthMismatch", "BatchLengthMismatch"];
                    prop_assert!(format!("{error:?}").starts_with(expected[invalid]), "{error:?}");
                    prop_assert_eq!(gather_all(&replay), before);
                }

                replay
                    .insert(position, batch(&observations, DType::F32), batch(&next_states, DType::F32), &truncateds, &slots)
                    .unwrap();
                for (env, end) in ends.into_iter().enumerate() {
                    let next = (end != EpisodeEnd::Terminate).then_some(next_states[env]);
                    model[position + env] = Some((observations[env], next));
                    observations[env] = if end == EpisodeEnd::Continue { next_states[env] } else { ids.next().unwrap() };
                }
                (position, inserted) = ((position + n) % capacity, inserted + n);

                let len = inserted.min(capacity);
                let sampleable = replay.sampleable_len(len);
                prop_assert_eq!(sampleable, len);
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

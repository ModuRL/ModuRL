use burn::tensor::{
    Bool, DType, Device, Float, Tensor,
    kind::{Autodiff, Kind},
};

use crate::buffers::experience_replay::{
    AlignedObservationReplay, ExperienceReplay, ExperienceReplayError, ReplayStorage,
    ReplayStorageError, TensorReplayColumn, replay_index_tensor,
};
use crate::tensor_operations::transfer_bool_tensor;

/// Invalid replay configuration or transition data.
#[derive(Debug, thiserror::Error)]
pub enum DeterministicReplayError {
    #[error("replay storage failed: {0}")]
    Storage(#[from] ReplayStorageError),
    #[error("replay {field} shape differs: expected {expected:?}, got {actual:?}")]
    Shape {
        field: &'static str,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("replay {field} dtype {dtype:?} does not match tensor kind {kind:?}")]
    InvalidDType {
        field: &'static str,
        dtype: DType,
        kind: Kind,
    },
    #[error("replay {field} item dimensions must be nonzero")]
    ZeroItemDimension { field: &'static str },
    #[error(
        "replay start {start} must be below capacity {capacity} and divisible by environment count {environment_count}"
    )]
    InvalidStart {
        start: usize,
        capacity: usize,
        environment_count: usize,
    },
}

/// Detached observations [batch_size, ...observation_shape] and Float actions [batch_size, ...action_shape].
/// Rewards and Bool termination flags have shape [batch_size, 1] on the storage device.
pub(super) struct DeterministicBatch<const O: usize, const A: usize, K: Autodiff> {
    pub(super) observations: Tensor<O, K>,
    pub(super) next_observations: Tensor<O, K>,
    pub(super) actions: Tensor<A>,
    pub(super) rewards: Tensor<2>,
    pub(super) terminated: Tensor<2, Bool>,
}

/// Native observations [num_envs, ...observation_shape] and Float actions [num_envs, ...action_shape].
/// Rewards and Bool termination flags have shape [num_envs, 1]; truncations have one flag per environment.
pub(super) struct DeterministicReplayInsert<const O: usize, const A: usize, K: Autodiff> {
    pub(super) observations: Tensor<O, K>,
    pub(super) next_observations: Tensor<O, K>,
    pub(super) actions: Tensor<A>,
    pub(super) rewards: Tensor<2>,
    pub(super) terminated: Tensor<2, Bool>,
    pub(super) truncateds: Vec<bool>,
}

/// Owns aligned replay for native rank-O observations, Float rank-A actions, and scalar rewards and Bool flags.
pub struct DeterministicReplay<const O: usize, const A: usize, K: Autodiff> {
    buffer:
        ExperienceReplay<DeterministicReplayInsert<O, A, K>, DeterministicReplayStorage<O, A, K>>,
}

impl<const O: usize, const A: usize, K: Autodiff> DeterministicReplay<O, A, K> {
    pub(super) fn new(storage: DeterministicReplayStorage<O, A, K>, batch_size: usize) -> Self {
        Self {
            buffer: ExperienceReplay::with_storage(storage, batch_size),
        }
    }

    pub(super) fn len(&self) -> usize {
        self.buffer.len()
    }

    pub(super) fn can_sample(&self) -> bool {
        self.len() >= self.buffer.get_batch_size()
    }

    pub(super) fn validate_environment_count(
        &mut self,
        count: usize,
    ) -> Result<(), DeterministicReplayError> {
        self.buffer.storage_mut().validate_environment_count(count)
    }

    /// Inserts native observations [num_envs, ...observation_shape], Float actions [num_envs, ...action_shape], and scalars [num_envs, 1].
    /// The storage implementation validates fields, transfers them to the storage device, and removes gradients.
    pub(super) fn insert(
        &mut self,
        transitions: DeterministicReplayInsert<O, A, K>,
    ) -> Result<(), DeterministicReplayError> {
        self.buffer.add(transitions).map_err(replay_error)
    }

    /// Samples detached native observations [batch_size, ...observation_shape], Float actions [batch_size, ...action_shape], and scalars [batch_size, 1].
    /// Uses the shared replay sampler and preserves the configured storage dtypes and device.
    pub(super) fn sample(&self) -> Result<DeterministicBatch<O, A, K>, DeterministicReplayError> {
        self.buffer.sample().map_err(replay_error)
    }

    #[cfg(test)]
    pub(super) fn storage_mut(&mut self) -> &mut DeterministicReplayStorage<O, A, K> {
        self.buffer.storage_mut()
    }
}

fn replay_error(
    error: ExperienceReplayError<DeterministicReplayError>,
) -> DeterministicReplayError {
    match error {
        ExperienceReplayError::ExperienceError(error) => error,
        ExperienceReplayError::InsertionExceedsCapacity { capacity, inserted } => {
            ReplayStorageError::InsertionExceedsCapacity { capacity, inserted }.into()
        }
    }
}

pub(super) struct DeterministicReplayStorage<const O: usize, const A: usize, K: Autodiff> {
    observations: AlignedObservationReplay<O, K>,
    actions: TensorReplayColumn<A>,
    rewards: TensorReplayColumn<2>,
    terminated: TensorReplayColumn<2, Bool>,
    environment_count: usize,
}

#[bon::bon]
impl<const O: usize, const A: usize, K: Autodiff> DeterministicReplayStorage<O, A, K> {
    /// Allocates observations [capacity, ...observation_shape], Float actions [capacity, ...action_shape], and scalars [capacity, 1].
    /// Ranks O and A include capacity and require at least one item axis.
    /// Preserves observation dtype; actions and rewards use dtype. All storage is detached on device.
    /// Returns an error for invalid dtypes, zero item dimensions, mismatched capacities, or invalid environment alignment.
    #[builder]
    pub(super) fn new(
        observation_shape: [usize; O],
        action_shape: [usize; A],
        observation_dtype: DType,
        environment_count: usize,
        dtype: DType,
        device: Device,
    ) -> Result<Self, DeterministicReplayError> {
        const {
            assert!(O >= 2 && A >= 2, "replay needs capacity and item axes");
        }
        validate_dtype::<K>("observations", observation_dtype)?;
        validate_dtype::<Float>("actions and rewards", dtype)?;
        for (field, shape) in [
            ("observations", observation_shape.as_slice()),
            ("actions", action_shape.as_slice()),
        ] {
            if shape[1..].contains(&0) {
                return Err(DeterministicReplayError::ZeroItemDimension { field });
            }
        }
        let capacity = observation_shape[0];
        if action_shape[0] != capacity {
            return Err(DeterministicReplayError::Shape {
                field: "action storage",
                expected: vec![capacity],
                actual: vec![action_shape[0]],
            });
        }
        let observations = AlignedObservationReplay::new(
            observation_shape,
            environment_count,
            (&device, observation_dtype),
        )?;
        Ok(Self {
            observations,
            actions: TensorReplayColumn::new(action_shape, (&device, dtype)),
            rewards: TensorReplayColumn::new([capacity, 1], (&device, dtype)),
            terminated: TensorReplayColumn::new([capacity, 1], &device),
            environment_count,
        })
    }

    pub(super) fn validate_environment_count(
        &self,
        count: usize,
    ) -> Result<(), DeterministicReplayError> {
        self.observations
            .validate_environment_count(count)
            .map_err(Into::into)
    }

    /// Checks Float actions [num_envs, ...action_shape] and scalars [num_envs, 1].
    /// Aligned storage validates observation shape and dtype; this validates environment count without changing storage.
    fn validate_insert(
        &self,
        start: usize,
        transitions: &DeterministicReplayInsert<O, A, K>,
    ) -> Result<usize, DeterministicReplayError> {
        let count = transitions.observations.dims()[0];
        if start >= self.capacity() || !start.is_multiple_of(self.environment_count) {
            return Err(DeterministicReplayError::InvalidStart {
                start,
                capacity: self.capacity(),
                environment_count: self.environment_count,
            });
        }
        let mut action_shape = self.actions.tensor().dims();
        action_shape[0] = count;
        validate_shape("actions", &transitions.actions, action_shape)?;
        validate_shape("rewards", &transitions.rewards, [count, 1])?;
        validate_shape("termination", &transitions.terminated, [count, 1])?;
        if transitions.truncateds.len() != count {
            return Err(ReplayStorageError::BatchLengthMismatch {
                field: "truncations",
                expected: count,
                actual: transitions.truncateds.len(),
            }
            .into());
        }
        Ok(count)
    }
}

fn validate_dtype<K: Autodiff>(
    field: &'static str,
    dtype: DType,
) -> Result<(), DeterministicReplayError> {
    let matches_kind = if const { matches!(K::KIND, Kind::Float) } {
        dtype.is_float()
    } else if const { matches!(K::KIND, Kind::Int) } {
        dtype.is_int() || dtype.is_uint()
    } else {
        dtype.is_bool()
    };
    if !matches_kind {
        return Err(DeterministicReplayError::InvalidDType {
            field,
            dtype,
            kind: K::KIND,
        });
    }
    Ok(())
}

/// Checks a rank-D tensor [batch_size, ...item_shape] against the required axes without converting values.
fn validate_shape<const D: usize, K: Autodiff>(
    field: &'static str,
    tensor: &Tensor<D, K>,
    expected: [usize; D],
) -> Result<(), DeterministicReplayError> {
    if tensor.dims() != expected {
        return Err(DeterministicReplayError::Shape {
            field,
            expected: expected.to_vec(),
            actual: tensor.dims().to_vec(),
        });
    }
    Ok(())
}

impl<const O: usize, const A: usize, K: Autodiff> ReplayStorage
    for DeterministicReplayStorage<O, A, K>
{
    type Insert = DeterministicReplayInsert<O, A, K>;
    type Batch = DeterministicBatch<O, A, K>;
    type Error = DeterministicReplayError;

    fn capacity(&self) -> usize {
        self.rewards.tensor().dims()[0]
    }

    /// Stores native observations [num_envs, ...observation_shape], Float actions [num_envs, ...action_shape], and scalars [num_envs, 1].
    /// Validates non-observation fields before writing. Preserves observation dtype and casts actions and rewards to the configured floating dtype.
    /// Aligned storage validates and transfers observations; truncations preserve terminal next observations.
    fn insert(&mut self, start: usize, transitions: Self::Insert) -> Result<usize, Self::Error> {
        let count = self.validate_insert(start, &transitions)?;
        let device = self.rewards.tensor().device();
        let dtype = self.rewards.tensor().dtype();
        self.observations.insert(
            start,
            transitions.observations,
            transitions.next_observations,
            &transitions.truncateds,
        )?;
        self.actions
            .write(start, transitions.actions.to_device(&device).cast(dtype))?;
        self.rewards
            .write(start, transitions.rewards.to_device(&device).cast(dtype))?;
        self.terminated
            .write(start, transfer_bool_tensor(transitions.terminated, &device))?;
        Ok(count)
    }

    /// Gathers host indices [sample_count] into native observations, Float actions, and scalars with a leading sample axis.
    /// Preserves item axes and storage dtypes on the storage device. Returned tensors are detached and unaffected by later writes.
    /// Terminal observations replace next observations for truncated transitions.
    fn gather(&self, indices: &[usize]) -> Result<Self::Batch, Self::Error> {
        let (observations, next_observations) = self.observations.gather(indices)?;
        let rows = replay_index_tensor(indices, &self.rewards.tensor().device())?;
        Ok(DeterministicBatch {
            observations,
            next_observations,
            actions: self.actions.gather(rows.clone()),
            rewards: self.rewards.gather(rows.clone()),
            terminated: self.terminated.gather(rows),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, TensorData};

    /// Creates U64 observations [2, 1] and Float actions, rewards, and Bool flags [2, 1].
    fn integer_transitions(device: &Device) -> DeterministicReplayInsert<2, 2, Int> {
        DeterministicReplayInsert {
            observations: Tensor::from_data([[u64::MAX - 1], [u64::MAX - 2]], (device, DType::U64)),
            next_observations: Tensor::from_data([[7u64], [8]], (device, DType::U64)),
            actions: Tensor::from_data([[0.25f32], [-0.5]], device),
            rewards: Tensor::from_data([[1.0f32], [2.0]], device),
            terminated: Tensor::from_data([[false], [true]], device),
            truncateds: vec![true, false],
        }
    }

    fn integer_storage(device: &Device) -> DeterministicReplayStorage<2, 2, Int> {
        DeterministicReplayStorage::builder()
            .observation_shape([4, 1])
            .action_shape([4, 1])
            .observation_dtype(DType::U64)
            .environment_count(2)
            .dtype(DType::F64)
            .device(device.clone())
            .build()
            .unwrap()
    }

    #[test]
    fn native_integer_precision_truncation_and_ring_alignment_are_preserved() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex();
        let mut storage = integer_storage(&device);
        assert_eq!(storage.capacity(), 4);
        storage.insert(0, integer_transitions(&device)).unwrap();
        let before = storage.gather(&[0, 1]).unwrap();
        assert_eq!(before.observations.dtype(), DType::U64);
        assert_eq!(
            before
                .observations
                .clone()
                .into_data()
                .try_to_vec::<u64>()
                .unwrap(),
            [u64::MAX - 1, u64::MAX - 2]
        );
        assert_eq!(before.actions.dtype(), DType::F64);
        assert_eq!(
            before.actions.into_data().try_to_vec::<f64>().unwrap(),
            [0.25, -0.5]
        );
        assert_eq!(before.rewards.dtype(), DType::F64);
        assert_eq!(
            before.rewards.into_data().try_to_vec::<f64>().unwrap(),
            [1.0, 2.0]
        );
        assert_eq!(
            before
                .terminated
                .into_data()
                .iter::<bool>()
                .collect::<Vec<_>>(),
            [false, true]
        );
        let mut next = integer_transitions(&device);
        next.observations = Tensor::from_data([[3u64], [4]], (&device, DType::U64));
        storage.insert(2, next).unwrap();
        assert_eq!(
            storage
                .gather(&[0, 1])
                .unwrap()
                .next_observations
                .into_data()
                .try_to_vec::<u64>()
                .unwrap(),
            [7, 4]
        );
        let mut wrapped = integer_transitions(&device);
        wrapped.observations = Tensor::from_data([[5u64], [6]], (&device, DType::U64));
        wrapped.truncateds = vec![false, false];
        storage.insert(0, wrapped).unwrap();
        let after_wrap = storage.gather(&[0, 1, 2, 3]).unwrap();
        assert_eq!(
            after_wrap
                .observations
                .into_data()
                .try_to_vec::<u64>()
                .unwrap(),
            [5, 6, 3, 4]
        );
        assert_eq!(
            after_wrap
                .next_observations
                .into_data()
                .try_to_vec::<u64>()
                .unwrap(),
            [7, 8, 7, 8]
        );
        assert_eq!(
            before.observations.into_data().try_to_vec::<u64>().unwrap(),
            [u64::MAX - 1, u64::MAX - 2]
        );
    }

    #[test]
    fn boolean_observations_keep_high_rank_item_axes_and_terminal_overrides() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex();
        let mut storage = DeterministicReplayStorage::<3, 3, Bool>::builder()
            .observation_shape([4, 1, 2])
            .action_shape([4, 1, 1])
            .observation_dtype(Tensor::<3, Bool>::zeros([1, 1, 2], &device).dtype())
            .environment_count(2)
            .dtype(DType::F32)
            .device(device.clone())
            .build()
            .unwrap();
        for start in [0, 2] {
            storage
                .insert(
                    start,
                    DeterministicReplayInsert {
                        observations: Tensor::from_data(
                            [[[true, true]], [[false, false]]],
                            &device,
                        ),
                        next_observations: Tensor::from_data(
                            [[[false, false]], [[true, true]]],
                            &device,
                        ),
                        actions: Tensor::ones([2, 1, 1], &device),
                        rewards: Tensor::ones([2, 1], &device),
                        terminated: Tensor::from_data([[false], [true]], &device),
                        truncateds: vec![true, false],
                    },
                )
                .unwrap();
        }
        let batch = storage.gather(&[0, 0]).unwrap();
        assert_eq!(batch.observations.dims(), [2, 1, 2]);
        assert_eq!(batch.actions.dims(), [2, 1, 1]);
        assert_eq!(batch.rewards.dims(), [2, 1]);
        assert_eq!(batch.terminated.dims(), [2, 1]);
        assert_eq!(
            batch
                .next_observations
                .into_data()
                .iter::<bool>()
                .collect::<Vec<_>>(),
            [false; 4]
        );
        let empty = storage.gather(&[]).unwrap();
        assert_eq!(empty.observations.dims(), [0, 1, 2]);
        assert_eq!(empty.actions.dims(), [0, 1, 1]);
    }

    #[test]
    fn float_experience_is_detached_and_compute_dtype_is_preserved() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        let storage_device = device.clone().without_autodiff();
        let mut storage = DeterministicReplayStorage::<2, 2, Float>::builder()
            .observation_shape([4, 1])
            .action_shape([4, 1])
            .observation_dtype(DType::F64)
            .environment_count(2)
            .dtype(DType::F64)
            .device(storage_device.clone())
            .build()
            .unwrap();
        storage
            .insert(
                0,
                DeterministicReplayInsert {
                    observations: Tensor::ones([2, 1], (&device, DType::F64)).require_grad(),
                    next_observations: Tensor::ones([2, 1], (&device, DType::F64)).require_grad(),
                    actions: Tensor::ones([2, 1], &device).require_grad(),
                    rewards: Tensor::ones([2, 1], &device).require_grad(),
                    terminated: Tensor::from_data([[false], [true]], &device),
                    truncateds: vec![true, false],
                },
            )
            .unwrap();
        let batch = storage.gather(&[0, 1]).unwrap();
        for tensor in [
            batch.observations,
            batch.next_observations,
            batch.actions,
            batch.rewards,
        ] {
            assert!(!tensor.is_autodiff());
            assert!(!tensor.is_require_grad());
            assert_eq!(tensor.dtype(), DType::F64);
            assert_eq!(tensor.device(), storage_device);
        }
        assert!(!batch.terminated.is_autodiff());
    }

    #[test]
    fn invalid_transitions_do_not_overwrite_existing_rows() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex();
        let mut storage = integer_storage(&device);
        storage.insert(0, integer_transitions(&device)).unwrap();
        for field in [
            "observations",
            "next observations",
            "actions",
            "rewards",
            "termination",
        ] {
            let mut transitions = integer_transitions(&device);
            match field {
                "observations" => {
                    transitions.observations = Tensor::zeros([2, 2], (&device, DType::U64))
                }
                "next observations" => {
                    transitions.next_observations = Tensor::zeros([2, 2], (&device, DType::U64))
                }
                "actions" => transitions.actions = Tensor::zeros([2, 2], &device),
                "rewards" => transitions.rewards = Tensor::zeros([2, 2], &device),
                "termination" => {
                    transitions.terminated =
                        Tensor::from_data([[false, false], [false, false]], &device)
                }
                _ => unreachable!(),
            }
            let error = storage.insert(0, transitions);
            if matches!(field, "observations" | "next observations") {
                assert!(matches!(
                    error,
                    Err(DeterministicReplayError::Storage(
                        ReplayStorageError::ShapeMismatch {
                            field: actual,
                            ..
                        }
                    )) if actual == field
                ));
            } else {
                assert!(matches!(
                    error,
                    Err(DeterministicReplayError::Shape { field: actual, .. }) if actual == field
                ));
            }
        }
        for next in [false, true] {
            let mut transitions = integer_transitions(&device);
            if next {
                transitions.next_observations = transitions.next_observations.cast(DType::U32);
            } else {
                transitions.observations = transitions.observations.cast(DType::U32);
            }
            let field = if next {
                "next observations"
            } else {
                "observations"
            };
            assert!(matches!(
                storage.insert(0, transitions),
                Err(DeterministicReplayError::Storage(
                    ReplayStorageError::DTypeMismatch {
                        field: actual,
                        expected: DType::U64,
                        actual: DType::U32,
                    }
                )) if actual == field
            ));
        }
        let mut truncateds = integer_transitions(&device);
        truncateds.truncateds.pop();
        assert!(matches!(
            storage.insert(0, truncateds),
            Err(DeterministicReplayError::Storage(
                ReplayStorageError::BatchLengthMismatch {
                    field: "truncations",
                    ..
                }
            ))
        ));
        assert!(matches!(
            storage.validate_environment_count(1),
            Err(DeterministicReplayError::Storage(
                ReplayStorageError::EnvironmentCountMismatch {
                    expected: 2,
                    actual: 1
                }
            ))
        ));
        for start in [1, 4] {
            assert!(matches!(
                storage.insert(start, integer_transitions(&device)),
                Err(DeterministicReplayError::InvalidStart { .. })
            ));
        }
        assert_eq!(
            storage
                .gather(&[0, 1])
                .unwrap()
                .observations
                .into_data()
                .try_to_vec::<u64>()
                .unwrap(),
            [u64::MAX - 1, u64::MAX - 2]
        );
        assert_eq!(
            storage
                .gather(&[0, 1])
                .unwrap()
                .next_observations
                .into_data()
                .try_to_vec::<u64>()
                .unwrap(),
            [7, 8]
        );
    }

    #[test]
    fn builder_rejects_invalid_shapes_dtypes_and_environment_alignment() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex();
        let build =
            |observation_shape, action_shape, observation_dtype, dtype, environment_count| {
                DeterministicReplayStorage::<2, 2, Int>::builder()
                    .observation_shape(observation_shape)
                    .action_shape(action_shape)
                    .observation_dtype(observation_dtype)
                    .environment_count(environment_count)
                    .dtype(dtype)
                    .device(device.clone())
                    .build()
            };
        assert!(matches!(
            build([4, 1], [6, 1], DType::U64, DType::F32, 2),
            Err(DeterministicReplayError::Shape {
                field: "action storage",
                ..
            })
        ));
        for (observations, actions) in [([4, 0], [4, 1]), ([4, 1], [4, 0])] {
            assert!(matches!(
                build(observations, actions, DType::U64, DType::F32, 2),
                Err(DeterministicReplayError::ZeroItemDimension { .. })
            ));
        }
        assert!(matches!(
            build([4, 1], [4, 1], DType::F32, DType::F32, 2),
            Err(DeterministicReplayError::InvalidDType {
                field: "observations",
                ..
            })
        ));
        assert!(matches!(
            build([4, 1], [4, 1], DType::U64, DType::I64, 2),
            Err(DeterministicReplayError::InvalidDType {
                field: "actions and rewards",
                ..
            })
        ));
        for (capacity, count) in [(0, 2), (2, 2), (3, 2), (4, 0)] {
            assert!(matches!(
                build([capacity, 1], [capacity, 1], DType::U64, DType::F32, count),
                Err(DeterministicReplayError::Storage(
                    ReplayStorageError::InvalidReplayAlignment { .. }
                ))
            ));
        }
        assert!(matches!(
            build([usize::MAX, 1], [usize::MAX, 1], DType::U64, DType::F32, 1),
            Err(DeterministicReplayError::Storage(
                ReplayStorageError::CapacityOverflow { .. }
            ))
        ));
        let bool_dtype =
            Tensor::<2, Bool>::from_data(TensorData::new(vec![true], [1, 1]), &device).dtype();
        assert!(matches!(
            build([4, 1], [4, 1], bool_dtype, DType::F32, 2),
            Err(DeterministicReplayError::InvalidDType { .. })
        ));
    }
}

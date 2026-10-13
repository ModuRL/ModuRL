use burn::tensor::{Bool, Cast, DType, Device, Tensor, kind::Autodiff};

use crate::buffers::experience_replay::{
    AlignedObservationReplay, ExperienceReplay, ExperienceReplayError, ReplayStorage,
    ReplayStorageError, TensorReplayColumn, replay_index_tensor,
};

#[derive(Debug, thiserror::Error)]
pub enum SACReplayError {
    #[error("replay storage failed: {0}")]
    Storage(#[from] ReplayStorageError),
    #[error("replay {field} shape differs: expected {expected:?}, got {actual:?}")]
    Shape {
        field: &'static str,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("replay observation dtype differs: expected {expected:?}, got {actual:?}")]
    ObservationDType { expected: DType, actual: DType },
}

/// Owns SAC's replay ring and aligned observation storage with native observation and action kinds.
pub(super) struct SACReplay<const O: usize, const A: usize, K: Autodiff, L: Autodiff>
where
    DType: Cast<A, L, OutputKind = L>,
{
    buffer: ExperienceReplay<SACReplayInsert<O, A, K, L>, SACReplayStorage<O, A, K, L>>,
}

impl<const O: usize, const A: usize, K: Autodiff, L: Autodiff> SACReplay<O, A, K, L>
where
    DType: Cast<A, L, OutputKind = L>,
{
    pub(super) fn new(storage: SACReplayStorage<O, A, K, L>, batch_size: usize) -> Self {
        Self {
            buffer: ExperienceReplay::with_storage(storage, batch_size),
        }
    }

    pub(super) fn validate_environment_count(
        &mut self,
        count: usize,
    ) -> Result<(), SACReplayError> {
        self.buffer.storage_mut().validate_environment_count(count)
    }

    pub(super) fn len(&self) -> usize {
        self.buffer.len()
    }

    pub(super) fn can_sample(&self) -> bool {
        self.buffer.len() >= self.buffer.get_batch_size()
    }

    /// Inserts native observations, actions, and statistics with one leading environment axis; detaches every stored field.
    pub(super) fn insert(
        &mut self,
        transitions: SACReplayInsert<O, A, K, L>,
    ) -> Result<(), SACReplayError> {
        self.buffer.add(transitions).map_err(replay_error)
    }

    /// Samples native observations and actions with a leading batch axis and scalar statistics `[batch_size, 1]`.
    /// Preserves observation kind and dtype and uses the existing replay RNG. Returned tensors have no gradients.
    pub(super) fn sample(&self) -> Result<SACOptimizationBatch<O, A, K, L>, SACReplayError> {
        self.buffer.sample().map_err(replay_error)
    }
}

fn replay_error(error: ExperienceReplayError<SACReplayError>) -> SACReplayError {
    match error {
        ExperienceReplayError::ExperienceError(error) => error,
        ExperienceReplayError::InsertionExceedsCapacity { capacity, inserted } => {
            ReplayStorageError::InsertionExceedsCapacity { capacity, inserted }.into()
        }
    }
}

pub(super) struct SACReplayInsert<const O: usize, const A: usize, K: Autodiff, L: Autodiff> {
    pub observations: Tensor<O, K>,
    pub next_observations: Tensor<O, K>,
    pub actions: Tensor<A, L>,
    pub rewards: Tensor<2>,
    pub terminated: Tensor<2, Bool>,
    pub collection_policy_entropies: Tensor<2>,
    pub entropy_change_weights: Tensor<2>,
    pub truncateds: Vec<bool>,
}

pub(super) struct SACOptimizationBatch<const O: usize, const A: usize, K: Autodiff, L: Autodiff> {
    pub observations: Tensor<O, K>,
    pub next_observations: Tensor<O, K>,
    pub actions: Tensor<A, L>,
    pub rewards: Tensor<2>,
    pub terminated: Tensor<2, Bool>,
    pub collection_policy_entropies: Tensor<2>,
    pub entropy_change_weights: Tensor<2>,
}

pub(super) struct SACReplayStorage<const O: usize, const A: usize, K: Autodiff, L: Autodiff> {
    observations: AlignedObservationReplay<O, K>,
    actions: TensorReplayColumn<A, L>,
    rewards: TensorReplayColumn<2>,
    terminated: TensorReplayColumn<2, Bool>,
    collection_policy_entropies: TensorReplayColumn<2>,
    entropy_change_weights: TensorReplayColumn<2>,
}

#[bon::bon]
impl<const O: usize, const A: usize, K: Autodiff, L: Autodiff> SACReplayStorage<O, A, K, L> {
    /// Allocates observations `[capacity, ...observation_shape]`, actions `[capacity, ...action_shape]`, and statistics `[capacity, 1]`.
    /// Observation and action dtypes come from collection tensors. Statistics use dtype; storage is detached on device.
    #[builder]
    pub(super) fn new(
        observation_shape: [usize; O],
        action_shape: [usize; A],
        observation_dtype: DType,
        action_dtype: DType,
        environment_count: usize,
        dtype: DType,
        device: Device,
    ) -> Result<Self, SACReplayError> {
        const {
            assert!(O >= 2 && A >= 2, "replay needs capacity and item axes");
        }
        let capacity = observation_shape[0];
        if action_shape[0] != capacity {
            return Err(SACReplayError::Shape {
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
            actions: TensorReplayColumn::new(action_shape, (&device, action_dtype)),
            rewards: TensorReplayColumn::new([capacity, 1], (&device, dtype)),
            terminated: TensorReplayColumn::new([capacity, 1], &device),
            collection_policy_entropies: TensorReplayColumn::new([capacity, 1], (&device, dtype)),
            entropy_change_weights: TensorReplayColumn::new([capacity, 1], (&device, dtype)),
        })
    }

    /// Validates observations `[num_envs, ...observation_shape]`, actions `[num_envs, ...action_shape]`, and statistics `[num_envs, 1]`.
    /// Checks observation dtypes and environment count without changing storage. Returns the number of transitions.
    fn validate_insert(
        &self,
        transitions: &SACReplayInsert<O, A, K, L>,
    ) -> Result<usize, SACReplayError> {
        let count = transitions.observations.dims()[0];
        let mut observation_shape = self.observations.shape();
        observation_shape[0] = count;
        let mut action_shape = self.actions.tensor().dims();
        action_shape[0] = count;
        validate_shape("observations", &transitions.observations, observation_shape)?;
        validate_shape(
            "next observations",
            &transitions.next_observations,
            observation_shape,
        )?;
        validate_shape("actions", &transitions.actions, action_shape)?;
        validate_shape("rewards", &transitions.rewards, [count, 1])?;
        validate_shape("termination", &transitions.terminated, [count, 1])?;
        validate_shape(
            "collection entropy",
            &transitions.collection_policy_entropies,
            [count, 1],
        )?;
        validate_shape(
            "entropy change weights",
            &transitions.entropy_change_weights,
            [count, 1],
        )?;
        if transitions.truncateds.len() != count {
            return Err(ReplayStorageError::BatchLengthMismatch {
                field: "truncations",
                expected: count,
                actual: transitions.truncateds.len(),
            }
            .into());
        }
        let dtype = self.observations.observations().dtype();
        for actual in [
            transitions.observations.dtype(),
            transitions.next_observations.dtype(),
        ] {
            if actual != dtype {
                return Err(SACReplayError::ObservationDType {
                    expected: dtype,
                    actual,
                });
            }
        }
        self.validate_environment_count(count)?;
        Ok(count)
    }

    pub(super) fn validate_environment_count(&self, count: usize) -> Result<(), SACReplayError> {
        self.observations
            .validate_environment_count(count)
            .map_err(Into::into)
    }
}

/// Validates a rank-D field `[batch_size, ...item_shape]` against its expected axes without changing values.
fn validate_shape<const D: usize, K: Autodiff>(
    field: &'static str,
    tensor: &Tensor<D, K>,
    expected: [usize; D],
) -> Result<(), SACReplayError> {
    if tensor.dims() != expected {
        return Err(SACReplayError::Shape {
            field,
            expected: expected.to_vec(),
            actual: tensor.dims().to_vec(),
        });
    }
    Ok(())
}

impl<const O: usize, const A: usize, K: Autodiff, L: Autodiff> ReplayStorage
    for SACReplayStorage<O, A, K, L>
where
    DType: Cast<A, L, OutputKind = L>,
{
    type Insert = SACReplayInsert<O, A, K, L>;
    type Batch = SACOptimizationBatch<O, A, K, L>;
    type Error = SACReplayError;

    fn capacity(&self) -> usize {
        self.rewards.tensor().dims()[0]
    }

    /// Inserts observations `[num_envs, ...observation_shape]`, actions `[num_envs, ...action_shape]`, and statistics `[num_envs, 1]`.
    /// Preserves observation kind and first-batch dtype. Explicitly casts actions to their storage dtype and statistics to compute dtype.
    /// Validates fields before writing; transfers and detaches each tensor on the storage device.
    fn insert(&mut self, start: usize, transitions: Self::Insert) -> Result<usize, Self::Error> {
        let count = self.validate_insert(&transitions)?;
        let device = self.rewards.tensor().device();
        let statistics_dtype = self.rewards.tensor().dtype();
        let action_dtype = self.actions.tensor().dtype();
        self.observations.insert(
            start,
            transitions.observations.to_device(&device),
            transitions.next_observations.to_device(&device),
            &transitions.truncateds,
        )?;
        self.actions.write(
            start,
            <DType as Cast<A, L>>::cast(transitions.actions.to_device(&device), action_dtype),
        )?;
        self.rewards.write(
            start,
            transitions
                .rewards
                .to_device(&device)
                .cast(statistics_dtype),
        )?;
        self.terminated
            .write(start, transitions.terminated.to_device(&device))?;
        self.collection_policy_entropies.write(
            start,
            transitions
                .collection_policy_entropies
                .to_device(&device)
                .cast(statistics_dtype),
        )?;
        self.entropy_change_weights.write(
            start,
            transitions
                .entropy_change_weights
                .to_device(&device)
                .cast(statistics_dtype),
        )?;
        Ok(count)
    }

    /// Gathers host indices `[sample_count]` into detached observations, actions, and statistics with one leading sample axis.
    /// Observations preserve native dtype and kind; terminal overrides replace reset observations for truncated transitions.
    fn gather(&self, indices: &[usize]) -> Result<Self::Batch, Self::Error> {
        let (observations, next_observations) = self.observations.gather(indices)?;
        let rows = replay_index_tensor(indices, &self.rewards.tensor().device())?;
        Ok(SACOptimizationBatch {
            observations,
            next_observations,
            actions: self.actions.gather(rows.clone()),
            rewards: self.rewards.gather(rows.clone()),
            terminated: self.terminated.gather(rows.clone()),
            collection_policy_entropies: self.collection_policy_entropies.gather(rows.clone()),
            entropy_change_weights: self.entropy_change_weights.gather(rows),
        })
    }

    fn sampleable_len(&self, len: usize) -> usize {
        self.observations.sampleable_len(len)
    }

    fn sample_index(&self, index: usize, len: usize) -> usize {
        self.observations.sample_index(index, len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Float, Int};

    /// Creates I64 observations/actions `[2, 1]` and scalar statistics `[2, 1]` with one truncated transition.
    fn integer_transitions(device: &Device, dtype: DType) -> SACReplayInsert<2, 2, Int, Int> {
        SACReplayInsert {
            observations: Tensor::from_data([[1i64], [2]], (device, dtype)),
            next_observations: Tensor::from_data([[10i64], [20]], (device, dtype)),
            actions: Tensor::from_data([[0i64], [1]], device),
            rewards: Tensor::ones([2, 1], device),
            terminated: Tensor::from_data([[false], [true]], device),
            collection_policy_entropies: Tensor::zeros([2, 1], device),
            entropy_change_weights: Tensor::zeros([2, 1], device),
            truncateds: vec![true, false],
        }
    }

    #[test]
    fn environment_count_is_enforced_before_and_after_first_insertion() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex();
        let mut storage = SACReplayStorage::<2, 2, Int, Int>::builder()
            .observation_shape([4, 1])
            .action_shape([4, 1])
            .observation_dtype(DType::I64)
            .action_dtype(DType::I32)
            .environment_count(2)
            .dtype(DType::F64)
            .device(device.clone())
            .build()
            .unwrap();
        storage.validate_environment_count(2).unwrap();
        for populated in [false, true] {
            if populated {
                storage
                    .insert(0, integer_transitions(&device, DType::I64))
                    .unwrap();
            }
            storage.validate_environment_count(2).unwrap();
            assert!(matches!(
                storage.validate_environment_count(1),
                Err(SACReplayError::Storage(
                    ReplayStorageError::EnvironmentCountMismatch {
                        expected: 2,
                        actual: 1,
                    }
                ))
            ));
        }
        assert_eq!(storage.capacity(), 4);
        assert_eq!(
            storage.gather(&[0]).unwrap().observations.dtype(),
            DType::I64
        );
    }

    #[test]
    fn native_observation_dtype_is_preserved_and_changes_do_not_overwrite_replay() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex();
        let mut storage = SACReplayStorage::<2, 2, Int, Int>::builder()
            .observation_shape([4, 1])
            .action_shape([4, 1])
            .observation_dtype(DType::I64)
            .action_dtype(DType::I32)
            .environment_count(2)
            .dtype(DType::F64)
            .device(device.clone())
            .build()
            .unwrap();
        storage
            .insert(0, integer_transitions(&device, DType::I64))
            .unwrap();
        let batch = storage.gather(&[0, 1]).unwrap();
        assert_eq!(batch.observations.dtype(), DType::I64);
        assert_eq!(batch.rewards.dtype(), DType::F64);
        assert_eq!(
            batch
                .terminated
                .into_data()
                .iter::<bool>()
                .collect::<Vec<_>>(),
            [false, true]
        );
        assert_eq!(
            batch
                .next_observations
                .into_data()
                .try_to_vec::<i64>()
                .unwrap(),
            [10, 20]
        );
        let invalid = integer_transitions(&device, DType::I32);
        assert!(matches!(
            storage.insert(0, invalid),
            Err(SACReplayError::ObservationDType {
                expected: DType::I64,
                actual: DType::I32
            })
        ));
        assert_eq!(
            storage
                .gather(&[0, 1])
                .unwrap()
                .observations
                .into_data()
                .try_to_vec::<i64>()
                .unwrap(),
            [1, 2]
        );
        let mut next = integer_transitions(&device, DType::I64);
        next.observations = Tensor::from_data([[3i64], [4]], (&device, DType::I64));
        storage.insert(2, next).unwrap();
        assert_eq!(
            storage
                .gather(&[0])
                .unwrap()
                .next_observations
                .into_data()
                .try_to_vec::<i64>()
                .unwrap(),
            [10]
        );
        assert_eq!(storage.sampleable_len(4), 4);
        storage
            .insert(0, integer_transitions(&device, DType::I64))
            .unwrap();
        assert_eq!(storage.sampleable_len(4), 2);
    }

    #[test]
    fn boolean_observations_and_terminal_overrides_keep_native_kind() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex();
        let mut storage = SACReplayStorage::<2, 2, Bool, Int>::builder()
            .observation_shape([4, 1])
            .action_shape([4, 1])
            .observation_dtype(Tensor::<2, Bool>::zeros([1, 1], &device).dtype())
            .action_dtype(DType::I32)
            .environment_count(2)
            .dtype(DType::F32)
            .device(device.clone())
            .build()
            .unwrap();
        for start in [0, 2] {
            storage
                .insert(
                    start,
                    SACReplayInsert {
                        observations: Tensor::from_data([[true], [false]], &device),
                        next_observations: Tensor::from_data([[false], [true]], &device),
                        actions: Tensor::zeros([2, 1], &device),
                        rewards: Tensor::ones([2, 1], &device),
                        terminated: Tensor::from_data([[false], [false]], &device),
                        collection_policy_entropies: Tensor::zeros([2, 1], &device),
                        entropy_change_weights: Tensor::zeros([2, 1], &device),
                        truncateds: vec![true, false],
                    },
                )
                .unwrap();
        }
        let batch = storage.gather(&[0]).unwrap();
        assert_eq!(
            batch
                .observations
                .into_data()
                .iter::<bool>()
                .collect::<Vec<_>>(),
            [true]
        );
        assert_eq!(
            batch
                .next_observations
                .into_data()
                .iter::<bool>()
                .collect::<Vec<_>>(),
            [false]
        );
    }

    #[test]
    fn shape_validation_precedes_writes_and_float_experience_is_detached() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        let mut storage = SACReplayStorage::<2, 2, Float, Int>::builder()
            .observation_shape([4, 1])
            .action_shape([4, 1])
            .observation_dtype(DType::F64)
            .action_dtype(DType::I32)
            .environment_count(2)
            .dtype(DType::F64)
            .device(Device::flex())
            .build()
            .unwrap();
        let mut transitions = SACReplayInsert {
            observations: Tensor::ones([2, 1], (&device, DType::F64)).require_grad(),
            next_observations: Tensor::ones([2, 1], (&device, DType::F64)).require_grad(),
            actions: Tensor::zeros([2, 1], &device),
            rewards: Tensor::ones([2, 1], &device).require_grad(),
            terminated: Tensor::from_data([[false], [false]], &device),
            collection_policy_entropies: Tensor::zeros([2, 1], &device),
            entropy_change_weights: Tensor::zeros([2, 1], &device),
            truncateds: vec![true, false],
        };
        transitions.rewards = Tensor::ones([2, 2], &device);
        assert!(matches!(
            storage.insert(0, transitions),
            Err(SACReplayError::Shape {
                field: "rewards",
                ..
            })
        ));
        assert_eq!(
            storage
                .gather(&[0])
                .unwrap()
                .observations
                .into_data()
                .try_to_vec::<f64>()
                .unwrap(),
            [0.0]
        );
        storage
            .insert(
                0,
                SACReplayInsert {
                    observations: Tensor::ones([2, 1], (&device, DType::F64)).require_grad(),
                    next_observations: Tensor::ones([2, 1], (&device, DType::F64)).require_grad(),
                    actions: Tensor::zeros([2, 1], &device),
                    rewards: Tensor::ones([2, 1], &device).require_grad(),
                    terminated: Tensor::from_data([[false], [false]], &device),
                    collection_policy_entropies: Tensor::zeros([2, 1], &device),
                    entropy_change_weights: Tensor::zeros([2, 1], &device),
                    truncateds: vec![true, false],
                },
            )
            .unwrap();
        let batch = storage.gather(&[0, 1]).unwrap();
        assert!(!batch.observations.is_autodiff());
        assert!(!batch.next_observations.is_autodiff());
        assert!(!batch.rewards.is_autodiff());
        assert_eq!(batch.observations.dtype(), DType::F64);
        assert_eq!(batch.rewards.dtype(), DType::F64);
    }
}

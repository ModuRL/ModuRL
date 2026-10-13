use burn::tensor::{
    Bool, DType, Device, Float, Tensor,
    kind::{Autodiff, Kind},
};

use crate::buffers::experience_replay::{
    AlignedObservationReplay, ReplayStorage, ReplayStorageError, TensorReplayColumn,
    replay_index_tensor,
};

/// Invalid replay configuration or transition data.
#[derive(Debug, thiserror::Error)]
pub(super) enum DeterministicReplayError {
    #[error("replay storage failed: {0}")]
    Storage(#[from] ReplayStorageError),
    #[error("replay {field} shape differs: expected {expected:?}, got {actual:?}")]
    Shape {
        field: &'static str,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("replay {field} dtype is {actual:?}, expected {expected:?}")]
    DType {
        field: &'static str,
        expected: DType,
        actual: DType,
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
    pub(super) fn builder(
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

    /// Checks observations [num_envs, ...observation_shape], Float actions [num_envs, ...action_shape], and scalars [num_envs, 1].
    /// Requires the configured observation dtype and environment count without changing storage.
    fn validate_insert(
        &self,
        start: usize,
        transitions: &DeterministicReplayInsert<O, A, K>,
    ) -> Result<usize, DeterministicReplayError> {
        let count = transitions.observations.dims()[0];
        self.validate_environment_count(count)?;
        if start >= self.capacity() || !start.is_multiple_of(self.environment_count) {
            return Err(DeterministicReplayError::InvalidStart {
                start,
                capacity: self.capacity(),
                environment_count: self.environment_count,
            });
        }
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
        if transitions.truncateds.len() != count {
            return Err(ReplayStorageError::BatchLengthMismatch {
                field: "truncations",
                expected: count,
                actual: transitions.truncateds.len(),
            }
            .into());
        }
        let expected = self.observations.observations().dtype();
        for (field, actual) in [
            ("observations", transitions.observations.dtype()),
            ("next observations", transitions.next_observations.dtype()),
        ] {
            if actual != expected {
                return Err(DeterministicReplayError::DType {
                    field,
                    expected,
                    actual,
                });
            }
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
    /// Validates all fields before writing. Preserves observation dtype and casts actions and rewards to the configured floating dtype.
    /// Transfers each tensor to the storage device and removes gradients. Truncations preserve terminal next observations.
    fn insert(&mut self, start: usize, transitions: Self::Insert) -> Result<usize, Self::Error> {
        let count = self.validate_insert(start, &transitions)?;
        let device = self.rewards.tensor().device();
        let dtype = self.rewards.tensor().dtype();
        self.observations.insert(
            start,
            transitions.observations.to_device(&device),
            transitions.next_observations.to_device(&device),
            &transitions.truncateds,
        )?;
        self.actions
            .write(start, transitions.actions.to_device(&device).cast(dtype))?;
        self.rewards
            .write(start, transitions.rewards.to_device(&device).cast(dtype))?;
        self.terminated
            .write(start, transitions.terminated.to_device(&device))?;
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

    fn sampleable_len(&self, len: usize) -> usize {
        self.observations.sampleable_len(len)
    }

    fn sample_index(&self, index: usize, len: usize) -> usize {
        self.observations.sample_index(index, len)
    }
}

#[cfg(test)]
#[path = "replay/tests.rs"]
mod tests;

use candle_core::{DType, Tensor};

use crate::buffers::experience_replay::{
    AlignedObservationReplay, ReplayStorage, ReplayStorageError, TensorReplayColumn,
    replay_index_tensor,
};

pub(super) struct DeterministicBatch {
    pub(super) states: Tensor,
    pub(super) next_states: Tensor,
    pub(super) actions: Tensor,
    pub(super) rewards: Tensor,
    pub(super) terminated: Tensor,
}

pub(super) struct DeterministicReplayInsert {
    pub(super) states: Tensor,
    pub(super) next_states: Tensor,
    pub(super) actions: Tensor,
    pub(super) rewards: Tensor,
    pub(super) terminated: Tensor,
    pub(super) truncateds: Vec<bool>,
}

pub(super) struct DeterministicReplayStorage {
    observations: AlignedObservationReplay,
    actions: TensorReplayColumn,
    rewards: TensorReplayColumn,
    terminated: TensorReplayColumn,
    capacity: usize,
    device: candle_core::Device,
}

impl DeterministicReplayStorage {
    pub(super) fn new(
        capacity: usize,
        state_shape: &[usize],
        action_shape: &[usize],
        dtype: DType,
        device: candle_core::Device,
    ) -> Result<Self, ReplayStorageError> {
        Ok(Self {
            observations: AlignedObservationReplay::new(capacity, state_shape, DType::F32, &device),
            actions: TensorReplayColumn::new(capacity, action_shape, dtype, &device)?,
            rewards: TensorReplayColumn::new(capacity, &[1], dtype, &device)?,
            terminated: TensorReplayColumn::new(capacity, &[1], dtype, &device)?,
            capacity,
            device,
        })
    }

    pub(super) fn initialize_environment_count(
        &mut self,
        environment_count: usize,
    ) -> Result<(), ReplayStorageError> {
        self.observations
            .initialize_environment_count(environment_count)
    }
}

impl ReplayStorage for DeterministicReplayStorage {
    type Insert = DeterministicReplayInsert;
    type Batch = DeterministicBatch;
    type Error = ReplayStorageError;

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn insert(&mut self, start: usize, transitions: Self::Insert) -> Result<usize, Self::Error> {
        let prepared = DeterministicReplayInsert {
            states: transitions.states,
            next_states: transitions.next_states,
            actions: self.actions.prepare(&transitions.actions)?,
            rewards: self.rewards.prepare(&transitions.rewards)?,
            terminated: self.terminated.prepare(&transitions.terminated)?,
            truncateds: transitions.truncateds,
        };
        let count = prepared.states.dim(0)?;
        for (name, tensor) in [
            ("actions", &prepared.actions),
            ("rewards", &prepared.rewards),
            ("terminated", &prepared.terminated),
        ] {
            let actual = tensor.dim(0)?;
            if actual != count {
                return Err(ReplayStorageError::BatchLengthMismatch {
                    field: name,
                    expected: count,
                    actual,
                });
            }
        }
        self.observations.insert(
            start,
            &prepared.states,
            &prepared.next_states,
            &prepared.truncateds,
        )?;
        for (column, tensor) in [
            (&self.actions, &prepared.actions),
            (&self.rewards, &prepared.rewards),
            (&self.terminated, &prepared.terminated),
        ] {
            column.write(start, tensor)?;
        }
        Ok(count)
    }

    fn gather(&self, indices: &[usize]) -> Result<Self::Batch, Self::Error> {
        let (states, next_states) = self.observations.gather(indices)?;
        let indices = replay_index_tensor(indices, &self.device)?;
        Ok(DeterministicBatch {
            states,
            next_states,
            actions: self.actions.gather(&indices)?,
            rewards: self.rewards.gather(&indices)?,
            terminated: self.terminated.gather(&indices)?,
        })
    }

    fn sampleable_len(&self, len: usize) -> usize {
        self.observations.sampleable_len(len)
    }

    fn sample_index(&self, index: usize, len: usize) -> usize {
        self.observations.sample_index(index, len)
    }
}

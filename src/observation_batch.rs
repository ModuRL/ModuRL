//! Operations on observation tensors with a leading environment batch axis.

use crate::tensor_rank::PrevRank;
use burn::tensor::{
    Bool, DType, Device, Int, IntDType, Slice, Tensor, TensorReadError,
    kind::{Autodiff, Kind},
};
use std::fmt::Debug;

/// Observation batches shaped `[batch_size, ...item_shape]`.
/// Batching and indexing preserve gradients until `without_autodiff` is called.
pub trait ObservationBatch: Clone + Debug + Send + Sized {
    /// One environment's observation, without the batch axis.
    type Item: Clone + Debug + Send;

    /// Stacks nonempty items with matching shapes, dtypes, and devices.
    fn stack(items: Vec<Self::Item>) -> Result<Self, ObservationBatchError>;
    /// Concatenates nonempty batches with matching item shapes, dtypes, and devices.
    fn cat(batches: Vec<Self>) -> Result<Self, ObservationBatchError>;
    /// Selects `[selection_size, ...item_shape]` using in-range indices on the batch's device.
    /// Reads indices to check their bounds; an empty selection returns an empty batch.
    fn select(self, indices: Tensor<1, Int>) -> Result<Self, ObservationBatchError>;
    /// Returns one item at an in-range batch index.
    fn row(&self, index: usize) -> Result<Self::Item, ObservationBatchError>;
    /// Returns the number of environment rows.
    fn batch_size(&self) -> usize;
    /// Moves the batch to a device supporting its numeric dtype, preserving shape and kind.
    /// Cross-device Bool transfers use the destination's default Bool storage dtype.
    fn to_device(self, device: &Device) -> Self;
    /// Removes the autodiff association without changing shape, kind, or dtype.
    fn without_autodiff(self) -> Self;
}

/// Observation batch inputs that cannot be stacked, concatenated, or indexed.
#[derive(Debug, thiserror::Error)]
pub enum ObservationBatchError {
    #[error("cannot batch an empty list")]
    EmptyBatch,
    #[error("tensor {index} has shape {actual:?}, expected {expected:?}")]
    ShapeMismatch {
        index: usize,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("tensor {index} has dtype {actual:?}, expected {expected:?}")]
    DTypeMismatch {
        index: usize,
        expected: DType,
        actual: DType,
    },
    #[error("tensors must be on the same device")]
    DeviceMismatch,
    #[error("row index {index} is outside a batch of size {batch_size}")]
    IndexOutOfBounds { index: i128, batch_size: usize },
    #[error("reading row indices failed: {0}")]
    TensorRead(#[from] TensorReadError),
}

impl<const R: usize, const D: usize, K: Autodiff> ObservationBatch for Tensor<R, K>
where
    Self: PrevRank<Prev = Tensor<D, K>>,
{
    type Item = Tensor<D, K>;

    fn stack(items: Vec<Self::Item>) -> Result<Self, ObservationBatchError> {
        validate_tensors(&items, false)?;
        Ok(Tensor::stack(items, 0))
    }

    fn cat(batches: Vec<Self>) -> Result<Self, ObservationBatchError> {
        validate_tensors(&batches, true)?;
        Ok(Tensor::cat(batches, 0))
    }

    /// Selects `[indices.len(), ...item_shape]`, preserving dtype and gradients.
    fn select(self, indices: Tensor<1, Int>) -> Result<Self, ObservationBatchError> {
        if indices.device() != self.device() {
            return Err(ObservationBatchError::DeviceMismatch);
        }
        let values: Vec<i128> = if indices.dtype().is_uint() {
            indices
                .try_to_vec_as::<u64>()?
                .into_iter()
                .map(i128::from)
                .collect()
        } else {
            indices
                .try_to_vec_as::<i64>()?
                .into_iter()
                .map(i128::from)
                .collect()
        };
        let batch_size = self.batch_size();
        for &index in &values {
            if index < 0 || index >= batch_size as i128 {
                return Err(ObservationBatchError::IndexOutOfBounds { index, batch_size });
            }
        }
        if K::KIND == Kind::Bool || !matches!(indices.dtype(), DType::I32 | DType::I64) {
            if values.is_empty() {
                let mut shape = self.dims();
                shape[0] = 0;
                return Ok(Tensor::empty(shape, (&self.device(), self.dtype())));
            }
            let rows = values
                .into_iter()
                .map(|index| {
                    let index = index as usize;
                    self.clone().slice([Slice::from(index..index + 1)])
                })
                .collect();
            return Ok(Tensor::cat(rows, 0));
        }
        Ok(Tensor::select(self, 0, indices))
    }

    fn row(&self, index: usize) -> Result<Self::Item, ObservationBatchError> {
        let batch_size = self.batch_size();
        if index >= batch_size {
            return Err(ObservationBatchError::IndexOutOfBounds {
                index: index as i128,
                batch_size,
            });
        }
        Ok(self
            .clone()
            .slice([Slice::from(index..index + 1)])
            .squeeze_dim(0))
    }

    fn batch_size(&self) -> usize {
        self.dims()[0]
    }

    fn to_device(self, device: &Device) -> Self {
        if K::KIND == Kind::Bool && self.device() != *device {
            let tensor = Tensor::<R, Bool>::from_dispatch(self.into_dispatch());
            let tensor = tensor.cast(IntDType::I32).to_device(device).bool();
            return Tensor::from_dispatch(tensor.into_dispatch());
        }
        Tensor::to_device(self, device)
    }

    fn without_autodiff(self) -> Self {
        Tensor::without_autodiff(self)
    }
}

/// Checks `[...item_shape]` or `[batch_size, ...item_shape]` tensors before combining them.
fn validate_tensors<const D: usize, K: Autodiff>(
    tensors: &[Tensor<D, K>],
    batched: bool,
) -> Result<(), ObservationBatchError> {
    let first = tensors.first().ok_or(ObservationBatchError::EmptyBatch)?;
    let start = usize::from(batched);
    let expected = first.dims()[start..].to_vec();
    for (index, tensor) in tensors.iter().enumerate().skip(1) {
        let actual = tensor.dims()[start..].to_vec();
        if actual != expected {
            return Err(ObservationBatchError::ShapeMismatch {
                index,
                expected,
                actual,
            });
        }
        if tensor.dtype() != first.dtype() {
            return Err(ObservationBatchError::DTypeMismatch {
                index,
                expected: first.dtype(),
                actual: tensor.dtype(),
            });
        }
        if tensor.device() != first.device() {
            return Err(ObservationBatchError::DeviceMismatch);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Float;

    /// Exercises `[batch_size, 2]` tensors, preserving their values, kind, and dtype.
    #[allow(deprecated)]
    fn exercise_tensor_ops<K: Autodiff>(items: Vec<Tensor<1, K>>, dtype: DType) {
        let batch = <Tensor<2, K> as ObservationBatch>::stack(items).unwrap();
        assert_eq!(batch.dims(), [2, 2]);
        assert_eq!(batch.dtype(), dtype);
        assert_eq!(ObservationBatch::batch_size(&batch), 2);
        assert_eq!(
            batch.try_to_vec_as::<f64>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        let row = ObservationBatch::row(&batch, 1).unwrap();
        assert_eq!(row.dims(), [2]);
        assert_eq!(row.dtype(), dtype);
        assert_eq!(row.try_to_vec_as::<f64>().unwrap(), vec![3.0, 4.0]);
        let combined = ObservationBatch::cat(vec![
            batch.clone(),
            batch.clone().slice([Slice::from(0..1)]),
        ])
        .unwrap();
        assert_eq!(combined.dims(), [3, 2]);
        assert_eq!(combined.dtype(), dtype);
        assert_eq!(
            combined.try_to_vec_as::<f64>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0]
        );
        let selected =
            ObservationBatch::select(combined, Tensor::from_data([2i32, 1, 1], &batch.device()))
                .unwrap();
        assert_eq!(selected.dims(), [3, 2]);
        assert_eq!(selected.dtype(), dtype);
        assert_eq!(
            selected.try_to_vec_as::<f64>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 3.0, 4.0]
        );
        let empty = ObservationBatch::select(
            batch.clone(),
            Tensor::empty([0], (&batch.device(), DType::I64)),
        )
        .unwrap();
        assert_eq!(ObservationBatch::batch_size(&empty), 0);
        assert_eq!(empty.dtype(), dtype);
        assert!(matches!(
            ObservationBatch::row(&empty, 0),
            Err(ObservationBatchError::IndexOutOfBounds { .. })
        ));
        let moved = ObservationBatch::to_device(batch.clone(), &Device::ndarray());
        assert_eq!(moved.device(), Device::ndarray());
        assert_eq!(moved.dtype(), dtype);
        assert_eq!(moved.dims(), batch.dims());
        assert_eq!(
            moved.try_to_vec_as::<f64>().unwrap(),
            batch.try_to_vec_as::<f64>().unwrap()
        );
        let detached = ObservationBatch::without_autodiff(batch.clone().autodiff());
        assert!(!detached.is_autodiff());
        assert_eq!(detached.dtype(), dtype);
        assert_eq!(detached.dims(), batch.dims());
        assert_eq!(detached.device(), batch.device());
        assert_eq!(
            detached.try_to_vec_as::<f64>().unwrap(),
            batch.try_to_vec_as::<f64>().unwrap()
        );
    }

    #[test]
    fn float_operations_preserve_values_and_dtype() {
        let device = Device::flex();
        exercise_tensor_ops::<Float>(
            vec![
                Tensor::from_data([1.0f64, 2.0], (&device, DType::F64)),
                Tensor::from_data([3.0f64, 4.0], (&device, DType::F64)),
            ],
            DType::F64,
        );
    }

    #[test]
    fn u8_operations_preserve_values_and_dtype() {
        let device = Device::flex();
        exercise_tensor_ops::<Int>(
            vec![
                Tensor::from_data([1u8, 2], (&device, DType::U8)),
                Tensor::from_data([3u8, 4], (&device, DType::U8)),
            ],
            DType::U8,
        );
    }

    #[test]
    fn boolean_detachment_preserves_values_and_dtype() {
        let device = Device::flex().autodiff();
        let batch = Tensor::<2, Bool>::from_data([[true, false], [false, true]], &device);
        assert!(batch.is_autodiff());
        let detached = ObservationBatch::without_autodiff(batch.clone());
        assert!(!detached.is_autodiff());
        assert_eq!(detached.dims(), batch.dims());
        assert_eq!(detached.dtype(), batch.dtype());
        assert_eq!(detached.device(), device.without_autodiff());
        assert_eq!(
            detached.try_to_vec_as::<bool>().unwrap(),
            vec![true, false, false, true]
        );
    }

    #[test]
    #[allow(deprecated)]
    fn boolean_selection_preserves_dtype_and_can_be_concatenated() {
        for device in [Device::flex(), Device::ndarray()] {
            let batch = Tensor::<2, Bool>::from_data([[true, false], [false, true]], &device);
            for rows in [vec![1i32], vec![]] {
                let indices = Tensor::from_data(rows.as_slice(), (&device, DType::I32));
                let selected = ObservationBatch::select(batch.clone(), indices).unwrap();
                let combined =
                    ObservationBatch::cat(vec![batch.clone(), selected.clone()]).unwrap();
                assert_eq!(selected.dtype(), batch.dtype());
                assert_eq!(selected.dims(), [rows.len(), 2]);
                assert_eq!(
                    selected.try_to_vec_as::<bool>().unwrap(),
                    if rows.is_empty() {
                        vec![]
                    } else {
                        vec![false, true]
                    }
                );
                assert_eq!(combined.dims(), [2 + rows.len(), 2]);
                let mut expected = vec![true, false, false, true];
                if !rows.is_empty() {
                    expected.extend([false, true]);
                }
                assert_eq!(combined.try_to_vec_as::<bool>().unwrap(), expected);
            }
        }
    }

    #[test]
    #[allow(deprecated)]
    fn boolean_transfer_uses_destination_storage_and_preserves_values() {
        for source in [Device::flex(), Device::ndarray()] {
            for destination in [Device::flex(), Device::ndarray()] {
                let source = source.clone().autodiff();
                let batch = Tensor::<2, Bool>::from_data([[true, false], [false, true]], &source);
                let transferred = ObservationBatch::to_device(batch, &destination);
                assert_eq!(transferred.device(), destination.clone().autodiff());
                assert_eq!(transferred.dims(), [2, 2]);
                assert_eq!(
                    transferred.try_to_vec_as::<bool>().unwrap(),
                    vec![true, false, false, true]
                );
                assert!(transferred.is_autodiff());
                let default = Tensor::<2, Bool>::from_data([[true, false]], &destination);
                assert_eq!(transferred.dtype(), default.dtype());
            }
        }
    }

    #[test]
    #[allow(deprecated)]
    fn selection_accepts_original_integer_index_dtypes() {
        for device in [Device::flex(), Device::ndarray()] {
            for dtype in [DType::I32, DType::I64, DType::U8] {
                let indices = Tensor::<1, Int>::from_data([1i32, 0], (&device, dtype));
                let floats = Tensor::<2>::from_data([[1.0f32], [2.0]], &device);
                let selected = ObservationBatch::select(floats, indices.clone()).unwrap();
                assert_eq!(selected.try_to_vec_as::<f32>().unwrap(), vec![2.0, 1.0]);
                let integers = Tensor::<2, Int>::from_data([[1u8], [2]], (&device, DType::U8));
                let selected = ObservationBatch::select(integers, indices).unwrap();
                assert_eq!(selected.dtype(), DType::U8);
                assert_eq!(selected.try_to_vec_as::<u8>().unwrap(), vec![2, 1]);
            }
        }
    }

    #[test]
    fn float_batching_and_indexing_preserve_gradients_until_detached() {
        let device = Device::flex().autodiff();
        let item = Tensor::<1>::from_data([1.0f32, 2.0], &device).require_grad();
        let batch = <Tensor<2> as ObservationBatch>::stack(vec![item.clone()]).unwrap();
        let combined = ObservationBatch::cat(vec![batch.clone(), batch]).unwrap();
        let selected =
            ObservationBatch::select(combined, Tensor::from_data([1i32, 0], &device)).unwrap();
        let row = ObservationBatch::row(&selected, 0).unwrap();
        let gradients = row.sum().backward();
        assert_eq!(
            item.grad(&gradients)
                .unwrap()
                .try_to_vec_as::<f32>()
                .unwrap(),
            vec![1.0, 1.0]
        );
        let detached = ObservationBatch::without_autodiff(selected);
        assert!(!detached.is_autodiff());
        assert!(!detached.is_require_grad());
    }

    /// Checks errors for combining `[2]` items or `[batch_size, 2]` batches.
    #[allow(deprecated)]
    fn exercise_combination_errors<K: Autodiff>(dtype: DType, other_dtype: DType) {
        let device = Device::flex();
        assert!(matches!(
            <Tensor<2, K> as ObservationBatch>::stack(vec![]),
            Err(ObservationBatchError::EmptyBatch)
        ));
        assert!(matches!(
            <Tensor<2, K> as ObservationBatch>::cat(vec![]),
            Err(ObservationBatchError::EmptyBatch)
        ));
        assert!(matches!(
            <Tensor<2, K> as ObservationBatch>::stack(vec![
                Tensor::empty([2], (&device, dtype)),
                Tensor::empty([3], (&device, dtype))
            ]),
            Err(ObservationBatchError::ShapeMismatch { index: 1, .. })
        ));
        assert!(matches!(
            <Tensor<2, K> as ObservationBatch>::cat(vec![
                Tensor::empty([1, 2], (&device, dtype)),
                Tensor::empty([2, 3], (&device, dtype))
            ]),
            Err(ObservationBatchError::ShapeMismatch { index: 1, .. })
        ));
        assert!(matches!(
            <Tensor<2, K> as ObservationBatch>::stack(vec![
                Tensor::empty([2], (&device, dtype)),
                Tensor::empty([2], (&device, other_dtype))
            ]),
            Err(ObservationBatchError::DTypeMismatch { index: 1, .. })
        ));
        assert!(matches!(
            <Tensor<2, K> as ObservationBatch>::cat(vec![
                Tensor::empty([1, 2], (&device, dtype)),
                Tensor::empty([2, 2], (&device, other_dtype))
            ]),
            Err(ObservationBatchError::DTypeMismatch { index: 1, .. })
        ));
        assert!(matches!(
            <Tensor<2, K> as ObservationBatch>::stack(vec![
                Tensor::empty([2], (&device, dtype)),
                Tensor::empty([2], (&Device::ndarray(), dtype))
            ]),
            Err(ObservationBatchError::DeviceMismatch)
        ));
        assert!(matches!(
            <Tensor<2, K> as ObservationBatch>::cat(vec![
                Tensor::empty([1, 2], (&device, dtype)),
                Tensor::empty([2, 2], (&Device::ndarray(), dtype))
            ]),
            Err(ObservationBatchError::DeviceMismatch)
        ));
    }

    #[test]
    fn combining_float_and_u8_tensors_returns_input_errors() {
        exercise_combination_errors::<Float>(DType::F32, DType::F64);
        exercise_combination_errors::<Int>(DType::U8, DType::I64);
    }

    /// Checks row and select bounds for `[2, 2]` batches without invoking backend indexing.
    #[allow(deprecated)]
    fn exercise_index_errors<K: Autodiff>(batch: Tensor<2, K>) {
        for index in [2, usize::MAX] {
            assert!(matches!(
                ObservationBatch::row(&batch, index),
                Err(ObservationBatchError::IndexOutOfBounds { .. })
            ));
        }
        for indices in [vec![-1i64], vec![2], vec![0, 1, 2], vec![i64::MAX]] {
            assert!(matches!(
                ObservationBatch::select(
                    batch.clone(),
                    Tensor::from_data(indices.as_slice(), (&batch.device(), DType::I64))
                ),
                Err(ObservationBatchError::IndexOutOfBounds { .. })
            ));
        }
        assert!(
            matches!(ObservationBatch::select(batch.clone(), Tensor::from_data([u64::MAX], (&batch.device(), DType::U64))), Err(ObservationBatchError::IndexOutOfBounds { index, .. }) if index == i128::from(u64::MAX))
        );
        assert!(matches!(
            ObservationBatch::select(batch, Tensor::from_data([0i32], &Device::ndarray())),
            Err(ObservationBatchError::DeviceMismatch)
        ));
    }

    #[test]
    fn indexing_float_and_u8_tensors_returns_input_errors() {
        let device = Device::flex();
        exercise_index_errors(Tensor::<2>::zeros([2, 2], &device));
        exercise_index_errors(Tensor::<2, Int>::zeros([2, 2], (&device, DType::U8)));
    }
}

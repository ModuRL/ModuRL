use burn::tensor::{Bool, DType, Device, Tensor, TensorData, kind::Basic};

pub trait Experience: Clone {
    type Batch;
    type Error;

    fn batch(experiences: &[Self]) -> Result<Self::Batch, Self::Error>;
}

/// Invalid shapes or incompatible tensors in an experience batch.
#[derive(Debug, thiserror::Error)]
pub enum ExperienceBatchError {
    #[error("cannot batch an empty experience slice")]
    EmptyBatch,
    #[error("experience {index} has shape {actual:?}, expected {expected:?}")]
    ShapeMismatch {
        index: usize,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("experience {index} has dtype {actual:?}, expected {expected:?}")]
    DTypeMismatch {
        index: usize,
        expected: DType,
        actual: DType,
    },
    #[error("experience {index} is on a different device")]
    DeviceMismatch { index: usize },
    #[error("experience {index} has {actual} boolean values, expected {expected}")]
    BoolFieldLengthMismatch {
        index: usize,
        expected: usize,
        actual: usize,
    },
}

/// Adds a leading experience axis to rank-`D` fields, returning rank `B = D + 1`.
/// Invalid rank combinations are rejected when the compiler builds the function.
/// Fields must share shape, dtype, and device. Gradients remain connected;
/// callers control detachment before storing or batching experience.
pub(crate) fn stack_tensor_field<T, const D: usize, const B: usize, K: Basic>(
    experiences: &[T],
    select: fn(&T) -> Tensor<D, K>,
) -> Result<Tensor<B, K>, ExperienceBatchError> {
    const {
        assert!(B == D + 1, "output rank must be input rank + 1");
    }
    if experiences.is_empty() {
        return Err(ExperienceBatchError::EmptyBatch);
    }
    let fields: Vec<_> = experiences.iter().map(select).collect();
    let first = &fields[0];
    for (index, field) in fields.iter().enumerate().skip(1) {
        if field.dims() != first.dims() {
            return Err(ExperienceBatchError::ShapeMismatch {
                index,
                expected: first.dims().to_vec(),
                actual: field.dims().to_vec(),
            });
        }
        if field.dtype() != first.dtype() {
            return Err(ExperienceBatchError::DTypeMismatch {
                index,
                expected: first.dtype(),
                actual: field.dtype(),
            });
        }
        if field.device() != first.device() {
            return Err(ExperienceBatchError::DeviceMismatch { index });
        }
    }
    Ok(Tensor::stack(fields, 0))
}

/// Converts one `[item_count]` boolean field from each experience into an
/// `Bool` tensor shaped `[batch, item_count, 1]`. Every field must have the same
/// length. Numeric mask conversion belongs to the algorithm consuming the batch.
pub(crate) fn stack_bool_field<T>(
    experiences: &[T],
    select: fn(&T) -> &[bool],
    device: &Device,
) -> Result<Tensor<3, Bool>, ExperienceBatchError> {
    let first = experiences
        .first()
        .ok_or(ExperienceBatchError::EmptyBatch)?;
    let item_count = select(first).len();
    let mut values = Vec::with_capacity(experiences.len() * item_count);
    for (index, experience) in experiences.iter().enumerate() {
        let field = select(experience);
        if field.len() != item_count {
            return Err(ExperienceBatchError::BoolFieldLengthMismatch {
                index,
                expected: item_count,
                actual: field.len(),
            });
        }
        values.extend_from_slice(field);
    }
    Ok(Tensor::from_data(
        TensorData::new(values, [experiences.len(), item_count, 1]),
        device,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Int;

    #[test]
    fn stacking_preserves_time_environment_axes_dtype_and_gradients() {
        let device = Device::flex().autodiff();
        let fields = [
            Tensor::<2>::from_data([[1.0f64, 2.0], [3.0, 4.0]], (&device, DType::F64))
                .require_grad(),
            Tensor::<2>::from_data([[5.0f64, 6.0], [7.0, 8.0]], (&device, DType::F64))
                .require_grad(),
        ];
        let batch: Tensor<3> = stack_tensor_field(&fields, Clone::clone).unwrap();
        assert_eq!(batch.dims(), [2, 2, 2]);
        assert_eq!(batch.dtype(), DType::F64);
        assert_eq!(batch.device(), device);
        assert_eq!(
            batch.clone().into_data().try_to_vec::<f64>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
        let gradients = batch.sum().backward();
        for field in fields {
            assert_eq!(
                field
                    .grad(&gradients)
                    .unwrap()
                    .into_data()
                    .try_to_vec::<f64>()
                    .unwrap(),
                vec![1.0; 4]
            );
        }
    }

    #[test]
    fn stacking_preserves_integer_kind() {
        let device = Device::flex();
        let fields = [
            Tensor::<1, Int>::from_data([1i32, 2], &device),
            Tensor::from_data([3i32, 4], &device),
        ];
        let batch: Tensor<2, Int> = stack_tensor_field(&fields, Clone::clone).unwrap();
        assert_eq!(batch.dims(), [2, 2]);
        assert_eq!(
            batch.into_data().try_to_vec::<i32>().unwrap(),
            vec![1, 2, 3, 4]
        );
    }

    #[test]
    fn invalid_tensor_batches_return_typed_errors() {
        let device = Device::flex();
        let empty: [Tensor<1>; 0] = [];
        let result: Result<Tensor<2>, _> = stack_tensor_field(&empty, Clone::clone);
        assert!(matches!(result, Err(ExperienceBatchError::EmptyBatch)));
        let fields = [
            Tensor::<1>::zeros([2], &device),
            Tensor::zeros([3], &device),
        ];
        let result: Result<Tensor<2>, _> = stack_tensor_field(&fields, Clone::clone);
        assert!(matches!(
            result,
            Err(ExperienceBatchError::ShapeMismatch { index: 1, .. })
        ));
        let fields = [
            Tensor::<1>::zeros([2], (&device, DType::F32)),
            Tensor::zeros([2], (&device, DType::F64)),
        ];
        let result: Result<Tensor<2>, _> = stack_tensor_field(&fields, Clone::clone);
        assert!(matches!(
            result,
            Err(ExperienceBatchError::DTypeMismatch { index: 1, .. })
        ));
        let fields = [
            Tensor::<1>::zeros([2], &device),
            Tensor::zeros([2], &device.clone().autodiff()),
        ];
        let result: Result<Tensor<2>, _> = stack_tensor_field(&fields, Clone::clone);
        assert_eq!(result.unwrap().dims(), [2, 2]);
    }

    #[test]
    fn boolean_fields_remain_boolean_and_validate_lengths() {
        let device = Device::flex();
        let fields = [vec![true, false], vec![false, true]];
        let batch = stack_bool_field(&fields, |field| field, &device).unwrap();
        assert_eq!(batch.dims(), [2, 2, 1]);
        assert!(batch.dtype().is_bool());
        assert_eq!(
            batch.into_data().try_to_vec::<bool>().unwrap(),
            vec![true, false, false, true]
        );
        let invalid = [vec![true, false], vec![true]];
        assert!(matches!(
            stack_bool_field(&invalid, |field| field, &device),
            Err(ExperienceBatchError::BoolFieldLengthMismatch { index: 1, .. })
        ));
        let empty: [Vec<bool>; 0] = [];
        assert!(matches!(
            stack_bool_field(&empty, |field| field, &device),
            Err(ExperienceBatchError::EmptyBatch)
        ));
    }
}

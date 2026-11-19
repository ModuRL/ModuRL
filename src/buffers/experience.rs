use burn::tensor::{DType, Tensor, kind::Basic};

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
}

/// Concatenates `[rows, ...]` fields without adding an axis. Trailing shapes,
/// dtypes, and devices must match; gradients remain connected.
pub(crate) fn cat_tensor_field<T, const D: usize, K: Basic>(
    experiences: &[T],
    select: fn(&T) -> Tensor<D, K>,
) -> Result<Tensor<D, K>, ExperienceBatchError> {
    if experiences.is_empty() {
        return Err(ExperienceBatchError::EmptyBatch);
    }
    let fields: Vec<_> = experiences.iter().map(select).collect();
    let first = &fields[0];
    for (index, field) in fields.iter().enumerate().skip(1) {
        if field.dims()[1..] != first.dims()[1..] {
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
    Ok(Tensor::cat(fields, 0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Int};

    #[test]
    fn concatenation_preserves_row_order_dtype_and_gradients() {
        let device = Device::flex().autodiff();
        let fields = [
            Tensor::<2>::from_data([[1.0f64, 2.0], [3.0, 4.0]], (&device, DType::F64))
                .require_grad(),
            Tensor::<2>::from_data([[5.0f64, 6.0], [7.0, 8.0]], (&device, DType::F64))
                .require_grad(),
        ];
        let batch: Tensor<2> = cat_tensor_field(&fields, Clone::clone).unwrap();
        assert_eq!(batch.dims(), [4, 2]);
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
    fn concatenation_preserves_integer_kind() {
        let device = Device::flex();
        let fields = [
            Tensor::<1, Int>::from_data([1i32, 2], &device),
            Tensor::from_data([3i32, 4], &device),
        ];
        let batch: Tensor<1, Int> = cat_tensor_field(&fields, Clone::clone).unwrap();
        assert_eq!(batch.dims(), [4]);
        assert_eq!(
            batch.into_data().try_to_vec::<i32>().unwrap(),
            vec![1, 2, 3, 4]
        );
    }

    #[test]
    fn invalid_tensor_batches_return_typed_errors() {
        let device = Device::flex();
        let empty: [Tensor<2>; 0] = [];
        let result: Result<Tensor<2>, _> = cat_tensor_field(&empty, Clone::clone);
        assert!(matches!(result, Err(ExperienceBatchError::EmptyBatch)));
        let fields = [
            Tensor::<2>::zeros([1, 2], &device),
            Tensor::zeros([1, 3], &device),
        ];
        let result: Result<Tensor<2>, _> = cat_tensor_field(&fields, Clone::clone);
        assert!(matches!(
            result,
            Err(ExperienceBatchError::ShapeMismatch { index: 1, .. })
        ));
        let fields = [
            Tensor::<2>::zeros([1, 2], (&device, DType::F32)),
            Tensor::zeros([1, 2], (&device, DType::F64)),
        ];
        let result: Result<Tensor<2>, _> = cat_tensor_field(&fields, Clone::clone);
        assert!(matches!(
            result,
            Err(ExperienceBatchError::DTypeMismatch { index: 1, .. })
        ));
        let fields = [
            Tensor::<2>::zeros([1, 2], &device),
            Tensor::zeros([1, 2], &device.clone().autodiff()),
        ];
        let result: Result<Tensor<2>, _> = cat_tensor_field(&fields, Clone::clone);
        assert_eq!(result.unwrap().dims(), [2, 2]);
    }
}

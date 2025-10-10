use candle_core::{DType, Error, Tensor, backprop::GradStore};

pub(crate) fn torch_like_min(a: &Tensor, b: &Tensor) -> Result<Tensor, Error> {
    // candle has a min but all it does is takes a tensor and a dim
    // we add a new dim at the end and then take the min along that dim
    let original_shape = a.dims().to_vec();
    let a = a.unsqueeze(original_shape.len()).unwrap();
    let b = b.unsqueeze(original_shape.len()).unwrap();
    let merged = Tensor::cat(&[a, b], original_shape.len()).unwrap();
    let result = merged.min(original_shape.len()).unwrap();

    Ok(result)
}

pub(crate) fn clip_gradients(
    grad_store: &mut GradStore,
    max_norm: f32,
) -> Result<f32, candle_core::Error> {
    let mut total_norm_sq: f32 = 0.0;
    let mut grads = vec![];

    for id in grad_store.get_ids() {
        if let Some(grad) = grad_store.get_id(*id) {
            let norm_sq = grad.sqr()?.sum_all()?.to_scalar::<f32>()?;
            total_norm_sq += norm_sq;
            grads.push((*id, grad.clone()));
        }
    }

    let total_norm = total_norm_sq.sqrt();
    if total_norm > max_norm {
        let scale = max_norm / total_norm;
        for (id, grad) in grads {
            let scale_t = Tensor::new(scale, &grad.device())?;
            let clipped = grad.broadcast_mul(&scale_t)?;
            grad_store.insert_id(id, clipped);
        }
    }

    Ok(total_norm)
}

// implement the tanh activation function
pub fn tanh(x: &Tensor) -> Result<Tensor, Error> {
    let e_pos = x.exp()?;
    let e_neg = (-1.0 * x)?.exp()?;
    let numerator = (&e_pos - &e_neg)?;
    let denominator = (&e_pos + &e_neg)?;
    numerator.broadcast_div(&denominator)
}

/// Check if a tensor contains any NaN values.
/// Use very sparingly extremely slow.
pub fn tensor_has_nan(t: &Tensor) -> Result<bool, candle_core::Error> {
    // First, ensure tensor is on CPU (or copy to CPU)
    let t_cpu = if t.device().is_cpu() {
        t.clone()
    } else {
        t.to_device(&candle_core::Device::Cpu)?
    }
    .flatten_all()?;

    // Now get raw buffer as Vec<f32> (if dtype is f32)
    match t_cpu.dtype() {
        DType::F32 => {
            // assuming there's a method to get the raw Vec<f32> or slice
            let data: Vec<f32> = t_cpu.to_vec1()?;
            // (you'd need to check how Candle exposes this; I didn't find `as_slice` in docs)
            for x in data {
                if x.is_nan() {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        DType::F64 => {
            let data: Vec<f64> = t_cpu.to_vec1()?;
            for x in data {
                if x.is_nan() {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        _ => {
            // for integer types etc, NaN isn't relevant
            // not sure about bfloat16 or float16
            Ok(false)
        }
    }
}

pub fn gen_range_int_tensor(
    start: u32,
    end: u32,
    device: &candle_core::Device,
) -> Result<u32, candle_core::Error> {
    Tensor::rand(start as f32, end as f32 + 1.0, &[], device)?
        .floor()?
        .to_dtype(candle_core::DType::U32)?
        .clamp(start, end)? // This makes sure that it can't land exactly on possible_values
        .to_vec0::<u32>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contains_nan() {
        let a = Tensor::full(1.0, &[2, 2], &candle_core::Device::Cpu).unwrap();
        assert!(!tensor_has_nan(&a).unwrap());
        let b = Tensor::full(f32::NAN, &[2, 2], &candle_core::Device::Cpu).unwrap();
        assert!(tensor_has_nan(&b).unwrap());
        let c =
            Tensor::from_vec(vec![1.0, f32::NAN, 3.0], &[3], &candle_core::Device::Cpu).unwrap();
        assert!(tensor_has_nan(&c).unwrap());
    }
}

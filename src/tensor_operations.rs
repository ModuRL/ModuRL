use candle_core::{DType, Error, Tensor, backprop::GradStore};

pub(crate) fn torch_like_max(a: &Tensor, b: &Tensor) -> Result<Tensor, Error> {
    // See torch_like_min for why this is a masked select; ties route to `b`.
    let mask = a.gt(b)?.to_dtype(a.dtype())?;
    let inv_mask = (1.0 - &mask)?;
    (a * &mask)? + (b * &inv_mask)?
}

pub(crate) fn torch_like_min(a: &Tensor, b: &Tensor) -> Result<Tensor, Error> {
    // candle's min-reduction backward sends full gradient to every element tied
    // for the minimum, so a cat+min implementation double-counts gradient when
    // a == b (e.g. PPO's first minibatch, where ratio == 1 makes both surrogates
    // equal). Select with a detached mask instead so exactly one branch receives
    // gradient; ties route to `b`.
    let mask = a.lt(b)?.to_dtype(a.dtype())?;
    let inv_mask = (1.0 - &mask)?;
    (a * &mask)? + (b * &inv_mask)?
}

pub(crate) fn clip_gradients(
    loss: &Tensor,
    grad_store: &mut GradStore,
    max_norm: f32,
) -> Result<f32, candle_core::Error> {
    let mut grads = vec![];

    // We will collect all norm squares first
    // Then sort them to ensure deterministic order since float addition is not associative
    // Also note that tensor IDs are not determinisitic across runs so we cannot sort by that
    // Candle's GradStore can retain local gradients for non-variable operands.
    // CleanRL/PyTorch clips registered parameters only, so select the variable
    // nodes reachable from this loss rather than every GradStore entry.
    let variable_ids = loss
        .sorted_nodes()
        .into_iter()
        .filter(|node| node.is_variable())
        .map(Tensor::id);

    // Sort the scalar norm contributions before summing because floating-point
    // addition is not associative. Tensor IDs are allocation-dependent and
    // therefore are not a deterministic ordering across runs.
    let mut norm_sqrs: Vec<f32> = vec![];

    for id in variable_ids {
        if let Some(grad) = grad_store.get_id(id) {
            let norm_sq = grad.sqr()?.sum_all()?.to_scalar::<f32>()?;
            norm_sqrs.push(norm_sq);
            grads.push((id, grad.clone()));
        }
    }

    norm_sqrs.sort_by(f32::total_cmp);
    let total_norm_sq = norm_sqrs.into_iter().sum::<f32>();

    let total_norm = total_norm_sq.sqrt();
    if total_norm > max_norm {
        // Match PyTorch's clip_grad_norm_ denominator, including its epsilon.
        let scale = max_norm / (total_norm + 1e-6);
        for (id, grad) in &grads {
            let scale_t = Tensor::new(scale, grad.device())?;
            let clipped = grad.broadcast_mul(&scale_t)?;
            grad_store.insert_id(*id, clipped);
        }
    }

    Ok(total_norm)
}

pub(crate) fn normalize_tensor(t: &Tensor) -> Result<Tensor, candle_core::Error> {
    let mean = t.mean_all()?.broadcast_as(t.shape())?;
    let diff = (t.clone() - mean.clone())?;

    // Unbiased (n-1) std to match torch .std() / reference PPO implementations.
    let n = t.elem_count().max(2) as f64;
    let std = (diff.sqr()?.sum_all()? / (n - 1.0))?.sqrt()?;
    let std_with_eps = (std + 1e-8)?.broadcast_as(t.shape())?;

    diff / std_with_eps
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
    Ok(Tensor::rand(start as f32, end as f32 + 1.0, &[], device)?
        .floor()?
        .to_dtype(candle_core::DType::U32)?
        .to_vec0::<u32>()?
        .clamp(start, end)) // ensure within bounds
}

pub fn fisher_yates_shuffle<T: Clone>(arr: &mut [T], device: &candle_core::Device) {
    // generate arr.len random floats from 0 to 1
    let random_floats_tensor = Tensor::rand(0.0f32, 1.0f32, &[arr.len()], device).unwrap();
    let random_floats_vec = random_floats_tensor.to_vec1::<f32>().unwrap();
    let n = arr.len();
    for i in (1..n).rev() {
        let rand_float = random_floats_vec[i];
        // now we scale it to 0..=i
        let mut j = (rand_float * (i as f32 + 1.0)).floor() as usize;
        // make sure j is in bounds
        j = j.clamp(0, i);
        arr.swap(i, j);
    }
}

pub fn resevoir_sample<T: Clone>(arr: &[T], size: usize, device: &candle_core::Device) -> Vec<T> {
    let arr_len = arr.len();
    if arr_len == 0 {
        return vec![];
    }
    if arr.len() < size {
        return arr.to_vec();
    }
    // fill the reservoir array
    let mut reservoir: Vec<T> = arr[0..size.min(arr_len)].to_vec();
    let random_floats_tensor =
        Tensor::rand(0.0f32, 1.0f32, &[arr_len - reservoir.len()], device).unwrap();
    let random_floats_vec = random_floats_tensor.to_vec1::<f32>().unwrap();
    for i in size..arr_len {
        let rand_float = random_floats_vec[i - size];
        let j = (rand_float * (i as f32 + 1.0)).floor() as usize;
        if j < size {
            reservoir[j] = arr[i].clone();
        }
    }
    reservoir
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_clipping_matches_pytorch_epsilon() {
        let device = candle_core::Device::Cpu;
        let variable = candle_core::Var::from_vec(vec![3.0f32, 4.0], 2, &device).unwrap();
        let loss = variable.sqr().unwrap().sum_all().unwrap();
        let mut gradients = loss.backward().unwrap();

        let original_norm = clip_gradients(&loss, &mut gradients, 5.0).unwrap();
        let clipped = gradients
            .get(variable.as_tensor())
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let expected_scale = 5.0 / (10.0 + 1e-6);

        assert!((original_norm - 10.0).abs() < 1e-6);
        assert!((clipped[0] - 6.0 * expected_scale).abs() < 1e-6);
        assert!((clipped[1] - 8.0 * expected_scale).abs() < 1e-6);
    }

    #[test]
    fn gradient_clipping_ignores_detached_operand_gradients() {
        let device = candle_core::Device::Cpu;
        let parameter = candle_core::Var::from_vec(vec![1_000.0f32, 1_000.0], 2, &device).unwrap();
        let detached_operand = candle_core::Var::from_vec(vec![1.0f32, 1.0], 2, &device).unwrap();
        let detached_operand = detached_operand.detach();
        let loss = (parameter.as_tensor() * detached_operand)
            .unwrap()
            .sum_all()
            .unwrap();
        let mut gradients = loss.backward().unwrap();

        // The parameter gradient is [1, 1], while Candle also retains the
        // detached operand's much larger local gradient [1000, 1000]. Only the
        // parameter gradient belongs in the clipping norm.
        let original_norm = clip_gradients(&loss, &mut gradients, 1.0).unwrap();
        let clipped = gradients
            .get(parameter.as_tensor())
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let expected_norm = 2.0f32.sqrt();
        let expected_scale = 1.0 / (expected_norm + 1e-6);

        assert!((original_norm - expected_norm).abs() < 1e-6);
        assert!((clipped[0] - expected_scale).abs() < 1e-6);
        assert!((clipped[1] - expected_scale).abs() < 1e-6);
    }

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

    #[cfg(any(feature = "cuda", feature = "metal"))]
    #[test]
    fn test_fisher_yates_shuffle_determinism() {
        #[cfg(feature = "cuda")]
        let device = candle_core::Device::new_cuda(0).unwrap();
        #[cfg(feature = "metal")]
        let device = candle_core::Device::new_metal(0).unwrap();

        let mut arr1: Vec<u32> = (0..100).collect();
        let mut arr2: Vec<u32> = (0..100).collect();

        device.set_seed(42).unwrap();
        fisher_yates_shuffle(&mut arr1, &device);
        fisher_yates_shuffle(&mut arr1, &device);

        device.set_seed(42).unwrap();
        fisher_yates_shuffle(&mut arr2, &device);
        fisher_yates_shuffle(&mut arr2, &device);

        assert_eq!(arr1, arr2);
    }
}

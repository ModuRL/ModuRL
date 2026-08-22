use candle_core::{Tensor, backprop::GradStore};

/// Clips gradients reachable from scalar `loss` shaped `[]`; gradient tensors
/// in `grad_store` retain their parameter shapes.
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
    let mut norm_sqrs: Vec<f64> = vec![];

    for id in variable_ids {
        if let Some(grad) = grad_store.get_id(id) {
            let norm_sq = grad
                .sqr()?
                .sum_all()?
                .to_dtype(candle_core::DType::F64)?
                .to_scalar::<f64>()?;
            norm_sqrs.push(norm_sq);
            grads.push((id, grad.clone()));
        }
    }

    norm_sqrs.sort_by(f64::total_cmp);
    let total_norm_sq = norm_sqrs.into_iter().sum::<f64>();

    let total_norm = total_norm_sq.sqrt();
    if total_norm > f64::from(max_norm) {
        // Match PyTorch's clip_grad_norm_ denominator, including its epsilon.
        let scale = f64::from(max_norm) / (total_norm + 1e-6);
        for (id, grad) in &grads {
            let scale_t = Tensor::new(scale, grad.device())?.to_dtype(grad.dtype())?;
            let clipped = grad.broadcast_mul(&scale_t)?;
            grad_store.insert_id(*id, clipped);
        }
    }

    Ok(total_norm as f32)
}

/// Normalizes `t` over all elements while preserving its arbitrary shape
/// `[...]`.
pub(crate) fn normalize_tensor(t: &Tensor) -> Result<Tensor, candle_core::Error> {
    let mean = t.mean_all()?.broadcast_as(t.shape())?;
    let diff = (t.clone() - mean.clone())?;

    // Unbiased (n-1) std to match torch .std() / reference PPO implementations.
    let n = t.elem_count().max(2) as f64;
    let std = (diff.sqr()?.sum_all()? / (n - 1.0))?.sqrt()?;
    let std_with_eps = (std + 1e-8)?.broadcast_as(t.shape())?;

    diff / std_with_eps
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
}

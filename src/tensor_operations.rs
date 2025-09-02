use candle_core::{backprop::GradStore, Error, Tensor};

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
pub(crate) fn tanh(x: &Tensor) -> Result<Tensor, Error> {
    let e_pos = x.exp()?;
    let e_neg = (-1.0 * x)?.exp()?;
    let numerator = (&e_pos - &e_neg)?;
    let denominator = (&e_pos + &e_neg)?;
    numerator.broadcast_div(&denominator)
}

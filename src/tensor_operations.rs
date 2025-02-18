use candle_core::{Error, Tensor};

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

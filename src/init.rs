//! Orthogonal weight initialization (PyTorch `nn.init.orthogonal_` semantics),
//! as used by reference PPO implementations: sqrt(2) gain for hidden layers,
//! 0.01 for policy heads, 1.0 for value heads, zero biases.

use candle_core::{Device, IndexOp, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

/// Creates a linear layer with orthogonally initialized weights (scaled by
/// `gain`) and zero-initialized bias, registered in the given VarBuilder.
pub fn linear_ortho(in_dim: usize, out_dim: usize, gain: f64, vb: VarBuilder) -> Result<Linear> {
    let device = vb.device();
    let weight_init = orthogonal_init(&[out_dim, in_dim], gain, device)?;

    let weight = vb.get((out_dim, in_dim), "weight")?;
    weight.slice_set(&weight_init, 0, 0)?;

    let bias = vb.get_with_hints(out_dim, "bias", candle_nn::Init::Const(0.0))?;

    Ok(Linear::new(weight, Some(bias)))
}

/// Generates an orthogonal tensor of `shape` scaled by `gain`, matching
/// PyTorch's flattening rule for tensors with two or more dimensions. The first
/// dimension is kept as rows and all trailing dimensions are flattened into
/// columns before orthogonalization.
pub fn orthogonal_init(shape: &[usize], gain: f64, device: &Device) -> Result<Tensor> {
    assert!(
        shape.len() >= 2,
        "orthogonal_init requires at least 2 dimensions"
    );
    let rows = shape[0];
    let cols = shape[1..].iter().product::<usize>();

    let random_matrix = Tensor::randn(0f32, 1f32, &[rows, cols], device)?;

    let q = if rows < cols {
        let transposed = random_matrix.t()?;
        qr_decomposition(&transposed)?.t()?
    } else {
        qr_decomposition(&random_matrix)?
    };

    q.affine(gain, 0.0)?.reshape(shape)
}

/// Q factor via modified Gram-Schmidt.
///
/// Candle does not expose a QR decomposition, so this keeps the unavoidable
/// column loop but leaves the projection and normalization work as tensor ops.
fn qr_decomposition(a: &Tensor) -> Result<Tensor> {
    let shape = a.dims();
    let cols = shape[1];

    let mut q_cols = Vec::with_capacity(cols);

    for j in 0..cols {
        let mut v = a.i((.., j))?;
        for q_i in q_cols.iter().take(j) {
            let dot = (q_i * &v)?.sum_all()?;
            let scaled_q = q_i.broadcast_mul(&dot)?;
            v = (v - scaled_q)?;
        }
        let norm = v.sqr()?.sum_all()?.sqrt()?;
        let q_j = v.broadcast_div(&norm)?;
        q_cols.push(q_j);
    }

    Tensor::stack(&q_cols, 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn orthogonal_rows_scaled_by_gain() {
        let device = Device::Cpu;
        let gain = 2.0f64.sqrt();
        // wide matrix: rows orthonormal, W W^T = gain^2 I
        let w = orthogonal_init(&[16, 64], gain, &device).unwrap();
        let product = w.matmul(&w.t().unwrap()).unwrap();
        let expected = (gain * gain) as f32;
        for i in 0..16 {
            for j in 0..16 {
                let v = product.i((i, j)).unwrap().to_vec0::<f32>().unwrap();
                let target = if i == j { expected } else { 0.0 };
                assert!(
                    (v - target).abs() < 1e-4,
                    "W W^T [{},{}] = {}, expected {}",
                    i,
                    j,
                    v,
                    target
                );
            }
        }
    }

    #[test]
    fn orthogonal_supports_conv_shape_by_flattening_trailing_dims() {
        let device = Device::Cpu;
        let gain = 2.0f64.sqrt();
        let w = orthogonal_init(&[8, 4, 3, 3], gain, &device).unwrap();
        assert_eq!(w.dims(), &[8, 4, 3, 3]);

        let flattened = w.reshape(&[8, 4 * 3 * 3]).unwrap();
        let product = flattened.matmul(&flattened.t().unwrap()).unwrap();
        let expected = (gain * gain) as f32;
        for i in 0..8 {
            for j in 0..8 {
                let v = product.i((i, j)).unwrap().to_vec0::<f32>().unwrap();
                let target = if i == j { expected } else { 0.0 };
                assert!(
                    (v - target).abs() < 1e-4,
                    "flattened W W^T [{},{}] = {}, expected {}",
                    i,
                    j,
                    v,
                    target
                );
            }
        }
    }

    #[test]
    fn orthogonal_tall_flattened_shape_has_orthonormal_columns() {
        let device = Device::Cpu;
        let w = orthogonal_init(&[16, 2, 2], 1.0, &device).unwrap();
        assert_eq!(w.dims(), &[16, 2, 2]);

        let flattened = w.reshape(&[16, 4]).unwrap();
        let product = flattened.t().unwrap().matmul(&flattened).unwrap();
        for i in 0..4 {
            for j in 0..4 {
                let v = product.i((i, j)).unwrap().to_vec0::<f32>().unwrap();
                let target = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (v - target).abs() < 1e-4,
                    "flattened W^T W [{},{}] = {}, expected {}",
                    i,
                    j,
                    v,
                    target
                );
            }
        }
    }

    #[test]
    fn linear_ortho_registers_vars_and_zero_bias() {
        let device = Device::Cpu;
        let var_map = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &device);
        let layer = linear_ortho(8, 4, 0.01, vb.pp("head")).unwrap();

        assert_eq!(var_map.all_vars().len(), 2);
        let bias: Vec<f32> = layer.bias().unwrap().to_vec1().unwrap();
        assert!(bias.iter().all(|b| *b == 0.0));

        // gain 0.01: every weight tiny
        let w: Vec<f32> = layer.weight().flatten_all().unwrap().to_vec1().unwrap();
        assert!(w.iter().all(|x| x.abs() < 0.011));
    }

    #[test]
    fn linear_ortho_bias_is_zero_with_output_shape() {
        let device = Device::Cpu;
        let var_map = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&var_map, candle_core::DType::F32, &device);
        let layer = linear_ortho(3, 5, 1.0, vb.pp("linear")).unwrap();

        let bias = layer.bias().expect("linear_ortho should create a bias");
        assert_eq!(bias.dims(), &[5]);
        let bias_values: Vec<f32> = bias.to_vec1().unwrap();
        assert_eq!(bias_values, vec![0.0; 5]);
    }
}

use bon::bon;
use candle_core::Error;
use candle_nn::{self, VarBuilder, linear};
pub mod probabilistic_model;

pub struct MLP {
    hidden_layers: Vec<candle_nn::Linear>,
    activation: Box<dyn candle_nn::Module>,
    output_layer: candle_nn::Linear,
    input_layer: candle_nn::Linear,
    output_activation: Option<Box<dyn candle_nn::Module>>,
}

#[bon]
impl MLP {
    #[builder]
    pub fn new(
        input_size: usize,
        output_size: usize,
        vb: VarBuilder<'_>,
        #[builder(default = vec![32, 32, 32])] hidden_layer_sizes: Vec<usize>,
        #[builder(default = Box::new(candle_nn::Activation::Relu))] activation: Box<
            dyn candle_nn::Module,
        >,
        output_activation: Option<Box<dyn candle_nn::Module>>,
        #[builder(default = "mlp".to_string())] name: String,
    ) -> Result<Self, Error> {
        let mut hidden_layers = Vec::new();
        let input_layer = linear(
            input_size,
            hidden_layer_sizes[0],
            vb.pp(format!("{name}_input_layer")),
        )?;
        for i in 0..hidden_layer_sizes.len() - 1 {
            hidden_layers.push(linear(
                hidden_layer_sizes[i],
                hidden_layer_sizes[i + 1],
                vb.pp(format!("{name}_hidden_layer_{i}")),
            )?);
        }
        let output_layer = linear(
            hidden_layer_sizes[hidden_layer_sizes.len() - 1],
            output_size,
            vb.pp(format!("{name}_output_layer")),
        )?;
        Ok(Self {
            hidden_layers,
            activation,
            output_layer,
            input_layer,
            output_activation,
        })
    }
}

impl candle_nn::Module for MLP {
    fn forward(&self, xs: &candle_core::Tensor) -> Result<candle_core::Tensor, Error> {
        let mut x = self.input_layer.forward(xs)?;
        for layer in &self.hidden_layers {
            x = layer.forward(&x)?;
            x = self.activation.forward(&x)?;
        }
        x = self.output_layer.forward(&x)?;
        if let Some(output_activation) = &self.output_activation {
            x = output_activation.forward(&x)?;
        }
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::{Module, VarMap};

    //#[cfg(any(feature = "cuda", feature = "metal"))]
    #[test]
    fn test_mlp_determinism() {
        #[cfg(feature = "cuda")]
        let mut device = Device::new_cuda(0).unwrap();
        #[cfg(feature = "metal")]
        let mut device = Device::new_metal(0).unwrap();

        let input = candle_core::Tensor::rand(0.0f32, 1.0, &[1, 4], &device).unwrap();
        let mut last_output: Option<candle_core::Tensor> = None;
        for i in 0..10 {
            device.set_seed(42).unwrap();
            let vm = VarMap::new();
            let vb = VarBuilder::from_varmap(&vm, candle_core::DType::F32, &device);
            let mlp1 = MLP::builder()
                .input_size(4)
                .output_size(2)
                .vb(vb.clone())
                .hidden_layer_sizes(vec![8, 8])
                .build()
                .unwrap();

            let current_output = mlp1.forward(&input).unwrap();
            if let Some(last_output) = &last_output {
                let max_diff = last_output
                    .sub(&current_output)
                    .unwrap()
                    .abs()
                    .unwrap()
                    .max_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap();

                assert!(
                    max_diff < 1e-6,
                    "Outputs differ at iteration {i} by {max_diff}"
                );
            }
            last_output = Some(current_output);
        }
    }
}

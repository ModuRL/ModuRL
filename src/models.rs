use bon::bon;
use candle_core::Error;
use candle_nn::{self, VarBuilder, linear};
pub mod probabilistic_model;

pub struct MLPArchitecture {
    pub input_size: usize,
    pub output_size: usize,
    pub hidden_layer_sizes: Vec<usize>,
    pub name: String,
}

pub struct MLPInitializedLayers {
    pub input_layer: candle_nn::Linear,
    pub hidden_layers: Vec<candle_nn::Linear>,
    pub output_layer: candle_nn::Linear,
}

pub trait MLPInitializer {
    fn initialize(
        &self,
        arch: &MLPArchitecture,
        vb: VarBuilder<'_>,
    ) -> Result<MLPInitializedLayers, Error>;
}

pub struct DefaultMLPInitializer;

impl MLPInitializer for DefaultMLPInitializer {
    fn initialize(
        &self,
        arch: &MLPArchitecture,
        vb: VarBuilder<'_>,
    ) -> Result<MLPInitializedLayers, Error> {
        let mut hidden_layers = Vec::new();
        let input_layer = linear(
            arch.input_size,
            arch.hidden_layer_sizes[0],
            vb.pp(format!("{}_input_layer", arch.name)),
        )?;
        for i in 0..arch.hidden_layer_sizes.len() - 1 {
            hidden_layers.push(linear(
                arch.hidden_layer_sizes[i],
                arch.hidden_layer_sizes[i + 1],
                vb.pp(format!("{}_hidden_layer_{i}", arch.name)),
            )?);
        }
        let output_layer = linear(
            arch.hidden_layer_sizes[arch.hidden_layer_sizes.len() - 1],
            arch.output_size,
            vb.pp(format!("{}_output_layer", arch.name)),
        )?;

        Ok(MLPInitializedLayers {
            input_layer,
            hidden_layers,
            output_layer,
        })
    }
}

pub struct OrthogonalMLPInitializer {
    pub hidden_gain: f64,
    pub output_gain: f64,
}

impl MLPInitializer for OrthogonalMLPInitializer {
    fn initialize(
        &self,
        arch: &MLPArchitecture,
        vb: VarBuilder<'_>,
    ) -> Result<MLPInitializedLayers, Error> {
        let mut hidden_layers = Vec::new();
        let input_layer = crate::init::linear_ortho(
            arch.input_size,
            arch.hidden_layer_sizes[0],
            self.hidden_gain,
            vb.pp(format!("{}_input_layer", arch.name)),
        )?;
        for i in 0..arch.hidden_layer_sizes.len() - 1 {
            hidden_layers.push(crate::init::linear_ortho(
                arch.hidden_layer_sizes[i],
                arch.hidden_layer_sizes[i + 1],
                self.hidden_gain,
                vb.pp(format!("{}_hidden_layer_{i}", arch.name)),
            )?);
        }
        let output_layer = crate::init::linear_ortho(
            arch.hidden_layer_sizes[arch.hidden_layer_sizes.len() - 1],
            arch.output_size,
            self.output_gain,
            vb.pp(format!("{}_output_layer", arch.name)),
        )?;

        Ok(MLPInitializedLayers {
            input_layer,
            hidden_layers,
            output_layer,
        })
    }
}

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
        #[builder(default = Box::new(DefaultMLPInitializer))] initializer: Box<dyn MLPInitializer>,
    ) -> Result<Self, Error> {
        let arch = MLPArchitecture {
            input_size,
            output_size,
            hidden_layer_sizes,
            name,
        };
        let layers = initializer.initialize(&arch, vb)?;
        Ok(Self {
            hidden_layers: layers.hidden_layers,
            activation,
            output_layer: layers.output_layer,
            input_layer: layers.input_layer,
            output_activation,
        })
    }
}

impl candle_nn::Module for MLP {
    fn forward(&self, xs: &candle_core::Tensor) -> Result<candle_core::Tensor, Error> {
        let mut x = self.input_layer.forward(xs)?;
        x = self.activation.forward(&x)?;
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

#[cfg(any(feature = "cuda", feature = "metal"))]
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::{Module, VarMap};

    #[test]
    fn test_mlp_determinism() {
        #[cfg(feature = "cuda")]
        let device = Device::new_cuda(0).unwrap();
        #[cfg(feature = "metal")]
        let device = Device::new_metal(0).unwrap();

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

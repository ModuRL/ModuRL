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

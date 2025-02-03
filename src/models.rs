use candle_core::Error;
use candle_nn::{self, linear, VarBuilder};

pub struct MLP {
    hidden_layers: Vec<candle_nn::Linear>,
    activation: candle_nn::Activation,
    output_layer: candle_nn::Linear,
    input_layer: candle_nn::Linear,
    output_activation: Option<candle_nn::Activation>,
}

impl MLP {
    pub fn new(
        input_size: usize,
        hidden_layer_sizes: Vec<usize>,
        output_size: usize,
        activation: candle_nn::Activation,
        output_activation: Option<candle_nn::Activation>,
        vb: &mut VarBuilder,
    ) -> Result<Self, Error> {
        let mut hidden_layers = Vec::new();
        let input_layer = linear(input_size, hidden_layer_sizes[0], vb.pp("input layer"))?;
        for i in 0..hidden_layer_sizes.len() - 1 {
            hidden_layers.push(linear(
                hidden_layer_sizes[i],
                hidden_layer_sizes[i + 1],
                vb.pp(format!("hidden layer {}", i)),
            )?);
        }
        let output_layer = linear(
            hidden_layer_sizes[hidden_layer_sizes.len() - 1],
            output_size,
            vb.pp("output layer"),
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

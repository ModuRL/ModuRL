use candle_core::Error;
use candle_nn::{self, linear, VarBuilder};
pub mod probabilistic_model;

pub struct MLP {
    hidden_layers: Vec<candle_nn::Linear>,
    activation: Box<dyn candle_nn::Module>,
    output_layer: candle_nn::Linear,
    input_layer: candle_nn::Linear,
    output_activation: Option<Box<dyn candle_nn::Module>>,
}

pub struct MLPBuilder<'a> {
    pub input_size: usize,
    pub output_size: usize,
    pub hidden_layer_sizes: Option<Vec<usize>>,
    pub activation: Option<Box<dyn candle_nn::Module>>,
    pub output_activation: Option<Box<dyn candle_nn::Module>>,
    pub vb: VarBuilder<'a>,
}

impl<'a> MLPBuilder<'a> {
    pub fn new(input_size: usize, output_size: usize, vb: VarBuilder<'a>) -> Self {
        Self {
            input_size,
            output_size,
            hidden_layer_sizes: None,
            activation: None,
            output_activation: None,
            vb: vb,
        }
    }

    pub fn hidden_layer_sizes(mut self, hidden_layer_sizes: Vec<usize>) -> Self {
        self.hidden_layer_sizes = Some(hidden_layer_sizes);
        self
    }

    pub fn activation(mut self, activation: Box<dyn candle_nn::Module>) -> Self {
        self.activation = Some(activation);
        self
    }

    pub fn output_activation(mut self, output_activation: Box<dyn candle_nn::Module>) -> Self {
        self.output_activation = Some(output_activation);
        self
    }

    pub fn build(mut self) -> Result<MLP, Error> {
        // TODO: make the default hidden layer size change based on input and output size
        let hidden_layer_sizes = self.hidden_layer_sizes.unwrap_or(vec![32, 32, 32]);
        let activation = self
            .activation
            .unwrap_or(Box::new(candle_nn::Activation::Relu));
        // output layer activation is optional and defaults to None

        MLP::new(
            self.input_size,
            hidden_layer_sizes,
            self.output_size,
            activation,
            self.output_activation,
            &mut self.vb,
        )
    }
}

impl MLP {
    fn new(
        input_size: usize,
        hidden_layer_sizes: Vec<usize>,
        output_size: usize,
        activation: Box<dyn candle_nn::Module>,
        output_activation: Option<Box<dyn candle_nn::Module>>,
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

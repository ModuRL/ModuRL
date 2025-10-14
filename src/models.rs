use bon::bon;
use burn::{
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
    tensor::Device,
};
//use candle_nn::{self, VarBuilder, linear};
//pub mod probabilistic_model;

#[derive(Module, Debug)]
pub struct MLP<B: Backend, A: Module<B> = Relu, OA: Module<B> = ()> {
    hidden_layers: Vec<Linear<B>>,
    activation: A,
    output_layer: Linear<B>,
    input_layer: Linear<B>,
    output_activation: Option<OA>,
}

#[bon]
impl<B: Backend, A: Module<B>, OA: Module<B>> MLP<B, A, OA> {
    #[builder]
    pub fn new(
        input_size: usize,
        output_size: usize,
        #[builder(default = vec![32, 32, 32])] hidden_layer_sizes: Vec<usize>,
        activation: A,
        output_activation: Option<OA>,
        device: Device<B>,
    ) -> Self {
        let mut hidden_layers = Vec::new();
        let input_config = LinearConfig::new(input_size, hidden_layer_sizes[0]);
        let input_layer = input_config.init(&device);
        for i in 0..hidden_layer_sizes.len() - 1 {
            let config = LinearConfig::new(hidden_layer_sizes[i], hidden_layer_sizes[i + 1]);
            hidden_layers.push(config.init(&device));
        }
        let output_config = if hidden_layer_sizes.is_empty() {
            LinearConfig::new(input_size, output_size)
        } else {
            LinearConfig::new(*hidden_layer_sizes.last().unwrap(), output_size)
        };
        let output_layer = output_config.init(&device);

        Self {
            hidden_layers,
            activation,
            output_layer,
            input_layer,
            output_activation,
        }
    }
}

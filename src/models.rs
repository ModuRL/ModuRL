use bon::bon;
use candle_core::{Error, Tensor};
use candle_nn::{self, VarBuilder, linear};
pub mod probabilistic_model;

pub trait MLPInitializer {
    fn initialize_hidden_layer(
        &self,
        input_size: usize,
        output_size: usize,
        vb: VarBuilder<'_>,
    ) -> Result<candle_nn::Linear, Error>;

    fn initialize_output_layer(
        &self,
        input_size: usize,
        output_size: usize,
        vb: VarBuilder<'_>,
    ) -> Result<candle_nn::Linear, Error>;
}

pub struct DefaultMLPInitializer;

impl MLPInitializer for DefaultMLPInitializer {
    fn initialize_hidden_layer(
        &self,
        input_size: usize,
        output_size: usize,
        vb: VarBuilder<'_>,
    ) -> Result<candle_nn::Linear, Error> {
        linear(input_size, output_size, vb)
    }

    fn initialize_output_layer(
        &self,
        input_size: usize,
        output_size: usize,
        vb: VarBuilder<'_>,
    ) -> Result<candle_nn::Linear, Error> {
        linear(input_size, output_size, vb)
    }
}

pub struct OrthogonalMLPInitializer {
    pub hidden_gain: f64,
    pub output_gain: f64,
}

impl MLPInitializer for OrthogonalMLPInitializer {
    fn initialize_hidden_layer(
        &self,
        input_size: usize,
        output_size: usize,
        vb: VarBuilder<'_>,
    ) -> Result<candle_nn::Linear, Error> {
        crate::init::linear_ortho(input_size, output_size, self.hidden_gain, vb)
    }

    fn initialize_output_layer(
        &self,
        input_size: usize,
        output_size: usize,
        vb: VarBuilder<'_>,
    ) -> Result<candle_nn::Linear, Error> {
        crate::init::linear_ortho(input_size, output_size, self.output_gain, vb)
    }
}

fn initialize_hidden_layers(
    initializer: &dyn MLPInitializer,
    input_size: usize,
    output_sizes: &[usize],
    vb: VarBuilder<'_>,
    layer_name: impl Fn(usize) -> String,
) -> Result<(Vec<candle_nn::Linear>, usize), Error> {
    let mut layers = Vec::with_capacity(output_sizes.len());
    let mut layer_input_size = input_size;
    for (index, &layer_output_size) in output_sizes.iter().enumerate() {
        layers.push(initializer.initialize_hidden_layer(
            layer_input_size,
            layer_output_size,
            vb.pp(layer_name(index)),
        )?);
        layer_input_size = layer_output_size;
    }
    Ok((layers, layer_input_size))
}

pub struct MLP {
    hidden_layers: Vec<candle_nn::Linear>,
    activation: Box<dyn candle_nn::Module>,
    output_layer: candle_nn::Linear,
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
        let (hidden_layers, hidden_output_size) = initialize_hidden_layers(
            initializer.as_ref(),
            input_size,
            &hidden_layer_sizes,
            vb.clone(),
            |index| {
                if index == 0 {
                    format!("{name}_input_layer")
                } else {
                    format!("{}_hidden_layer_{}", name, index - 1)
                }
            },
        )?;
        let output_layer = initializer.initialize_output_layer(
            hidden_output_size,
            output_size,
            vb.pp(format!("{name}_output_layer")),
        )?;
        Ok(Self {
            hidden_layers,
            activation,
            output_layer,
            output_activation,
        })
    }
}

impl candle_nn::Module for MLP {
    /// Maps `xs` shaped `[batch, input_size]` to `[batch, output_size]`.
    fn forward(&self, xs: &candle_core::Tensor) -> Result<candle_core::Tensor, Error> {
        let mut x = xs.clone();
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

/// A dueling Q-network with a shared MLP trunk and value/advantage heads.
///
/// The network produces one scalar state value and one advantage per action,
/// then combines them as `Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)`.
/// Its output is shaped `[batch, output_size]`, so it can be passed directly
/// to the DQN and DDQN agents.
pub struct DuelingMLP {
    shared_layers: Vec<candle_nn::Linear>,
    value_hidden_layers: Vec<candle_nn::Linear>,
    value_output_layer: candle_nn::Linear,
    advantage_hidden_layers: Vec<candle_nn::Linear>,
    advantage_output_layer: candle_nn::Linear,
    activation: Box<dyn candle_nn::Module>,
}

#[bon]
impl DuelingMLP {
    #[builder]
    pub fn new(
        input_size: usize,
        output_size: usize,
        vb: VarBuilder<'_>,
        #[builder(default = vec![32, 32, 32])] hidden_layer_sizes: Vec<usize>,
        #[builder(default = Vec::new())] value_hidden_layer_sizes: Vec<usize>,
        #[builder(default = Vec::new())] advantage_hidden_layer_sizes: Vec<usize>,
        #[builder(default = Box::new(candle_nn::Activation::Relu))] activation: Box<
            dyn candle_nn::Module,
        >,
        #[builder(default = "dueling_mlp".to_string())] name: String,
        #[builder(default = Box::new(DefaultMLPInitializer))] initializer: Box<dyn MLPInitializer>,
    ) -> Result<Self, Error> {
        if output_size == 0 {
            candle_core::bail!("a dueling MLP requires at least one output action");
        }

        let (shared_layers, shared_output_size) = initialize_hidden_layers(
            initializer.as_ref(),
            input_size,
            &hidden_layer_sizes,
            vb.clone(),
            |index| format!("{name}_shared_layer_{index}"),
        )?;
        let (value_hidden_layers, value_output_size) = initialize_hidden_layers(
            initializer.as_ref(),
            shared_output_size,
            &value_hidden_layer_sizes,
            vb.clone(),
            |index| format!("{name}_value_hidden_layer_{index}"),
        )?;
        let (advantage_hidden_layers, advantage_output_size) = initialize_hidden_layers(
            initializer.as_ref(),
            shared_output_size,
            &advantage_hidden_layer_sizes,
            vb.clone(),
            |index| format!("{name}_advantage_hidden_layer_{index}"),
        )?;
        let value_output_layer = initializer.initialize_output_layer(
            value_output_size,
            1,
            vb.pp(format!("{name}_value_output_layer")),
        )?;
        let advantage_output_layer = initializer.initialize_output_layer(
            advantage_output_size,
            output_size,
            vb.pp(format!("{name}_advantage_output_layer")),
        )?;

        Ok(Self {
            shared_layers,
            value_hidden_layers,
            value_output_layer,
            advantage_hidden_layers,
            advantage_output_layer,
            activation,
        })
    }

    /// Applies hidden `layers` to `input` shaped `[batch, input_size]`.
    fn forward_hidden_layers(
        &self,
        layers: &[candle_nn::Linear],
        input: &Tensor,
    ) -> Result<Tensor, Error> {
        let mut output = input.clone();
        for layer in layers {
            output = candle_core::Module::forward(layer, &output)?;
            output = self.activation.forward(&output)?;
        }
        Ok(output)
    }

    /// Combines values shaped `[batch, 1]` and advantages shaped
    /// `[batch, output_size]` into Q-values shaped `[batch, output_size]`.
    fn combine_streams(&self, value: &Tensor, advantages: &Tensor) -> Result<Tensor, Error> {
        let centered_advantages = advantages.broadcast_sub(&advantages.mean_keepdim(1)?)?;
        value.broadcast_add(&centered_advantages)
    }
}

impl candle_nn::Module for DuelingMLP {
    /// Maps `xs` shaped `[batch, input_size]` to one Q-value per action,
    /// shaped `[batch, output_size]`.
    fn forward(&self, xs: &Tensor) -> Result<Tensor, Error> {
        let features = self.forward_hidden_layers(&self.shared_layers, xs)?;
        let value_features = self.forward_hidden_layers(&self.value_hidden_layers, &features)?;
        let advantage_features =
            self.forward_hidden_layers(&self.advantage_hidden_layers, &features)?;
        let value = self.value_output_layer.forward(&value_features)?;
        let advantages = self.advantage_output_layer.forward(&advantage_features)?;
        self.combine_streams(&value, &advantages)
    }
}

#[cfg(test)]
mod dueling_tests {
    use super::DuelingMLP;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{Module, VarBuilder, VarMap};

    fn dueling_mlp(output_size: usize) -> Result<(DuelingMLP, VarMap), candle_core::Error> {
        let vars = VarMap::new();
        let network = DuelingMLP::builder()
            .input_size(4)
            .output_size(output_size)
            .vb(VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu))
            .hidden_layer_sizes(vec![8])
            .value_hidden_layer_sizes(vec![6])
            .advantage_hidden_layer_sizes(vec![7])
            .activation(Box::new(Tensor::tanh))
            .build()?;
        Ok((network, vars))
    }

    #[test]
    fn outputs_one_q_value_per_action() {
        let (network, _) = dueling_mlp(3).unwrap();
        let input = Tensor::zeros(&[5, 4], DType::F32, &Device::Cpu).unwrap();

        assert_eq!(network.forward(&input).unwrap().dims(), &[5, 3]);
    }

    #[test]
    fn combines_value_and_mean_centered_advantages() {
        let (network, _) = dueling_mlp(3).unwrap();
        let values = Tensor::new(&[[10.0f32], [-2.0]], &Device::Cpu).unwrap();
        let advantages =
            Tensor::new(&[[1.0f32, 2.0, 3.0], [-3.0, 0.0, 3.0]], &Device::Cpu).unwrap();

        let q_values = network
            .combine_streams(&values, &advantages)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();

        assert_eq!(q_values, vec![vec![9.0, 10.0, 11.0], vec![-5.0, -2.0, 1.0]]);
    }

    #[test]
    fn explicit_stream_parameters_receive_q_loss_gradients() {
        let (network, vars) = dueling_mlp(3).unwrap();
        let input = Tensor::new(&[[0.5f32, -0.25, 1.0, 0.75]], &Device::Cpu).unwrap();
        let selected_q_value = network.forward(&input).unwrap().narrow(1, 0, 1).unwrap();
        let gradients = selected_q_value.sum_all().unwrap().backward().unwrap();
        let variables = vars.data().lock().unwrap();

        for name in [
            "dueling_mlp_shared_layer_0.weight",
            "dueling_mlp_value_hidden_layer_0.weight",
            "dueling_mlp_value_output_layer.weight",
            "dueling_mlp_advantage_hidden_layer_0.weight",
            "dueling_mlp_advantage_output_layer.weight",
        ] {
            assert!(
                gradients.get(variables[name].as_tensor()).is_some(),
                "{name} did not receive a Q-loss gradient"
            );
        }
    }

    #[test]
    fn supports_one_empty_and_one_nonempty_stream() {
        let vars = VarMap::new();
        let network = DuelingMLP::builder()
            .input_size(4)
            .output_size(3)
            .vb(VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu))
            .hidden_layer_sizes(vec![8])
            .advantage_hidden_layer_sizes(vec![7])
            .build()
            .unwrap();
        let variables = vars.data().lock().unwrap();

        assert!(!variables.contains_key("dueling_mlp_value_hidden_layer_0.weight"));
        assert!(variables.contains_key("dueling_mlp_value_output_layer.weight"));
        assert!(variables.contains_key("dueling_mlp_advantage_hidden_layer_0.weight"));
        assert_eq!(
            network
                .forward(&Tensor::zeros(&[2, 4], DType::F32, &Device::Cpu).unwrap())
                .unwrap()
                .dims(),
            &[2, 3]
        );
    }

    #[test]
    fn rejects_zero_actions() {
        assert!(dueling_mlp(0).is_err());
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

use bon::bon;
use burn::{
    module::{Module, ModuleMapper, Param},
    nn::{Initializer, Linear, LinearConfig, Relu, Tanh, activation::Activation},
    tensor::{DType, Float, Tensor, TensorCreationOptions, TensorData, kind::Basic},
};

pub mod probabilistic_model;

/// Executes a tensor-to-tensor computation with explicit ranks and observation kind.
/// Parameter-owning models also implement Burn's `Module` so optimizers can visit their parameters.
pub trait Forward<const I: usize, const O: usize, K: Basic = Float> {
    type Error;

    /// Maps a rank-`I` tensor of kind `K` to a rank-`O` float tensor.
    /// Implementations define layouts, dtype, device, gradient behavior, and any explicit observation conversions.
    fn forward(&self, input: Tensor<I, K>) -> Result<Tensor<O>, Self::Error>;
}

impl<const D: usize> Forward<D, D> for Relu {
    type Error = std::convert::Infallible;

    /// Applies ReLU to rank-`D` input, preserving its shape, floating-point dtype, device, and gradient paths.
    fn forward(&self, input: Tensor<D>) -> Result<Tensor<D>, Self::Error> {
        Ok(Relu::forward(self, input))
    }
}

impl<const D: usize> Forward<D, D> for Tanh {
    type Error = std::convert::Infallible;

    /// Applies tanh to rank-`D` input, preserving its shape, floating-point dtype, device, and gradient paths.
    fn forward(&self, input: Tensor<D>) -> Result<Tensor<D>, Self::Error> {
        Ok(Tanh::forward(self, input))
    }
}

/// Errors in model configuration or the input tensor contract.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("model feature counts must be positive, got {0}")]
    ZeroFeatures(usize),
    #[error("model weight shape [{input}, {output}] is too large")]
    ShapeTooLarge { input: usize, output: usize },
    #[error(
        "model initializer requires finite values, ordered uniform bounds, positive uniform gains, and nonnegative standard deviations"
    )]
    InvalidInitializer,
    #[error("model parameters require a supported floating-point dtype, got {0:?}")]
    UnsupportedDType(DType),
    #[error("model initializer dtype conversion failed: {0}")]
    InitializerData(#[from] burn::tensor::DataError),
    #[error("model input has {actual} features, expected {expected}")]
    InputFeatures { expected: usize, actual: usize },
    #[error("model input dtype is {actual:?}, expected {expected:?}")]
    InputDType { expected: DType, actual: DType },
    #[error("model input must use the parameter device")]
    InputDevice,
    #[error("SwiGlu is a projected layer and cannot serve as a shape-preserving MLP activation")]
    ProjectedActivation,
    #[error("PRelu has {actual} parameters, expected 1 or {expected}")]
    ActivationFeatures { expected: usize, actual: usize },
}

/// Checks distribution arguments before Burn initializes model weights.
fn validate_initializer(initializer: &Initializer) -> Result<(), ModelError> {
    let valid = match initializer {
        Initializer::Constant { value } => value.is_finite(),
        Initializer::Uniform { min, max } => min.is_finite() && max.is_finite() && min < max,
        Initializer::Normal { mean, std } => mean.is_finite() && std.is_finite() && *std >= 0.0,
        Initializer::KaimingUniform { gain, .. } | Initializer::XavierUniform { gain } => {
            gain.is_finite() && *gain > 0.0
        }
        Initializer::KaimingNormal { gain, .. } | Initializer::XavierNormal { gain } => {
            gain.is_finite() && *gain >= 0.0
        }
        Initializer::Orthogonal { gain } => gain.is_finite(),
        Initializer::Ones | Initializer::Zeros => true,
    };
    if !valid {
        return Err(ModelError::InvalidInitializer);
    }
    Ok(())
}

/// Checks uniform bounds after conversion to the dtype used for weight initialization.
/// Rejects bounds that become equal, infinite, or too far apart for the sampler's arithmetic.
fn validate_uniform_bounds(min: f64, max: f64, dtype: DType) -> Result<(), ModelError> {
    let data = TensorData::from([min, max, max - min]).try_cast(dtype)?;
    let mut bounds = data.iter::<f64>();
    if !matches!((bounds.next(), bounds.next(), bounds.next()),
        (Some(min), Some(max), Some(width))
            if min.is_finite() && max.is_finite() && width.is_finite() && min < max)
    {
        return Err(ModelError::InvalidInitializer);
    }
    Ok(())
}

/// Checks that an activation preserves each configured layer's feature count.
fn validate_activation(activation: &Activation, widths: &[usize]) -> Result<(), ModelError> {
    match activation {
        Activation::SwiGlu(_) => return Err(ModelError::ProjectedActivation),
        Activation::PRelu(layer) => {
            let actual = layer.alpha.shape().dims::<1>()[0];
            for &expected in widths {
                if actual != 1 && actual != expected {
                    return Err(ModelError::ActivationFeatures { expected, actual });
                }
            }
        }
        _ => {}
    }
    Ok(())
}

impl Forward<2, 2> for Activation {
    type Error = ModelError;

    /// Applies an activation to `[batch_size, features]` and preserves both axes, dtype, device, and gradient paths.
    /// Learned activation parameters must match the input dtype and device. PRelu accepts 1 or `features` parameters.
    /// SwiGlu is not accepted because its projection can change the feature count.
    fn forward(&self, input: Tensor<2>) -> Result<Tensor<2>, ModelError> {
        if matches!(self, Activation::SwiGlu(_)) {
            return Err(ModelError::ProjectedActivation);
        }
        if let Activation::PRelu(layer) = self {
            validate_activation(self, &[input.dims()[1]])?;
            let alpha = layer.alpha.val();
            if alpha.dtype() != input.dtype() {
                return Err(ModelError::InputDType {
                    expected: alpha.dtype(),
                    actual: input.dtype(),
                });
            }
            if alpha.device() != input.device() {
                return Err(ModelError::InputDevice);
            }
        }
        Ok(Activation::forward(self, input))
    }
}

impl Forward<2, 2> for Linear {
    type Error = ModelError;

    /// Maps `[batch_size, input_features]` to `[batch_size, output_features]`, preserving dtype, device, and gradient paths.
    /// Input features must match the weight's first axis. Input dtype and device must match the weights.
    fn forward(&self, input: Tensor<2>) -> Result<Tensor<2>, ModelError> {
        validate_input(&input, self)?;
        Ok(Linear::forward(self, input))
    }
}

/// Builds a sequence of native linear layers with the selected weight and bias initializers.
/// Validates feature counts and weight sizes, and supplies fan counts for Kaiming and Xavier initialization.
/// Each weight has shape `[input_features, output_features]`; each bias has shape `[output_features]`.
fn initialize_hidden_layers(
    input_size: usize,
    output_sizes: &[usize],
    initializer: &Initializer,
    bias_initializer: Option<&Initializer>,
    options: &TensorCreationOptions,
) -> Result<(Vec<Linear>, usize), ModelError> {
    validate_initializer(initializer)?;
    if input_size == 0 {
        return Err(ModelError::ZeroFeatures(input_size));
    }
    let mut layers = Vec::with_capacity(output_sizes.len());
    let mut input = input_size;
    for &output in output_sizes {
        layers.push(initialize_linear(
            input,
            output,
            initializer,
            bias_initializer,
            options,
        )?);
        input = output;
    }
    Ok((layers, input))
}

/// Builds a native linear layer after checking feature counts, weight size, and initializer arguments.
/// Supplies fan counts for weight initialization. Bias uses its own initializer, uniform bounds by default, or zeros for orthogonal weights.
/// Weights have shape `[input, output]`. Bias has shape `[output]`.
fn initialize_linear(
    input: usize,
    output: usize,
    initializer: &Initializer,
    bias_initializer: Option<&Initializer>,
    options: &TensorCreationOptions,
) -> Result<Linear, ModelError> {
    validate_initializer(initializer)?;
    if input == 0 || output == 0 {
        return Err(ModelError::ZeroFeatures(0));
    }
    input
        .checked_mul(output)
        .ok_or(ModelError::ShapeTooLarge { input, output })?;
    let default_bias = if matches!(initializer, Initializer::Orthogonal { .. }) {
        Initializer::Zeros
    } else {
        let bound = 1.0 / (input as f64).sqrt();
        Initializer::Uniform {
            min: -bound,
            max: bound,
        }
    };
    let bias_initializer = bias_initializer.unwrap_or(&default_bias);
    validate_initializer(bias_initializer)?;
    if matches!(bias_initializer, Initializer::Orthogonal { .. }) {
        return Err(ModelError::InvalidInitializer);
    }
    let dtype = options.device.settings().float_dtype.into();
    for initializer in [initializer, bias_initializer] {
        let bounds = match initializer {
            Initializer::Uniform { min, max } => Some((*min, *max)),
            Initializer::KaimingUniform { gain, fan_out_only } => {
                let fan = if *fan_out_only { output } else { input };
                let bound = 3.0f64.sqrt() * gain / (fan as f64).sqrt();
                Some((-bound, bound))
            }
            Initializer::XavierUniform { gain } => {
                let fan = input
                    .checked_add(output)
                    .ok_or(ModelError::ShapeTooLarge { input, output })?;
                let bound = 3.0f64.sqrt() * gain * (2.0 / fan as f64).sqrt();
                Some((-bound, bound))
            }
            _ => None,
        };
        if let Some((min, max)) = bounds {
            validate_uniform_bounds(min, max, dtype)?;
        }
    }
    let mut layer = LinearConfig::new(input, output)
        .with_initializer(initializer.clone())
        .with_bias(false)
        .init(&options.device);
    layer.bias =
        Some(bias_initializer.init_with([output], Some(input), Some(output), &options.device));
    Ok(layer)
}

/// Casts every floating-point model parameter, including learned activation parameters, to the configured dtype.
struct ModelDType {
    dtype: DType,
}

impl ModuleMapper for ModelDType {
    /// Keeps each rank-`D` parameter's shape and device, and casts its values while retaining its gradient requirement.
    /// Disables gradient tracking during the cast so initialized parameters remain leaves that Burn can optimize.
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let dtype = self.dtype;
        param.init_mapper(move |tensor| {
            let require_grad = tensor.is_require_grad();
            tensor
                .set_require_grad(false)
                .cast(dtype)
                .set_require_grad(require_grad)
        })
    }
}

/// Selects a supported parameter dtype without changing shared device settings.
/// The dtype applies to model weights `[input_features, output_features]`, biases `[output_features]`, and learned activation parameters.
fn model_dtype(options: &TensorCreationOptions) -> Result<DType, ModelError> {
    let dtype = options.dtype_or(options.device.settings().float_dtype.into());
    if !dtype.is_float() || !options.device.supports_dtype(dtype) {
        return Err(ModelError::UnsupportedDType(dtype));
    }
    Ok(dtype)
}

/// Checks rank-2 input `[batch_size, input_features]` against weight shape `[input_features, output_features]`.
/// Batch size can vary. Input must share the first layer's floating-point dtype and device.
fn validate_input(input: &Tensor<2>, layer: &Linear) -> Result<(), ModelError> {
    let expected = layer.weight.shape().dims::<2>()[0];
    let actual = input.dims()[1];
    if actual != expected {
        return Err(ModelError::InputFeatures { expected, actual });
    }
    let weight = layer.weight.val();
    if input.dtype() != weight.dtype() {
        return Err(ModelError::InputDType {
            expected: weight.dtype(),
            actual: input.dtype(),
        });
    }
    if input.device() != weight.device() {
        return Err(ModelError::InputDevice);
    }
    Ok(())
}

/// Applies linear layers and activation to `[batch_size, input_features]`, returning `[batch_size, last_layer_features]`.
/// An empty layer sequence returns the input unchanged. Operations preserve dtype, device, batch size, and gradient paths.
fn forward_hidden_layers(
    layers: &[Linear],
    activation: &Activation,
    mut input: Tensor<2>,
) -> Tensor<2> {
    for layer in layers {
        input = activation.forward(layer.forward(input));
    }
    input
}

/// A multilayer perceptron with native Burn parameters and configurable activations.
/// Inputs and outputs use the `Float` tensor kind with one feature vector per batch item.
#[derive(Module, Debug)]
pub struct MLP {
    hidden_layers: Vec<Linear>,
    activation: Activation,
    output_layer: Linear,
    output_activation: Option<Activation>,
}

#[bon]
impl MLP {
    /// Builds a network with positive feature counts, native weight and bias initializers.
    /// The network maps `[batch_size, input_size]` to `[batch_size, output_size]` without changing the batch axis.
    /// Creation options accept a device or `(device, dtype)`. Dtype defaults to the device's floating-point dtype.
    /// Initialization uses the device's default dtype, then casts all parameters to the requested dtype.
    /// Activations must preserve layer widths. PRelu parameter counts must be 1 or match every affected width.
    /// Supplied learned activations retain their training state. Burn's `Module::train` enables their configured gradients when needed.
    /// Weights default to Kaiming normal with gain sqrt(2). Biases default to uniform bounds based on input width.
    /// Orthogonal weights use zero biases by default. An explicit bias initializer applies to every linear layer.
    #[builder]
    pub fn builder(
        input_size: usize,
        output_size: usize,
        #[builder(into)] options: TensorCreationOptions,
        #[builder(default = vec![32, 32, 32])] hidden_layer_sizes: Vec<usize>,
        #[builder(default = Activation::Relu(Relu), into)] activation: Activation,
        #[builder(into)] output_activation: Option<Activation>,
        #[builder(default = Initializer::KaimingNormal { gain: 2.0f64.sqrt(), fan_out_only: false })]
        hidden_initializer: Initializer,
        #[builder(default = Initializer::KaimingNormal { gain: 2.0f64.sqrt(), fan_out_only: false })]
        output_initializer: Initializer,
        bias_initializer: Option<Initializer>,
    ) -> Result<Self, ModelError> {
        let dtype = model_dtype(&options)?;
        validate_activation(&activation, &hidden_layer_sizes)?;
        if let Some(activation) = &output_activation {
            validate_activation(activation, &[output_size])?;
        }
        let (hidden_layers, hidden_output_size) = initialize_hidden_layers(
            input_size,
            &hidden_layer_sizes,
            &hidden_initializer,
            bias_initializer.as_ref(),
            &options,
        )?;
        let output_layer = initialize_linear(
            hidden_output_size,
            output_size,
            &output_initializer,
            bias_initializer.as_ref(),
            &options,
        )?;
        Ok(Self {
            hidden_layers,
            activation,
            output_layer,
            output_activation,
        }
        .fork(&options.device)
        .map(&mut ModelDType { dtype }))
    }
}

impl Forward<2, 2> for MLP {
    type Error = ModelError;

    /// Maps `[batch_size, input_size]` to `[batch_size, output_size]`, preserving the batch axis and gradient paths.
    /// Input must share the model parameters' floating-point dtype and device. Incorrect feature counts return an error.
    fn forward(&self, input: Tensor<2>) -> Result<Tensor<2>, ModelError> {
        validate_input(
            &input,
            self.hidden_layers.first().unwrap_or(&self.output_layer),
        )?;
        let features = forward_hidden_layers(&self.hidden_layers, &self.activation, input);
        let output = self.output_layer.forward(features);
        Ok(match &self.output_activation {
            Some(activation) => activation.forward(output),
            None => output,
        })
    }
}

/// Predicts one Q-value per discrete action with a shared trunk and separate value and advantage branches.
/// The value branch predicts one observation value. The advantage branch predicts each action's value relative to a baseline.
/// Subtracts the mean advantage across actions, then adds the observation value to each action.
#[derive(Module, Debug)]
pub struct DuelingMLP {
    shared_layers: Vec<Linear>,
    value_hidden_layers: Vec<Linear>,
    value_output_layer: Linear,
    advantage_hidden_layers: Vec<Linear>,
    advantage_output_layer: Linear,
    activation: Activation,
}

#[bon]
impl DuelingMLP {
    /// Builds a Q-network with positive feature and action counts, native weight and bias initializers.
    /// The network maps `[batch_size, input_size]` to `[batch_size, output_size]`, with one Q-value per discrete action.
    /// Creation options set parameter device and dtype. Initialization uses the device's default dtype before casting.
    /// Activations must preserve layer widths. PRelu parameter counts must be 1 or match every hidden width.
    /// Supplied learned activations retain their training state. Burn's `Module::train` enables their configured gradients when needed.
    /// Weights default to Kaiming normal with gain sqrt(2). Biases default to uniform bounds based on input width.
    /// Orthogonal weights use zero biases by default. An explicit bias initializer applies to every linear layer.
    #[builder]
    pub fn builder(
        input_size: usize,
        output_size: usize,
        #[builder(into)] options: TensorCreationOptions,
        #[builder(default = vec![32, 32, 32])] hidden_layer_sizes: Vec<usize>,
        #[builder(default = Vec::new())] value_hidden_layer_sizes: Vec<usize>,
        #[builder(default = Vec::new())] advantage_hidden_layer_sizes: Vec<usize>,
        #[builder(default = Activation::Relu(Relu), into)] activation: Activation,
        #[builder(default = Initializer::KaimingNormal { gain: 2.0f64.sqrt(), fan_out_only: false })]
        hidden_initializer: Initializer,
        #[builder(default = Initializer::KaimingNormal { gain: 2.0f64.sqrt(), fan_out_only: false })]
        output_initializer: Initializer,
        bias_initializer: Option<Initializer>,
    ) -> Result<Self, ModelError> {
        let dtype = model_dtype(&options)?;
        validate_activation(&activation, &hidden_layer_sizes)?;
        validate_activation(&activation, &value_hidden_layer_sizes)?;
        validate_activation(&activation, &advantage_hidden_layer_sizes)?;
        let (shared_layers, shared_output_size) = initialize_hidden_layers(
            input_size,
            &hidden_layer_sizes,
            &hidden_initializer,
            bias_initializer.as_ref(),
            &options,
        )?;
        let (value_hidden_layers, value_size) = initialize_hidden_layers(
            shared_output_size,
            &value_hidden_layer_sizes,
            &hidden_initializer,
            bias_initializer.as_ref(),
            &options,
        )?;
        let (advantage_hidden_layers, advantage_size) = initialize_hidden_layers(
            shared_output_size,
            &advantage_hidden_layer_sizes,
            &hidden_initializer,
            bias_initializer.as_ref(),
            &options,
        )?;
        let value_output_layer = initialize_linear(
            value_size,
            1,
            &output_initializer,
            bias_initializer.as_ref(),
            &options,
        )?;
        let advantage_output_layer = initialize_linear(
            advantage_size,
            output_size,
            &output_initializer,
            bias_initializer.as_ref(),
            &options,
        )?;
        Ok(Self {
            shared_layers,
            value_hidden_layers,
            value_output_layer,
            advantage_hidden_layers,
            advantage_output_layer,
            activation,
        }
        .fork(&options.device)
        .map(&mut ModelDType { dtype }))
    }

    /// Combines `[batch_size, 1]` values with `[batch_size, action_count]` advantages into `[batch_size, action_count]` Q-values.
    /// Inputs must share batch size, floating-point dtype, and device. The action mean keeps a size-one action axis for broadcasting.
    /// The result preserves batch and action axes, dtype, device, and gradient paths through both inputs.
    fn combine_streams(value: Tensor<2>, advantages: Tensor<2>) -> Tensor<2> {
        let mean = advantages.clone().mean_dim(1);
        value + (advantages - mean)
    }
}

impl Forward<2, 2> for DuelingMLP {
    type Error = ModelError;

    /// Maps `[batch_size, input_size]` to Q-values `[batch_size, output_size]`, preserving batch size and gradient paths.
    /// Input must share the model parameters' floating-point dtype and device. Incorrect feature counts return an error.
    fn forward(&self, input: Tensor<2>) -> Result<Tensor<2>, ModelError> {
        let first = self
            .shared_layers
            .first()
            .or_else(|| self.value_hidden_layers.first())
            .unwrap_or(&self.value_output_layer);
        validate_input(&input, first)?;
        let features = forward_hidden_layers(&self.shared_layers, &self.activation, input);
        let value_features = forward_hidden_layers(
            &self.value_hidden_layers,
            &self.activation,
            features.clone(),
        );
        let advantage_features =
            forward_hidden_layers(&self.advantage_hidden_layers, &self.activation, features);
        Ok(Self::combine_streams(
            self.value_output_layer.forward(value_features),
            self.advantage_output_layer.forward(advantage_features),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        nn::{PReluConfig, Tanh},
        optim::{GradientsParams, SgdConfig},
        tensor::Device,
    };

    #[test]
    fn mlp_preserves_dtype_shape_and_gradients() {
        let device = Device::flex().autodiff();
        for dtype in [DType::F32, DType::F64] {
            let network = MLP::builder()
                .input_size(4)
                .output_size(2)
                .options((&device, dtype))
                .bias_initializer(Initializer::Zeros)
                .hidden_layer_sizes(vec![3])
                .hidden_initializer(Initializer::Constant { value: 0.5 })
                .output_initializer(Initializer::Constant { value: 0.25 })
                .activation(Tanh)
                .output_activation(Tanh)
                .build()
                .unwrap();
            let input = Tensor::<2>::ones([5, 4], (&device, dtype)).require_grad();
            let output = network.forward(input.clone()).unwrap();
            assert_eq!(output.dims(), [5, 2]);
            assert_eq!(output.dtype(), dtype);
            let gradients = output.sum().backward();
            assert!(input.grad(&gradients).is_some());
            for layer in network.hidden_layers.iter().chain([&network.output_layer]) {
                assert_eq!(layer.weight.val().grad(&gradients).unwrap().dtype(), dtype);
                assert!(
                    layer
                        .bias
                        .as_ref()
                        .unwrap()
                        .val()
                        .grad(&gradients)
                        .is_some()
                );
            }
        }
    }

    #[test]
    fn empty_hidden_layers_and_zero_biases_use_native_initializers() {
        let device = Device::flex();
        for initializer in [
            Initializer::Orthogonal { gain: 1.0 },
            Initializer::KaimingNormal {
                gain: 1.0,
                fan_out_only: false,
            },
            Initializer::XavierUniform { gain: 1.0 },
        ] {
            let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
            let network = MLP::builder()
                .input_size(4)
                .output_size(2)
                .options(&device)
                .bias_initializer(Initializer::Zeros)
                .hidden_layer_sizes(vec![])
                .output_initializer(initializer)
                .build()
                .unwrap();
            let output = network.forward(Tensor::zeros([3, 4], &device)).unwrap();
            assert_eq!(output.dims(), [3, 2]);
            assert_eq!(output.abs().max().into_scalar::<f32>(), 0.0);
        }
    }

    #[test]
    fn default_biases_are_uniform_and_orthogonal_biases_are_zero() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex();
        let network = MLP::builder()
            .input_size(4)
            .output_size(16)
            .options(&device)
            .hidden_layer_sizes(vec![])
            .build()
            .unwrap();
        let bias = network.output_layer.bias.as_ref().unwrap().val();
        let values = bias.into_data().try_to_vec::<f32>().unwrap();
        assert!(values.iter().all(|value| *value >= -0.5 && *value < 0.5));
        assert!(values.iter().any(|value| *value != 0.0));
        assert_eq!(network.num_params(), 80);
        let orthogonal = MLP::builder()
            .input_size(4)
            .output_size(16)
            .options(&device)
            .hidden_layer_sizes(vec![])
            .output_initializer(Initializer::Orthogonal { gain: 1.0 })
            .build()
            .unwrap();
        assert_eq!(
            orthogonal
                .output_layer
                .bias
                .as_ref()
                .unwrap()
                .val()
                .abs()
                .max()
                .into_scalar::<f32>(),
            0.0
        );
    }

    #[test]
    fn dueling_combines_values_and_mean_centered_advantages() {
        let device = Device::flex();
        for dtype in [DType::F32, DType::F64] {
            let values = Tensor::from_data([[10.0f64], [-2.0]], (&device, dtype));
            let advantages =
                Tensor::from_data([[1.0f64, 2.0, 3.0], [-3.0, 0.0, 3.0]], (&device, dtype));
            let output = DuelingMLP::combine_streams(values, advantages);
            assert_eq!(output.dims(), [2, 3]);
            assert_eq!(
                output
                    .into_data()
                    .convert::<f64>()
                    .try_to_vec::<f64>()
                    .unwrap(),
                vec![9.0, 10.0, 11.0, -5.0, -2.0, 1.0]
            );
        }
    }

    #[test]
    fn all_dueling_branches_receive_selected_action_gradients() {
        let device = Device::flex().autodiff();
        for dtype in [DType::F32, DType::F64] {
            let network = DuelingMLP::builder()
                .input_size(4)
                .output_size(3)
                .options((&device, dtype))
                .bias_initializer(Initializer::Zeros)
                .hidden_layer_sizes(vec![8])
                .value_hidden_layer_sizes(vec![6])
                .advantage_hidden_layer_sizes(vec![7])
                .activation(Tanh)
                .hidden_initializer(Initializer::Constant { value: 0.1 })
                .output_initializer(Initializer::Constant { value: 0.2 })
                .build()
                .unwrap();
            let output = network
                .forward(Tensor::ones([2, 4], (&device, dtype)))
                .unwrap();
            assert_eq!(output.dims(), [2, 3]);
            let gradients = output.slice([0..2, 0..1]).sum().backward();
            for layer in network
                .shared_layers
                .iter()
                .chain(&network.value_hidden_layers)
                .chain(&network.advantage_hidden_layers)
                .chain([&network.value_output_layer, &network.advantage_output_layer])
            {
                let gradient = layer.weight.val().grad(&gradients).unwrap();
                assert_eq!(gradient.dtype(), dtype);
                assert!(gradient.abs().max().into_scalar::<f64>() > 0.0);
            }
        }
    }

    #[test]
    fn dueling_supports_empty_trunk_and_branches() {
        let device = Device::flex();
        for (shared, value, advantage) in [(vec![], vec![], vec![]), (vec![8], vec![], vec![7])] {
            let network = DuelingMLP::builder()
                .input_size(4)
                .output_size(1)
                .options(&device)
                .bias_initializer(Initializer::Zeros)
                .hidden_layer_sizes(shared)
                .value_hidden_layer_sizes(value)
                .advantage_hidden_layer_sizes(advantage)
                .hidden_initializer(Initializer::Zeros)
                .output_initializer(Initializer::Zeros)
                .build()
                .unwrap();
            assert_eq!(
                network
                    .forward(Tensor::zeros([2, 4], &device))
                    .unwrap()
                    .dims(),
                [2, 1]
            );
        }
    }

    #[test]
    fn learned_activation_parameters_use_model_dtype_and_gradients() {
        let device = Device::flex().autodiff();
        let network = MLP::builder()
            .input_size(2)
            .output_size(1)
            .options((&device, DType::F64))
            .bias_initializer(Initializer::Zeros)
            .hidden_layer_sizes(vec![2])
            .activation(PReluConfig::new().init(&device))
            .hidden_initializer(Initializer::Constant { value: -1.0 })
            .output_initializer(Initializer::Ones)
            .build()
            .unwrap();
        let Activation::PRelu(activation) = &network.activation else {
            panic!("expected PRelu");
        };
        assert!(activation.alpha.val().is_require_grad());
        assert!(activation.alpha.val().is_autodiff());
        let output = network
            .forward(Tensor::ones([1, 2], (&device, DType::F64)))
            .unwrap();
        let gradients = output.sum().backward();
        let Activation::PRelu(activation) = &network.activation else {
            panic!("expected PRelu");
        };
        assert_eq!(activation.alpha.val().dtype(), DType::F64);
        assert_eq!(
            activation.alpha.val().grad(&gradients).unwrap().dtype(),
            DType::F64
        );
    }

    #[test]
    fn initialized_activation_parameters_remain_optimizable_after_casting() {
        let device = Device::flex().autodiff();
        let activation = PReluConfig::new().init(&device);
        let before = activation.alpha.val().into_scalar::<f32>();
        let network = MLP::builder()
            .input_size(2)
            .output_size(1)
            .options((&device, DType::F64))
            .bias_initializer(Initializer::Zeros)
            .hidden_layer_sizes(vec![2])
            .activation(activation)
            .hidden_initializer(Initializer::Constant { value: -1.0 })
            .output_initializer(Initializer::Ones)
            .build()
            .unwrap();
        let input = Tensor::ones([1, 2], (&device, DType::F64));
        let gradients = network.forward(input.clone()).unwrap().sum().backward();
        let gradients = GradientsParams::from_grads(gradients, &network);
        let mut optimizer = SgdConfig::new().init();
        let network = optimizer.step(0.1, network, gradients);
        let Activation::PRelu(activation) = &network.activation else {
            panic!("expected PRelu");
        };
        assert!(
            (activation.alpha.val().into_scalar::<f64>() - (f64::from(before) + 0.4)).abs() < 1e-10
        );
        assert!(activation.alpha.val().is_require_grad());
        let gradients = network.forward(input).unwrap().sum().backward();
        assert!(activation.alpha.val().grad(&gradients).is_some());
    }

    #[test]
    fn invalid_configuration_and_inputs_return_typed_errors() {
        let device = Device::flex();
        assert!(matches!(
            MLP::builder()
                .input_size(0)
                .output_size(2)
                .options(&device)
                .bias_initializer(Initializer::Zeros)
                .build(),
            Err(ModelError::ZeroFeatures(0))
        ));
        assert!(matches!(
            DuelingMLP::builder()
                .input_size(4)
                .output_size(0)
                .options(&device)
                .bias_initializer(Initializer::Zeros)
                .build(),
            Err(ModelError::ZeroFeatures(0))
        ));
        assert!(matches!(
            MLP::builder()
                .input_size(4)
                .output_size(2)
                .options(&device)
                .bias_initializer(Initializer::Zeros)
                .hidden_layer_sizes(vec![0])
                .build(),
            Err(ModelError::ZeroFeatures(0))
        ));
        assert!(matches!(
            MLP::builder()
                .input_size(usize::MAX)
                .output_size(2)
                .options(&device)
                .bias_initializer(Initializer::Zeros)
                .hidden_layer_sizes(vec![])
                .build(),
            Err(ModelError::ShapeTooLarge { .. })
        ));
        assert!(matches!(
            MLP::builder()
                .input_size(4)
                .output_size(2)
                .options((&device, DType::U32))
                .bias_initializer(Initializer::Zeros)
                .build(),
            Err(ModelError::UnsupportedDType(DType::U32))
        ));
        for initializer in [
            Initializer::Orthogonal { gain: f64::NAN },
            Initializer::KaimingUniform {
                gain: 0.0,
                fan_out_only: false,
            },
            Initializer::XavierUniform { gain: 0.0 },
            Initializer::KaimingNormal {
                gain: -1.0,
                fan_out_only: false,
            },
            Initializer::Uniform { min: 1.0, max: 0.0 },
            Initializer::Uniform {
                min: 1.0,
                max: 1.0 + f64::EPSILON,
            },
            Initializer::Uniform {
                min: -f64::MAX,
                max: f64::MAX,
            },
            Initializer::Normal {
                mean: 0.0,
                std: -1.0,
            },
        ] {
            assert!(matches!(
                MLP::builder()
                    .input_size(4)
                    .output_size(2)
                    .options(&device)
                    .bias_initializer(Initializer::Zeros)
                    .output_initializer(initializer)
                    .build(),
                Err(ModelError::InvalidInitializer)
            ));
        }
        let network = MLP::builder()
            .input_size(4)
            .output_size(2)
            .options(&device)
            .bias_initializer(Initializer::Zeros)
            .hidden_initializer(Initializer::Zeros)
            .output_initializer(Initializer::Zeros)
            .build()
            .unwrap();
        assert!(matches!(
            network.forward(Tensor::zeros([2, 3], &device)),
            Err(ModelError::InputFeatures { .. })
        ));
        assert!(matches!(
            network.forward(Tensor::zeros([2, 4], (&device, DType::F64))),
            Err(ModelError::InputDType { .. })
        ));
        assert!(matches!(
            MLP::builder()
                .input_size(4)
                .output_size(2)
                .options(&device)
                .bias_initializer(Initializer::Zeros)
                .activation(PReluConfig::new().with_num_parameters(3).init(&device))
                .build(),
            Err(ModelError::ActivationFeatures { .. })
        ));
        assert!(matches!(
            MLP::builder()
                .input_size(4)
                .output_size(2)
                .options(&device)
                .bias_initializer(Initializer::Orthogonal { gain: 1.0 })
                .build(),
            Err(ModelError::InvalidInitializer)
        ));
    }
}

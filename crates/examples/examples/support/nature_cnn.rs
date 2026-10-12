use burn::{
    module::{Module, Param},
    nn::{
        Initializer,
        activation::Activation,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{DType, Device, Float, Tensor},
};
use modurl::models::{Forward, MLP, ModelError};

/// Encodes four 84-by-84 grayscale frames into 512 features with the Nature CNN architecture.
/// Inputs and parameters use F32 on the configured device.
#[derive(Module, Debug)]
pub(crate) struct NatureCnn {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    hidden: MLP,
}

impl NatureCnn {
    /// Builds the fixed encoder with orthogonal weights and zero biases on `device`.
    pub(crate) fn new(device: &Device) -> Result<Self, ModelError> {
        Ok(Self {
            conv1: convolution([4, 32], 8, 4, device),
            conv2: convolution([32, 64], 4, 2, device),
            conv3: convolution([64, 64], 3, 1, device),
            hidden: MLP::builder()
                .input_size(64 * 7 * 7)
                .output_size(512)
                .hidden_layer_sizes(vec![])
                .output_activation(Activation::Relu)
                .output_initializer(Initializer::Orthogonal {
                    gain: 2.0f64.sqrt(),
                })
                .options((device, DType::F32))
                .build()?,
        })
    }
}

/// Initializes rank-4 convolution weights and rank-1 zero biases on the same device, without changing channel or kernel axes.
fn convolution(channels: [usize; 2], kernel: usize, stride: usize, device: &Device) -> Conv2d {
    let mut layer = Conv2dConfig::new(channels, [kernel, kernel])
        .with_stride([stride, stride])
        .with_bias(false)
        .with_initializer(Initializer::Orthogonal {
            gain: 2.0f64.sqrt(),
        })
        .init(device);
    layer.bias = Some(Param::from_tensor(Tensor::zeros(
        [channels[1]],
        (device, DType::F32),
    )));
    layer
}

impl Forward<4, 2> for NatureCnn {
    type InputKind = Float;
    type OutputKind = Float;
    type Error = NatureCnnError;

    /// Maps F32 frames `[batch_size, 4, 84, 84]` to F32 features `[batch_size, 512]`.
    /// Preserves the batch axis and gradients. Inputs must use the model device.
    fn forward(&self, observations: Tensor<4>) -> Result<Tensor<2>, Self::Error> {
        let [batch_size, channels, height, width] = observations.dims();
        if [channels, height, width] != [4, 84, 84] {
            return Err(NatureCnnError::InputShape(observations.dims()));
        }
        if observations.dtype() != DType::F32 {
            return Err(ModelError::InputDType {
                expected: DType::F32,
                actual: observations.dtype(),
            }
            .into());
        }
        if observations.device() != self.conv1.weight.val().device() {
            return Err(ModelError::InputDevice.into());
        }
        let features = self.conv1.forward(observations).relu();
        let features = self.conv2.forward(features).relu();
        let features = self.conv3.forward(features).relu();
        self.hidden
            .forward(features.reshape([batch_size, 64 * 7 * 7]))
            .map_err(NatureCnnError::from)
    }
}

#[derive(Debug)]
pub(crate) enum NatureCnnError {
    InputShape([usize; 4]),
    Model(ModelError),
}

impl std::fmt::Display for NatureCnnError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InputShape(actual) => write!(
                formatter,
                "CNN input shape is {actual:?}, expected [batch_size, 4, 84, 84]"
            ),
            Self::Model(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for NatureCnnError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InputShape(_) => None,
            Self::Model(error) => Some(error),
        }
    }
}

impl From<ModelError> for NatureCnnError {
    fn from(error: ModelError) -> Self {
        Self::Model(error)
    }
}

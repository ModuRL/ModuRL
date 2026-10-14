use burn::{
    module::{Module, Param},
    nn::{
        Initializer, Relu,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{DType, Device, Distribution, Float, Int, Tensor},
};
use modurl::models::{Forward, MLP, ModelError};

/// Encodes four 84-by-84 grayscale frames into 512 features with the Nature CNN architecture.
/// Inputs use U8 Int tensors; explicit normalization produces F32 values on the parameter device.
#[derive(Module, Debug)]
pub(crate) struct NatureCnn {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    hidden: MLP,
}

impl NatureCnn {
    /// Builds the fixed encoder with PPO orthogonal weights when orthogonal is true, or DQN fan-in uniform weights otherwise.
    pub(crate) fn new(device: &Device, orthogonal: bool) -> Result<Self, ModelError> {
        let hidden_initializer = if orthogonal {
            Initializer::Orthogonal {
                gain: 2.0f64.sqrt(),
            }
        } else {
            let bound = 1.0 / (64.0f64 * 7.0 * 7.0).sqrt();
            Initializer::Uniform {
                min: -bound,
                max: bound,
            }
        };
        Ok(Self {
            conv1: convolution([4, 32], 8, 4, device, orthogonal),
            conv2: convolution([32, 64], 4, 2, device, orthogonal),
            conv3: convolution([64, 64], 3, 1, device, orthogonal),
            hidden: MLP::builder()
                .input_size(64 * 7 * 7)
                .output_size(512)
                .hidden_layer_sizes(vec![])
                .output_activation(Relu)
                .output_initializer(hidden_initializer)
                .options((device, DType::F32))
                .build()?,
        })
    }
}

/// Initializes convolution weights [output_channels, input_channels, kernel, kernel] and biases [output_channels] on device.
fn convolution(
    channels: [usize; 2],
    kernel: usize,
    stride: usize,
    device: &Device,
    orthogonal: bool,
) -> Conv2d {
    let bound = 1.0 / ((channels[0] * kernel * kernel) as f64).sqrt();
    let initializer = if orthogonal {
        Initializer::Orthogonal {
            gain: 2.0f64.sqrt(),
        }
    } else {
        Initializer::Uniform {
            min: -bound,
            max: bound,
        }
    };
    let mut layer = Conv2dConfig::new(channels, [kernel, kernel])
        .with_stride([stride, stride])
        .with_bias(false)
        .with_initializer(initializer)
        .init(device);
    let bias = if orthogonal {
        Tensor::zeros([channels[1]], (device, DType::F32))
    } else {
        Tensor::random(
            [channels[1]],
            Distribution::Uniform(-bound, bound),
            (device, DType::F32),
        )
    };
    layer.bias = Some(Param::from_tensor(bias));
    layer
}

impl Forward<4, 2> for NatureCnn {
    type InputKind = Int;
    type OutputKind = Float;
    type Error = NatureCnnError;

    /// Maps U8 Int frames `[batch_size, 4, 84, 84]` to F32 features `[batch_size, 512]`.
    /// Preserves the batch axis and gradients. Inputs must use the model device.
    fn forward(&self, observations: Tensor<4, Int>) -> Result<Tensor<2>, Self::Error> {
        let [batch_size, channels, height, width] = observations.dims();
        if [channels, height, width] != [4, 84, 84] {
            return Err(NatureCnnError::InputShape(observations.dims()));
        }
        if observations.dtype() != DType::U8 {
            return Err(ModelError::InputDType {
                expected: DType::U8,
                actual: observations.dtype(),
            }
            .into());
        }
        if observations.device() != self.conv1.weight.val().device() {
            return Err(ModelError::InputDevice.into());
        }
        let observations = observations.float().cast(DType::F32) / 255.0;
        let features = Relu.forward(self.conv1.forward(observations));
        let features = Relu.forward(self.conv2.forward(features));
        let features = Relu.forward(self.conv3.forward(features));
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{module::ModuleVisitor, optim::GradientsParams};

    #[test]
    fn orthogonal_convolutions_preserve_gain_and_zero_biases() {
        let device = Device::flex();
        let layer = convolution([4, 8], 3, 1, &device, true);
        let weights = layer.weight.val().reshape::<2, _>([8, 4 * 3 * 3]);
        let gram = weights
            .clone()
            .matmul(weights.transpose())
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        for (index, value) in gram.into_iter().enumerate() {
            let expected = if index / 8 == index % 8 { 2.0 } else { 0.0 };
            assert!((value - expected).abs() < 1e-4);
        }
        assert_eq!(
            layer
                .bias
                .unwrap()
                .val()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            vec![0.0; 8]
        );
    }

    #[test]
    fn byte_inputs_preserve_shape_and_parameter_gradients() {
        #[cfg(feature = "cuda")]
        let device = Device::cuda(0).autodiff();
        #[cfg(all(feature = "metal", not(feature = "cuda")))]
        let device = Device::metal(burn::tensor::DeviceKind::DefaultDevice).autodiff();
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let device = Device::flex().autodiff();
        let model = NatureCnn::new(&device, false).unwrap().train();
        let input = Tensor::<4, Int>::full([1, 4, 84, 84], 127, (&device, DType::U8));
        let output = model.forward(input.clone()).unwrap();
        assert_eq!(output.dims(), [1, 512]);
        assert_eq!(output.dtype(), DType::F32);
        let gradients = GradientsParams::from_grads(output.mean().backward(), &model);
        struct GradientCount<'a> {
            gradients: &'a GradientsParams,
            count: usize,
        }
        impl ModuleVisitor for GradientCount<'_> {
            /// Counts rank-D parameter gradients, preserving each parameter's shape and device.
            fn visit_float<const D: usize>(&mut self, parameter: &Param<Tensor<D>>) {
                assert!(self.gradients.get::<D>(parameter.id).is_some());
                self.count += 1;
            }
        }
        let mut count = GradientCount {
            gradients: &gradients,
            count: 0,
        };
        model.visit(&mut count);
        assert_eq!(count.count, 8);
        assert_eq!(input.dtype(), DType::U8);
        assert!(matches!(
            model.forward(input.cast(DType::I32)),
            Err(NatureCnnError::Model(ModelError::InputDType { .. }))
        ));
        assert!(matches!(
            model.forward(Tensor::zeros([1, 3, 84, 84], (&device, DType::U8))),
            Err(NatureCnnError::InputShape(_))
        ));
    }

    #[test]
    fn ppo_checkpoint_round_trip_preserves_predictions() {
        use modurl::{
            distributions::CategoricalDistribution,
            models::probabilistic_model::ProbabilisticPolicyModel,
        };
        let device = Device::flex();
        let make_models = || {
            let cnn = NatureCnn::new(&device, false).unwrap();
            let head = |output| {
                MLP::builder()
                    .input_size(512)
                    .output_size(output)
                    .hidden_layer_sizes(vec![])
                    .options(&device)
                    .build()
                    .unwrap()
            };
            (
                cnn,
                ProbabilisticPolicyModel::new(head(2), CategoricalDistribution),
                head(1),
            )
        };
        let models = make_models();
        let input = Tensor::<4, Int>::full([1, 4, 84, 84], 127, (&device, DType::U8));
        let features = models.0.forward(input.clone()).unwrap();
        let expected_policy = models
            .1
            .module()
            .forward(features.clone())
            .unwrap()
            .into_data();
        let expected_value = models.2.forward(features).unwrap().into_data();
        let path =
            std::env::temp_dir().join(format!("modurl-nature-cnn-{}.bpk", std::process::id()));
        models.save_file(&path).unwrap();
        let loaded = make_models().try_load_file(&path).unwrap();
        let features = loaded.0.forward(input).unwrap();
        assert_eq!(
            loaded
                .1
                .module()
                .forward(features.clone())
                .unwrap()
                .into_data(),
            expected_policy
        );
        assert_eq!(
            loaded.2.forward(features).unwrap().into_data(),
            expected_value
        );
        std::fs::remove_file(path).unwrap();
    }
}

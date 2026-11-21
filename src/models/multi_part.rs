//! Per-field feature extraction for flat-transport multipart observations.

use super::Forward;
use crate::spaces::{MultiPartSpace, ObservationPart, ObservationParts, SpaceError};
use burn::{
    module::Module,
    tensor::{Float, Tensor, kind::Basic},
};

/// Encodes one native observation batch into floating-point feature vectors.
/// Every matching [`Forward`] implementation automatically implements this trait.
pub trait PartEncoder<B> {
    /// Failure returned by the field encoder.
    type Error: std::fmt::Display;
    /// Returns `[batch, features]` without detaching gradient paths.
    fn encode(&self, batch: B) -> Result<Tensor<2>, Self::Error>;
}

impl<M, const R: usize, K: Basic> PartEncoder<Tensor<R, K>> for M
where
    M: Forward<R, 2, InputKind = K, OutputKind = Float>,
    M::Error: std::fmt::Display,
{
    type Error = M::Error;
    /// Encodes `[batch, ...field_shape]` to `[batch, features]`.
    fn encode(&self, batch: Tensor<R, K>) -> Result<Tensor<2>, Self::Error> {
        self.forward(batch)
    }
}

/// Errors in multipart transport or field feature extraction.
#[derive(Debug, thiserror::Error)]
pub enum MultiPartEncoderError {
    /// The flat input does not match the space.
    #[error(transparent)]
    Space(#[from] SpaceError),
    /// A field encoder failed; `part` is its zero-based tuple index.
    #[error("encoder for part {part} failed: {message}")]
    Part { part: usize, message: String },
    /// Encoder outputs must share the input batch size and each other's device.
    #[error("field encoder outputs have incompatible batch sizes or devices")]
    IncompatibleOutputs,
}

/// Tuple dispatch for 2–8 typed field encoders.
pub trait EncodeObservationParts<P: ObservationParts> {
    /// Runs each field encoder and concatenates its output along axis one.
    fn encode_parts(
        &self,
        parts: P::Parts,
        batch_size: usize,
    ) -> Result<Tensor<2>, MultiPartEncoderError>;
}

/// A Burn module that unpacks flat observations and concatenates field features.
///
/// `E` is a tuple of modules implementing [`PartEncoder`] for their respective
/// native batches. Burn visits all tuple parameters; the space is skipped.
/// Compose its [`Forward`] output with an MLP trunk or policy/value heads in a
/// custom module, just as with any other feature extractor.
#[derive(Module)]
pub struct MultiPartEncoder<E, P: Clone + Send + Sync + 'static> {
    encoders: E,
    #[module(skip)]
    space: MultiPartSpace<P>,
}

impl<E: std::fmt::Debug, P: Clone + Send + Sync + 'static> std::fmt::Debug
    for MultiPartEncoder<E, P>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiPartEncoder")
            .field("encoders", &self.encoders)
            .field("space", &self.space)
            .finish()
    }
}

impl<E, P: ObservationParts + Send + Sync + 'static> MultiPartEncoder<E, P> {
    /// Creates a feature extractor with one encoder per observation part.
    pub fn new(encoders: E, space: MultiPartSpace<P>) -> Self {
        Self { encoders, space }
    }
    /// Returns the transport layout.
    pub fn space(&self) -> &MultiPartSpace<P> {
        &self.space
    }
    /// Returns the typed field encoder modules.
    pub fn encoders(&self) -> &E {
        &self.encoders
    }
}

impl<E, P> Forward<2, 2> for MultiPartEncoder<E, P>
where
    P: ObservationParts + Send + Sync + 'static,
    E: EncodeObservationParts<P>,
{
    type InputKind = Float;
    type OutputKind = Float;
    type Error = MultiPartEncoderError;
    /// Encodes flat `[batch, flat_len]` to concatenated `[batch, features]`.
    fn forward(&self, batch: Tensor<2>) -> Result<Tensor<2>, Self::Error> {
        let count = batch.dims()[0];
        self.encoders.encode_parts(self.space.unpack(batch)?, count)
    }
}

macro_rules! impl_encoders {
    ($($P:ident : $E:ident : $i:tt),+) => {
        impl<$($P, $E),+> EncodeObservationParts<($($P,)+)> for ($($E,)+)
        where $($P: ObservationPart + Clone, $E: PartEncoder<$P::Batch>),+ {
            fn encode_parts(&self, parts: ($($P::Batch,)+), batch_size: usize) -> Result<Tensor<2>, MultiPartEncoderError> {
                let fields = vec![$(self.$i.encode(parts.$i).map_err(|error| MultiPartEncoderError::Part { part: $i, message: error.to_string() })?),+];
                let device = fields[0].device();
                if fields.iter().any(|f| f.dims()[0] != batch_size || f.device() != device) {
                    return Err(MultiPartEncoderError::IncompatibleOutputs);
                }
                let dtype = if fields.iter().any(|f| f.dtype() == burn::tensor::DType::F64) { burn::tensor::DType::F64 } else { burn::tensor::DType::F32 };
                Ok(Tensor::cat(fields.into_iter().map(|f| f.cast(dtype)).collect(), 1))
            }
        }
    };
}
impl_encoders!(A:EA:0, B:EB:1);
impl_encoders!(A:EA:0, B:EB:1, C:EC:2);
impl_encoders!(A:EA:0, B:EB:1, C:EC:2, D:ED:3);
impl_encoders!(A:EA:0, B:EB:1, C:EC:2, D:ED:3, E:EE:4);
impl_encoders!(A:EA:0, B:EB:1, C:EC:2, D:ED:3, E:EE:4, F:EF:5);
impl_encoders!(A:EA:0, B:EB:1, C:EC:2, D:ED:3, E:EE:4, F:EF:5, G:EG:6);
impl_encoders!(A:EA:0, B:EB:1, C:EC:2, D:ED:3, E:EE:4, F:EF:5, G:EG:6, H:EH:7);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        models::{MLP, ModelError},
        spaces::{BoxSpace, Discrete},
    };
    use burn::tensor::{DType, Device, Int};
    use std::convert::Infallible;

    #[derive(Module, Debug)]
    struct ImageEncoder {
        model: MLP,
    }
    impl Forward<4, 2> for ImageEncoder {
        type InputKind = Float;
        type OutputKind = Float;
        type Error = ModelError;
        /// Maps images `[batch, 1, 2, 2]` to `[batch, 2]` features.
        fn forward(&self, batch: Tensor<4>) -> Result<Tensor<2>, Self::Error> {
            let count = batch.dims()[0];
            self.model.forward(batch.reshape([count, 4]))
        }
    }
    #[derive(Module, Debug)]
    struct ModeEncoder;
    impl Forward<2, 2> for ModeEncoder {
        type InputKind = Int;
        type OutputKind = Float;
        type Error = Infallible;
        /// Casts category indices `[batch, 1]` to float features of the same shape.
        fn forward(&self, batch: Tensor<2, Int>) -> Result<Tensor<2>, Self::Error> {
            Ok(batch.float())
        }
    }

    #[derive(Module, Debug)]
    struct WrongBatchEncoder;
    impl Forward<2, 2> for WrongBatchEncoder {
        type InputKind = Float;
        type OutputKind = Float;
        type Error = Infallible;
        /// Returns an intentionally invalid `[batch + 1, 1]` fixture output.
        fn forward(&self, batch: Tensor<2>) -> Result<Tensor<2>, Self::Error> {
            Ok(Tensor::zeros(
                [batch.dims()[0] + 1, 1],
                (&batch.device(), batch.dtype()),
            ))
        }
    }

    fn mlp(input: usize, output: usize, device: &Device, dtype: DType) -> MLP {
        MLP::builder()
            .input_size(input)
            .output_size(output)
            .hidden_layer_sizes(vec![])
            .options((device, dtype))
            .build()
            .unwrap()
    }

    #[test]
    fn mixed_encoder_shape_gradient_and_errors() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        for dtype in [DType::F32, DType::F64] {
            let device = Device::flex().autodiff();
            let space = MultiPartSpace::new((
                BoxSpace::new_unbounded([1, 1, 2, 2], &device),
                BoxSpace::new_unbounded([1, 3], &device),
                Discrete::new(2).unwrap(),
            ));
            let encoder = MultiPartEncoder::new(
                (
                    ImageEncoder {
                        model: mlp(4, 2, &device, dtype),
                    },
                    mlp(3, 2, &device, dtype),
                    ModeEncoder,
                ),
                space.clone(),
            );
            let image = Tensor::ones([2, 1, 2, 2], (&device, dtype)).require_grad();
            let joints = Tensor::ones([2, 3], (&device, dtype)).require_grad();
            let flat = space
                .pack((
                    image.clone(),
                    joints.clone(),
                    Tensor::zeros([2, 1], &device),
                ))
                .unwrap();
            let output = encoder.forward(flat).unwrap();
            assert_eq!(output.dims(), [2, 5]);
            assert_eq!(output.dtype(), dtype);
            let gradients = output.sum().backward();
            assert!(image.grad(&gradients).is_some());
            assert!(joints.grad(&gradients).is_some());
            assert!(
                encoder
                    .encoders
                    .0
                    .model
                    .output_layer
                    .weight
                    .val()
                    .grad(&gradients)
                    .is_some()
            );
            assert!(
                encoder
                    .encoders
                    .1
                    .output_layer
                    .weight
                    .val()
                    .grad(&gradients)
                    .is_some()
            );
            assert!(matches!(
                encoder.forward(Tensor::zeros([2, 7], (&device, dtype))),
                Err(MultiPartEncoderError::Space(_))
            ));
            assert!(matches!(
                encoder.forward(Tensor::zeros([0, 8], (&device, dtype))),
                Err(MultiPartEncoderError::Space(
                    SpaceError::EmptyObservationBatch
                ))
            ));
            let bad = MultiPartEncoder::new(
                (mlp(5, 2, &device, dtype), mlp(3, 2, &device, dtype)),
                MultiPartSpace::new((
                    BoxSpace::new_unbounded([1, 4], &device),
                    BoxSpace::new_unbounded([1, 3], &device),
                )),
            );
            assert!(matches!(
                bad.forward(Tensor::zeros([2, 7], (&device, dtype))),
                Err(MultiPartEncoderError::Part { part: 0, .. })
            ));
            let bad = MultiPartEncoder::new(
                (WrongBatchEncoder, mlp(3, 2, &device, dtype)),
                bad.space().clone(),
            );
            assert!(matches!(
                bad.forward(Tensor::zeros([2, 7], (&device, dtype))),
                Err(MultiPartEncoderError::IncompatibleOutputs)
            ));
        }
    }

    #[test]
    fn existing_dqn_trains_with_typed_observations() {
        use crate::{
            agents::{Agent, q_learning::dqn::DQNAgent, test_support::FixedEnv},
            gym::{Gym, MultiGym, ResetInfo, StepInfo, VectorizedGymError, VectorizedGymWrapper},
            parameter_schedule::ConstantSchedule,
        };
        use burn::optim::SgdConfig;
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        type Layout = (BoxSpace<2>, BoxSpace<2>);
        struct Env {
            inner: FixedEnv,
            space: MultiPartSpace<Layout>,
        }
        impl Gym for Env {
            type Error = Infallible;
            type ActionSpace = Discrete;
            type ObservationSpace = MultiPartSpace<Layout>;
            /// Steps using a scalar Int action `[1]`.
            fn step(&mut self, action: Tensor<1, Int>) -> Result<StepInfo, Self::Error> {
                self.inner.step(action)
            }
            fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
                self.inner.reset()
            }
            fn action_space(&self) -> Discrete {
                self.inner.action_space()
            }
            fn observation_space(&self) -> Self::ObservationSpace {
                self.space.clone()
            }
        }
        let device = Device::flex().autodiff();
        let space = MultiPartSpace::new((
            BoxSpace::new_unbounded([1, 2], &device),
            BoxSpace::new_unbounded([1, 2], &device),
        ));
        let network = MultiPartEncoder::new(
            (
                mlp(2, 1, &device, DType::F32),
                mlp(2, 1, &device, DType::F32),
            ),
            space.clone(),
        );
        let mut env = VectorizedGymWrapper::new(vec![Env {
            inner: FixedEnv::new(device.clone().without_autodiff()),
            space: space.clone(),
        }])
        .unwrap();
        let mut agent = DQNAgent::<_, _, VectorizedGymError<Infallible>>::builder()
            .action_space(Discrete::new(2).unwrap())
            .observation_space(space)
            .online_q_network(network)
            .optimizer(SgdConfig::new().init())
            .learning_rate(0.01)
            .epsilon_schedule(ConstantSchedule::new(0.0))
            .replay_capacity(4)
            .batch_size(1)
            .training_start(1)
            .update_frequency(1)
            .target_update_interval(2)
            .training_horizon(4)
            .build()
            .unwrap();
        let observations = env.reset().unwrap();
        assert_eq!(agent.act(&observations, &mut ()).unwrap().dims(), [1, 1]);
        agent.learn(&mut env, 4).unwrap();
    }
}

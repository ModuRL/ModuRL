use std::num::NonZeroUsize;

use burn::tensor::{Float, Tensor, TensorReadError, activation::softplus};

use super::{
    DifferentiableExpectation, DistEval, Distribution, DistributionTensorError, ExpectationTerms,
    validate_statistics,
};

/// Changes each action component with a transform that has an inverse.
/// An event is one complete action; its event shape excludes batch and candidate axes.
/// Methods accept the `Float` tensor kind and preserve the input shape, dtype, and device.
/// For vector actions, layouts are `[batch_size, action_components]` or `[batch_size, candidate_count, action_components]`.
pub trait DistributionTransform {
    type Error;

    /// Transforms rank-`R` actions shaped `[batch_size, ...candidate_axes, ...event_shape]` and preserves every axis.
    /// The candidate axis is present when evaluating candidate actions. Rank `R` counts all input axes.
    fn forward<const R: usize>(&self, input: Tensor<R>) -> Result<Tensor<R>, Self::Error>;

    /// Recovers base actions from rank-`R` transformed actions shaped `[batch_size, ...candidate_axes, ...event_shape]`.
    /// The result preserves every axis. The candidate axis is present when evaluating candidate actions.
    fn inverse<const R: usize>(&self, output: Tensor<R>) -> Result<Tensor<R>, Self::Error>;

    /// Returns one log absolute derivative per action component to account for changes in action spacing and probability density.
    /// Each term is `log |dy/dx|`, where `x` is the base action component and `y` is the transformed component.
    /// Both inputs and the rank-`R` result have shape `[batch_size, ...candidate_axes, ...event_shape]`.
    /// Inputs must have matching shapes, dtypes, and devices. The result preserves these properties.
    /// Summing terms over the event axes gives the log absolute Jacobian determinant for one complete action.
    fn log_abs_det_jacobian<const R: usize>(
        &self,
        input: Tensor<R>,
        output: Tensor<R>,
    ) -> Result<Tensor<R>, Self::Error>;

    /// Returns the constant adjustment to target entropy, the entropy level used to tune policy randomness.
    fn target_entropy_adjustment(&self) -> Result<f64, Self::Error> {
        Ok(0.0)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TanhTransform;

impl DistributionTransform for TanhTransform {
    type Error = DistributionTensorError;

    /// Applies tanh to rank-`R` actions shaped `[batch_size, ...candidate_axes, ...event_shape]`.
    /// The result preserves every axis, the dtype, and the device.
    fn forward<const R: usize>(&self, input: Tensor<R>) -> Result<Tensor<R>, Self::Error> {
        Ok(input.tanh())
    }

    /// Applies inverse tanh to rank-`R` actions shaped `[batch_size, ...candidate_axes, ...event_shape]`.
    /// Clamps values to `[-1 + 1e-6, 1 - 1e-6]` to avoid infinite results at the bounds.
    /// The result preserves every axis, the dtype, and the device.
    fn inverse<const R: usize>(&self, output: Tensor<R>) -> Result<Tensor<R>, Self::Error> {
        let bounded = output.clamp(-1.0 + 1e-6, 1.0 - 1e-6);
        let denominator = bounded.clone().neg() + 1.0;
        Ok(((bounded + 1.0) / denominator).log() * 0.5)
    }

    /// Returns log absolute derivatives with the inputs' rank-`R` shape `[batch_size, ...candidate_axes, ...event_shape]`.
    /// Inputs must share shape, dtype, and device. The result preserves these properties.
    /// Computes derivatives from the base actions so values and gradients remain accurate when tanh rounds to -1 or 1.
    fn log_abs_det_jacobian<const R: usize>(
        &self,
        input: Tensor<R>,
        _output: Tensor<R>,
    ) -> Result<Tensor<R>, Self::Error> {
        let correction = softplus(input.clone() * -2.0, 1.0);
        Ok((input.neg() + std::f64::consts::LN_2 - correction) * 2.0)
    }
}

/// Scales and shifts each action component: `y = scale * x + shift`.
/// Rank-`E` parameters describe one action and have shape `[...event_shape]`, without batch or candidate axes.
/// Parameters broadcast over the batch axis and any candidate axes of rank-`R` actions, where `R > E`.
/// Event dimensions must match exactly. Parameter broadcasting does not change the action shape.
#[derive(Clone, Debug)]
pub struct AffineTransform<const E: usize = 1> {
    scale: Tensor<E>,
    shift: Tensor<E>,
}

#[derive(Debug, thiserror::Error)]
pub enum AffineTransformError {
    #[error("affine transform tensor validation failed: {0}")]
    TensorError(#[from] DistributionTensorError),
    #[error("affine transform parameter read failed: {0}")]
    ReadError(#[from] TensorReadError),
    #[error("an affine transform scale cannot contain zero")]
    ZeroScale,
    #[error("affine transform parameters must be finite")]
    NonFiniteParameters,
    #[error("affine transform bounds must be finite and strictly ordered")]
    InvalidBounds,
}

impl<const E: usize> AffineTransform<E> {
    /// Creates a transform from rank-`E` scale and shift tensors shaped `[...event_shape]`.
    /// Both inputs must share shape, dtype, and device. They have no batch or candidate axes.
    /// Both inputs must be finite, and every scale value must be nonzero so the transform has an inverse.
    pub fn new(scale: Tensor<E>, shift: Tensor<E>) -> Result<Self, AffineTransformError> {
        validate_statistics("affine shift", &scale, &shift)?;
        if scale
            .clone()
            .equal_elem(0.0)
            .any()
            .try_into_scalar::<bool>()?
        {
            return Err(AffineTransformError::ZeroScale);
        }
        if !scale
            .clone()
            .is_finite()
            .bool_and(shift.clone().is_finite())
            .all()
            .try_into_scalar::<bool>()?
        {
            return Err(AffineTransformError::NonFiniteParameters);
        }
        Ok(Self { scale, shift })
    }

    /// Returns the stored rank-`E` scale shaped `[...event_shape]`, without batch or candidate axes.
    pub fn scale(&self) -> &Tensor<E> {
        &self.scale
    }

    /// Returns the stored rank-`E` shift shaped `[...event_shape]`, without batch or candidate axes.
    pub fn shift(&self) -> &Tensor<E> {
        &self.shift
    }

    /// Creates a transform that maps each action component from `[-1, 1]` to its lower and upper bounds.
    /// Rank-`E` inputs `low` and `high` must share shape `[...event_shape]`, dtype, and device.
    /// Bounds must be finite, with `low < high` for every component.
    /// Stored scale and shift keep the bounds' event shape, dtype, and device.
    pub fn from_bounds(low: Tensor<E>, high: Tensor<E>) -> Result<Self, AffineTransformError> {
        validate_statistics("affine upper bounds", &low, &high)?;
        let valid = low
            .clone()
            .is_finite()
            .bool_and(high.clone().is_finite())
            .bool_and(low.clone().lower(high.clone()));
        if !valid.all().try_into_scalar::<bool>()? {
            return Err(AffineTransformError::InvalidBounds);
        }
        // Halve each bound first to avoid overflow when finite bounds approach the dtype's largest magnitude.
        let scale = high.clone() * 0.5 - low.clone() * 0.5;
        let shift = high * 0.5 + low * 0.5;
        Self::new(scale, shift)
    }

    /// Checks event dimensions and adds size-one batch and candidate axes to the scale and shift.
    /// Rank-`R` input has shape `[batch_size, ...candidate_axes, ...event_shape]`; rank-`E` parameters must match every event dimension.
    /// Ordinary actions have no candidate axis. Expectation actions have a candidate axis after the batch axis.
    /// Rank `R` must exceed `E` so the input includes a batch axis.
    /// Returned parameters have rank `R`, with size-one axes before the unchanged event shape.
    /// Moves parameters to the input device and casts them to the input dtype. The input keeps its shape.
    fn broadcast_parameters<const R: usize>(
        &self,
        input: &Tensor<R>,
    ) -> Result<(Tensor<R>, Tensor<R>), AffineTransformError> {
        const {
            assert!(
                R > E,
                "affine input requires a batch axis before event axes"
            );
        }
        let actual = input.dims();
        let parameter_shape = self.scale.dims();
        // Require each parameter dimension to match the corresponding trailing action dimension.
        // Parameters can broadcast over batch and candidate axes, but not within an action.
        for (axis, size) in parameter_shape.iter().enumerate() {
            if *size != actual[R - E + axis] {
                let mut expected = actual.to_vec();
                expected[R - E + axis] = *size;
                return Err(DistributionTensorError::ShapeMismatch {
                    field: "affine input",
                    expected,
                    actual: actual.to_vec(),
                }
                .into());
            }
        }
        let device = input.device();
        Ok((
            self.scale
                .clone()
                .to_device(&device)
                .cast(input.dtype())
                .unsqueeze::<R>(),
            self.shift
                .clone()
                .to_device(&device)
                .cast(input.dtype())
                .unsqueeze::<R>(),
        ))
    }
}

impl<const E: usize> DistributionTransform for AffineTransform<E> {
    type Error = AffineTransformError;

    /// Scales and shifts rank-`R` actions shaped `[batch_size, ...candidate_axes, ...event_shape]`.
    /// Rank `R` must exceed `E`, and event dimensions must match the stored parameters.
    /// Parameters broadcast over batch and candidate axes after conversion to the input dtype and device.
    /// The result preserves the input shape, dtype, device, and gradient paths.
    fn forward<const R: usize>(&self, input: Tensor<R>) -> Result<Tensor<R>, Self::Error> {
        let (scale, shift) = self.broadcast_parameters(&input)?;
        Ok(input * scale + shift)
    }

    /// Removes the shift and scale from rank-`R` actions shaped `[batch_size, ...candidate_axes, ...event_shape]`.
    /// Rank `R` must exceed `E`, and event dimensions must match the stored parameters.
    /// Parameters broadcast over batch and candidate axes after conversion to the input dtype and device.
    /// The result preserves the input shape, dtype, device, and gradient paths.
    fn inverse<const R: usize>(&self, output: Tensor<R>) -> Result<Tensor<R>, Self::Error> {
        let (scale, shift) = self.broadcast_parameters(&output)?;
        Ok((output - shift) / scale)
    }

    /// Returns `log |scale|` for each component of rank-`R` actions shaped `[batch_size, ...candidate_axes, ...event_shape]`.
    /// Both inputs must share shape, dtype, and device. Event dimensions must match the parameters, and `R > E`.
    /// Converts scales to the input dtype and device, then broadcasts them over batch and candidate axes.
    /// The result keeps the input shape, dtype, and device.
    fn log_abs_det_jacobian<const R: usize>(
        &self,
        input: Tensor<R>,
        _output: Tensor<R>,
    ) -> Result<Tensor<R>, Self::Error> {
        let (scale, _) = self.broadcast_parameters(&input)?;
        Ok(scale.abs().log().expand(input.dims()))
    }

    /// Sums `log |scale|` over all event axes of the stored rank-`E` scale shaped `[...event_shape]`.
    /// Returns a scalar read to the host, with no batch or candidate axes.
    fn target_entropy_adjustment(&self) -> Result<f64, Self::Error> {
        Ok(self
            .scale
            .clone()
            .abs()
            .log()
            .sum()
            .try_into_scalar::<f64>()?)
    }
}

/// Applies an invertible transform to actions from a base distribution.
#[derive(Clone, Debug, Default)]
pub struct TransformedDistribution<D, T> {
    distribution: D,
    transform: T,
}

#[derive(Debug, thiserror::Error)]
pub enum TransformedDistributionError<DE, TE> {
    #[error("distribution failed: {0}")]
    DistributionError(#[source] DE),
    #[error("distribution transform failed: {0}")]
    TransformError(#[source] TE),
    #[error("transformed distribution tensor validation failed: {0}")]
    TensorError(#[from] DistributionTensorError),
}

impl<D, T> TransformedDistribution<D, T> {
    pub fn new(distribution: D, transform: T) -> Self {
        Self {
            distribution,
            transform,
        }
    }

    pub fn distribution(&self) -> &D {
        &self.distribution
    }

    pub fn transform(&self) -> &T {
        &self.transform
    }
}

/// Sums values over the trailing event axes and removes those axes.
/// Rank `P` counts the leading axes to keep. Ranks must satisfy `R >= P >= 1`.
/// Rank-`R` input `[batch_size, ...event_shape]` becomes rank-1 output `[batch_size]` when `P = 1`.
/// Input `[batch_size, candidate_count, ...event_shape]` becomes `[batch_size, candidate_count]` when `P = 2`.
/// The result keeps the leading axes in order and preserves dtype, device, and gradient paths.
fn sum_event_dimensions<const R: usize, const P: usize>(values: Tensor<R>) -> Tensor<P> {
    const {
        assert!(P >= 1, "event reduction requires a prefix axis");
        assert!(R >= P, "event reduction rank must cover prefix axes");
    }
    let shape = values.dims();
    let mut prefix = [1; P];
    prefix.copy_from_slice(&shape[..P]);
    let prefix_size = prefix.iter().product::<usize>();
    let event_size = shape[P..].iter().product::<usize>();
    values
        .reshape([prefix_size, event_size])
        .sum_dim(1)
        .reshape(prefix)
}

impl<D, T, const P: usize, const A: usize> Distribution<P, A> for TransformedDistribution<D, T>
where
    D: Distribution<P, A>,
    T: DistributionTransform,
{
    type Error = TransformedDistributionError<D::Error, T::Error>;

    /// Samples base actions and applies the transform.
    /// Rank-`P` distribution parameters have shape `[batch_size, ...parameter_shape]`.
    /// Rank-`A` returned actions have shape `[batch_size, ...event_shape]`, with one action per batch item.
    /// The transform preserves the base samples' shape, dtype, device, and gradient paths.
    fn sample(&self, outputs: Tensor<P>) -> Result<Tensor<A>, Self::Error> {
        let sample = self
            .distribution
            .sample(outputs)
            .map_err(TransformedDistributionError::DistributionError)?;
        self.transform
            .forward(sample)
            .map_err(TransformedDistributionError::TransformError)
    }

    /// Applies the transform to the base distribution's mode, its most likely action.
    /// Rank-`P` distribution parameters have shape `[batch_size, ...parameter_shape]`.
    /// Rank-`A` returned actions have shape `[batch_size, ...event_shape]`.
    /// The transform preserves the base modes' shape, dtype, device, and gradient paths.
    fn mode(&self, outputs: Tensor<P>) -> Result<Tensor<A>, Self::Error> {
        let mode = self
            .distribution
            .mode(outputs)
            .map_err(TransformedDistributionError::DistributionError)?;
        self.transform
            .forward(mode)
            .map_err(TransformedDistributionError::TransformError)
    }

    /// Computes transformed action log density and an entropy estimate for each batch item.
    /// Rank-`P` distribution parameters have shape `[batch_size, ...parameter_shape]`.
    /// Rank-`A` actions have shape `[batch_size, ...event_shape]`; batch sizes must match.
    /// The transform changes action spacing and therefore changes probability density.
    /// Sums the transform's log absolute derivatives over event axes, then subtracts that sum from the base action log density.
    /// Both returned statistics have shape `[batch_size]`. Negative log density at the supplied action gives the entropy estimate.
    /// The transform preserves action dtype and device. Statistics follow the base distribution's dtype and device requirements.
    /// The calculation retains gradient paths through the inverse transform and density correction.
    fn dist_eval(&self, outputs: Tensor<P>, actions: Tensor<A>) -> Result<DistEval, Self::Error> {
        let base_actions = self
            .transform
            .inverse(actions.clone())
            .map_err(TransformedDistributionError::TransformError)?;
        let evaluation = self
            .distribution
            .dist_eval(outputs, base_actions.clone())
            .map_err(TransformedDistributionError::DistributionError)?;
        let jacobian = self
            .transform
            .log_abs_det_jacobian(base_actions, actions)
            .map_err(TransformedDistributionError::TransformError)?;
        let correction: Tensor<1> = sum_event_dimensions(jacobian);
        let log_prob = evaluation.log_prob().clone() - correction;
        // Negative log density estimates entropy when actions are sampled from the distribution.
        let entropy = log_prob.clone().neg();
        Ok(DistEval::new(log_prob, entropy)?)
    }
}

impl<D, T, const P: usize, const A: usize, const C: usize> DifferentiableExpectation<P, A, C>
    for TransformedDistribution<D, T>
where
    D: DifferentiableExpectation<P, A, C, CandidateKind = Float>,
    T: DistributionTransform,
{
    type CandidateKind = Float;

    /// Transforms continuous candidate actions and adjusts their log densities.
    /// Rank-`P` distribution parameters have shape `[batch_size, ...parameter_shape]`.
    /// Returned rank-`C` actions have shape `[batch_size, candidate_count, ...event_shape]`.
    /// Log densities and unchanged weights have shape `[batch_size, candidate_count]`.
    /// Sums log absolute derivatives over event axes and subtracts the sum from each base candidate's log density.
    /// The transform preserves candidate shape, dtype, device, and gradient paths.
    /// Log densities and weights follow the base distribution's dtype and device requirements.
    fn expectation(
        &self,
        outputs: Tensor<P>,
        samples: NonZeroUsize,
    ) -> Result<ExpectationTerms<C>, Self::Error> {
        let terms = self
            .distribution
            .expectation(outputs, samples)
            .map_err(TransformedDistributionError::DistributionError)?;
        let (base_actions, base_log_probabilities, weights) = terms.into_parts();
        let actions = self
            .transform
            .forward(base_actions.clone())
            .map_err(TransformedDistributionError::TransformError)?;
        let jacobian = self
            .transform
            .log_abs_det_jacobian(base_actions, actions.clone())
            .map_err(TransformedDistributionError::TransformError)?;
        let correction: Tensor<2> = sum_event_dimensions(jacobian);
        let log_probabilities = base_log_probabilities - correction;
        Ok(ExpectationTerms::new(actions, log_probabilities, weights)?)
    }

    /// Adds the transform's constant adjustment to the base distribution's scalar target entropy.
    /// Rank-`P` distribution parameters have shape `[batch_size, ...parameter_shape]` and follow the base distribution's dtype and device requirements.
    /// The result has no batch, candidate, or event axes.
    fn default_target_entropy(&self, outputs: &Tensor<P>) -> Result<f64, Self::Error> {
        Ok(self
            .distribution
            .default_target_entropy(outputs)
            .map_err(TransformedDistributionError::DistributionError)?
            + self
                .transform
                .target_entropy_adjustment()
                .map_err(TransformedDistributionError::TransformError)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::{GaussianDistribution, GaussianDistributionError};
    use burn::tensor::{DType, Device};

    #[test]
    fn tanh_round_trip_and_jacobian_match_closed_form() {
        let device = Device::flex();
        for dtype in [DType::F32, DType::F64] {
            let input = Tensor::<2>::from_data([[-2.0f64, -0.5, 0.0, 0.5, 2.0]], (&device, dtype));
            let output = TanhTransform.forward(input.clone()).unwrap();
            let recovered = TanhTransform.inverse(output.clone()).unwrap();
            assert!((input.clone() - recovered).abs().max().into_scalar::<f64>() < 1e-5);
            let jacobian = TanhTransform
                .log_abs_det_jacobian(input, output.clone())
                .unwrap();
            let expected = (output.square().neg() + 1.0).log();
            assert!((jacobian - expected).abs().max().into_scalar::<f64>() < 1e-5);
        }
    }

    #[test]
    fn saturated_tanh_jacobian_retains_values_and_gradients() {
        let device = Device::flex().autodiff();
        let input = Tensor::<2>::from_floats([[-20.0, 20.0]], &device).require_grad();
        let output = TanhTransform.forward(input.clone()).unwrap();
        let jacobian = TanhTransform
            .log_abs_det_jacobian(input.clone(), output)
            .unwrap();
        let expected = 2.0 * (2.0f32.ln() - 20.0);
        for value in jacobian.clone().into_data().try_to_vec::<f32>().unwrap() {
            assert!((value - expected).abs() < 1e-4);
        }
        let gradients = jacobian.sum().backward();
        let values = input
            .grad(&gradients)
            .unwrap()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert!((values[0] - 2.0).abs() < 1e-5);
        assert!((values[1] + 2.0).abs() < 1e-5);
    }

    #[test]
    fn affine_bounds_round_trip_preserves_input_precision_and_candidate_axes() {
        let device = Device::flex();
        let affine = AffineTransform::<1>::from_bounds(
            Tensor::from_floats([-1.0, -4.0], &device),
            Tensor::from_floats([3.0, 2.0], &device),
        )
        .unwrap();
        let input =
            Tensor::<3>::from_data([[[-1.0f64, 1.0], [0.25, -0.25]]], (&device, DType::F64));
        let output = affine.forward(input.clone()).unwrap();
        assert_eq!(output.dims(), [1, 2, 2]);
        assert_eq!(output.dtype(), DType::F64);
        assert_eq!(
            output.clone().into_data().try_to_vec::<f64>().unwrap(),
            vec![-1.0, 2.0, 1.5, -1.75]
        );
        assert!(
            (affine.inverse(output).unwrap() - input)
                .abs()
                .max()
                .into_scalar::<f64>()
                < 1e-12
        );
        assert!(matches!(
            affine.forward(Tensor::<2>::zeros([2, 3], &device)),
            Err(AffineTransformError::TensorError(
                DistributionTensorError::ShapeMismatch { .. }
            ))
        ));
        let wide = AffineTransform::<1>::from_bounds(
            Tensor::from_data([-f64::MAX], (&device, DType::F64)),
            Tensor::from_data([f64::MAX], (&device, DType::F64)),
        )
        .unwrap();
        let endpoints = wide
            .forward(Tensor::<2>::from_data(
                [[-1.0f64], [1.0]],
                (&device, DType::F64),
            ))
            .unwrap();
        assert_eq!(
            endpoints.into_data().try_to_vec::<f64>().unwrap(),
            vec![-f64::MAX, f64::MAX]
        );
    }

    #[test]
    fn affine_requires_exact_event_shapes_and_counts_each_scale_once() {
        let device = Device::flex();
        let scalar_scale = AffineTransform::<1>::new(
            Tensor::from_floats([2.0], &device),
            Tensor::zeros([1], &device),
        )
        .unwrap();
        let actions = Tensor::<2>::zeros([2, 3], &device);
        let candidates = Tensor::<3>::zeros([2, 4, 3], &device);
        assert!(matches!(
            scalar_scale.forward(actions.clone()),
            Err(AffineTransformError::TensorError(
                DistributionTensorError::ShapeMismatch { .. }
            ))
        ));
        assert!(scalar_scale.inverse(actions.clone()).is_err());
        assert!(
            scalar_scale
                .log_abs_det_jacobian(actions.clone(), actions.clone())
                .is_err()
        );
        assert!(scalar_scale.forward(candidates.clone()).is_err());

        let affine = AffineTransform::<1>::new(
            Tensor::from_floats([2.0, 2.0, 2.0], &device),
            Tensor::zeros([3], &device),
        )
        .unwrap();
        let adjustment = affine.target_entropy_adjustment().unwrap();
        assert!((adjustment - 3.0 * 2.0f64.ln()).abs() < 1e-6);
        let output = affine.forward(actions.clone()).unwrap();
        assert_eq!(output.dims(), [2, 3]);
        let jacobian = affine.log_abs_det_jacobian(actions, output).unwrap();
        let corrections: Tensor<1> = sum_event_dimensions(jacobian);
        for correction in corrections.into_data().try_to_vec::<f32>().unwrap() {
            assert!((f64::from(correction) - adjustment).abs() < 1e-6);
        }
        assert_eq!(affine.forward(candidates).unwrap().dims(), [2, 4, 3]);
    }

    #[test]
    fn affine_parameters_and_bounds_are_validated() {
        let device = Device::flex();
        let values = |array: [f32; 2]| Tensor::<1>::from_floats(array, &device);
        assert!(matches!(
            AffineTransform::new(values([1.0, 1.0]), Tensor::zeros([1], &device)),
            Err(AffineTransformError::TensorError(
                DistributionTensorError::ShapeMismatch { .. }
            ))
        ));
        assert!(matches!(
            AffineTransform::new(values([0.0, 1.0]), values([0.0, 0.0])),
            Err(AffineTransformError::ZeroScale)
        ));
        for invalid in [f32::INFINITY, f32::NAN] {
            assert!(matches!(
                AffineTransform::new(values([invalid, 1.0]), values([0.0, 0.0])),
                Err(AffineTransformError::NonFiniteParameters)
            ));
        }
        assert!(matches!(
            AffineTransform::from_bounds(values([-1.0, 2.0]), values([1.0, 2.0])),
            Err(AffineTransformError::InvalidBounds)
        ));
        assert!(matches!(
            AffineTransform::from_bounds(values([-1.0, 3.0]), values([1.0, 2.0])),
            Err(AffineTransformError::InvalidBounds)
        ));
    }

    #[test]
    fn affine_scale_adjusts_density_and_target_entropy() {
        let device = Device::flex();
        let affine = AffineTransform::<1>::new(
            Tensor::from_floats([2.0, 4.0], &device),
            Tensor::from_floats([1.0, -1.0], &device),
        )
        .unwrap();
        let base = TransformedDistribution::<GaussianDistribution, TanhTransform>::default();
        let transformed = TransformedDistribution::new(base.clone(), affine.clone());
        let outputs = Tensor::<2>::zeros([1, 4], &device);
        let base_actions = Tensor::from_floats([[0.1, -0.2]], &device);
        let actions = affine.forward(base_actions.clone()).unwrap();
        let base_log_prob = base
            .dist_eval(outputs.clone(), base_actions)
            .unwrap()
            .log_prob()
            .clone()
            .into_scalar::<f32>();
        let log_prob = transformed
            .dist_eval(outputs.clone(), actions)
            .unwrap()
            .log_prob()
            .clone()
            .into_scalar::<f32>();
        let log_scale = 8.0f32.ln();
        assert!((log_prob - (base_log_prob - log_scale)).abs() < 1e-5);
        assert!(
            (transformed.default_target_entropy(&outputs).unwrap()
                - base.default_target_entropy(&outputs).unwrap()
                - f64::from(log_scale))
            .abs()
                < 1e-6
        );
    }

    #[test]
    fn transformed_expectation_preserves_weights_bounds_and_reparameterization() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        let outputs = Tensor::<2>::zeros([4, 6], &device).require_grad();
        let distribution =
            TransformedDistribution::<GaussianDistribution, TanhTransform>::default();
        let terms = distribution
            .expectation(outputs.clone(), NonZeroUsize::new(3).unwrap())
            .unwrap();
        assert_eq!(terms.actions().dims(), [4, 3, 3]);
        assert!(terms.actions().clone().abs().max().into_scalar::<f32>() <= 1.0);
        assert_eq!(
            terms
                .weights()
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            vec![1.0 / 3.0; 12]
        );
        assert!(
            terms
                .log_probabilities()
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
        let gradients = terms.actions().clone().sum().backward();
        let gradient = outputs
            .grad(&gradients)
            .unwrap()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        for row in gradient.chunks(6) {
            assert!(row[..3].iter().all(|value| *value > 0.0));
        }
    }

    #[test]
    fn component_errors_remain_distinguishable() {
        let device = Device::flex();
        let distribution =
            TransformedDistribution::<GaussianDistribution, TanhTransform>::default();
        assert!(matches!(
            distribution.sample(Tensor::zeros([1, 3], &device)),
            Err(TransformedDistributionError::DistributionError(
                GaussianDistributionError::InvalidOutputWidth { output_width: 3 }
            ))
        ));
        let affine = AffineTransform::<1>::new(
            Tensor::from_floats([1.0, 1.0, 1.0], &device),
            Tensor::zeros([3], &device),
        )
        .unwrap();
        let distribution = TransformedDistribution::new(GaussianDistribution::default(), affine);
        assert!(matches!(
            distribution.mode(Tensor::zeros([1, 4], &device)),
            Err(TransformedDistributionError::TransformError(
                AffineTransformError::TensorError(DistributionTensorError::ShapeMismatch { .. })
            ))
        ));
    }

    #[test]
    fn corrected_log_density_keeps_the_reparameterized_mean_gradient() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        let outputs = Tensor::<2>::zeros([1, 4], &device).require_grad();
        let distribution =
            TransformedDistribution::<GaussianDistribution, TanhTransform>::default();
        let terms = distribution
            .expectation(outputs.clone(), NonZeroUsize::MIN)
            .unwrap();
        let actions = terms
            .actions()
            .clone()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let gradients = terms.log_probabilities().clone().sum().backward();
        let gradients = outputs
            .grad(&gradients)
            .unwrap()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        for (gradient, action) in gradients[..2].iter().zip(actions) {
            assert!((gradient - 2.0 * action).abs() < 1e-5);
        }
    }

    #[test]
    fn transforms_preserve_high_rank_and_scalar_events() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex();
        let distribution = TransformedDistribution::new(
            GaussianDistribution::<6>::new([2, 1, 2, 1, 2]).unwrap(),
            TanhTransform,
        );
        let terms = distribution
            .expectation(
                Tensor::zeros([2, 16], &device),
                NonZeroUsize::new(3).unwrap(),
            )
            .unwrap();
        assert_eq!(terms.actions().dims(), [2, 3, 2, 1, 2, 1, 2]);
        assert_eq!(terms.log_probabilities().dims(), [2, 3]);
        assert!(
            terms
                .log_probabilities()
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
        let scalar = TransformedDistribution::new(
            GaussianDistribution::<1>::new([]).unwrap(),
            TanhTransform,
        );
        assert_eq!(
            scalar.mode(Tensor::zeros([2, 2], &device)).unwrap().dims(),
            [2]
        );
    }
}

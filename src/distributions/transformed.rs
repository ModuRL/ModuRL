use std::num::NonZeroUsize;

use candle_core::{DType, Tensor};

use super::{DifferentiableExpectation, DistEval, Distribution, ExpectationTerms};

/// An invertible, elementwise distribution transform.
pub trait DistributionTransform {
    type Error;

    /// Maps `input` shaped `[..., ...event_shape]` to an output with the same
    /// shape.
    fn forward(&self, input: &Tensor) -> Result<Tensor, Self::Error>;
    /// Maps `output` shaped `[..., ...event_shape]` back to an input with the
    /// same shape.
    fn inverse(&self, output: &Tensor) -> Result<Tensor, Self::Error>;

    /// Returns the elementwise `log |dy/dx|` at `input` and `output`, which
    /// must have the same shape `[..., ...event_shape]`.
    fn log_abs_det_jacobian(&self, input: &Tensor, output: &Tensor) -> Result<Tensor, Self::Error>;

    /// Alias for [`DistributionTransform::log_abs_det_jacobian`]. Both tensors
    /// must have the same shape `[..., ...event_shape]`.
    fn forward_log_abs_det_jacobian(
        &self,
        input: &Tensor,
        output: &Tensor,
    ) -> Result<Tensor, Self::Error> {
        self.log_abs_det_jacobian(input, output)
    }

    /// Constant change in target entropy introduced by this transform.
    fn target_entropy_adjustment(&self) -> Result<f64, Self::Error> {
        Ok(0.0)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TanhTransform;

impl DistributionTransform for TanhTransform {
    type Error = candle_core::Error;

    /// Applies tanh while preserving the arbitrary input shape `[...]`.
    fn forward(&self, input: &Tensor) -> Result<Tensor, Self::Error> {
        input.tanh()
    }

    /// Applies inverse tanh while preserving the arbitrary output shape
    /// `[...]`.
    fn inverse(&self, output: &Tensor) -> Result<Tensor, Self::Error> {
        let bounded = output.clamp(-1.0 + 1e-6, 1.0 - 1e-6)?;
        let numerator = (&bounded + 1.0)?;
        let denominator = (1.0 - &bounded)?;
        (numerator / denominator)?.log()? * 0.5
    }

    /// Returns an elementwise Jacobian with the same arbitrary shape `[...]`
    /// as `input` and `output`.
    fn log_abs_det_jacobian(
        &self,
        input: &Tensor,
        _output: &Tensor,
    ) -> Result<Tensor, Self::Error> {
        // Stable equivalent of log(1 - tanh(x)^2). Computing this from the
        // output loses both value accuracy and gradients once tanh saturates.
        let negative_double_input = (input * -2.0)?;
        let softplus = (&negative_double_input.clamp(0.0, f64::INFINITY)?
            + &negative_double_input
                .abs()?
                .neg()?
                .exp()?
                .affine(1.0, 1.0)?
                .log()?)?;
        (input.neg()?.affine(1.0, std::f64::consts::LN_2)? - softplus)? * 2.0
    }
}

/// A runtime-configurable elementwise affine transform `scale * x + shift`.
#[derive(Clone, Debug)]
pub struct AffineTransform {
    scale: Tensor,
    shift: Tensor,
}

#[derive(Debug)]
pub enum AffineTransformError {
    TensorError(candle_core::Error),
    ZeroScale,
    NonFiniteParameters,
    InvalidBounds,
}

impl From<candle_core::Error> for AffineTransformError {
    fn from(error: candle_core::Error) -> Self {
        Self::TensorError(error)
    }
}

impl AffineTransform {
    /// Creates a transform from `scale` and `shift` tensors with identical
    /// shape `event_shape`.
    pub fn new(scale: Tensor, shift: Tensor) -> Result<Self, AffineTransformError> {
        if scale.shape() != shift.shape() {
            return Err(AffineTransformError::TensorError(
                candle_core::Error::ShapeMismatchBinaryOp {
                    lhs: scale.shape().clone(),
                    rhs: shift.shape().clone(),
                    op: "affine transform scale and shift",
                },
            ));
        }
        if scale.eq(0.0)?.max_all()?.to_scalar::<u8>()? != 0 {
            return Err(AffineTransformError::ZeroScale);
        }
        let non_finite = scale
            .ne(&scale)?
            .maximum(&scale.abs()?.eq(f64::INFINITY)?)?
            .maximum(&shift.ne(&shift)?)?
            .maximum(&shift.abs()?.eq(f64::INFINITY)?)?;
        if non_finite.max_all()?.to_scalar::<u8>()? != 0 {
            return Err(AffineTransformError::NonFiniteParameters);
        }
        Ok(Self { scale, shift })
    }

    pub fn scale(&self) -> &Tensor {
        &self.scale
    }

    pub fn shift(&self) -> &Tensor {
        &self.shift
    }

    /// Maps `[-1, 1]` elementwise onto bounds `low` and `high`, which must have
    /// identical shape `event_shape`.
    pub fn from_bounds(low: &Tensor, high: &Tensor) -> Result<Self, AffineTransformError> {
        if low.shape() != high.shape() {
            return Err(AffineTransformError::TensorError(
                candle_core::Error::ShapeMismatchBinaryOp {
                    lhs: low.shape().clone(),
                    rhs: high.shape().clone(),
                    op: "affine transform bounds",
                },
            ));
        }
        let invalid = low
            .ne(low)?
            .maximum(&low.abs()?.eq(f64::INFINITY)?)?
            .maximum(&high.ne(high)?)?
            .maximum(&high.abs()?.eq(f64::INFINITY)?)?
            .maximum(&low.ge(high)?)?;
        if invalid.max_all()?.to_scalar::<u8>()? != 0 {
            return Err(AffineTransformError::InvalidBounds);
        }
        let scale = ((high - low)? * 0.5)?;
        let shift = ((high + low)? * 0.5)?;
        Self::new(scale, shift)
    }
}

impl DistributionTransform for AffineTransform {
    type Error = AffineTransformError;

    /// Transforms `input` shaped `[..., ...event_shape]`, preserving its shape.
    fn forward(&self, input: &Tensor) -> Result<Tensor, Self::Error> {
        let scale = self
            .scale
            .to_device(input.device())?
            .to_dtype(input.dtype())?;
        let shift = self
            .shift
            .to_device(input.device())?
            .to_dtype(input.dtype())?;
        Ok(input.broadcast_mul(&scale)?.broadcast_add(&shift)?)
    }

    /// Inverts `output` shaped `[..., ...event_shape]`, preserving its shape.
    fn inverse(&self, output: &Tensor) -> Result<Tensor, Self::Error> {
        let scale = self
            .scale
            .to_device(output.device())?
            .to_dtype(output.dtype())?;
        let shift = self
            .shift
            .to_device(output.device())?
            .to_dtype(output.dtype())?;
        Ok(output.broadcast_sub(&shift)?.broadcast_div(&scale)?)
    }

    /// Returns an elementwise Jacobian shaped `[..., ...event_shape]`, matching
    /// `input` and `output`.
    fn log_abs_det_jacobian(
        &self,
        input: &Tensor,
        _output: &Tensor,
    ) -> Result<Tensor, Self::Error> {
        let scale = self
            .scale
            .to_device(input.device())?
            .to_dtype(input.dtype())?;
        Ok(scale.abs()?.log()?.broadcast_as(input.shape())?)
    }

    fn target_entropy_adjustment(&self) -> Result<f64, Self::Error> {
        let adjustment = self.scale.abs()?.log()?.sum_all()?;
        match adjustment.dtype() {
            DType::F64 => Ok(adjustment.to_scalar::<f64>()?),
            _ => Ok(adjustment.to_dtype(DType::F32)?.to_scalar::<f32>()? as f64),
        }
    }
}

/// A distribution followed by an invertible action transform.
#[derive(Clone, Debug, Default)]
pub struct TransformedDistribution<D, T> {
    distribution: D,
    transform: T,
}

#[derive(Debug)]
pub enum TransformedDistributionError<DE, TE> {
    DistributionError(DE),
    TransformError(TE),
    TensorError(candle_core::Error),
}

impl<DE, TE> From<candle_core::Error> for TransformedDistributionError<DE, TE> {
    fn from(error: candle_core::Error) -> Self {
        Self::TensorError(error)
    }
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

/// Reduces `values` shaped `prefix_shape + event_shape` across every dimension
/// beginning at `first_event_dimension`, returning `prefix_shape`.
fn sum_event_dimensions(
    values: &Tensor,
    first_event_dimension: usize,
) -> candle_core::Result<Tensor> {
    let mut result = values.clone();
    while result.rank() > first_event_dimension {
        result = result.sum(result.rank() - 1)?;
    }
    Ok(result)
}

impl<D, T> Distribution for TransformedDistribution<D, T>
where
    D: Distribution,
    T: DistributionTransform,
{
    type Error = TransformedDistributionError<D::Error, T::Error>;

    /// Maps parameters `[batch, ...parameter_shape]` to transformed samples
    /// `[batch, ...event_shape]`.
    fn sample(&self, outputs: &Tensor) -> Result<Tensor, Self::Error> {
        let sample = self
            .distribution
            .sample(outputs)
            .map_err(TransformedDistributionError::DistributionError)?;
        self.transform
            .forward(&sample)
            .map_err(TransformedDistributionError::TransformError)
    }

    /// Maps parameters `[batch, ...parameter_shape]` to transformed modes
    /// `[batch, ...event_shape]`.
    fn mode(&self, outputs: &Tensor) -> Result<Tensor, Self::Error> {
        let mode = self
            .distribution
            .mode(outputs)
            .map_err(TransformedDistributionError::DistributionError)?;
        self.transform
            .forward(&mode)
            .map_err(TransformedDistributionError::TransformError)
    }

    /// Evaluates transformed actions `[batch, ...event_shape]` under parameters
    /// `[batch, ...parameter_shape]`, returning statistics `[batch]`.
    fn dist_eval(&self, outputs: &Tensor, actions: &Tensor) -> Result<DistEval, Self::Error> {
        let base_actions = self
            .transform
            .inverse(actions)
            .map_err(TransformedDistributionError::TransformError)?;
        let base_evaluation = self
            .distribution
            .dist_eval(outputs, &base_actions)
            .map_err(TransformedDistributionError::DistributionError)?;
        let correction = sum_event_dimensions(
            &self
                .transform
                .log_abs_det_jacobian(&base_actions, actions)
                .map_err(TransformedDistributionError::TransformError)?,
            1,
        )?;
        let log_prob = (base_evaluation.log_prob() - correction)?;
        // In general transformed entropy has no closed form. `-log p(y)` is
        // the unbiased one-sample estimate needed by policy diagnostics.
        let entropy = log_prob.neg()?;
        Ok(DistEval::new(log_prob, entropy))
    }
}

impl<D, T> DifferentiableExpectation for TransformedDistribution<D, T>
where
    D: DifferentiableExpectation,
    T: DistributionTransform,
{
    /// Builds candidates `[batch, candidates, ...event_shape]` from parameters
    /// `[batch, ...parameter_shape]`.
    fn expectation(
        &self,
        outputs: &Tensor,
        samples: NonZeroUsize,
    ) -> Result<ExpectationTerms, Self::Error> {
        let terms = self
            .distribution
            .expectation(outputs, samples)
            .map_err(TransformedDistributionError::DistributionError)?;
        let (base_actions, base_log_probabilities, weights) = terms.into_parts();
        let actions = self
            .transform
            .forward(&base_actions)
            .map_err(TransformedDistributionError::TransformError)?;
        let correction = sum_event_dimensions(
            &self
                .transform
                .log_abs_det_jacobian(&base_actions, &actions)
                .map_err(TransformedDistributionError::TransformError)?,
            2,
        )?;
        let log_probabilities = (&base_log_probabilities - correction)?;
        Ok(ExpectationTerms::new(actions, log_probabilities, weights)?)
    }

    /// Computes target entropy from parameters
    /// `[batch, ...parameter_shape]`.
    fn default_target_entropy(&self, outputs: &Tensor) -> Result<f64, Self::Error> {
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
    use crate::distributions::GaussianDistribution;
    use candle_core::{Device, Tensor, Var};

    #[test]
    fn tanh_inverse_and_jacobian_are_consistent() {
        let transform = TanhTransform;
        let input =
            Tensor::from_vec(vec![-2.0f32, -0.5, 0.0, 0.5, 2.0], (1, 5), &Device::Cpu).unwrap();
        let output = transform.forward(&input).unwrap();
        let recovered = transform.inverse(&output).unwrap();
        let error = (&input - recovered)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(error < 1e-5);

        let jacobian = transform.log_abs_det_jacobian(&input, &output).unwrap();
        let expected = (1.0 - output.sqr().unwrap()).unwrap().log().unwrap();
        let error = (jacobian - expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(error < 1e-6);
    }

    #[test]
    fn tanh_jacobian_remains_accurate_and_differentiable_when_saturated() {
        let input = Var::from_vec(vec![-20.0f32, 20.0], (1, 2), &Device::Cpu).unwrap();
        let output = TanhTransform.forward(input.as_tensor()).unwrap();
        let jacobian = TanhTransform
            .log_abs_det_jacobian(input.as_tensor(), &output)
            .unwrap();
        let expected = 2.0 * (2.0_f32.ln() - 20.0);
        for value in jacobian.to_vec2::<f32>().unwrap()[0].iter() {
            assert!((value - expected).abs() < 1e-4, "{value} != {expected}");
        }
        let gradients = jacobian.sum_all().unwrap().backward().unwrap();
        let gradient = gradients
            .get(input.as_tensor())
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert!((gradient[0][0] - 2.0).abs() < 1e-5);
        assert!((gradient[0][1] + 2.0).abs() < 1e-5);
    }

    #[test]
    fn affine_transform_composes_with_tanh_bounds() {
        let scale = Tensor::from_vec(vec![2.0f32, 3.0], 2, &Device::Cpu).unwrap();
        let shift = Tensor::from_vec(vec![1.0f32, -1.0], 2, &Device::Cpu).unwrap();
        let affine = AffineTransform::new(scale, shift).unwrap();
        let input = Tensor::from_vec(vec![-1.0f32, 1.0], (1, 2), &Device::Cpu).unwrap();
        let squashed = TanhTransform.forward(&input).unwrap();
        let output = affine.forward(&squashed).unwrap();
        let recovered = affine.inverse(&output).unwrap();
        let error = (recovered - squashed)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(error < 1e-6);
        let values = output.to_vec2::<f32>().unwrap();
        assert!(values[0][0] >= -1.0 && values[0][0] <= 3.0);
        assert!(values[0][1] >= -4.0 && values[0][1] <= 2.0);
    }

    #[test]
    fn affine_transform_rejects_invalid_parameters_and_bounds() {
        let values = |values: Vec<f32>| Tensor::from_vec(values, 2, &Device::Cpu).unwrap();
        let mismatched = Tensor::zeros(1, DType::F32, &Device::Cpu).unwrap();
        assert!(matches!(
            AffineTransform::new(values(vec![1.0, 1.0]), mismatched),
            Err(AffineTransformError::TensorError(
                candle_core::Error::ShapeMismatchBinaryOp { .. }
            ))
        ));
        assert!(matches!(
            AffineTransform::new(values(vec![0.0, 1.0]), values(vec![0.0, 0.0])),
            Err(AffineTransformError::ZeroScale)
        ));
        assert!(matches!(
            AffineTransform::new(values(vec![f32::INFINITY, 1.0]), values(vec![0.0, 0.0])),
            Err(AffineTransformError::NonFiniteParameters)
        ));
        assert!(matches!(
            AffineTransform::from_bounds(&values(vec![-1.0, 2.0]), &values(vec![1.0, 2.0])),
            Err(AffineTransformError::InvalidBounds)
        ));
        assert!(matches!(
            AffineTransform::from_bounds(&values(vec![-1.0, 3.0]), &values(vec![1.0, 2.0])),
            Err(AffineTransformError::InvalidBounds)
        ));
    }

    #[test]
    fn affine_scale_adjusts_log_probability_and_default_target_entropy() {
        let scale = Tensor::from_vec(vec![2.0f32, 4.0], 2, &Device::Cpu).unwrap();
        let shift = Tensor::from_vec(vec![1.0f32, -1.0], 2, &Device::Cpu).unwrap();
        let affine = AffineTransform::new(scale, shift).unwrap();
        let base = TransformedDistribution::<GaussianDistribution, TanhTransform>::default();
        let transformed = TransformedDistribution::new(base.clone(), affine.clone());
        let outputs = Tensor::zeros((1, 4), DType::F32, &Device::Cpu).unwrap();
        let base_actions = Tensor::from_vec(vec![0.1f32, -0.2], (1, 2), &Device::Cpu).unwrap();
        let actions = affine.forward(&base_actions).unwrap();
        let base_log_probability = base
            .dist_eval(&outputs, &base_actions)
            .unwrap()
            .log_prob()
            .to_vec1::<f32>()
            .unwrap()[0];
        let transformed_log_probability = transformed
            .dist_eval(&outputs, &actions)
            .unwrap()
            .log_prob()
            .to_vec1::<f32>()
            .unwrap()[0];
        let log_scale = 8.0_f32.ln();
        assert!((transformed_log_probability - (base_log_probability - log_scale)).abs() < 1e-5);
        assert!(
            (transformed.default_target_entropy(&outputs).unwrap()
                - (base.default_target_entropy(&outputs).unwrap() + log_scale as f64))
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn squashed_gaussian_expectation_stays_in_bounds_and_keeps_weights() {
        let outputs = Tensor::zeros((4, 6), candle_core::DType::F32, &Device::Cpu).unwrap();
        let distribution =
            TransformedDistribution::<GaussianDistribution, TanhTransform>::default();
        let terms = distribution
            .expectation(&outputs, NonZeroUsize::MIN)
            .unwrap();
        let max = terms
            .actions()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(max <= 1.0);
        assert_eq!(
            terms.weights().to_vec2::<f32>().unwrap(),
            vec![vec![1.0]; 4]
        );
        assert!(
            terms
                .log_probabilities()
                .to_vec2::<f32>()
                .unwrap()
                .iter()
                .flatten()
                .all(|x| x.is_finite())
        );
    }

    #[test]
    fn transformed_distribution_preserves_component_error_kinds() {
        let distribution =
            TransformedDistribution::<GaussianDistribution, TanhTransform>::default();
        let outputs = Tensor::zeros((1, 3), DType::F32, &Device::Cpu).unwrap();
        assert!(matches!(
            distribution.sample(&outputs),
            Err(TransformedDistributionError::DistributionError(
                crate::distributions::GaussianDistributionError::InvalidOutputWidth {
                    output_width: 3
                }
            ))
        ));
    }

    #[test]
    fn squashed_gaussian_preserves_five_dimensional_action_events() {
        let distribution = TransformedDistribution::new(
            GaussianDistribution::new(vec![2, 1, 2, 1, 2]).unwrap(),
            TanhTransform,
        );
        let outputs = Tensor::zeros((2, 16), candle_core::DType::F32, &Device::Cpu).unwrap();
        let terms = distribution
            .expectation(&outputs, NonZeroUsize::new(3).unwrap())
            .unwrap();

        assert_eq!(terms.actions().dims(), &[2, 3, 2, 1, 2, 1, 2]);
        assert_eq!(terms.log_probabilities().dims(), &[2, 3]);
        assert!(
            terms
                .log_probabilities()
                .to_vec2::<f32>()
                .unwrap()
                .iter()
                .flatten()
                .all(|value| value.is_finite())
        );
    }
}

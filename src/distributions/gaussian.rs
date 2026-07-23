use std::num::NonZeroUsize;

use candle_core::Tensor;

use crate::distributions::{DifferentiableExpectation, DistEval, Distribution, ExpectationTerms};

/// Operations for batches of independent Gaussian distributions.
///
/// `GaussianDistribution` expects a rank-2 parameter tensor with shape
/// `[batch_size, 2 * action_size]`. The first half of dimension 1 contains the
/// action means. The second half contains their log standard deviations. By
/// default, sampling returns `[batch_size, action_size]`. A configured action
/// shape reshapes those flat parameters and samples to
/// `[batch_size, ...action_shape]`; log probability and entropy always reduce
/// every action-event dimension to `[batch_size]`.
///
/// The distribution does not squash or clamp samples. A bounded [`BoxSpace`]
/// clamps the separate action tensor sent to an environment while PPO retains
/// the original sample for probability calculations.
///
/// [`BoxSpace`]: crate::spaces::BoxSpace
#[derive(Clone, Debug, Default)]
pub struct GaussianDistribution {
    action_shape: Option<Vec<usize>>,
}

#[derive(Debug)]
pub enum GaussianDistributionError {
    TensorError(candle_core::Error),
    ActionShapeTooLarge,
    ZeroActionDimension,
    InvalidOutputWidth { output_width: usize },
}

impl From<candle_core::Error> for GaussianDistributionError {
    fn from(error: candle_core::Error) -> Self {
        Self::TensorError(error)
    }
}

impl GaussianDistribution {
    /// Configures the shape of one action event. Model parameters remain flat.
    pub fn new(action_shape: Vec<usize>) -> Result<Self, GaussianDistributionError> {
        let event_size = action_shape.iter().try_fold(1usize, |size, dimension| {
            size.checked_mul(*dimension)
                .ok_or(GaussianDistributionError::ActionShapeTooLarge)
        })?;
        if event_size == 0 {
            return Err(GaussianDistributionError::ZeroActionDimension);
        }
        Ok(Self {
            action_shape: Some(action_shape),
        })
    }

    pub fn action_shape(&self) -> Option<&[usize]> {
        self.action_shape.as_deref()
    }

    /// Splits `outputs` shaped `[batch_size, 2 * event_size]` into mean and
    /// log-standard-deviation tensors shaped `[batch_size, ...action_shape]`.
    fn parameters(&self, outputs: &Tensor) -> Result<(Tensor, Tensor), GaussianDistributionError> {
        let (_, output_size) = outputs.dims2()?;
        if output_size == 0 || output_size % 2 != 0 {
            return Err(GaussianDistributionError::InvalidOutputWidth {
                output_width: output_size,
            });
        }
        let half = output_size / 2;
        let Some(action_shape) = &self.action_shape else {
            return Ok((outputs.narrow(1, 0, half)?, outputs.narrow(1, half, half)?));
        };
        let event_size = action_shape.iter().product::<usize>();
        if event_size != half {
            return Err(GaussianDistributionError::TensorError(
                candle_core::Error::UnexpectedShape {
                    msg: "Gaussian model output must describe the configured action shape".into(),
                    expected: (outputs.dim(0)?, event_size * 2).into(),
                    got: outputs.shape().clone(),
                },
            ));
        }
        let mut batched_shape = Vec::with_capacity(action_shape.len() + 1);
        batched_shape.push(outputs.dim(0)?);
        batched_shape.extend(action_shape);
        Ok((
            outputs
                .narrow(1, 0, half)?
                .reshape(batched_shape.as_slice())?,
            outputs
                .narrow(1, half, half)?
                .reshape(batched_shape.as_slice())?,
        ))
    }
}

/// Reduces `values` shaped `[batch_size, ...event_shape]` to `[batch_size]`.
fn sum_event_dimensions(values: &Tensor) -> candle_core::Result<Tensor> {
    let mut result = values.clone();
    while result.rank() > 1 {
        result = result.sum(result.rank() - 1)?;
    }
    Ok(result)
}

impl Distribution for GaussianDistribution {
    type Error = GaussianDistributionError;

    /// Samples actions `[batch_size, ...action_shape]` from parameters
    /// `[batch_size, 2 * event_size]`.
    fn sample(&self, outputs: &Tensor) -> Result<Tensor, Self::Error> {
        let (action_mean, action_log_std) = self.parameters(outputs)?;
        let mut unit_normal = Tensor::randn(0.0, 1.0, action_mean.dims(), action_mean.device())?;
        let dtype = action_mean.dtype();
        unit_normal = unit_normal.to_dtype(dtype)?;
        let action_std = action_log_std.exp()?;

        Ok((action_mean + unit_normal * action_std)?)
    }

    /// Returns means `[batch_size, ...action_shape]` from parameters
    /// `[batch_size, 2 * event_size]`.
    fn mode(&self, outputs: &Tensor) -> Result<Tensor, Self::Error> {
        Ok(self.parameters(outputs)?.0)
    }

    /// Evaluates actions `[batch_size, ...action_shape]` under parameters
    /// `[batch_size, 2 * event_size]`, returning statistics `[batch_size]`.
    fn dist_eval(&self, outputs: &Tensor, actions: &Tensor) -> Result<DistEval, Self::Error> {
        let (action_mean, action_log_std) = self.parameters(outputs)?;
        let action_std = action_log_std.exp()?;
        let action = actions;
        // Break down each component
        let action_diff = (action.clone() - action_mean)?;
        let normalized_diff = (action_diff.powf(2.0)? / action_std.powf(2.0))?;
        let log_det_term = (2.0 * &action_log_std)?;
        // This term is applied per action component before the reduction.
        let normalization = (2.0 * std::f64::consts::PI).ln();

        let total_inside = ((&normalized_diff + &log_det_term)? + normalization)?;
        let log_prob = -0.5 * total_inside;

        let log_prob = sum_event_dimensions(&log_prob?)?;
        // H[N(mu, sigma^2)] = log(sigma) + 0.5 * log(2*pi*e)
        // for each independent action component.
        let entropy = sum_event_dimensions(
            &(&action_log_std + 0.5 * (std::f64::consts::PI * 2.0 * std::f64::consts::E).ln())?,
        )?;

        Ok(DistEval::new(log_prob, entropy))
    }
}

impl DifferentiableExpectation for GaussianDistribution {
    /// Samples candidates `[batch_size, samples, ...action_shape]` from
    /// parameters `[batch_size, 2 * event_size]`.
    fn expectation(
        &self,
        outputs: &Tensor,
        samples: NonZeroUsize,
    ) -> Result<ExpectationTerms, Self::Error> {
        let sample_count = samples.get();
        let mut actions = Vec::with_capacity(sample_count);
        let mut log_probabilities = Vec::with_capacity(sample_count);
        for _ in 0..sample_count {
            let sample = self.sample(outputs)?;
            log_probabilities.push(self.dist_eval(outputs, &sample)?.log_prob().clone());
            actions.push(sample);
        }
        let actions = Tensor::stack(&actions, 1)?;
        let log_probabilities = Tensor::stack(&log_probabilities, 1)?;
        let weights = Tensor::ones(
            log_probabilities.shape(),
            log_probabilities.dtype(),
            log_probabilities.device(),
        )?
        .affine(1.0 / sample_count as f64, 0.0)?;
        Ok(ExpectationTerms::new(actions, log_probabilities, weights)?)
    }

    /// Computes target entropy from parameters
    /// `[batch_size, 2 * event_size]`.
    fn default_target_entropy(&self, outputs: &Tensor) -> Result<f64, Self::Error> {
        let event_size = self.parameters(outputs)?.0.elem_count() / outputs.dim(0)?;
        Ok(-(event_size as f64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn diagonal_gaussian_entropy_and_log_prob_match_closed_form() {
        const ACTION_DIMS: usize = 3;
        let log_std = 0.5_f32.ln();
        let outputs = Tensor::from_vec(
            [vec![0.0_f32; ACTION_DIMS], vec![log_std; ACTION_DIMS]].concat(),
            (1, ACTION_DIMS * 2),
            &Device::Cpu,
        )
        .unwrap();
        let actions =
            Tensor::zeros((1, ACTION_DIMS), candle_core::DType::F32, &Device::Cpu).unwrap();

        let distribution = GaussianDistribution::default();
        let evaluation = distribution.dist_eval(&outputs, &actions).unwrap();
        let actual_entropy = evaluation.entropy().to_vec1::<f32>().unwrap()[0];
        let actual_log_prob = evaluation.log_prob().to_vec1::<f32>().unwrap()[0];

        let component_entropy =
            log_std + 0.5 * (2.0 * std::f32::consts::PI * std::f32::consts::E).ln();
        let expected_entropy = ACTION_DIMS as f32 * component_entropy;
        let component_log_prob = -log_std - 0.5 * (2.0 * std::f32::consts::PI).ln();
        let expected_log_prob = ACTION_DIMS as f32 * component_log_prob;

        assert!((actual_entropy - expected_entropy).abs() < 1e-6);
        assert!((actual_log_prob - expected_log_prob).abs() < 1e-6);
    }

    #[test]
    fn mode_is_the_configured_mean() {
        let outputs = Tensor::from_vec(
            vec![0.25_f32, -0.5, 0.75, -1.0, 0.0, 1.0],
            (1, 6),
            &Device::Cpu,
        )
        .unwrap();
        let distribution = GaussianDistribution::default();

        assert_eq!(
            distribution
                .mode(&outputs)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            vec![vec![0.25, -0.5, 0.75]]
        );
    }

    #[test]
    fn differentiable_expectation_is_one_reparameterized_sample() {
        let mean = candle_core::Var::from_vec(vec![0.0f32, 0.0], (1, 2), &Device::Cpu).unwrap();
        let log_std = Tensor::zeros((1, 2), candle_core::DType::F32, &Device::Cpu).unwrap();
        let outputs = Tensor::cat(&[mean.as_tensor(), &log_std], 1).unwrap();
        let distribution = GaussianDistribution::default();
        let terms = distribution
            .expectation(&outputs, NonZeroUsize::MIN)
            .unwrap();

        assert_eq!(terms.actions().dims(), &[1, 1, 2]);
        assert_eq!(terms.log_probabilities().dims(), &[1, 1]);
        assert_eq!(terms.weights().to_vec2::<f32>().unwrap(), vec![vec![1.0]]);
        assert_eq!(distribution.default_target_entropy(&outputs).unwrap(), -2.0);

        let loss = terms.actions().sum_all().unwrap();
        let gradients = loss.backward().unwrap();
        assert_eq!(
            gradients
                .get(mean.as_tensor())
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            vec![vec![1.0, 1.0]]
        );
    }

    #[test]
    fn expectation_uses_requested_sample_count_and_uniform_weights() {
        let outputs = Tensor::zeros((2, 4), candle_core::DType::F32, &Device::Cpu).unwrap();
        let distribution = GaussianDistribution::default();
        let terms = distribution
            .expectation(&outputs, NonZeroUsize::new(4).unwrap())
            .unwrap();

        assert_eq!(terms.actions().dims(), &[2, 4, 2]);
        assert_eq!(terms.log_probabilities().dims(), &[2, 4]);
        assert_eq!(
            terms.weights().to_vec2::<f32>().unwrap(),
            vec![vec![0.25; 4]; 2]
        );
    }

    #[test]
    fn configured_event_shape_preserves_five_action_dimensions() {
        let action_shape = vec![2, 1, 2, 1, 2];
        let event_size = action_shape.iter().product::<usize>();
        let distribution = GaussianDistribution::new(action_shape).unwrap();
        let outputs =
            Tensor::zeros((3, event_size * 2), candle_core::DType::F32, &Device::Cpu).unwrap();

        let actions = distribution.sample(&outputs).unwrap();
        assert_eq!(actions.dims(), &[3, 2, 1, 2, 1, 2]);
        let evaluation = distribution.dist_eval(&outputs, &actions).unwrap();
        assert_eq!(evaluation.log_prob().dims(), &[3]);
        assert_eq!(evaluation.entropy().dims(), &[3]);
        assert_eq!(distribution.default_target_entropy(&outputs).unwrap(), -8.0);

        let terms = distribution
            .expectation(&outputs, NonZeroUsize::new(4).unwrap())
            .unwrap();
        assert_eq!(terms.actions().dims(), &[3, 4, 2, 1, 2, 1, 2]);
        assert_eq!(terms.log_probabilities().dims(), &[3, 4]);
        assert_eq!(terms.weights().dims(), &[3, 4]);
    }

    #[test]
    fn configured_event_shape_must_match_model_output_width() {
        let distribution = GaussianDistribution::new(vec![2, 3]).unwrap();
        let outputs = Tensor::zeros((1, 10), candle_core::DType::F32, &Device::Cpu).unwrap();
        assert!(matches!(
            distribution.sample(&outputs),
            Err(GaussianDistributionError::TensorError(
                candle_core::Error::UnexpectedShape { .. }
            ))
        ));
        assert!(matches!(
            GaussianDistribution::new(vec![2, 0, 3]),
            Err(GaussianDistributionError::ZeroActionDimension)
        ));
        let odd_outputs = Tensor::zeros((1, 3), candle_core::DType::F32, &Device::Cpu).unwrap();
        assert!(matches!(
            GaussianDistribution::default().sample(&odd_outputs),
            Err(GaussianDistributionError::InvalidOutputWidth { output_width: 3 })
        ));
    }
}

use burn::{
    module::{Module, ModuleVisitor, Param, ParamId},
    optim::GradientsParams,
    tensor::{DType, Tensor, TensorReadError},
};
use std::collections::HashSet;

/// Clips one global norm across the model's parameter gradients, preserving every parameter shape, dtype, and device.
/// Gradients must come from this model's backward pass. Missing gradients are skipped; shared parameters count once.
/// A tuple of models shares one norm. Gradients outside those models remain unchanged.
/// Returns the original norm as a host `f32`; reading norm contributions can fail.
pub(crate) fn clip_gradients<M: Module>(
    model: &M,
    gradients: &mut GradientsParams,
    max_norm: f32,
) -> Result<f32, TensorReadError> {
    let mut norms = GradientNorms {
        gradients,
        visited: HashSet::new(),
        groups: Vec::new(),
    };
    model.visit(&mut norms);
    let mut norm_sqrs = Vec::new();
    for group in norms.groups {
        // Read each device's scalar contributions together to limit device-to-host transfers.
        norm_sqrs.extend(
            Tensor::<1>::cat(group, 0)
                .try_into_data_as::<f64>()?
                .try_to_vec::<f64>()?,
        );
    }
    // Sort contributions because parameter traversal order must not change floating-point accumulation.
    norm_sqrs.sort_by(f64::total_cmp);
    let total_norm = norm_sqrs.into_iter().sum::<f64>().sqrt();
    if total_norm > f64::from(max_norm) {
        model.visit(&mut GradientScale {
            gradients,
            visited: HashSet::new(),
            scale: f64::from(max_norm) / (total_norm + 1e-6),
        });
    }
    Ok(total_norm as f32)
}

struct GradientNorms<'a> {
    gradients: &'a GradientsParams,
    visited: HashSet<ParamId>,
    groups: Vec<Vec<Tensor<1>>>,
}

impl ModuleVisitor for GradientNorms<'_> {
    /// Reduces a parameter gradient of rank `D` to an F64 squared norm `[1]` on its device.
    /// Computes squares and their sum in the gradient's dtype, then casts the scalar contribution to F64.
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        if !self.visited.insert(param.id) {
            return;
        }
        let Some(gradient) = self.gradients.get::<D>(param.id) else {
            return;
        };
        let norm_sq = gradient.square().sum().cast(DType::F64);
        if let Some(group) = self
            .groups
            .iter_mut()
            .find(|group| group[0].device() == norm_sq.device())
        {
            group.push(norm_sq);
        } else {
            self.groups.push(vec![norm_sq]);
        }
    }
}

struct GradientScale<'a> {
    gradients: &'a mut GradientsParams,
    visited: HashSet<ParamId>,
    scale: f64,
}

impl ModuleVisitor for GradientScale<'_> {
    /// Scales a parameter gradient of rank `D`, preserving its axes, dtype, and device.
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        if !self.visited.insert(param.id) {
            return;
        }
        if let Some(gradient) = self.gradients.remove::<D>(param.id) {
            self.gradients.register(param.id, gradient * self.scale);
        }
    }
}

/// Normalizes all elements of a rank-`D` Float tensor, preserving every axis, dtype, device, and gradient path.
/// PPO inputs such as `[batch_size, 1]` or `[time, num_envs, 1]` use one mean and standard deviation across all entries.
/// Uses the sample standard deviation with denominator `max(element_count, 2) - 1` and adds `1e-8` after the square root.
pub(crate) fn normalize_tensor<const D: usize>(t: &Tensor<D>) -> Tensor<D> {
    let mean = t.clone().mean().reshape([1; D]);
    let diff = t.clone() - mean;
    let n = t.shape().num_elements().max(2) as f64;
    let std = (diff.clone().square().sum() / (n - 1.0)).sqrt();
    diff / (std + 1e-8).reshape([1; D])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, TensorData};

    #[test]
    fn normalization_preserves_shape_precision_and_sample_variance() {
        let device = Device::flex();
        for dtype in [DType::F32, DType::F64] {
            let input = Tensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], (&device, dtype));
            let output = normalize_tensor(&input);
            assert_eq!(output.dims(), [2, 2]);
            assert_eq!(output.dtype(), dtype);
            assert_eq!(output.device(), device);
            let actual = output
                .cast(DType::F64)
                .into_data()
                .try_to_vec::<f64>()
                .unwrap();
            let denominator = (5.0f64 / 3.0).sqrt() + 1e-8;
            for (actual, centered) in actual.into_iter().zip([-1.5, -0.5, 0.5, 1.5]) {
                assert!((actual - centered / denominator).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn normalization_handles_singletons_constants_and_empty_tensors() {
        let device = Device::flex();
        for values in [vec![4.0f32], vec![4.0; 3], vec![]] {
            let size = values.len();
            let input = Tensor::<1>::from_data(TensorData::new(values, [size]), &device);
            let output = normalize_tensor(&input);
            assert_eq!(output.dims(), [size]);
            assert_eq!(
                output.into_data().try_to_vec::<f32>().unwrap(),
                vec![0.0; size]
            );
        }
    }

    #[test]
    fn normalization_preserves_gradients() {
        let device = Device::flex().autodiff();
        let input =
            Tensor::<1>::from_data([1.0f64, 2.0, 4.0], (&device, DType::F64)).require_grad();
        let weights = Tensor::<1>::from_data([1.0f64, 0.0, 0.0], (&device, DType::F64));
        let gradients = (normalize_tensor(&input) * weights).sum().backward();
        let actual = input
            .grad(&gradients)
            .unwrap()
            .into_data()
            .try_to_vec::<f64>()
            .unwrap();
        let step = 1e-5;
        for index in 0..3 {
            let evaluate = |offset: f64| {
                let mut values = [1.0, 2.0, 4.0];
                values[index] += offset;
                let mean = values.iter().sum::<f64>() / 3.0;
                let variance = values
                    .iter()
                    .map(|value| (value - mean).powi(2))
                    .sum::<f64>()
                    / 2.0;
                (values[0] - mean) / (variance.sqrt() + 1e-8)
            };
            let expected = (evaluate(step) - evaluate(-step)) / (2.0 * step);
            assert!((actual[index] - expected).abs() < 1e-8);
        }
    }

    #[test]
    fn gradient_clipping_matches_pytorch_epsilon() {
        let device = Device::flex().autodiff();
        for dtype in [DType::F32, DType::F64] {
            let variable =
                Param::from_tensor(Tensor::<1>::from_data([3.0f64, 4.0], (&device, dtype)));
            let loss = variable.val().square().sum();
            let mut gradients = GradientsParams::from_grads(loss.backward(), &variable);
            let gradient_device = gradients.get::<1>(variable.id).unwrap().device();
            let original_norm = clip_gradients(&variable, &mut gradients, 5.0).unwrap();
            let clipped = gradients.get::<1>(variable.id).unwrap();
            assert_eq!(clipped.dims(), [2]);
            assert_eq!(clipped.dtype(), dtype);
            assert_eq!(clipped.device(), gradient_device);
            let clipped = clipped
                .cast(DType::F64)
                .into_data()
                .try_to_vec::<f64>()
                .unwrap();
            let expected_scale = 5.0 / (10.0 + 1e-6);
            assert!((original_norm - 10.0).abs() < 1e-6);
            assert!((clipped[0] - 6.0 * expected_scale).abs() < 1e-6);
            assert!((clipped[1] - 8.0 * expected_scale).abs() < 1e-6);
        }
    }

    #[test]
    fn gradient_clipping_ignores_detached_operand_gradients() {
        let device = Device::flex().autodiff();
        let parameter = Param::from_tensor(Tensor::<1>::from_data([1000.0f32, 1000.0], &device));
        let operand = Tensor::<1>::ones([2], &device).require_grad().detach();
        let loss = (parameter.val() * operand).sum();
        let mut gradients = GradientsParams::from_grads(loss.backward(), &parameter);
        let original_norm = clip_gradients(&parameter, &mut gradients, 1.0).unwrap();
        let clipped = gradients
            .get::<1>(parameter.id)
            .unwrap()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let expected_norm = 2.0f32.sqrt();
        let expected_scale = 1.0 / (expected_norm + 1e-6);

        assert!((original_norm - expected_norm).abs() < 1e-6);
        assert!((clipped[0] - expected_scale).abs() < 1e-6);
        assert!((clipped[1] - expected_scale).abs() < 1e-6);
    }

    #[test]
    fn gradient_clipping_uses_one_norm_for_multiple_models_and_shared_parameters() {
        let device = Device::flex().autodiff();
        let first = Param::from_tensor(Tensor::<1>::ones([1], &device));
        let second = Param::from_tensor(Tensor::<2>::ones([1, 1], &device));
        let models = (first.clone(), second.clone(), first.clone());
        let loss = first.val().sum() * 3.0 + second.val().sum() * 4.0;
        let mut gradients = GradientsParams::from_grads(loss.backward(), &models);
        assert_eq!(clip_gradients(&models, &mut gradients, 4.0).unwrap(), 5.0);
        let scale = 4.0 / (5.0 + 1e-6);
        assert!(
            (gradients.get::<1>(first.id).unwrap().into_scalar::<f64>() - 3.0 * scale).abs() < 1e-6
        );
        assert!(
            (gradients.get::<2>(second.id).unwrap().into_scalar::<f64>() - 4.0 * scale).abs()
                < 1e-6
        );
    }

    #[test]
    fn gradient_clipping_is_deterministic_across_traversal_order_and_parameter_ids() {
        let device = Device::flex().autodiff();
        let mut coefficients = vec![1e8f64];
        coefficients.extend([1.0; 32]);

        // A large contribution followed by small ones loses precision unless accumulation order is fixed.
        let forward_sum = coefficients.iter().map(|value| value * value).sum::<f64>();
        let reverse_sum = coefficients
            .iter()
            .rev()
            .map(|value| value * value)
            .sum::<f64>();
        assert_ne!(forward_sum.to_bits(), reverse_sum.to_bits());

        let evaluate = |order: usize| {
            let parameters = coefficients
                .iter()
                .map(|_| Param::from_tensor(Tensor::<1>::ones([1], (&device, DType::F64))))
                .collect::<Vec<_>>();
            let loss = parameters.iter().zip(&coefficients).fold(
                Tensor::<1>::zeros([1], (&device, DType::F64)),
                |sum, (parameter, coefficient)| sum + parameter.val().sum() * *coefficient,
            );
            let mut models = parameters.clone();
            match order {
                0 => {}
                1 => models.reverse(),
                2 => models.rotate_left(9),
                3 => models.extend(parameters.iter().rev().cloned()),
                _ => unreachable!(),
            }
            let mut gradients = GradientsParams::from_grads(loss.backward(), &models);
            let norm = clip_gradients(&models.clone(), &mut gradients, 0.5).unwrap();
            let clipped = parameters
                .iter()
                .map(|parameter| {
                    gradients
                        .get::<1>(parameter.id)
                        .unwrap()
                        .into_scalar::<f64>()
                        .to_bits()
                })
                .collect::<Vec<_>>();
            (norm.to_bits(), clipped)
        };

        let expected = evaluate(0);
        for _ in 0..3 {
            for order in 0..4 {
                assert_eq!(evaluate(order), expected);
            }
        }
    }

    #[test]
    fn gradient_clipping_skips_missing_gradients_and_leaves_other_parameters_unchanged() {
        let device = Device::flex().autodiff();
        let selected = Param::from_tensor(Tensor::<1>::ones([1], &device));
        let unused = Param::from_tensor(Tensor::<1>::ones([1], &device));
        let other = Param::from_tensor(Tensor::<1>::ones([1], &device));
        let loss = selected.val().sum() * 3.0 + other.val().sum() * 100.0;
        let mut gradients = GradientsParams::from_grads(
            loss.backward(),
            &(selected.clone(), unused.clone(), other.clone()),
        );
        assert_eq!(
            clip_gradients(&(selected.clone(), unused.clone()), &mut gradients, 2.0).unwrap(),
            3.0
        );
        assert!(gradients.get::<1>(unused.id).is_none());
        assert_eq!(
            gradients.get::<1>(other.id).unwrap().into_scalar::<f32>(),
            100.0
        );
        assert_eq!(clip_gradients(&unused, &mut gradients, 2.0).unwrap(), 0.0);
        let before = gradients.get::<1>(selected.id).unwrap().into_data();
        clip_gradients(&selected, &mut gradients, 10.0).unwrap();
        assert_eq!(gradients.get::<1>(selected.id).unwrap().into_data(), before);
    }
}

use burn::tensor::{Bool, Tensor};

/// Builds one-step Bellman targets from `rewards`, `terminated`, and
/// `next_values`, all shaped `[batch]`, returning `[batch]`.
///
/// `terminated` is a boolean mask: true means the episode terminated and the
/// next value must not contribute. Truncation alone should not set this mask.
/// Rewards and next values remain differentiable; callers control detachment.
pub fn bellman_targets(
    rewards: Tensor<1>,
    terminated: Tensor<1, Bool>,
    next_values: Tensor<1>,
    gamma: f64,
) -> Tensor<1> {
    rewards + next_values.mask_fill(terminated, 0.0) * gamma
}

/// Returns a clipped value loss shaped `[1]` from `prediction`, `target`, and
/// `anchor`, all shaped `[batch]`.
///
/// The prediction update is clipped around `anchor`, and the larger of the
/// clipped and unclipped squared errors is averaged. This function does not
/// detach any input.
pub fn clipped_value_loss(
    prediction: Tensor<1>,
    target: Tensor<1>,
    anchor: Tensor<1>,
    epsilon: f64,
) -> Tensor<1> {
    let delta = (prediction.clone() - anchor.clone()).clamp(-epsilon, epsilon);
    let clipped = anchor + delta;
    let loss = (prediction - target.clone()).square();
    let clipped_loss = (clipped - target).square();
    loss.max_pair(clipped_loss).mean()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;

    #[test]
    fn bellman_targets_mask_termination_without_implicit_detachment() {
        let device = Device::flex().autodiff();
        let rewards = Tensor::<1>::from_floats([1.0, 2.0], &device).require_grad();
        let terminated = Tensor::<1, Bool>::from_data([false, true], &device);
        let next_values = Tensor::<1>::from_floats([10.0, 20.0], &device).require_grad();
        let targets = bellman_targets(rewards.clone(), terminated, next_values.clone(), 0.9);

        assert_eq!(
            targets.clone().into_data().try_to_vec::<f32>().unwrap(),
            vec![10.0, 2.0]
        );
        let gradients = targets.sum().backward();
        assert_eq!(
            rewards
                .grad(&gradients)
                .unwrap()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            vec![1.0, 1.0]
        );
        assert_eq!(
            next_values
                .grad(&gradients)
                .unwrap()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            vec![0.9, 0.0]
        );
    }

    #[test]
    fn clipped_value_loss_leaves_gradient_boundaries_to_the_caller() {
        let device = Device::flex().autodiff();
        let prediction = Tensor::<1>::from_floats([0.0, 2.0], &device).require_grad();
        let anchor = Tensor::<1>::from_floats([2.0, 0.0], &device).require_grad();
        let target = Tensor::<1>::from_floats([0.0, 0.0], &device).require_grad();
        let loss = clipped_value_loss(prediction.clone(), target.clone(), anchor.clone(), 0.5);

        assert_eq!(loss.dims(), [1]);
        assert_eq!(
            loss.clone().into_data().try_to_vec::<f32>().unwrap(),
            vec![3.125]
        );
        let gradients = loss.backward();
        for (tensor, expected) in [
            (prediction, vec![0.0, 2.0]),
            (anchor, vec![1.5, 0.0]),
            (target, vec![-1.5, -2.0]),
        ] {
            assert_eq!(
                tensor
                    .grad(&gradients)
                    .unwrap()
                    .into_data()
                    .try_to_vec::<f32>()
                    .unwrap(),
                expected
            );
        }
    }
}

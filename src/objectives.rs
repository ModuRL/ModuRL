use crate::tensor_operations::clamp_with_boundary_gradients;
use burn::tensor::{Bool, Tensor};

/// Builds one-step Bellman targets from `rewards`, `terminated`, and
/// `next_values`, all shaped `[batch_size, 1]`, returning `[batch_size, 1]`.
///
/// `terminated` is a boolean mask: true means the episode terminated and the
/// next value must not contribute. Truncation alone should not set this mask.
/// Rewards and next values remain differentiable; callers control detachment.
pub fn bellman_targets(
    rewards: Tensor<2>,
    terminated: Tensor<2, Bool>,
    next_values: Tensor<2>,
    gamma: f64,
) -> Tensor<2> {
    assert_eq!(rewards.dims()[1], 1, "rewards require one scalar per row");
    assert_eq!(
        terminated.dims(),
        rewards.dims(),
        "termination shape must match rewards"
    );
    assert_eq!(
        next_values.dims(),
        rewards.dims(),
        "next-value shape must match rewards"
    );
    rewards + next_values.mask_fill(terminated, 0.0) * gamma
}

/// Returns a clipped value loss shaped `[1]` from `prediction`, `target`, and
/// `anchor`, all shaped `[batch_size, 1]`. The batch mean removes the batch axis.
///
/// The prediction update is clipped around `anchor`, and the larger of the
/// clipped and unclipped squared errors is averaged. This function does not
/// detach any input.
pub fn clipped_value_loss(
    prediction: Tensor<2>,
    target: Tensor<2>,
    anchor: Tensor<2>,
    epsilon: f64,
) -> Tensor<1> {
    assert_eq!(
        prediction.dims()[1],
        1,
        "prediction requires one scalar per row"
    );
    assert_eq!(
        target.dims(),
        prediction.dims(),
        "target shape must match prediction"
    );
    assert_eq!(
        anchor.dims(),
        prediction.dims(),
        "anchor shape must match prediction"
    );
    let delta =
        clamp_with_boundary_gradients(prediction.clone() - anchor.clone(), -epsilon, epsilon);
    let clipped = anchor + delta;
    let loss = (prediction - target.clone()).square();
    let clipped_loss = (clipped - target).square();
    // Equal errors split their gradients between the clipped and unclipped predictions.
    let ties = loss.clone().equal(clipped_loss.clone());
    loss.clone()
        .max_pair(clipped_loss.clone())
        .mask_where(ties, (loss + clipped_loss) * 0.5)
        .mean()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;

    #[test]
    fn clipped_value_loss_splits_gradients_at_the_clamp_boundary() {
        let device = Device::flex().autodiff();
        let prediction = Tensor::<2>::from_floats([[1.0]], &device).require_grad();
        let anchor = Tensor::<2>::from_floats([[0.0]], &device).require_grad();
        let target = Tensor::<2>::from_floats([[0.0]], &device).require_grad();
        let gradients =
            clipped_value_loss(prediction.clone(), target.clone(), anchor.clone(), 1.0).backward();
        for (input, expected) in [(prediction, 1.5), (anchor, 0.5), (target, -2.0)] {
            assert_eq!(
                input.grad(&gradients).unwrap().into_scalar::<f32>(),
                expected
            );
        }
    }

    #[test]
    fn clipped_value_loss_splits_tied_branch_gradients() {
        let device = Device::flex().autodiff();
        let prediction = Tensor::<2>::from_floats([[2.0]], &device).require_grad();
        let anchor = Tensor::<2>::from_floats([[0.0]], &device).require_grad();
        let target = Tensor::<2>::from_floats([[1.5]], &device).require_grad();
        let loss = clipped_value_loss(prediction.clone(), target.clone(), anchor.clone(), 1.0);
        assert_eq!(loss.clone().try_into_scalar::<f32>().unwrap(), 0.25);
        let gradients = loss.backward();
        assert_eq!(
            prediction
                .grad(&gradients)
                .unwrap()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            [0.5]
        );
        assert_eq!(
            anchor
                .grad(&gradients)
                .unwrap()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            [-0.5]
        );
        assert_eq!(
            target
                .grad(&gradients)
                .unwrap()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            [0.0]
        );
    }

    #[test]
    fn bellman_targets_mask_termination_without_implicit_detachment() {
        let device = Device::flex().autodiff();
        let rewards = Tensor::<2>::from_floats([[1.0], [2.0]], &device).require_grad();
        let terminated = Tensor::<2, Bool>::from_data([[false], [true]], &device);
        let next_values = Tensor::<2>::from_floats([[10.0], [20.0]], &device).require_grad();
        let targets = bellman_targets(rewards.clone(), terminated, next_values.clone(), 0.9);

        assert_eq!(targets.dims(), [2, 1]);
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
        let prediction = Tensor::<2>::from_floats([[0.0], [2.0]], &device).require_grad();
        let anchor = Tensor::<2>::from_floats([[2.0], [0.0]], &device).require_grad();
        let target = Tensor::<2>::from_floats([[0.0], [0.0]], &device).require_grad();
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

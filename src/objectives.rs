use candle_core::Tensor;

/// Builds one-step Bellman targets from `rewards`, `terminated`, and
/// `next_values`, all shaped `[batch]`, returning `[batch]`.
///
/// This function does not detach any input. Callers control the gradient
/// boundary appropriate for their algorithm.
pub fn bellman_targets(
    rewards: &Tensor,
    terminated: &Tensor,
    next_values: &Tensor,
    gamma: f64,
) -> candle_core::Result<Tensor> {
    let continuation = (1.0 - terminated)?;
    rewards + ((next_values * continuation)? * gamma)?
}

/// Returns a scalar clipped value loss from `prediction`, `target`, and
/// `anchor`, all shaped `[batch]`.
///
/// The prediction update is clipped around `anchor`, and the larger of the
/// clipped and unclipped squared errors is averaged. This function does not
/// detach any input.
pub fn clipped_value_loss(
    prediction: &Tensor,
    target: &Tensor,
    anchor: &Tensor,
    epsilon: f64,
) -> candle_core::Result<Tensor> {
    let delta = (prediction - anchor)?.clamp(-epsilon, epsilon)?;
    let clipped = (anchor + delta)?;
    let loss = (prediction - target)?.sqr()?;
    let clipped_loss = (clipped - target)?.sqr()?;
    loss.maximum(&clipped_loss)?.mean_all()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    #[test]
    fn bellman_targets_mask_termination_without_implicit_detachment() {
        let rewards = Var::from_vec(vec![1.0_f32, 2.0], 2, &Device::Cpu).unwrap();
        let terminated = Var::from_vec(vec![0.0_f32, 1.0], 2, &Device::Cpu).unwrap();
        let next_values = Var::from_vec(vec![10.0_f32, 20.0], 2, &Device::Cpu).unwrap();
        let targets = bellman_targets(
            rewards.as_tensor(),
            terminated.as_tensor(),
            next_values.as_tensor(),
            0.9,
        )
        .unwrap();

        assert_eq!(targets.to_vec1::<f32>().unwrap(), vec![10.0, 2.0]);
        let gradients = targets.sum_all().unwrap().backward().unwrap();
        assert!(gradients.get(rewards.as_tensor()).is_some());
        assert!(gradients.get(terminated.as_tensor()).is_some());
        assert!(gradients.get(next_values.as_tensor()).is_some());
    }

    #[test]
    fn clipped_value_loss_leaves_gradient_boundaries_to_the_caller() {
        let prediction = Var::from_vec(vec![0.0_f32, 0.0], 2, &Device::Cpu).unwrap();
        let anchor = Var::from_vec(vec![2.0_f32, 0.0], 2, &Device::Cpu).unwrap();
        let target = Var::from_vec(vec![0.0_f32, 0.0], 2, &Device::Cpu).unwrap();
        let loss = clipped_value_loss(
            prediction.as_tensor(),
            target.as_tensor(),
            anchor.as_tensor(),
            0.5,
        )
        .unwrap();

        assert_eq!(loss.to_scalar::<f32>().unwrap(), 1.125);
        let gradients = loss.backward().unwrap();
        assert!(gradients.get(prediction.as_tensor()).is_some());
        assert!(gradients.get(anchor.as_tensor()).is_some());
        assert!(gradients.get(target.as_tensor()).is_some());
    }
}

use std::{fmt::Debug, ops::Deref};

use candle_core::Tensor;
use candle_nn::VarMap;

use super::DeterministicActorCriticError;

pub(super) fn copy_actor_var_map<GE: Debug, SE: Debug>(
    online: &VarMap,
    target: &mut VarMap,
    tau: f64,
) -> Result<(), DeterministicActorCriticError<GE, SE>> {
    let online_values = online
        .data()
        .lock()
        .unwrap()
        .deref()
        .iter()
        .map(|(name, var)| (name.clone(), var.as_tensor().clone()))
        .collect::<Vec<_>>();
    let online_names = online_values
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<std::collections::HashSet<_>>();
    let target_names = target
        .data()
        .lock()
        .unwrap()
        .keys()
        .cloned()
        .collect::<std::collections::HashSet<_>>();
    if online_names != target_names {
        let mut online_only = online_names
            .difference(&target_names)
            .cloned()
            .collect::<Vec<_>>();
        online_only.sort();
        let mut target_only = target_names
            .difference(&online_names)
            .cloned()
            .collect::<Vec<_>>();
        target_only.sort();
        return Err(DeterministicActorCriticError::ActorParameterMapMismatch {
            online_only,
            target_only,
        });
    }
    for (name, online_value) in online_values {
        let value = if tau == 1.0 {
            online_value
        } else {
            let target_value = target
                .data()
                .lock()
                .unwrap()
                .get(&name)
                .expect("validated target actor parameter name")
                .as_tensor()
                .clone();
            ((online_value * tau)? + (target_value * (1.0 - tau))?)?
        };
        target.set_one(name, &value)?;
    }
    Ok(())
}

/// Adds elementwise noise to arbitrarily shaped `values`, preserving its
/// shape.
pub(super) fn add_gaussian_noise(
    values: &Tensor,
    standard_deviation: f64,
    clip: Option<f64>,
) -> candle_core::Result<Tensor> {
    if standard_deviation == 0.0 {
        return Ok(values.clone());
    }
    let mut noise = (Tensor::randn(0.0f32, 1.0, values.shape(), values.device())?
        .to_dtype(values.dtype())?
        * standard_deviation)?;
    if let Some(clip) = clip {
        noise = noise.clamp(-clip, clip)?;
    }
    values + noise
}

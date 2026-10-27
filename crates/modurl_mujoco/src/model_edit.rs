use mujoco_rs::prelude::MjModel;

use crate::MujocoError;

fn validate_unchanged<T: PartialEq + ?Sized>(
    field: &str,
    original: &T,
    candidate: &T,
) -> Result<(), MujocoError> {
    if original == candidate {
        Ok(())
    } else {
        Err(MujocoError::InvalidInput(format!(
            "runtime edits to {field} are not allowed"
        )))
    }
}

pub(crate) fn validate_runtime_edit(
    original: &MjModel,
    candidate: &MjModel,
) -> Result<(), MujocoError> {
    macro_rules! frozen {
        ($($field:ident),* $(,)?) => {
            $(validate_unchanged(
                stringify!($field),
                original.$field(),
                candidate.$field(),
            )?;)*
        };
    }

    // Other safe-mutable non-float fields use bounded or enum-checked MuJoCo 3.9.0 paths.
    frozen! {
        body_simple, dof_simplenum, tree_sleep_policy, site_type, flex_internal,
        flex_selfcollide, flex_activelayers, flex_passive, flex_elemlayer,
        flex_edgeequality, flex_rigid, flexedge_rigid, flex_centered, tex_type,
        tendon_treenum, actuator_gaintype, actuator_biastype, sensor_datatype,
        sensor_needstage, sensor_intprm,
    }

    let original_opt = original.opt();
    let candidate_opt = candidate.opt();
    macro_rules! unchanged_options {
        ($($field:ident),* $(,)?) => {
            $(validate_unchanged(
                concat!("opt.", stringify!($field)),
                &original_opt.$field,
                &candidate_opt.$field,
            )?;)*
        };
    }
    unchanged_options! {
        integrator, cone, jacobian, solver, ccd_iterations, sdf_initpoints, sdf_iterations,
    }

    for (field, value) in [
        ("opt.iterations", candidate_opt.iterations),
        ("opt.ls_iterations", candidate_opt.ls_iterations),
        ("opt.noslip_iterations", candidate_opt.noslip_iterations),
    ] {
        if value < 0 {
            return Err(MujocoError::InvalidInput(format!(
                "{field} must be non-negative"
            )));
        }
    }

    for (index, eq_type) in candidate.eq_type().iter().enumerate() {
        let flexstrain = mujoco_rs::mujoco_c::mjtEq::mjEQ_FLEXSTRAIN;
        if *eq_type == flexstrain || original.eq_type()[index] == flexstrain {
            let original_row = &original.eq_data()[index];
            let candidate_row = &candidate.eq_data()[index];
            if original_row
                .iter()
                .zip(candidate_row)
                .any(|(original, candidate)| original.to_bits() != candidate.to_bits())
            {
                return Err(MujocoError::InvalidInput(format!(
                    "eq_data for flexstrain equality {index} cannot be changed"
                )));
            }
        }
    }
    Ok(())
}

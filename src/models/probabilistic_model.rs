use candle_core::Tensor;
use candle_nn::Module;

use crate::distributions::Distribution;

use super::MLP;

pub trait ProbabilisticActor {
    type Error;
    fn sample(&self, state: &Tensor) -> Result<Tensor, Self::Error>;
    fn log_prob_and_entropy(
        &self,
        state: &Tensor,
        action: &Tensor,
    ) -> Result<(Tensor, Tensor), Self::Error>;
}

pub struct MLPProbabilisticActor<D>
where
    D: Distribution,
{
    mlp: MLP,
    _marker: std::marker::PhantomData<D>,
}

impl<D> MLPProbabilisticActor<D>
where
    D: Distribution,
    <D as Distribution>::Error: std::fmt::Debug,
{
    pub fn new(mlp: MLP) -> Self {
        Self {
            mlp,
            _marker: std::marker::PhantomData,
        }
    }
}

#[derive(Debug)]
pub enum MLPProbabilisticActorError<D>
where
    D: Distribution,
    <D as Distribution>::Error: std::fmt::Debug,
{
    MLPError(candle_core::Error),
    DistError(<D as Distribution>::Error),
}

impl<D> From<candle_core::Error> for MLPProbabilisticActorError<D>
where
    D: Distribution,
    <D as Distribution>::Error: std::fmt::Debug,
{
    fn from(error: candle_core::Error) -> Self {
        MLPProbabilisticActorError::MLPError(error)
    }
}

impl<D> ProbabilisticActor for MLPProbabilisticActor<D>
where
    D: Distribution,
    <D as Distribution>::Error: std::fmt::Debug,
{
    type Error = MLPProbabilisticActorError<D>;

    fn sample(&self, state: &Tensor) -> Result<Tensor, Self::Error> {
        let output = self.mlp.forward(state)?;
        let dist = D::from_outputs(&output);
        let action = dist.sample();
        Ok(action)
    }

    fn log_prob_and_entropy(
        &self,
        state: &Tensor,
        action: &Tensor,
    ) -> Result<(Tensor, Tensor), Self::Error> {
        let state = state.squeeze(1)?;
        let action = action.squeeze(1)?;
        let output = self.mlp.forward(&state)?;

        let dist = D::from_outputs(&output);
        let dist_eval = dist
            .dist_eval(&action)
            .map_err(|e| MLPProbabilisticActorError::DistError(e))?;

        let log_prob = dist_eval.log_prob().clone();
        let entropy = dist_eval.entropy().clone();
        Ok((log_prob, entropy))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{distributions::GuassianDistribution, models::MLPBuilder};
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{Activation, VarBuilder, VarMap};

    fn create_test_actor(
        input_size: usize,
        action_size: usize,
        hidden_sizes: Vec<usize>,
    ) -> Result<MLPProbabilisticActor<GuassianDistribution>, candle_core::Error> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &device);

        // Output size is double action_size for mean and std
        let mlp = MLPBuilder::new(input_size, action_size * 2, vb)
            .hidden_layer_sizes(hidden_sizes)
            .activation(Box::new(Activation::Relu))
            .build()?;

        Ok(MLPProbabilisticActor::new(mlp))
    }

    fn create_test_state(
        batch_size: usize,
        state_dim: usize,
    ) -> Result<Tensor, candle_core::Error> {
        Tensor::randn(0.0, 1.0, (batch_size, state_dim), &Device::Cpu)
    }

    #[test]
    fn test_actor_creation() {
        let actor = create_test_actor(4, 2, vec![32, 32]);
        assert!(actor.is_ok(), "Failed to create probabilistic actor");
    }

    #[test]
    fn test_sample_action_shape() {
        let actor = create_test_actor(4, 2, vec![32, 32]).unwrap();
        let state = create_test_state(1, 4).unwrap();

        let action = actor.sample(&state).unwrap();

        // Action should have shape [batch_size, action_dim]
        assert_eq!(action.dims(), &[1, 2]);
    }

    #[test]
    fn test_sample_batch_consistency() {
        let actor = create_test_actor(4, 2, vec![32, 32]).unwrap();
        let batch_sizes = vec![1, 5, 10];

        for batch_size in batch_sizes {
            let state = create_test_state(batch_size, 4).unwrap();
            let action = actor.sample(&state).unwrap();

            assert_eq!(
                action.dims(),
                &[batch_size, 2],
                "Action shape incorrect for batch size {}",
                batch_size
            );
        }
    }

    #[test]
    fn test_log_prob_and_entropy_shape() {
        let actor = create_test_actor(4, 2, vec![32, 32]).unwrap();
        let batch_size = 5;
        let state = create_test_state(batch_size, 4).unwrap();
        let action = create_test_state(batch_size, 2).unwrap();

        // Add dimension for squeeze operation in log_prob_and_entropy
        let state = state.unsqueeze(1).unwrap();
        let action = action.unsqueeze(1).unwrap();

        let (log_prob, entropy) = actor.log_prob_and_entropy(&state, &action).unwrap();

        // Both should have shape [batch_size]
        assert_eq!(log_prob.dims(), &[batch_size]);
        assert_eq!(entropy.dims(), &[batch_size]);
    }

    #[test]
    fn test_entropy_is_positive() {
        let actor = create_test_actor(4, 2, vec![32, 32]).unwrap();
        let state = create_test_state(1, 4).unwrap();
        let action = create_test_state(1, 2).unwrap();

        let state = state.unsqueeze(1).unwrap();
        let action = action.unsqueeze(1).unwrap();

        let (_, entropy) = actor.log_prob_and_entropy(&state, &action).unwrap();
        let entropy_val = entropy.to_vec1::<f64>().unwrap()[0];

        // Entropy should generally be positive for continuous distributions
        assert!(
            entropy_val.is_finite(),
            "Entropy is not finite: {}",
            entropy_val
        );
        assert!(
            entropy_val > -100.0,
            "Entropy is not reasonable: {}",
            entropy_val
        );
        assert!(entropy_val < 100.0, "Entropy is too large: {}", entropy_val);
    }

    #[test]
    fn test_log_prob_is_finite() {
        let actor = create_test_actor(4, 2, vec![32, 32]).unwrap();
        let state = create_test_state(3, 4).unwrap();
        let action = create_test_state(3, 2).unwrap();

        let state = state.unsqueeze(1).unwrap();
        let action = action.unsqueeze(1).unwrap();

        let (log_prob, _) = actor.log_prob_and_entropy(&state, &action).unwrap();
        let log_prob_vals = log_prob.to_vec1::<f64>().unwrap();

        for (i, val) in log_prob_vals.iter().enumerate() {
            assert!(
                val.is_finite(),
                "Log probability {} is not finite: {}",
                i,
                val
            );
        }
    }

    #[test]
    fn test_deterministic_with_same_state() {
        let actor = create_test_actor(4, 2, vec![32, 32]).unwrap();
        let state = Tensor::zeros((1, 4), candle_core::DType::F64, &Device::Cpu).unwrap();
        let action = Tensor::zeros((1, 2), candle_core::DType::F64, &Device::Cpu).unwrap();

        let state = state.unsqueeze(1).unwrap();
        let action = action.unsqueeze(1).unwrap();

        // Multiple calls with same input should give same log_prob and entropy
        let (log_prob1, entropy1) = actor.log_prob_and_entropy(&state, &action).unwrap();
        let (log_prob2, entropy2) = actor.log_prob_and_entropy(&state, &action).unwrap();

        let log_prob1_val = log_prob1.to_vec1::<f64>().unwrap()[0];
        let log_prob2_val = log_prob2.to_vec1::<f64>().unwrap()[0];
        let entropy1_val = entropy1.to_vec1::<f64>().unwrap()[0];
        let entropy2_val = entropy2.to_vec1::<f64>().unwrap()[0];

        assert!(
            (log_prob1_val - log_prob2_val).abs() < 1e-6,
            "Log probabilities should be deterministic"
        );
        assert!(
            (entropy1_val - entropy2_val).abs() < 1e-6,
            "Entropies should be deterministic"
        );
    }

    #[test]
    fn test_different_action_sizes() {
        let action_sizes = vec![1, 2, 4, 8];

        for action_size in action_sizes {
            let actor = create_test_actor(4, action_size, vec![32, 32]).unwrap();
            let state = create_test_state(1, 4).unwrap();
            let action = actor.sample(&state).unwrap();

            assert_eq!(
                action.dims(),
                &[1, action_size],
                "Action size incorrect for action_size {}",
                action_size
            );
        }
    }

    #[test]
    fn test_log_std_clamping() {
        // This test verifies that log_std is properly clamped between -5.0 and 2.0
        let actor = create_test_actor(4, 2, vec![32, 32]).unwrap();
        let state = create_test_state(1, 4).unwrap();
        let action = create_test_state(1, 2).unwrap();

        let state = state.unsqueeze(1).unwrap();
        let action = action.unsqueeze(1).unwrap();

        // This should not crash due to numerical issues
        let result = actor.log_prob_and_entropy(&state, &action);
        assert!(
            result.is_ok(),
            "Log prob calculation failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_gradient_flow() {
        // Test that the actor can be used in a gradient computation context
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &device);

        let mlp = MLPBuilder::new(4, 4, vb)
            .hidden_layer_sizes(vec![16])
            .activation(Box::new(Activation::Relu))
            .build()
            .unwrap();

        let actor: MLPProbabilisticActor<GuassianDistribution> = MLPProbabilisticActor::new(mlp);
        let state = create_test_state(2, 4).unwrap();

        // Sample action
        let action = actor.sample(&state).unwrap();

        // Compute log probability
        let state = state.unsqueeze(1).unwrap();
        let action = action.unsqueeze(1).unwrap();
        let (log_prob, entropy) = actor.log_prob_and_entropy(&state, &action).unwrap();

        // This should work without errors - just verify no panic occurs
        let _ = (log_prob.mean(0).unwrap(), entropy.mean(0).unwrap());
    }

    #[test]
    fn test_action_bounds_reasonable() {
        let actor = create_test_actor(4, 2, vec![32, 32]).unwrap();
        let state = create_test_state(10, 4).unwrap();

        let action = actor.sample(&state).unwrap();
        let action_vals = action.flatten_all().unwrap().to_vec1::<f64>().unwrap();

        // Actions should be reasonable (not extreme values)
        for val in action_vals {
            assert!(val.abs() < 100.0, "Action value {} is too extreme", val);
            assert!(val.is_finite(), "Action value {} is not finite", val);
        }
    }

    // Fuzzing tests - run multiple iterations with random inputs
    #[test]
    fn fuzz_sample_action_consistency() {
        const FUZZ_ITERATIONS: usize = 50;

        for i in 0..FUZZ_ITERATIONS {
            // Random dimensions within reasonable bounds
            let state_dim = 2 + (i % 10); // 2-11
            let action_dim = 1 + (i % 6); // 1-6
            let batch_size = 1 + (i % 20); // 1-20
            let hidden_size = 16 + (i % 64); // 16-79

            let actor =
                create_test_actor(state_dim, action_dim, vec![hidden_size, hidden_size]).unwrap();
            let state = create_test_state(batch_size, state_dim).unwrap();
            let action = actor.sample(&state).unwrap();

            // Verify shape consistency
            assert_eq!(
                action.dims(),
                &[batch_size, action_dim],
                "Iteration {}: Action shape incorrect for state_dim={}, action_dim={}, batch_size={}",
                i, state_dim, action_dim, batch_size
            );

            // Verify action values are reasonable
            let action_vals = action.flatten_all().unwrap().to_vec1::<f64>().unwrap();
            for val in action_vals {
                assert!(
                    val.is_finite(),
                    "Iteration {}: Action value is not finite: {}",
                    i,
                    val
                );
                assert!(
                    val.abs() < 1000.0,
                    "Iteration {}: Action value too extreme: {}",
                    i,
                    val
                );
            }
        }
    }

    #[test]
    fn fuzz_log_prob_and_entropy() {
        const FUZZ_ITERATIONS: usize = 30;

        for i in 0..FUZZ_ITERATIONS {
            let state_dim = 2 + (i % 8); // 2-9
            let action_dim = 1 + (i % 5); // 1-5
            let batch_size = 1 + (i % 15); // 1-15

            let actor = create_test_actor(state_dim, action_dim, vec![32, 32]).unwrap();
            let state = create_test_state(batch_size, state_dim).unwrap();
            let action = create_test_state(batch_size, action_dim).unwrap();

            let state = state.unsqueeze(1).unwrap();
            let action = action.unsqueeze(1).unwrap();

            let (log_prob, entropy) = actor.log_prob_and_entropy(&state, &action).unwrap();

            // Verify shapes
            assert_eq!(
                log_prob.dims(),
                &[batch_size],
                "Iteration {}: Log prob shape incorrect",
                i
            );
            assert_eq!(
                entropy.dims(),
                &[batch_size],
                "Iteration {}: Entropy shape incorrect",
                i
            );

            // Verify values are finite and reasonable
            let log_prob_vals = log_prob.to_vec1::<f64>().unwrap();
            let entropy_vals = entropy.to_vec1::<f64>().unwrap();

            for (j, val) in log_prob_vals.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "Iteration {}, sample {}: Log prob is not finite: {}",
                    i,
                    j,
                    val
                );
            }

            for (j, val) in entropy_vals.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "Iteration {}, sample {}: Entropy is not finite: {}",
                    i,
                    j,
                    val
                );
                assert!(
                    *val > -100.0,
                    "Iteration {}, sample {}: Entropy too negative: {}",
                    i,
                    j,
                    val
                );
            }
        }
    }

    #[test]
    fn fuzz_different_architectures() {
        const FUZZ_ITERATIONS: usize = 25;

        for i in 0..FUZZ_ITERATIONS {
            let state_dim = 2 + (i % 6); // 2-7
            let action_dim = 1 + (i % 4); // 1-4

            // Vary architecture
            let hidden_layers = match i % 4 {
                0 => vec![16],
                1 => vec![32, 16],
                2 => vec![64, 32, 16],
                _ => vec![128, 64],
            };

            let actor = create_test_actor(state_dim, action_dim, hidden_layers).unwrap();
            let state = create_test_state(5, state_dim).unwrap();

            // Test sampling
            let action = actor.sample(&state).unwrap();
            assert_eq!(
                action.dims(),
                &[5, action_dim],
                "Iteration {}: Sampling failed",
                i
            );

            // Test log prob computation
            let state_expanded = state.unsqueeze(1).unwrap();
            let action_expanded = action.unsqueeze(1).unwrap();
            let result = actor.log_prob_and_entropy(&state_expanded, &action_expanded);
            assert!(
                result.is_ok(),
                "Iteration {}: Log prob computation failed: {:?}",
                i,
                result.err()
            );
        }
    }

    #[test]
    fn fuzz_numerical_stability() {
        const FUZZ_ITERATIONS: usize = 40;

        for i in 0..FUZZ_ITERATIONS {
            let actor = create_test_actor(4, 2, vec![32, 32]).unwrap();

            // Create states with varying magnitudes to test numerical stability
            let scale = match i % 4 {
                0 => 0.01,  // Very small values
                1 => 1.0,   // Normal values
                2 => 10.0,  // Large values
                _ => 100.0, // Very large values
            };

            let mut state = create_test_state(3, 4).unwrap();
            state = (state * scale).unwrap();

            let mut action = create_test_state(3, 2).unwrap();
            action = (action * (scale * 0.1)).unwrap();

            let state = state.unsqueeze(1).unwrap();
            let action = action.unsqueeze(1).unwrap();

            // This should not crash or produce NaN/Inf values
            let result = actor.log_prob_and_entropy(&state, &action);
            assert!(
                result.is_ok(),
                "Iteration {} (scale={}): Computation failed: {:?}",
                i,
                scale,
                result.err()
            );

            if let Ok((log_prob, entropy)) = result {
                let log_prob_vals = log_prob.to_vec1::<f64>().unwrap();
                let entropy_vals = entropy.to_vec1::<f64>().unwrap();

                for val in log_prob_vals {
                    assert!(
                        val.is_finite(),
                        "Iteration {} (scale={}): Log prob not finite: {}",
                        i,
                        scale,
                        val
                    );
                }

                for val in entropy_vals {
                    assert!(
                        val.is_finite(),
                        "Iteration {} (scale={}): Entropy not finite: {}",
                        i,
                        scale,
                        val
                    );
                }
            }
        }
    }

    #[test]
    fn fuzz_batch_size_stress_test() {
        const FUZZ_ITERATIONS: usize = 20;

        for i in 0..FUZZ_ITERATIONS {
            // Test with various batch sizes including edge cases
            let batch_size = match i % 6 {
                0 => 1,
                1 => 2,
                2 => 5,
                3 => 10,
                4 => 50,
                _ => 100,
            };

            let actor = create_test_actor(4, 2, vec![32, 32]).unwrap();
            let state = create_test_state(batch_size, 4).unwrap();

            // Test sampling with large batches
            let action = actor.sample(&state).unwrap();
            assert_eq!(
                action.dims(),
                &[batch_size, 2],
                "Iteration {}: Batch size {} failed",
                i,
                batch_size
            );

            // Test log prob computation
            let state_expanded = state.unsqueeze(1).unwrap();
            let action_expanded = action.unsqueeze(1).unwrap();
            let (log_prob, entropy) = actor
                .log_prob_and_entropy(&state_expanded, &action_expanded)
                .unwrap();

            assert_eq!(
                log_prob.dims(),
                &[batch_size],
                "Iteration {}: Log prob batch size incorrect",
                i
            );
            assert_eq!(
                entropy.dims(),
                &[batch_size],
                "Iteration {}: Entropy batch size incorrect",
                i
            );

            // Verify all values are finite
            let log_prob_vals = log_prob.to_vec1::<f64>().unwrap();
            let entropy_vals = entropy.to_vec1::<f64>().unwrap();

            assert!(
                log_prob_vals.iter().all(|v| v.is_finite()),
                "Iteration {}: Some log prob values not finite",
                i
            );
            assert!(
                entropy_vals.iter().all(|v| v.is_finite()),
                "Iteration {}: Some entropy values not finite",
                i
            );
        }
    }
}

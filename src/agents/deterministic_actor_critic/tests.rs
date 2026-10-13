use super::{
    CriticCountRequirement, DeterministicActorCriticConfigurationError as Error,
    DeterministicActorCriticConfigurationValidator, add_gaussian_noise,
};
use candle_core::{Device, Tensor};

#[test]
fn validates_shared_and_strategy_configuration() {
    assert_eq!(
        DeterministicActorCriticConfigurationValidator::validate_configuration()
            .critic_count_requirement(CriticCountRequirement::Exact(2))
            .actual_critic_count(1)
            .replay_capacity(10)
            .batch_size(2)
            .gamma(0.99)
            .tau(0.005)
            .exploration_noise(0.1)
            .target_policy_noise(0.2)
            .target_noise_clip(0.5)
            .actor_update_interval(2)
            .update_frequency(1)
            .training_horizon(10)
            .call(),
        Err(Error::IncorrectCriticCount {
            expected: 2,
            actual: 1,
        })
    );
    assert_eq!(
        DeterministicActorCriticConfigurationValidator::validate_configuration()
            .critic_count_requirement(CriticCountRequirement::NonEmpty)
            .actual_critic_count(0)
            .replay_capacity(10)
            .batch_size(2)
            .gamma(0.99)
            .tau(0.005)
            .exploration_noise(0.1)
            .target_policy_noise(0.0)
            .target_noise_clip(0.0)
            .actor_update_interval(1)
            .update_frequency(1)
            .training_horizon(10)
            .call(),
        Err(Error::NoCritics)
    );
    assert_eq!(
        DeterministicActorCriticConfigurationValidator::validate_configuration()
            .critic_count_requirement(CriticCountRequirement::Exact(1))
            .actual_critic_count(1)
            .replay_capacity(10)
            .batch_size(2)
            .gamma(0.99)
            .tau(0.005)
            .exploration_noise(-0.1)
            .target_policy_noise(0.0)
            .target_noise_clip(0.0)
            .actor_update_interval(1)
            .update_frequency(1)
            .training_horizon(10)
            .call(),
        Err(Error::InvalidExplorationNoise)
    );
    assert_eq!(
        DeterministicActorCriticConfigurationValidator::validate_configuration()
            .critic_count_requirement(CriticCountRequirement::NonEmpty)
            .actual_critic_count(3)
            .replay_capacity(10)
            .batch_size(2)
            .gamma(0.99)
            .tau(0.005)
            .exploration_noise(0.1)
            .target_policy_noise(0.0)
            .target_noise_clip(0.0)
            .actor_update_interval(1)
            .update_frequency(1)
            .training_horizon(10)
            .call(),
        Ok(())
    );
}

#[test]
fn zero_noise_preserves_values() {
    let values = Tensor::new(&[[1.0f32, 2.0]], &Device::Cpu).unwrap();
    assert_eq!(
        add_gaussian_noise(&values, 0.0, Some(0.0))
            .unwrap()
            .to_vec2::<f32>()
            .unwrap(),
        vec![vec![1.0, 2.0]]
    );
}

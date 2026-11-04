use super::*;
use crate::tensor_rank::TensorBatchTypes;
use crate::{
    gym::{Gym, ResetInfo, StepInfo},
    models::{Forward, MLP},
    spaces::{BoxSpace, Discrete},
};
use burn::{
    module::{Module, ModuleVisitor, Param},
    nn::{Initializer, Tanh},
    optim::GradientsParams,
    tensor::{Bool, DType, Device, Float},
};
use std::convert::Infallible;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

pub(super) fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "expected {expected}, got {actual}"
    );
}

pub(super) fn mlp(input: usize, output: usize, device: &Device) -> MLP {
    MLP::builder()
        .input_size(input)
        .output_size(output)
        .hidden_layer_sizes(vec![2])
        .activation(Tanh)
        .hidden_initializer(Initializer::Constant { value: 0.1 })
        .output_initializer(Initializer::Constant { value: 0.1 })
        .bias_initializer(Initializer::Constant { value: 0.0 })
        .options((device, DType::F32))
        .build()
        .unwrap()
}

#[derive(Default)]
pub(super) struct NoopLogger {
    pub(super) updates: usize,
}

impl<I> PPOLogger<I> for NoopLogger {
    fn log(&mut self, _info: &PPOLogEntry) {
        self.updates += 1;
    }
}

pub(super) struct TwoStepTestGym {
    device: Device,
    steps: usize,
    truncates: bool,
}

impl TwoStepTestGym {
    pub(super) fn new(device: Device, truncates: bool) -> Self {
        Self {
            device,
            steps: 0,
            truncates,
        }
    }
}

impl Gym<usize> for TwoStepTestGym {
    type Error = Infallible;
    type ObservationSpace = BoxSpace;
    type ActionSpace = Discrete;

    /// Steps with one scalar Int action `[1]`, returning Float observations `[4]` on the fixture device.
    fn step(&mut self, _action: Tensor<1, Int>) -> Result<StepInfo<usize>, Self::Error> {
        self.steps += 1;
        Ok(StepInfo {
            observation: Tensor::zeros([4], (&self.device, DType::F32)),
            reward: 1.0,
            done: self.steps == 2 && !self.truncates,
            truncated: self.steps == 2 && self.truncates,
            info: self.steps,
        })
    }

    /// Returns Float observations `[4]` on the fixture device.
    fn reset(&mut self) -> Result<ResetInfo<usize>, Self::Error> {
        self.steps = 0;
        Ok(ResetInfo {
            observation: Tensor::zeros([4], (&self.device, DType::F32)),
            info: 0,
        })
    }

    fn observation_space(&self) -> Self::ObservationSpace {
        BoxSpace::new_with_universal_bounds([1, 4], -1.0, 1.0, &self.device)
    }

    fn action_space(&self) -> Self::ActionSpace {
        Discrete::new(2).unwrap()
    }
}

#[cfg(test)]
mod rollout_batch_tests {
    use burn::tensor::{DType, Device, Tensor};

    use super::{PPOExperience, PPOPreparedExperience};
    use crate::buffers::experience::{Experience, ExperienceBatchError};

    /// Creates Float rank-3 observations `[2, 1, 2]`, rank-2 actions `[2, 1]`, and scalar statistics `[2, 1]`.
    /// Uses F64 on the supplied device and retains observation gradients.
    fn transition(value: f64, device: &Device) -> PPOExperience<Tensor<3>, Tensor<2>> {
        let observations = Tensor::from_data(
            [[[value, value + 1.0]], [[value + 2.0, value + 3.0]]],
            (device, DType::F64),
        )
        .require_grad();
        let rewards = Tensor::from_data([[value + 2.0], [value + 3.0]], (device, DType::F64));
        PPOExperience::builder()
            .observations(observations.clone())
            .next_observations(observations + 10.0)
            .actions(Tensor::from_data(
                [[value], [value + 1.0]],
                (device, DType::F64),
            ))
            .rewards(rewards.clone())
            .training_rewards(rewards.clone())
            .next_dones(vec![true, false])
            .truncateds(vec![false, true])
            .log_probs(Tensor::zeros([2, 1], (device, DType::F64)))
            .values(Tensor::zeros([2, 1], (device, DType::F64)))
            .build()
    }

    #[test]
    fn rollout_batch_preserves_axes_statistics_dtype_and_gradients() {
        let device = Device::flex().autodiff();
        let mut experiences = [transition(1.0, &device), transition(10.0, &device)];
        for item in &mut experiences {
            item.prepared = Some(PPOPreparedExperience {
                advantages: item.rewards.clone(),
                returns: item.rewards.clone() + 100.0,
                old_values: Some(item.log_probs.clone()),
            });
        }
        let batch = PPOExperience::batch(&experiences).unwrap();
        let observations: Tensor<4> = batch.observations;
        let actions: Tensor<3> = batch.actions;
        assert_eq!(observations.dims(), [2, 2, 1, 2]);
        assert_eq!(actions.dims(), [2, 2, 1]);
        assert_eq!(observations.dtype(), DType::F64);
        assert_eq!(observations.device(), device);
        assert_eq!(batch.rewards.dims(), [2, 2, 1]);
        assert_eq!(batch.rewards.dtype(), DType::F64);
        assert_eq!(
            batch.rewards.into_data().try_to_vec::<f64>().unwrap(),
            [3.0, 4.0, 12.0, 13.0]
        );
        assert_eq!(batch.log_probs.dims(), [2, 2, 1]);
        let prepared = batch.prepared.unwrap();
        assert_eq!(prepared.advantages.dims(), [2, 2, 1]);
        assert_eq!(prepared.returns.dims(), [2, 2, 1]);
        assert_eq!(
            prepared.returns.into_data().try_to_vec::<f64>().unwrap(),
            [103.0, 104.0, 112.0, 113.0]
        );
        assert_eq!(prepared.old_values.unwrap().dims(), [2, 2, 1]);

        let gradients = observations.sum().backward();
        for item in experiences {
            assert_eq!(
                item.observations
                    .grad(&gradients)
                    .unwrap()
                    .into_data()
                    .try_to_vec::<f64>()
                    .unwrap(),
                [1.0; 4]
            );
        }
    }

    #[test]
    fn rollout_batch_reports_empty_and_mismatched_columns() {
        assert!(matches!(
            <PPOExperience as Experience>::batch(&[]),
            Err(ExperienceBatchError::EmptyBatch)
        ));
        let device = Device::flex().autodiff();
        let mut experiences = [transition(1.0, &device), transition(10.0, &device)];
        experiences[1].rewards = experiences[1].rewards.clone().cast(DType::F32);
        assert!(matches!(
            PPOExperience::batch(&experiences),
            Err(ExperienceBatchError::DTypeMismatch { index: 1, .. })
        ));
    }

    #[test]
    fn rollout_batch_preserves_optional_prepared_values() {
        let device = Device::flex().autodiff();
        let mut experiences = [transition(1.0, &device), transition(10.0, &device)];
        assert!(
            PPOExperience::batch(&experiences)
                .unwrap()
                .prepared
                .is_none()
        );
        for item in &mut experiences {
            item.prepared = Some(PPOPreparedExperience {
                advantages: item.rewards.clone(),
                returns: item.rewards.clone() + 100.0,
                old_values: None,
            });
        }
        assert!(
            PPOExperience::batch(&experiences)
                .unwrap()
                .prepared
                .unwrap()
                .old_values
                .is_none()
        );
    }
}

#[cfg(test)]
mod schedule_tests {
    use std::collections::HashSet;

    use burn::{
        module::ParamId,
        nn::{Dropout, DropoutConfig, Initializer, LinearConfig},
        optim::SgdConfig,
        tensor::kind::Kind,
    };
    use std::convert::Infallible;

    use super::*;
    use crate::{
        advantage_estimation::AdvantageEstimate,
        agents::{Agent, ToEnvironmentActions},
        distributions::{CategoricalDistribution, GaussianDistribution},
        gym::{Gym, MultiGym, ResetInfo, StepInfo, VectorizedGymError, VectorizedGymWrapper},
        models::{MLP, probabilistic_model::ProbabilisticPolicyModel},
        spaces::{ActionMap, BoxSpace, Discrete},
    };

    #[derive(Module, Debug)]
    struct DropoutTrunk {
        dropout: Dropout,
    }

    impl Forward<2, 2> for DropoutTrunk {
        type InputKind = Float;
        type OutputKind = Float;
        type Error = Infallible;

        /// Applies dropout to `[batch_size, features]`, preserving shape, dtype, and device.
        fn forward(&self, observations: Tensor<2>) -> Result<Tensor<2>, Self::Error> {
            Ok(self.dropout.forward(observations))
        }
    }

    #[derive(Module, Debug)]
    struct ObservationModel<K: Basic> {
        model: MLP,
        double_precision: bool,
        #[module(skip)]
        kind: PhantomData<K>,
    }

    impl<K: Basic> Forward<2, 2> for ObservationModel<K>
    where
        burn::tensor::FloatDType: burn::tensor::Cast<2, K, OutputKind = Float>,
    {
        type InputKind = K;
        type OutputKind = Float;
        type Error = crate::models::ModelError;

        /// Converts native observations [batch_size, input_features] inside the model and returns Float outputs [batch_size, output_features].
        /// Uses the model's floating-point precision and preserves parameter gradients.
        fn forward(&self, input: Tensor<2, K>) -> Result<Tensor<2>, Self::Error> {
            use burn::tensor::{Cast, FloatDType};
            let dtype = if self.double_precision {
                FloatDType::F64
            } else {
                FloatDType::F32
            };
            if const { matches!(K::KIND, Kind::Int) } {
                assert_eq!(input.dtype(), DType::I64);
            } else if const { matches!(K::KIND, Kind::Float) } {
                assert_eq!(input.dtype(), DType::F32);
            }
            self.model.forward(FloatDType::cast(input, dtype))
        }
    }

    #[test]
    fn act_for_training_samples_in_validation_mode() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        let network = SharedPPONetwork::builder()
            .shared_network(DropoutTrunk {
                dropout: DropoutConfig::new(0.5).init(),
            })
            .actor_head(ProbabilisticPolicyModel::new(
                mlp(64, 16, &device),
                GaussianDistribution::default(),
            ))
            .critic_head(
                LinearConfig::new(64, 1)
                    .with_initializer(Initializer::Constant { value: 0.5 })
                    .init(&device)
                    .no_grad(),
            )
            .optimizer(SgdConfig::new().init())
            .learning_rate(1e-3)
            .build()
            .unwrap();
        let mut agent = PPOAgent::<_, _, _, Infallible>::builder()
            .observation_space(BoxSpace::new_unbounded([1, 64], &device))
            .action_space(BoxSpace::new_with_universal_bounds(
                [1, 8],
                -10.0,
                10.0,
                &device,
            ))
            .network_info(network)
            .batch_size(1)
            .mini_batch_size(1)
            .num_epochs(1)
            .training_horizon(1)
            .build()
            .unwrap();
        let observations = Tensor::<2>::ones([1, 64], &device);

        device.seed(153);
        let training_actions = agent
            .act_for_training(&observations)
            .unwrap()
            .neurons()
            .clone();
        device.seed(153);
        let validation = agent.network_info.valid();
        let collection_actions = agent
            .sample_collection_action(&validation, &observations)
            .unwrap()
            .policy_actions;

        assert_eq!(training_actions.to_data(), collection_actions.to_data());
    }

    #[test]
    fn act_deterministic_ignores_dropout() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        let network = SharedPPONetwork::builder()
            .shared_network(DropoutTrunk {
                dropout: DropoutConfig::new(0.5).init(),
            })
            .actor_head(ProbabilisticPolicyModel::new(
                mlp(64, 16, &device),
                GaussianDistribution::default(),
            ))
            .critic_head(
                LinearConfig::new(64, 1)
                    .with_initializer(Initializer::Constant { value: 0.5 })
                    .init(&device)
                    .no_grad(),
            )
            .optimizer(SgdConfig::new().init())
            .learning_rate(1e-3)
            .build()
            .unwrap();
        let mut agent = PPOAgent::<_, _, _, Infallible>::builder()
            .observation_space(BoxSpace::new_unbounded([1, 64], &device))
            .action_space(BoxSpace::new_with_universal_bounds(
                [1, 8],
                -10.0,
                10.0,
                &device,
            ))
            .network_info(network)
            .batch_size(1)
            .mini_batch_size(1)
            .num_epochs(1)
            .training_horizon(1)
            .build()
            .unwrap();
        let observations = Tensor::<2>::ones([1, 64], &device);

        device.seed(1);
        let first_actions = agent.act_deterministic(&observations).unwrap();
        device.seed(2);
        let second_actions = agent.act_deterministic(&observations).unwrap();
        let validation = agent.network_info.valid();
        let latent = validation
            .forward_observations(observations.clone().without_autodiff())
            .unwrap();
        let expected_actions = agent
            .action_space
            .tensor_from_neurons(validation.mode(latent).unwrap())
            .unwrap();

        assert_eq!(first_actions.to_data(), second_actions.to_data());
        assert_eq!(first_actions.to_data(), expected_actions.to_data());
    }

    #[test]
    fn collection_values_reuse_shared_features_and_remain_detached() {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };

        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        #[derive(Module, Debug)]
        struct CountingTrunk {
            #[module(skip)]
            calls: Arc<AtomicUsize>,
        }

        impl Forward<2, 2> for CountingTrunk {
            type InputKind = Float;
            type OutputKind = Float;
            type Error = Infallible;

            /// Returns observations `[batch, 4]` unchanged and counts encoder calls.
            fn forward(&self, observations: Tensor<2>) -> Result<Tensor<2>, Self::Error> {
                self.calls.fetch_add(1, Ordering::Relaxed);
                Ok(observations)
            }
        }

        #[derive(Module, Debug)]
        struct ConstantCritic {
            value: Param<Tensor<1>>,
        }

        impl Forward<2, 2> for ConstantCritic {
            type InputKind = Float;
            type OutputKind = Float;
            type Error = Infallible;

            /// Expands the learned value `[1]` to `[batch, 1]` for observations `[batch, 4]`.
            fn forward(&self, observations: Tensor<2>) -> Result<Tensor<2>, Self::Error> {
                Ok(self
                    .value
                    .val()
                    .reshape([1, 1])
                    .expand([observations.dims()[0], 1]))
            }
        }

        let device = Device::flex().autodiff();
        let calls = Arc::new(AtomicUsize::new(0));
        let mut env: VectorizedGymWrapper<TwoStepTestGym, usize> =
            TwoStepTestGym::new(device.clone(), false).into();
        let network = SharedPPONetwork::builder()
            .shared_network(CountingTrunk {
                calls: calls.clone(),
            })
            .actor_head(ProbabilisticPolicyModel::new(
                mlp(4, 2, &device),
                CategoricalDistribution,
            ))
            .critic_head(ConstantCritic {
                value: Param::from_tensor(Tensor::ones([1], &device)),
            })
            .optimizer(SgdConfig::new().init())
            .learning_rate(1e-3)
            .build()
            .unwrap();
        let mut agent = PPOAgent::builder()
            .observation_space(env.observation_space())
            .action_space(env.action_space())
            .network_info(network)
            .batch_size(2)
            .mini_batch_size(2)
            .clip_value_loss(true)
            .training_horizon(2)
            .build()
            .unwrap();
        let mut observations = agent.prepare_rollout_collection(&mut env).unwrap();
        let validation = agent.network_info.valid();
        agent
            .collect_rollout(&validation, &mut env, &mut observations)
            .unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        for transition in &agent.rollout_buffer {
            assert!(!transition.values.is_require_grad());
            assert_eq!(transition.values.clone().into_scalar::<f32>(), 1.0);
        }

        agent.add_advantages_and_returns(&validation).unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 3);
        for (transition, expected) in agent.rollout_buffer.iter().zip([1.99f32, 1.0]) {
            let prepared = transition.prepared.as_ref().unwrap();
            let old_values = prepared.old_values.as_ref().unwrap();
            assert!(!old_values.is_require_grad());
            assert_eq!(old_values.clone().into_scalar::<f32>(), 1.0);
            assert!((prepared.returns.clone().into_scalar::<f32>() - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn ppo_learns_with_custom_advantage_estimator() {
        struct RewardMinusValue;

        impl AdvantageEstimator for RewardMinusValue {
            /// Returns reward-minus-value advantages and reward value targets, shaped `[time, env_count, 1]`.
            fn estimate(
                &self,
                rewards: &Tensor<3>,
                values: &Tensor<3>,
                _next_dones: &Tensor<3, Bool>,
                _next_truncateds: &Tensor<3, Bool>,
                _bootstrapped_values: &Tensor<2>,
                _gamma: f32,
            ) -> AdvantageEstimate {
                AdvantageEstimate {
                    advantages: rewards.clone() - values.clone(),
                    value_targets: rewards.clone(),
                }
            }
        }

        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        let mut env: VectorizedGymWrapper<TwoStepTestGym, usize> = (0..3)
            .map(|_| TwoStepTestGym::new(device.clone(), false))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let actor = ProbabilisticPolicyModel::new(mlp(4, 2, &device), CategoricalDistribution);
        let networks = SeparatePPONetwork::builder()
            .actor_network(actor)
            .critic_network(mlp(4, 1, &device))
            .actor_optimizer(SgdConfig::new().init())
            .critic_optimizer(SgdConfig::new().init())
            .actor_learning_rate(1e-3)
            .critic_learning_rate(1e-3)
            .build()
            .unwrap();
        let mut agent = PPOAgent::builder()
            .observation_space(env.observation_space())
            .action_space(env.action_space())
            .network_info(networks)
            .batch_size(6)
            .mini_batch_size(6)
            .num_epochs(1)
            .training_horizon(6)
            .advantage_estimator(RewardMinusValue)
            .build()
            .unwrap();

        agent.learn(&mut env, 6).unwrap();
    }

    #[test]
    fn ppo_rejects_mismatched_advantage_estimate() {
        struct MismatchedEstimator;

        impl AdvantageEstimator for MismatchedEstimator {
            /// Returns advantages shaped `[time, env_count, 2]` and value targets shaped `[time, env_count, 1]`.
            fn estimate(
                &self,
                rewards: &Tensor<3>,
                values: &Tensor<3>,
                _next_dones: &Tensor<3, Bool>,
                _next_truncateds: &Tensor<3, Bool>,
                _bootstrapped_values: &Tensor<2>,
                _gamma: f32,
            ) -> AdvantageEstimate {
                let advantages = rewards.clone() - values.clone();
                AdvantageEstimate {
                    advantages: Tensor::cat(vec![advantages.clone(), advantages], 2),
                    value_targets: rewards.clone(),
                }
            }
        }

        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        let mut env: VectorizedGymWrapper<TwoStepTestGym, usize> = (0..3)
            .map(|_| TwoStepTestGym::new(device.clone(), false))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let actor = ProbabilisticPolicyModel::new(mlp(4, 2, &device), CategoricalDistribution);
        let networks = SeparatePPONetwork::builder()
            .actor_network(actor)
            .critic_network(mlp(4, 1, &device))
            .actor_optimizer(SgdConfig::new().init())
            .critic_optimizer(SgdConfig::new().init())
            .actor_learning_rate(1e-3)
            .critic_learning_rate(1e-3)
            .build()
            .unwrap();
        let mut agent = PPOAgent::builder()
            .observation_space(env.observation_space())
            .action_space(env.action_space())
            .network_info(networks)
            .batch_size(6)
            .mini_batch_size(6)
            .num_epochs(1)
            .training_horizon(6)
            .advantage_estimator(MismatchedEstimator)
            .build()
            .unwrap();

        assert!(matches!(
            agent.learn(&mut env, 6),
            Err(PPOError::AdvantageEstimateMismatch { .. })
        ));
    }

    #[test]
    fn shared_model_can_change_observation_rank() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        #[derive(Module, Debug)]
        struct FlattenModel;

        impl Forward<3, 2> for FlattenModel {
            type InputKind = Float;
            type OutputKind = Float;
            type Error = Infallible;

            /// Combines matrix item axes in [batch_size, 2, 2] into features [batch_size, 4].
            /// Preserves Float dtype, device, and gradient connections.
            fn forward(&self, input: Tensor<3>) -> Result<Tensor<2>, Self::Error> {
                Ok(input.flatten(1, 2))
            }
        }

        struct MatrixGym {
            device: Device,
        }

        impl Gym<(), 3> for MatrixGym {
            type Error = Infallible;
            type ObservationSpace = BoxSpace<3>;
            type ActionSpace = Discrete;

            /// Accepts an Int scalar [1] and returns unbatched Float observations [2, 2] on the fixture device.
            fn step(
                &mut self,
                _action: Tensor<1, Int>,
            ) -> Result<StepInfo<(), Tensor<2>>, Self::Error> {
                Ok(StepInfo {
                    observation: Tensor::ones([2, 2], (&self.device, DType::F32)),
                    reward: 1.0,
                    done: false,
                    truncated: false,
                    info: (),
                })
            }

            /// Returns Float observations [2, 2] on the fixture device.
            fn reset(&mut self) -> Result<ResetInfo<(), Tensor<2>>, Self::Error> {
                Ok(ResetInfo {
                    observation: Tensor::ones([2, 2], (&self.device, DType::F32)),
                    info: (),
                })
            }

            fn observation_space(&self) -> Self::ObservationSpace {
                BoxSpace::new_unbounded([1, 2, 2], &self.device)
            }

            fn action_space(&self) -> Self::ActionSpace {
                Discrete::new(2).unwrap()
            }
        }

        let device = Device::flex().autodiff();
        let mut env: VectorizedGymWrapper<MatrixGym, (), 3> = MatrixGym {
            device: device.clone(),
        }
        .into();
        let actor = ProbabilisticPolicyModel::new(mlp(4, 2, &device), CategoricalDistribution);
        let networks = SharedPPONetwork::builder()
            .shared_network(FlattenModel)
            .actor_head(actor)
            .critic_head(mlp(4, 1, &device))
            .optimizer(SgdConfig::new().init())
            .learning_rate(1e-3)
            .build()
            .unwrap();
        let mut logger = NoopLogger::default();
        let mut agent = PPOAgent::builder()
            .observation_space(env.observation_space())
            .action_space(env.action_space())
            .network_info(networks)
            .batch_size(2)
            .mini_batch_size(2)
            .num_epochs(1)
            .training_horizon(2)
            .logging_info(&mut logger)
            .build()
            .unwrap();
        agent.learn(&mut env, 2).unwrap();
        assert_eq!(
            agent.current_observations.as_ref().unwrap().dims(),
            [1, 2, 2]
        );
        assert_eq!(
            agent
                .act_deterministic(&Tensor::ones([1, 2, 2], (&device, DType::F32)))
                .unwrap()
                .dims(),
            [1, 1]
        );
        assert_eq!(logger.updates, 1);
    }

    #[test]
    fn integer_and_boolean_observations_reach_native_models() {
        struct BoolSpace;

        impl TensorBatchTypes for BoolSpace {
            type Kind = Bool;
            type Item = Tensor<1, Bool>;
            type Batch = Tensor<2, Bool>;
        }

        impl ObservationSpace<2> for BoolSpace {
            /// Accepts Bool observations [batch_size, 1] without changing their dtype, device, or axes.
            fn contains(&self, values: &Tensor<2, Bool>) -> bool {
                values.dims()[1] == 1
            }

            fn shape(&self) -> Vec<usize> {
                vec![1]
            }
        }
        let device = Device::flex().autodiff();
        let actor = ProbabilisticPolicyModel::new(
            ObservationModel::<Bool> {
                model: mlp(1, 2, &device),
                double_precision: false,
                kind: PhantomData,
            },
            CategoricalDistribution,
        );
        let networks = SeparatePPONetwork::builder()
            .actor_network(actor)
            .critic_network(ObservationModel::<Bool> {
                model: mlp(1, 1, &device),
                double_precision: false,
                kind: PhantomData,
            })
            .actor_optimizer(SgdConfig::new().init())
            .critic_optimizer(SgdConfig::new().init())
            .actor_learning_rate(1e-3)
            .critic_learning_rate(1e-3)
            .build()
            .unwrap();
        let mut boolean_agent = PPOAgent::<_, _, _, Infallible>::builder()
            .observation_space(BoolSpace)
            .action_space(Discrete::new(2).unwrap())
            .network_info(networks)
            .training_horizon(1)
            .build()
            .unwrap();
        assert_eq!(
            boolean_agent
                .act_deterministic(&Tensor::from_data([[false], [true]], &device))
                .unwrap()
                .dims(),
            [2, 1]
        );

        let actor = ProbabilisticPolicyModel::new(
            ObservationModel::<Int> {
                model: mlp(1, 2, &device),
                double_precision: false,
                kind: PhantomData,
            },
            CategoricalDistribution,
        );
        let networks = SeparatePPONetwork::builder()
            .actor_network(actor)
            .critic_network(ObservationModel::<Int> {
                model: mlp(1, 1, &device),
                double_precision: false,
                kind: PhantomData,
            })
            .actor_optimizer(SgdConfig::new().init())
            .critic_optimizer(SgdConfig::new().init())
            .actor_learning_rate(1e-3)
            .critic_learning_rate(1e-3)
            .build()
            .unwrap();
        let mut integer_agent = PPOAgent::<_, _, _, Infallible>::builder()
            .observation_space(Discrete::new(2).unwrap())
            .action_space(Discrete::new(2).unwrap())
            .network_info(networks)
            .training_horizon(1)
            .build()
            .unwrap();
        assert_eq!(
            integer_agent
                .act_deterministic(&Tensor::from_data([[0i64], [1]], (&device, DType::I64)))
                .unwrap()
                .dims(),
            [2, 1]
        );
    }

    #[test]
    fn shared_models_preserve_boolean_observations_and_use_float_features() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        #[derive(Module, Debug)]
        struct BooleanTrunk;

        impl Forward<2, 2> for BooleanTrunk {
            type InputKind = Bool;
            type OutputKind = Float;
            type Error = Infallible;

            /// Converts Bool observations [batch_size, 1] into F64 features of the same shape inside the model.
            fn forward(&self, input: Tensor<2, Bool>) -> Result<Tensor<2>, Self::Error> {
                Ok(input.cast(burn::tensor::FloatDType::F64))
            }
        }

        #[derive(Module, Debug)]
        struct FloatHead {
            weight: Param<Tensor<2>>,
        }

        impl Forward<2, 2> for FloatHead {
            type InputKind = Float;
            type OutputKind = Float;
            type Error = Infallible;

            /// Maps F64 features [batch_size, 1] to F64 outputs [batch_size, output_features].
            /// Keeps the batch axis and gradients through the weight [1, output_features].
            fn forward(&self, input: Tensor<2>) -> Result<Tensor<2>, Self::Error> {
                Ok(input.matmul(self.weight.val()))
            }
        }

        struct BooleanSpace;

        impl TensorBatchTypes for BooleanSpace {
            type Kind = Bool;
            type Item = Tensor<1, Bool>;
            type Batch = Tensor<2, Bool>;
        }

        impl ObservationSpace<2> for BooleanSpace {
            /// Checks Bool observations [batch_size, 1] without changing their layout, dtype, or device.
            fn contains(&self, values: &Tensor<2, Bool>) -> bool {
                values.dims()[1] == 1
            }

            fn shape(&self) -> Vec<usize> {
                vec![1]
            }
        }

        struct BooleanGym {
            device: Device,
        }

        impl Gym for BooleanGym {
            type Error = Infallible;
            type ObservationSpace = BooleanSpace;
            type ActionSpace = Discrete;

            /// Accepts an Int action [1] and returns terminal Bool observations [1] with an F32 reward.
            fn step(
                &mut self,
                _action: Tensor<1, Int>,
            ) -> Result<StepInfo<(), Tensor<1, Bool>>, Self::Error> {
                Ok(StepInfo {
                    observation: Tensor::from_data([true], &self.device),
                    reward: 0.25,
                    done: false,
                    truncated: true,
                    info: (),
                })
            }

            /// Returns reset Bool observations [1] on the environment device.
            fn reset(&mut self) -> Result<ResetInfo<(), Tensor<1, Bool>>, Self::Error> {
                Ok(ResetInfo {
                    observation: Tensor::from_data([false], &self.device),
                    info: (),
                })
            }

            fn observation_space(&self) -> BooleanSpace {
                BooleanSpace
            }

            fn action_space(&self) -> Discrete {
                Discrete::new(2).unwrap()
            }
        }

        let device = Device::flex().autodiff();
        let env: VectorizedGymWrapper<BooleanGym> = BooleanGym {
            device: device.clone(),
        }
        .into();
        let use_f64_rewards = std::cell::Cell::new(false);
        let mut env =
            crate::wrappers::OutputMapMultiGymWrapper::new(env, Ok::<_, Infallible>, |mut step| {
                if use_f64_rewards.get() {
                    step.rewards = step.rewards.cast(DType::F64);
                }
                Ok::<_, Infallible>(step)
            });
        let actor = ProbabilisticPolicyModel::new(
            FloatHead {
                weight: Param::from_tensor(Tensor::from_data(
                    [[0.1f64, 0.2]],
                    (&device, DType::F64),
                )),
            },
            CategoricalDistribution,
        );
        let network = SharedPPONetwork::builder()
            .shared_network(BooleanTrunk)
            .actor_head(actor)
            .critic_head(FloatHead {
                weight: Param::from_tensor(Tensor::from_data(
                    [[1.0000000001f64]],
                    (&device, DType::F64),
                )),
            })
            .optimizer(SgdConfig::new().init())
            .learning_rate(1e-3)
            .build()
            .unwrap();
        let mut agent = PPOAgent::builder()
            .observation_space(env.observation_space())
            .action_space(env.action_space())
            .network_info(network)
            .batch_size(2)
            .mini_batch_size(2)
            .num_epochs(1)
            .training_horizon(2)
            .build()
            .unwrap();
        let mut observations = agent.prepare_rollout_collection(&mut env).unwrap();
        assert!(matches!(
            agent.collect_rollout(&agent.network_info.valid(), &mut env, &mut observations),
            Err(PPOError::CriticRewardDTypeMismatch {
                critic: DType::F64,
                reward: DType::F32,
            })
        ));
        assert_eq!(agent.rollout_buffer.len(), 0);
        use_f64_rewards.set(true);
        agent
            .collect_rollout(&agent.network_info.valid(), &mut env, &mut observations)
            .unwrap();
        let transition = &agent.rollout_buffer[0];
        assert!(matches!(transition.observations.dtype(), DType::Bool(_)));
        assert_eq!(
            transition
                .next_observations
                .clone()
                .into_data()
                .try_to_vec::<bool>()
                .unwrap(),
            [true]
        );
        assert_eq!(transition.rewards.dtype(), DType::F64);
        assert_eq!(transition.training_rewards.dtype(), DType::F64);
        let expected = 0.25 + 1.0000000001 * f64::from(agent.gamma);
        assert!(
            (transition.training_rewards.clone().into_scalar::<f64>() - expected).abs() < 1e-12
        );
        let explained_variance = agent
            .add_advantages_and_returns(&agent.network_info.valid())
            .unwrap();
        let prepared = agent.rollout_buffer[0].prepared.as_ref().unwrap();
        assert_eq!(prepared.returns.dtype(), DType::F64);
        assert!((prepared.returns.clone().into_scalar::<f64>() - expected).abs() < 1e-12);
        agent.optimize(explained_variance).unwrap();
        assert_eq!(
            agent.act_deterministic(&observations).unwrap().dims(),
            [1, 1]
        );
    }

    #[test]
    fn critic_requires_reward_dtype_and_preserves_observation_dtype() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        let env: VectorizedGymWrapper<TwoStepTestGym, usize> =
            TwoStepTestGym::new(device.clone(), false).into();
        let use_f64_rewards = std::cell::Cell::new(false);
        let mut env =
            crate::wrappers::OutputMapMultiGymWrapper::new(env, Ok::<_, Infallible>, |mut step| {
                if use_f64_rewards.get() {
                    step.rewards = step.rewards.cast(DType::F64);
                }
                Ok::<_, Infallible>(step)
            });
        let model = |output| {
            MLP::builder()
                .input_size(4)
                .output_size(output)
                .hidden_layer_sizes(vec![2])
                .hidden_initializer(Initializer::Constant { value: 0.1 })
                .output_initializer(Initializer::Constant { value: 0.1 })
                .bias_initializer(Initializer::Constant { value: 0.0 })
                .options((&device, DType::F64))
                .build()
                .unwrap()
        };
        let actor = ProbabilisticPolicyModel::new(
            ObservationModel::<Float> {
                model: model(2),
                double_precision: true,
                kind: PhantomData,
            },
            CategoricalDistribution,
        );
        let networks = SeparatePPONetwork::builder()
            .actor_network(actor)
            .critic_network(ObservationModel::<Float> {
                model: model(1),
                double_precision: true,
                kind: PhantomData,
            })
            .actor_optimizer(SgdConfig::new().init())
            .critic_optimizer(SgdConfig::new().init())
            .actor_learning_rate(1e-3)
            .critic_learning_rate(1e-3)
            .build()
            .unwrap();
        struct DTypeLogger {
            updates: usize,
        }

        impl PPOLogger<usize> for DTypeLogger {
            fn log(&mut self, entry: &PPOLogEntry) {
                for tensor in [
                    &entry.actor_loss,
                    &entry.critic_loss,
                    &entry.entropy,
                    &entry.explained_variance,
                ] {
                    assert_eq!(tensor.dtype(), DType::F64);
                }
                assert_eq!(entry.returns.dtype(), DType::F64);
                assert_eq!(entry.advantages.dtype(), DType::F64);
                assert_eq!(entry.rewards.dtype(), DType::F64);
                self.updates += 1;
            }
        }
        let mut logger = DTypeLogger { updates: 0 };
        let mut agent = PPOAgent::builder()
            .observation_space(env.observation_space())
            .action_space(env.action_space())
            .network_info(networks)
            .batch_size(2)
            .mini_batch_size(2)
            .num_epochs(1)
            .training_horizon(2)
            .logging_info(&mut logger)
            .build()
            .unwrap();
        assert!(matches!(
            agent.learn(&mut env, 2),
            Err(PPOError::CriticRewardDTypeMismatch {
                critic: DType::F64,
                reward: DType::F32,
            })
        ));
        use_f64_rewards.set(true);
        agent.reset_current_observations();
        agent.learn(&mut env, 2).unwrap();
        assert_eq!(
            agent.current_observations.as_ref().unwrap().dtype(),
            DType::F32
        );
        assert_eq!(logger.updates, 1);
    }

    #[test]
    fn policy_surrogates_use_actor_dtype_with_mixed_reward_dtypes() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();

        for (actor_dtype, reward_dtype) in [(DType::F64, DType::F32), (DType::F32, DType::F64)] {
            for clipped in [true, false] {
                let device = Device::flex().autodiff();
                let env = VectorizedGymWrapper::from(crate::agents::test_support::FixedEnv::new(
                    device.clone(),
                ));
                let mut env = crate::wrappers::OutputMapMultiGymWrapper::new(
                    env,
                    Ok::<_, Infallible>,
                    move |mut step| {
                        if reward_dtype == DType::F64 {
                            step.rewards = step.rewards.cast(DType::F64);
                        }
                        Ok::<_, Infallible>(step)
                    },
                );
                let model = |output_size, dtype| {
                    MLP::builder()
                        .input_size(4)
                        .output_size(output_size)
                        .hidden_layer_sizes(vec![2])
                        .options((&device, dtype))
                        .build()
                        .unwrap()
                };
                let actor = ProbabilisticPolicyModel::new(
                    ObservationModel::<Float> {
                        model: model(2, actor_dtype),
                        double_precision: actor_dtype == DType::F64,
                        kind: PhantomData,
                    },
                    CategoricalDistribution,
                );
                let critic = ObservationModel::<Float> {
                    model: model(1, reward_dtype),
                    double_precision: reward_dtype == DType::F64,
                    kind: PhantomData,
                };
                let networks = SeparatePPONetwork::builder()
                    .actor_network(actor)
                    .critic_network(critic)
                    .actor_optimizer(SgdConfig::new().init())
                    .critic_optimizer(SgdConfig::new().init())
                    .actor_learning_rate(1e-3)
                    .critic_learning_rate(1e-3)
                    .build()
                    .unwrap();

                struct DTypeLogger {
                    actor_dtype: DType,
                    advantages_dtype: DType,
                    updates: usize,
                }

                impl PPOLogger for DTypeLogger {
                    fn log(&mut self, entry: &PPOLogEntry) {
                        assert_eq!(entry.actor_loss.dtype(), self.actor_dtype);
                        assert_eq!(entry.advantages.dtype(), self.advantages_dtype);
                        self.updates += 1;
                    }
                }

                let mut logger = DTypeLogger {
                    actor_dtype,
                    advantages_dtype: reward_dtype,
                    updates: 0,
                };
                let mut agent = PPOAgent::builder()
                    .observation_space(env.observation_space())
                    .action_space(env.action_space())
                    .network_info(networks)
                    .clipped(clipped)
                    .normalize_advantage(false)
                    .batch_size(2)
                    .mini_batch_size(2)
                    .num_epochs(1)
                    .training_horizon(2)
                    .logging_info(&mut logger)
                    .build()
                    .unwrap();

                agent.learn(&mut env, 2).unwrap();
                assert_eq!(logger.updates, 1);
            }
        }
    }

    #[test]
    fn seeded_collection_and_training_are_repeatable() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        let mut runs = Vec::new();
        for _ in 0..2 {
            let mut env: VectorizedGymWrapper<TwoStepTestGym, usize> =
                TwoStepTestGym::new(device.clone(), false).into();
            let actor = ProbabilisticPolicyModel::new(mlp(4, 2, &device), CategoricalDistribution);
            let networks = SeparatePPONetwork::builder()
                .actor_network(actor)
                .critic_network(mlp(4, 1, &device))
                .actor_optimizer(SgdConfig::new().init())
                .critic_optimizer(SgdConfig::new().init())
                .actor_learning_rate(1e-3)
                .critic_learning_rate(1e-3)
                .build()
                .unwrap();
            struct Logger(Vec<(f32, f32)>);

            impl PPOLogger<usize> for Logger {
                fn log(&mut self, entry: &PPOLogEntry) {
                    self.0.push((
                        entry.actor_loss.clone().try_into_scalar::<f32>().unwrap(),
                        entry.critic_loss.clone().try_into_scalar::<f32>().unwrap(),
                    ));
                }
            }
            let mut logger = Logger(Vec::new());
            let mut agent = PPOAgent::builder()
                .observation_space(env.observation_space())
                .action_space(env.action_space())
                .network_info(networks)
                .batch_size(4)
                .mini_batch_size(2)
                .num_epochs(2)
                .training_horizon(8)
                .logging_info(&mut logger)
                .build()
                .unwrap();
            device.seed(42);
            agent.learn(&mut env, 8).unwrap();
            let evaluation = agent
                .act_deterministic(&Tensor::zeros([2, 4], (&device, DType::F32)))
                .unwrap()
                .into_data()
                .try_to_vec::<i32>()
                .unwrap();
            runs.push((logger.0, evaluation));
        }
        assert_eq!(runs[0], runs[1]);
    }

    #[test]
    fn truncation_uses_terminal_observations_and_preserves_reset_observations() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        #[derive(Module, Debug)]
        struct ObservationCritic {
            weight: Param<Tensor<1>>,
        }

        impl Forward<2, 2> for ObservationCritic {
            type InputKind = Float;
            type OutputKind = Float;
            type Error = Infallible;

            /// Returns the first feature times the learned weight, mapping [batch_size, 4] to [batch_size, 1].
            /// Preserves Float dtype, device, and parameter gradients.
            fn forward(&self, input: Tensor<2>) -> Result<Tensor<2>, Self::Error> {
                Ok(input.slice_dim(1, 0..1) * self.weight.val().unsqueeze::<2>())
            }
        }
        struct TruncatedGym {
            device: Device,
        }

        impl Gym for TruncatedGym {
            type Error = Infallible;
            type ObservationSpace = BoxSpace;
            type ActionSpace = Discrete;

            /// Accepts one Int action [1] and returns terminal Float observations [4] filled with seven.
            fn step(&mut self, _action: Tensor<1, Int>) -> Result<StepInfo, Self::Error> {
                Ok(StepInfo {
                    observation: Tensor::full([4], 7.0, (&self.device, DType::F32)),
                    reward: 1.0,
                    done: false,
                    truncated: true,
                    info: (),
                })
            }

            /// Returns reset Float observations [4] filled with zero on the fixture device.
            fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
                Ok(ResetInfo {
                    observation: Tensor::zeros([4], (&self.device, DType::F32)),
                    info: (),
                })
            }

            fn observation_space(&self) -> Self::ObservationSpace {
                BoxSpace::new_unbounded([1, 4], &self.device)
            }

            fn action_space(&self) -> Self::ActionSpace {
                Discrete::new(2).unwrap()
            }
        }
        let device = Device::flex().autodiff();
        let mut env: VectorizedGymWrapper<TruncatedGym> = TruncatedGym {
            device: device.clone(),
        }
        .into();
        let actor = ProbabilisticPolicyModel::new(mlp(4, 2, &device), CategoricalDistribution);
        let critic = ObservationCritic {
            weight: Param::from_tensor(Tensor::ones([1], (&device, DType::F32))),
        };
        let networks = SeparatePPONetwork::builder()
            .actor_network(actor)
            .critic_network(critic)
            .actor_optimizer(SgdConfig::new().init())
            .critic_optimizer(SgdConfig::new().init())
            .actor_learning_rate(1e-3)
            .critic_learning_rate(1e-3)
            .build()
            .unwrap();
        let mut agent = PPOAgent::builder()
            .observation_space(env.observation_space())
            .action_space(env.action_space())
            .network_info(networks)
            .batch_size(1)
            .mini_batch_size(1)
            .num_epochs(1)
            .gamma(0.9)
            .training_horizon(1)
            .build()
            .unwrap();
        let mut next_observations = agent.prepare_rollout_collection(&mut env).unwrap();
        agent
            .collect_rollout(
                &agent.network_info.valid(),
                &mut env,
                &mut next_observations,
            )
            .unwrap();
        assert_eq!(
            next_observations.into_data().try_to_vec::<f32>().unwrap(),
            [0.0; 4]
        );
        let item = &agent.rollout_buffer[0];
        assert_eq!(
            item.next_observations
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            [7.0; 4]
        );
        assert_eq!(
            item.rewards
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            [1.0]
        );
        assert!(
            (item
                .training_rewards
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap()[0]
                - 7.3)
                .abs()
                < 1e-6
        );
        agent
            .add_advantages_and_returns(&agent.network_info.valid())
            .unwrap();
        let returns = agent.rollout_buffer[0]
            .prepared
            .as_ref()
            .unwrap()
            .returns
            .clone();
        assert!((returns.into_data().try_to_vec::<f32>().unwrap()[0] - 7.3).abs() < 1e-6);
    }

    #[test]
    fn surrogate_minimum_splits_tied_gradients() {
        let device = Device::flex().autodiff();
        let left = Tensor::<2>::from_floats([[1.0]], &device).require_grad();
        let right = Tensor::<2>::from_floats([[1.0]], &device).require_grad();
        let ties = left.clone().equal(right.clone());
        let loss = left
            .clone()
            .min_pair(right.clone())
            .mask_where(ties, (left.clone() + right.clone()) * 0.5)
            .mean();
        let gradients = loss.backward();
        for operand in [left, right] {
            assert_eq!(
                operand
                    .grad(&gradients)
                    .unwrap()
                    .into_data()
                    .try_to_vec::<f32>()
                    .unwrap(),
                [0.5]
            );
        }
    }

    #[test]
    fn explained_variance_uses_population_variance() {
        use burn::tensor::Tensor;

        let device = burn::tensor::Device::flex();
        let returns = Tensor::<2>::from_data([[1.0f32], [2.0], [3.0]], &device);
        let rollout_values = Tensor::<2>::from_data([[1.0f32], [2.0], [2.0]], &device);

        let actual = compute_explained_variance(&returns, &rollout_values)
            .try_into_scalar::<f32>()
            .unwrap();

        assert!((actual - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn explained_variance_is_nan_for_constant_returns() {
        use burn::tensor::Tensor;

        let device = burn::tensor::Device::flex();
        let returns = Tensor::<2>::from_data([[1.0f32], [1.0], [1.0]], &device);
        let rollout_values = Tensor::<2>::from_data([[0.0f32], [1.0], [2.0]], &device);

        let actual = compute_explained_variance(&returns, &rollout_values)
            .try_into_scalar::<f32>()
            .unwrap();

        assert!(actual.is_nan());

        assert!(
            compute_explained_variance(&returns, &returns)
                .try_into_scalar::<f32>()
                .unwrap()
                .is_nan()
        );
    }

    #[test]
    fn population_variance_preserves_f64_device_and_gradients() {
        use burn::tensor::Tensor;

        let device = burn::tensor::Device::flex().autodiff();
        let values = Tensor::<3>::from_data(
            [[[1.0f64], [2.0], [3.0]]],
            (&device, burn::tensor::DType::F64),
        )
        .require_grad();
        let variance = population_variance(&values);
        assert_eq!(variance.dims(), [1]);
        assert_eq!(variance.dtype(), burn::tensor::DType::F64);
        assert_eq!(variance.device(), device);
        assert_close(
            variance.clone().try_into_scalar::<f64>().unwrap(),
            2.0 / 3.0,
        );

        let gradients = variance.backward();
        let gradient = values
            .grad(&gradients)
            .unwrap()
            .into_data()
            .try_to_vec::<f64>()
            .unwrap();
        for (actual, expected) in gradient.into_iter().zip([-2.0 / 3.0, 0.0, 2.0 / 3.0]) {
            assert_close(actual, expected);
        }
    }

    #[test]
    fn simultaneous_termination_and_truncation_does_not_bootstrap() {
        #[derive(Module, Debug)]
        struct UnitCritic;

        impl Forward<2, 2> for UnitCritic {
            type InputKind = Float;
            type OutputKind = Float;
            type Error = Infallible;

            /// Returns Float values [batch_size, 1] for observations [batch_size, 4], preserving dtype and device.
            fn forward(&self, input: Tensor<2>) -> Result<Tensor<2>, Self::Error> {
                Ok(Tensor::ones(
                    [input.dims()[0], 1],
                    (&input.device(), input.dtype()),
                ))
            }
        }
        let device = Device::flex().autodiff();
        let actor = ProbabilisticPolicyModel::new(mlp(4, 2, &device), CategoricalDistribution);
        let networks = SeparatePPONetwork::builder()
            .actor_optimizer(SgdConfig::new().init())
            .critic_optimizer(SgdConfig::new().init())
            .actor_learning_rate(1e-3)
            .critic_learning_rate(1e-3)
            .actor_network(actor)
            .critic_network(UnitCritic)
            .build()
            .unwrap();
        let mut agent = PPOAgent::<_, _, _, Infallible>::builder()
            .observation_space(BoxSpace::new_unbounded([1, 4], &device))
            .action_space(Discrete::new(2).unwrap())
            .network_info(networks)
            .batch_size(2)
            .mini_batch_size(2)
            .training_horizon(2)
            .build()
            .unwrap();
        let observations = Tensor::zeros([2, 4], (&device, DType::F32));
        let actual = agent
            .bootstrap_truncated_rewards(
                &agent.network_info.valid(),
                Tensor::zeros([2, 1], (&device, DType::F32)),
                &observations,
                &[true, false],
                &[true, true],
            )
            .unwrap();
        assert_eq!(actual.into_data().try_to_vec::<f32>().unwrap(), [0.0, 0.99]);
        assert!(matches!(
            agent.bootstrap_truncated_rewards(
                &agent.network_info.valid(),
                Tensor::zeros([2, 1], (&device, DType::F64)),
                &observations,
                &[true, false],
                &[true, true],
            ),
            Err(PPOError::CriticRewardDTypeMismatch {
                critic: DType::F32,
                reward: DType::F64,
            })
        ));
    }

    #[test]
    fn collection_logs_report_fresh_rewards_and_completed_episodes() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        let mut env: VectorizedGymWrapper<TwoStepTestGym, usize> = vec![
            TwoStepTestGym::new(device.clone(), false),
            TwoStepTestGym::new(device.clone(), true),
        ]
        .try_into()
        .unwrap();
        let actor = ProbabilisticPolicyModel::new(mlp(4, 2, &device), CategoricalDistribution);
        let networks = SeparatePPONetwork::builder()
            .actor_optimizer(SgdConfig::new().init())
            .critic_optimizer(SgdConfig::new().init())
            .actor_learning_rate(1e-3)
            .critic_learning_rate(1e-3)
            .actor_network(actor)
            .critic_network(mlp(4, 1, &device))
            .build()
            .unwrap();
        struct CollectionLogger {
            collection_rewards: Vec<Vec<f32>>,
            optimization_rewards: Vec<Vec<f32>>,
            infos: Vec<Vec<usize>>,
            completed_episodes: Vec<(usize, f32, usize, bool, bool, usize)>,
        }
        impl PPOLogger<usize> for CollectionLogger {
            fn log(&mut self, entry: &PPOLogEntry) {
                self.optimization_rewards.push(
                    entry
                        .rewards
                        .clone()
                        .into_data()
                        .try_to_vec::<f32>()
                        .unwrap(),
                );
            }

            fn log_collection(&mut self, entry: &PPOCollectionLogEntry<usize>) {
                self.collection_rewards.push(
                    entry
                        .collection_rewards
                        .clone()
                        .into_data()
                        .try_to_vec::<f32>()
                        .unwrap(),
                );
                self.infos.push(entry.infos.clone());
                self.completed_episodes
                    .extend(entry.completed_episodes.iter().map(|episode| {
                        (
                            episode.environment_index,
                            episode.episode_return,
                            episode.episode_length,
                            episode.terminated,
                            episode.truncated,
                            episode.collection_timestep,
                        )
                    }));
            }
        }
        let mut logger = CollectionLogger {
            collection_rewards: Vec::new(),
            optimization_rewards: Vec::new(),
            infos: Vec::new(),
            completed_episodes: Vec::new(),
        };
        let mut agent = PPOAgent::builder()
            .observation_space(env.observation_space())
            .action_space(env.action_space())
            .network_info(networks)
            .batch_size(2)
            .mini_batch_size(2)
            .num_epochs(1)
            .training_horizon(4)
            .logging_info(&mut logger)
            .build()
            .unwrap();
        agent.learn(&mut env, 2).unwrap();
        agent.learn(&mut env, 2).unwrap();
        assert_eq!(logger.collection_rewards, [vec![1.0, 1.0], vec![1.0, 1.0]]);
        assert_eq!(
            logger.optimization_rewards,
            [vec![1.0, 1.0], vec![1.0, 1.0]]
        );
        assert_eq!(logger.infos, [vec![1, 1], vec![2, 2]]);
        assert_eq!(
            logger.completed_episodes,
            [(0, 2.0, 2, true, false, 3), (1, 2.0, 2, false, true, 4)]
        );
    }

    #[test]
    fn reset_current_observations_restarts_caller_episodes() {
        struct EpisodeLogger(Vec<(f32, usize)>);

        impl PPOLogger<usize> for EpisodeLogger {
            fn log(&mut self, _entry: &PPOLogEntry) {}

            fn log_collection(&mut self, entry: &PPOCollectionLogEntry<usize>) {
                self.0.extend(
                    entry
                        .completed_episodes
                        .iter()
                        .map(|episode| (episode.episode_return, episode.episode_length)),
                );
            }
        }

        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        let mut env = VectorizedGymWrapper::from(TwoStepTestGym::new(device.clone(), false));
        let mut logger = EpisodeLogger(Vec::new());
        let networks = SeparatePPONetwork::builder()
            .actor_network(ProbabilisticPolicyModel::new(
                mlp(4, 2, &device),
                CategoricalDistribution,
            ))
            .critic_network(mlp(4, 1, &device))
            .actor_optimizer(SgdConfig::new().init())
            .critic_optimizer(SgdConfig::new().init())
            .actor_learning_rate(1e-3)
            .critic_learning_rate(1e-3)
            .build()
            .unwrap();
        let mut agent = PPOAgent::<_, _, _, VectorizedGymError<Infallible>, usize>::builder()
            .observation_space(env.observation_space())
            .action_space(env.action_space())
            .network_info(networks)
            .batch_size(8)
            .mini_batch_size(8)
            .num_epochs(1)
            .training_horizon(4)
            .logging_info(&mut logger)
            .build()
            .unwrap();

        let observations = env.reset().unwrap();
        let actions = agent.act_for_training(&observations).unwrap();
        let step = env.step(actions.to_environment_actions().unwrap()).unwrap();
        let next_observations = step.observations.clone();
        agent
            .learn_from_timesteps(&observations, actions, step)
            .unwrap();

        agent.reset_current_observations();

        let actions = agent.act_for_training(&next_observations).unwrap();
        let step = env.step(actions.to_environment_actions().unwrap()).unwrap();
        agent
            .learn_from_timesteps(&next_observations, actions, step)
            .unwrap();

        assert_eq!(logger.0, [(1.0, 1)]);
    }

    #[test]
    fn learn_after_reset_restarts_episode_totals() {
        #[derive(Debug, thiserror::Error)]
        #[error("scripted step failure")]
        struct ScriptedGymError;

        struct ScriptedGym {
            device: Device,
            steps: usize,
        }

        impl Gym<usize> for ScriptedGym {
            type Error = ScriptedGymError;
            type ObservationSpace = BoxSpace;
            type ActionSpace = Discrete;

            /// Accepts Int actions [1] and returns Float observations [4] on the fixture device.
            fn step(&mut self, _action: Tensor<1, Int>) -> Result<StepInfo<usize>, Self::Error> {
                self.steps += 1;
                match self.steps {
                    1 => Ok(StepInfo {
                        observation: Tensor::zeros([4], (&self.device, DType::F32)),
                        reward: 1.0,
                        done: false,
                        truncated: false,
                        info: self.steps,
                    }),
                    2 | 4 => Err(ScriptedGymError),
                    3 => Ok(StepInfo {
                        observation: Tensor::zeros([4], (&self.device, DType::F32)),
                        reward: 2.0,
                        done: true,
                        truncated: false,
                        info: self.steps,
                    }),
                    _ => unreachable!(),
                }
            }

            /// Returns Float observations [4] on the fixture device.
            fn reset(&mut self) -> Result<ResetInfo<usize>, Self::Error> {
                Ok(ResetInfo {
                    observation: Tensor::zeros([4], (&self.device, DType::F32)),
                    info: 0,
                })
            }

            fn observation_space(&self) -> Self::ObservationSpace {
                BoxSpace::new_with_universal_bounds([1, 4], -1.0, 1.0, &self.device)
            }

            fn action_space(&self) -> Self::ActionSpace {
                Discrete::new(2).unwrap()
            }
        }

        struct EpisodeLogger(Vec<(f32, usize)>);

        impl PPOLogger<usize> for EpisodeLogger {
            fn log(&mut self, _entry: &PPOLogEntry) {}

            fn log_collection(&mut self, entry: &PPOCollectionLogEntry<usize>) {
                self.0.extend(
                    entry
                        .completed_episodes
                        .iter()
                        .map(|episode| (episode.episode_return, episode.episode_length)),
                );
            }
        }

        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        let mut env = VectorizedGymWrapper::from(ScriptedGym {
            device: device.clone(),
            steps: 0,
        });
        let mut logger = EpisodeLogger(Vec::new());
        let networks = SeparatePPONetwork::builder()
            .actor_network(ProbabilisticPolicyModel::new(
                mlp(4, 2, &device),
                CategoricalDistribution,
            ))
            .critic_network(mlp(4, 1, &device))
            .actor_optimizer(SgdConfig::new().init())
            .critic_optimizer(SgdConfig::new().init())
            .actor_learning_rate(1e-3)
            .critic_learning_rate(1e-3)
            .build()
            .unwrap();
        let mut agent = PPOAgent::<_, _, _, VectorizedGymError<ScriptedGymError>, usize>::builder()
            .observation_space(env.observation_space())
            .action_space(env.action_space())
            .network_info(networks)
            .batch_size(8)
            .mini_batch_size(8)
            .num_epochs(1)
            .training_horizon(8)
            .logging_info(&mut logger)
            .build()
            .unwrap();

        assert!(agent.learn(&mut env, 2).is_err());
        assert!(agent.learn(&mut env, 1).is_err());
        assert_eq!(logger.0, [(2.0, 1)]);
    }

    #[test]
    fn caller_collected_rollout_matches_learn() {
        struct Parameters(Vec<f32>);

        impl ModuleVisitor for Parameters {
            /// Visits network parameters shaped `[parameter_shape]`.
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
                self.0
                    .extend(param.val().into_data().try_to_vec::<f32>().unwrap());
            }
        }

        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        let build_env = || {
            VectorizedGymWrapper::new(vec![
                TwoStepTestGym::new(device.clone(), false),
                TwoStepTestGym::new(device.clone(), true),
            ])
            .unwrap()
        };
        let build_agent = |observation_space, action_space| {
            let networks = SeparatePPONetwork::builder()
                .actor_network(ProbabilisticPolicyModel::new(
                    mlp(4, 2, &device),
                    CategoricalDistribution,
                ))
                .critic_network(mlp(4, 1, &device))
                .actor_optimizer(SgdConfig::new().init())
                .critic_optimizer(SgdConfig::new().init())
                .actor_learning_rate(1e-3)
                .critic_learning_rate(1e-3)
                .build()
                .unwrap();
            PPOAgent::<_, _, _, VectorizedGymError<Infallible>, usize>::builder()
                .observation_space(observation_space)
                .action_space(action_space)
                .network_info(networks)
                .batch_size(4)
                .mini_batch_size(4)
                .num_epochs(1)
                .training_horizon(8)
                .build()
                .unwrap()
        };

        device.seed(153);
        let mut learned_env = build_env();
        let mut learned_agent =
            build_agent(learned_env.observation_space(), learned_env.action_space());
        learned_agent.learn(&mut learned_env, 4).unwrap();
        let mut learned_parameters = Parameters(Vec::new());
        learned_agent.network().visit(&mut learned_parameters);

        device.seed(153);
        let mut caller_env = build_env();
        let mut caller_agent =
            build_agent(caller_env.observation_space(), caller_env.action_space());
        let mut observations = caller_env.reset().unwrap();
        for _ in 0..2 {
            let training_actions = caller_agent.act_for_training(&observations).unwrap();
            let step = caller_env
                .step(training_actions.to_environment_actions().unwrap())
                .unwrap();
            let next_observations = step.observations.clone();
            caller_agent
                .learn_from_timesteps(&observations, training_actions, step)
                .unwrap();
            observations = next_observations;
        }
        let mut caller_parameters = Parameters(Vec::new());
        caller_agent.network().visit(&mut caller_parameters);
        assert_eq!(learned_parameters.0, caller_parameters.0);
    }

    struct FailOnceEstimator {
        fail_next: Arc<AtomicBool>,
        time_steps: Arc<Mutex<Vec<usize>>>,
    }

    impl AdvantageEstimator for FailOnceEstimator {
        /// Records rollout lengths and returns mismatched advantages shaped `[time, env_count, 2]` once.
        fn estimate(
            &self,
            rewards: &Tensor<3>,
            values: &Tensor<3>,
            _next_dones: &Tensor<3, Bool>,
            _next_truncateds: &Tensor<3, Bool>,
            _bootstrapped_values: &Tensor<2>,
            _gamma: f32,
        ) -> AdvantageEstimate {
            self.time_steps.lock().unwrap().push(rewards.dims()[0]);
            let advantages = rewards.clone() - values.clone();
            if self.fail_next.swap(false, Ordering::Relaxed) {
                AdvantageEstimate {
                    advantages: Tensor::cat(vec![advantages.clone(), advantages], 2),
                    value_targets: rewards.clone(),
                }
            } else {
                AdvantageEstimate {
                    advantages,
                    value_targets: rewards.clone(),
                }
            }
        }
    }

    #[test]
    fn caller_rollout_discards_steps_after_failed_update() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        let fail_next = Arc::new(AtomicBool::new(true));
        let time_steps = Arc::new(Mutex::new(Vec::new()));
        let mut env: VectorizedGymWrapper<TwoStepTestGym, usize> = (0..3)
            .map(|index| TwoStepTestGym::new(device.clone(), index == 1))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let networks = SeparatePPONetwork::builder()
            .actor_network(ProbabilisticPolicyModel::new(
                mlp(4, 2, &device),
                CategoricalDistribution,
            ))
            .critic_network(mlp(4, 1, &device))
            .actor_optimizer(SgdConfig::new().init())
            .critic_optimizer(SgdConfig::new().init())
            .actor_learning_rate(1e-3)
            .critic_learning_rate(1e-3)
            .build()
            .unwrap();
        let mut agent = PPOAgent::<_, _, _, VectorizedGymError<Infallible>, usize>::builder()
            .observation_space(env.observation_space())
            .action_space(env.action_space())
            .network_info(networks)
            .batch_size(6)
            .mini_batch_size(6)
            .num_epochs(1)
            .training_horizon(12)
            .advantage_estimator(FailOnceEstimator {
                fail_next,
                time_steps: time_steps.clone(),
            })
            .build()
            .unwrap();

        let mut observations = env.reset().unwrap();
        for step_index in 0..4 {
            let training_actions = agent.act_for_training(&observations).unwrap();
            let step = env
                .step(training_actions.to_environment_actions().unwrap())
                .unwrap();
            let next_observations = step.observations.clone();
            let result = agent.learn_from_timesteps(&observations, training_actions, step);
            if step_index == 1 {
                assert!(result.is_err());
            } else {
                result.unwrap();
            }
            observations = next_observations;
        }

        assert_eq!(*time_steps.lock().unwrap(), vec![2, 2]);
    }

    #[test]
    fn caller_rollout_discards_rollout_after_failed_learn() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        let fail_next = Arc::new(AtomicBool::new(true));
        let time_steps = Arc::new(Mutex::new(Vec::new()));
        let mut env: VectorizedGymWrapper<TwoStepTestGym, usize> = (0..3)
            .map(|index| TwoStepTestGym::new(device.clone(), index == 1))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let networks = SeparatePPONetwork::builder()
            .actor_network(ProbabilisticPolicyModel::new(
                mlp(4, 2, &device),
                CategoricalDistribution,
            ))
            .critic_network(mlp(4, 1, &device))
            .actor_optimizer(SgdConfig::new().init())
            .critic_optimizer(SgdConfig::new().init())
            .actor_learning_rate(1e-3)
            .critic_learning_rate(1e-3)
            .build()
            .unwrap();
        let mut agent = PPOAgent::<_, _, _, VectorizedGymError<Infallible>, usize>::builder()
            .observation_space(env.observation_space())
            .action_space(env.action_space())
            .network_info(networks)
            .batch_size(6)
            .mini_batch_size(6)
            .num_epochs(1)
            .training_horizon(12)
            .advantage_estimator(FailOnceEstimator {
                fail_next,
                time_steps: time_steps.clone(),
            })
            .build()
            .unwrap();

        assert!(agent.learn(&mut env, 6).is_err());
        let mut observations = env.reset().unwrap();
        for _ in 0..2 {
            let training_actions = agent.act_for_training(&observations).unwrap();
            let step = env
                .step(training_actions.to_environment_actions().unwrap())
                .unwrap();
            let next_observations = step.observations.clone();
            agent
                .learn_from_timesteps(&observations, training_actions, step)
                .unwrap();
            observations = next_observations;
        }

        assert_eq!(*time_steps.lock().unwrap(), vec![2, 2]);
    }

    #[test]
    fn caller_collected_rollout_rejects_environment_count_changes() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        let mut env: VectorizedGymWrapper<TwoStepTestGym, usize> = vec![
            TwoStepTestGym::new(device.clone(), false),
            TwoStepTestGym::new(device.clone(), true),
        ]
        .try_into()
        .unwrap();
        let mut agent = PPOAgent::<_, _, _, VectorizedGymError<Infallible>, usize>::builder()
            .observation_space(env.observation_space())
            .action_space(env.action_space())
            .network_info(
                SeparatePPONetwork::builder()
                    .actor_network(ProbabilisticPolicyModel::new(
                        mlp(4, 2, &device),
                        CategoricalDistribution,
                    ))
                    .critic_network(mlp(4, 1, &device))
                    .actor_optimizer(SgdConfig::new().init())
                    .critic_optimizer(SgdConfig::new().init())
                    .actor_learning_rate(1e-3)
                    .critic_learning_rate(1e-3)
                    .build()
                    .unwrap(),
            )
            .batch_size(4)
            .mini_batch_size(4)
            .num_epochs(1)
            .training_horizon(8)
            .build()
            .unwrap();

        let observations = env.reset().unwrap();
        let training_actions = agent.act_for_training(&observations).unwrap();
        let step = env
            .step(training_actions.to_environment_actions().unwrap())
            .unwrap();
        let next_observations = step.observations.clone();
        agent
            .learn_from_timesteps(&observations, training_actions, step)
            .unwrap();

        let mut single_env = VectorizedGymWrapper::from(TwoStepTestGym::new(device.clone(), false));
        let single_observations = single_env.reset().unwrap();
        let single_actions = agent.act_for_training(&single_observations).unwrap();
        let single_step = single_env
            .step(single_actions.to_environment_actions().unwrap())
            .unwrap();
        assert!(matches!(
            agent.learn_from_timesteps(&single_observations, single_actions, single_step),
            Err(PPOError::BatchLengthMismatch {
                field: "environment count",
                expected: 2,
                actual: 1,
            })
        ));

        let training_actions = agent.act_for_training(&next_observations).unwrap();
        let step = env
            .step(training_actions.to_environment_actions().unwrap())
            .unwrap();
        agent
            .learn_from_timesteps(&next_observations, training_actions, step)
            .unwrap();
    }

    #[test]
    fn separate_ppo_losses_only_reach_their_parameters() {
        struct ParameterIds(HashSet<ParamId>);

        impl ModuleVisitor for ParameterIds {
            /// Records rank-D parameter IDs without reading values or changing tensor shapes.
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
                self.0.insert(param.id);
            }
        }
        struct GradientIds<'a> {
            gradients: &'a GradientsParams,
            ids: HashSet<ParamId>,
        }

        impl ModuleVisitor for GradientIds<'_> {
            /// Records IDs with rank-D gradients, preserving gradient shapes and the gradient store.
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
                if self.gradients.get::<D>(param.id).is_some() {
                    self.ids.insert(param.id);
                }
            }
        }
        let device = Device::flex().autodiff();
        let actor =
            ProbabilisticPolicyModel::new(mlp(4, 6, &device), GaussianDistribution::default());
        let critic = mlp(4, 1, &device);
        let mut actor_ids = ParameterIds(HashSet::new());
        let mut critic_ids = ParameterIds(HashSet::new());
        actor.visit(&mut actor_ids);
        critic.visit(&mut critic_ids);
        let models = (actor.clone(), critic.clone());
        let networks = SeparatePPONetwork::builder()
            .actor_network(actor)
            .critic_network(critic)
            .actor_optimizer(SgdConfig::new().init())
            .critic_optimizer(SgdConfig::new().init())
            .actor_learning_rate(3e-4)
            .critic_learning_rate(3e-4)
            .combined_loss(true)
            .build()
            .unwrap();
        let mut agent = PPOAgent::<_, _, _, Infallible>::builder()
            .observation_space(BoxSpace::new_unbounded([1, 4], &device))
            .action_space(BoxSpace::new_with_universal_bounds(
                [1, 3],
                -1.0,
                1.0,
                &device,
            ))
            .network_info(networks)
            .batch_size(4)
            .mini_batch_size(4)
            .ent_coef(0.001)
            .clip_value_loss(true)
            .training_horizon(4)
            .build()
            .unwrap();

        let observations = Tensor::<2>::from_floats(
            [
                [0.1, 0.2, 0.3, 0.4],
                [-0.1, 0.3, -0.2, 0.5],
                [0.7, -0.4, 0.2, -0.3],
                [-0.5, -0.2, 0.6, 0.1],
            ],
            &device,
        )
        .require_grad();
        let actions = Tensor::<2>::full([4, 3], 0.1, (&device, DType::F32)).require_grad();
        let log_probs = Tensor::<2>::full([4, 1], -1.0, (&device, DType::F32)).require_grad();
        let advantages =
            Tensor::<2>::from_floats([[1.0], [-1.0], [0.5], [-0.5]], &device).require_grad();
        let returns =
            Tensor::<2>::from_floats([[1.0], [2.0], [3.0], [4.0]], &device).require_grad();
        let rewards = Tensor::<2>::full([4, 1], 0.1, (&device, DType::F32)).require_grad();
        let old_values = Tensor::<2>::zeros([4, 1], (&device, DType::F32)).require_grad();
        for branch in 0..3 {
            let losses = agent
                .compute_loss()
                .observations(observations.clone().detach())
                .actions(actions.clone().detach())
                .old_log_probs(log_probs.clone().detach())
                .advantages(advantages.clone().detach())
                .returns(returns.clone().detach())
                .rewards(rewards.clone().detach())
                .old_values(old_values.clone().detach())
                .explained_variance(Tensor::zeros([1], (&device, DType::F32)))
                .clip_range(0.2)
                .call()
                .unwrap();
            let gradients = match branch {
                0 => losses.actor_loss.backward(),
                1 => losses.critic_loss.backward(),
                _ => (losses.actor_loss + losses.critic_loss).backward(),
            };
            assert!(observations.grad(&gradients).is_none());
            assert!(actions.grad(&gradients).is_none());
            for tensor in [&log_probs, &advantages, &returns, &rewards, &old_values] {
                assert!(tensor.grad(&gradients).is_none());
            }
            let gradients = GradientsParams::from_grads(gradients, &models);
            let mut ids = GradientIds {
                gradients: &gradients,
                ids: HashSet::new(),
            };
            models.visit(&mut ids);
            match branch {
                0 => {
                    assert!(actor_ids.0.is_subset(&ids.ids));
                    assert!(critic_ids.0.is_disjoint(&ids.ids));
                }
                1 => {
                    assert!(critic_ids.0.is_subset(&ids.ids));
                    assert!(actor_ids.0.is_disjoint(&ids.ids));
                }
                _ => {
                    assert!(actor_ids.0.is_subset(&ids.ids));
                    assert!(critic_ids.0.is_subset(&ids.ids));
                }
            }
        }
    }
}

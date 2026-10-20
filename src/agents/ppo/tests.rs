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
    tensor::{DType, Device, Float},
};
use std::convert::Infallible;

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
        Discrete::new(2)
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

    use burn::{module::ParamId, nn::Initializer, optim::SgdConfig, tensor::kind::Kind};
    use std::convert::Infallible;

    use super::*;
    use crate::{
        agents::Agent,
        distributions::{CategoricalDistribution, GaussianDistribution},
        gym::{Gym, MultiGym, ResetInfo, StepInfo, VectorizedGymWrapper},
        models::{MLP, probabilistic_model::ProbabilisticPolicyModel},
        spaces::{BoxSpace, Discrete},
    };

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
            .device(device)
            .build()
            .unwrap();
        let mut observations = agent.prepare_rollout_collection(&mut env).unwrap();
        let validation = agent.network_info.valid();
        agent
            .collect_rollout(&validation, &mut env, &mut observations)
            .unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        for transition in agent.rollout_buffer.get_raw() {
            assert!(!transition.values.is_require_grad());
            assert_eq!(transition.values.clone().into_scalar::<f32>(), 1.0);
        }

        agent.add_advantages_and_returns(&validation).unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 3);
        for (transition, expected) in agent.rollout_buffer.get_raw().iter().zip([1.99f32, 1.0]) {
            let prepared = transition.prepared.as_ref().unwrap();
            let old_values = prepared.old_values.as_ref().unwrap();
            assert!(!old_values.is_require_grad());
            assert_eq!(old_values.clone().into_scalar::<f32>(), 1.0);
            assert!((prepared.returns.clone().into_scalar::<f32>() - expected).abs() < 1e-6);
        }
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
                Discrete::new(2)
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
            .device(device.clone())
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
            .action_space(Discrete::new(2))
            .network_info(networks)
            .training_horizon(1)
            .device(device.clone())
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
            .observation_space(Discrete::new(2))
            .action_space(Discrete::new(2))
            .network_info(networks)
            .training_horizon(1)
            .device(device.clone())
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
                Discrete::new(2)
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
            .device(device)
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
        let transition = &agent.rollout_buffer.get_raw()[0];
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
        let prepared = agent.rollout_buffer.get_raw()[0].prepared.as_ref().unwrap();
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
            .device(device)
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
                .device(device.clone())
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
                Discrete::new(2)
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
            .device(device)
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
        let item = &agent.rollout_buffer.get_raw()[0];
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
        let returns = agent.rollout_buffer.get_raw()[0]
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
            .action_space(Discrete::new(2))
            .network_info(networks)
            .batch_size(2)
            .mini_batch_size(2)
            .training_horizon(2)
            .device(device.clone())
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
            .device(device)
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
            .device(device.clone())
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

    #[test]
    fn gae_matches_scalar_reference_across_environments_and_episode_boundaries() {
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
            .gamma(0.9)
            .gae_lambda(0.8)
            .device(device.clone())
            .build()
            .unwrap();
        agent.learn(&mut env, 0).unwrap();
        let rewards = [[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let values = [[0.2_f32, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]];
        let dones = [
            [false, false, false],
            [true, false, false],
            [false, true, false],
        ];
        let truncateds = [
            [false, false, false],
            [false, true, false],
            [false, false, true],
        ];
        let bootstrap = [1.1_f32, 1.2, 1.3];
        let reward_tensor = Tensor::<3>::from_data(
            TensorData::new(rewards.into_iter().flatten().collect::<Vec<_>>(), [3, 3, 1]),
            &device,
        );
        let value_tensor = Tensor::<3>::from_data(
            TensorData::new(values.into_iter().flatten().collect::<Vec<_>>(), [3, 3, 1]),
            &device,
        );
        let done_tensor = Tensor::from_data(
            TensorData::new(dones.into_iter().flatten().collect::<Vec<_>>(), [3, 3, 1]),
            &device,
        );
        let truncated_tensor = Tensor::from_data(
            TensorData::new(
                truncateds.into_iter().flatten().collect::<Vec<_>>(),
                [3, 3, 1],
            ),
            &device,
        );
        let bootstrap_tensor =
            Tensor::<2>::from_data(TensorData::new(bootstrap.to_vec(), [3, 1]), &device);
        for dtype in [DType::F32, DType::F64] {
            let advantages = agent.compute_gae(
                &reward_tensor.clone().cast(dtype),
                &value_tensor.clone().cast(dtype),
                &done_tensor,
                &truncated_tensor,
                &bootstrap_tensor.clone().cast(dtype),
            );
            assert_eq!(advantages.dtype(), dtype);
            let actual = advantages
                .into_data()
                .convert::<f64>()
                .try_to_vec::<f64>()
                .unwrap();
            for env_idx in 0..3 {
                let mut next_value = bootstrap[env_idx];
                let mut gae = 0.0_f32;
                for time in (0..3).rev() {
                    let mask = if dones[time][env_idx] || truncateds[time][env_idx] {
                        0.0
                    } else {
                        1.0
                    };
                    let value = values[time][env_idx];
                    let delta = rewards[time][env_idx] + 0.9 * next_value * mask - value;
                    gae = delta + 0.9 * 0.8 * gae * mask;
                    assert!((actual[time * 3 + env_idx] - f64::from(gae)).abs() < 1e-5);
                    next_value = value;
                }
            }
        }
    }
}

use bon::bon;
use burn::{
    module::{Module, ModuleMapper, ModuleVisitor, Param},
    optim::{GradientsParams, ModuleOptimizer},
    tensor::{
        Device, Float, Tensor, TensorReadError,
        kind::{Autodiff, Basic},
    },
};
use std::{marker::PhantomData, sync::Arc};

use super::PPOConfigurationError;
use crate::{
    models::{Forward, probabilistic_model::ProbabilisticPolicy},
    parameter_schedule::{ParameterSchedule, ScheduleProgress},
    tensor_rank::{NextRank, PrevRank},
};

struct PPOGradientSelection<'a> {
    gradients: &'a GradientsParams,
    selected: GradientsParams,
}

impl ModuleVisitor for PPOGradientSelection<'_> {
    /// Copies a parameter's rank-D gradient into the actor's store, preserving shape, dtype, and device.
    /// Keeps the combined store intact so shared parameter IDs remain available to the critic optimizer.
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        if let Some(gradient) = self.gradients.get::<D>(param.id) {
            self.selected.register(param.id, gradient);
        }
    }
}

/// Owns a shared model whose output feeds both actor and critic heads.
/// One optimizer updates all three models using one learning rate and schedule.
/// Module operations affect the models. Validation snapshots clone optimizer state and share schedules.
/// Learning-rate schedules must implement `Send + Sync`.
#[derive(Clone)]
pub struct SharedPPONetwork<T: Module, A: Module, C: Module> {
    optimizer: ModuleOptimizer,
    learning_rate: f64,
    model: (T, A, C),
    lr_scheduler: Option<Arc<dyn ParameterSchedule + Send + Sync>>,
}

impl<T: Module, A: Module, C: Module> std::fmt::Debug for SharedPPONetwork<T, A, C> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SharedPPONetwork")
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl<T: Module, A: Module, C: Module> Module for SharedPPONetwork<T, A, C> {
    fn collect_devices(&self, devices: Vec<Device>) -> Vec<Device> {
        self.model.collect_devices(devices)
    }

    fn fork(mut self, device: &Device) -> Self {
        self.model = self.model.fork(device);
        self
    }

    fn to_device(mut self, device: &Device) -> Self {
        self.model = self.model.to_device(device);
        self
    }

    fn visit<Visitor: ModuleVisitor>(&self, visitor: &mut Visitor) {
        self.model.visit(visitor);
    }

    fn map<Mapper: ModuleMapper>(mut self, mapper: &mut Mapper) -> Self {
        self.model = self.model.map(mapper);
        self
    }

    fn train(mut self) -> Self {
        self.model = self.model.train();
        self
    }

    fn valid(&self) -> Self {
        Self {
            model: self.model.valid(),
            ..self.clone()
        }
    }

    fn materialize(mut self) -> Self {
        self.model = self.model.materialize();
        self
    }
}

#[bon]
impl<T: Module, A: Module, C: Module> SharedPPONetwork<T, A, C> {
    /// Returns a snapshot of the shared model, actor head, and critic head, in that order.
    /// Parameter tensors retain their shape, dtype, device, and current values.
    pub fn models(&self) -> (T, A, C) {
        self.model.clone()
    }

    /// Owns the models and optimizer, and enables training on the models.
    /// Model input and output contracts must agree. Models must share a device; Critic outputs and PPO statistics must use the environment reward dtype.
    /// Rejects a nonpositive or non-finite learning rate.
    #[builder]
    pub fn new(
        #[builder(into)] optimizer: ModuleOptimizer,
        learning_rate: f64,
        shared_network: T,
        critic_head: C,
        actor_head: A,
        #[builder(with = |schedule: impl ParameterSchedule + Send + Sync + 'static| Arc::new(schedule))]
        lr_scheduler: Option<Arc<dyn ParameterSchedule + Send + Sync>>,
    ) -> Result<Self, PPOConfigurationError> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(PPOConfigurationError::InvalidLearningRate);
        }
        Ok(Self {
            optimizer,
            learning_rate,
            model: (shared_network, actor_head, critic_head).train(),
            lr_scheduler,
        })
    }
}

/// Owns separate actor and critic models with independently configurable optimizers, rates, and schedules.
/// `combined_loss` selects a shared gradient-clipping norm instead of independent actor and critic norms.
/// Module operations affect the models. Validation snapshots clone optimizer state and share schedules.
/// Learning-rate schedules must implement `Send + Sync`.
#[derive(Clone)]
pub struct SeparatePPONetwork<A: Module, C: Module> {
    actor_optimizer: ModuleOptimizer,
    critic_optimizer: ModuleOptimizer,
    actor_learning_rate: f64,
    critic_learning_rate: f64,
    model: (A, C),
    actor_lr_scheduler: Option<Arc<dyn ParameterSchedule + Send + Sync>>,
    critic_lr_scheduler: Option<Arc<dyn ParameterSchedule + Send + Sync>>,
    combined_loss: bool,
}

impl<A: Module, C: Module> std::fmt::Debug for SeparatePPONetwork<A, C> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SeparatePPONetwork")
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl<A: Module, C: Module> Module for SeparatePPONetwork<A, C> {
    fn collect_devices(&self, devices: Vec<Device>) -> Vec<Device> {
        self.model.collect_devices(devices)
    }

    fn fork(mut self, device: &Device) -> Self {
        self.model = self.model.fork(device);
        self
    }

    fn to_device(mut self, device: &Device) -> Self {
        self.model = self.model.to_device(device);
        self
    }

    fn visit<Visitor: ModuleVisitor>(&self, visitor: &mut Visitor) {
        self.model.visit(visitor);
    }

    fn map<Mapper: ModuleMapper>(mut self, mapper: &mut Mapper) -> Self {
        self.model = self.model.map(mapper);
        self
    }

    fn train(mut self) -> Self {
        self.model = self.model.train();
        self
    }

    fn valid(&self) -> Self {
        Self {
            model: self.model.valid(),
            ..self.clone()
        }
    }

    fn materialize(mut self) -> Self {
        self.model = self.model.materialize();
        self
    }
}

#[bon]
impl<A: Module, C: Module> SeparatePPONetwork<A, C> {
    /// Returns a snapshot of the actor and critic models, in that order.
    /// Parameter tensors retain their shape, dtype, device, and current values.
    pub fn models(&self) -> (A, C) {
        self.model.clone()
    }

    /// Owns the models and optimizers, and enables training on both models.
    /// Models must accept the same native observation kind and share a device. Critic outputs and PPO statistics must use the environment reward dtype.
    /// Rejects nonpositive or non-finite learning rates.
    #[builder]
    pub fn new(
        #[builder(into)] actor_optimizer: ModuleOptimizer,
        #[builder(into)] critic_optimizer: ModuleOptimizer,
        actor_learning_rate: f64,
        critic_learning_rate: f64,
        actor_network: A,
        critic_network: C,
        #[builder(with = |schedule: impl ParameterSchedule + Send + Sync + 'static| Arc::new(schedule))]
        actor_lr_scheduler: Option<Arc<dyn ParameterSchedule + Send + Sync>>,
        #[builder(with = |schedule: impl ParameterSchedule + Send + Sync + 'static| Arc::new(schedule))]
        critic_lr_scheduler: Option<Arc<dyn ParameterSchedule + Send + Sync>>,
        #[builder(default = false)] combined_loss: bool,
    ) -> Result<Self, PPOConfigurationError> {
        if !actor_learning_rate.is_finite()
            || actor_learning_rate <= 0.0
            || !critic_learning_rate.is_finite()
            || critic_learning_rate <= 0.0
        {
            return Err(PPOConfigurationError::InvalidLearningRate);
        }
        Ok(Self {
            actor_optimizer,
            critic_optimizer,
            actor_learning_rate,
            critic_learning_rate,
            model: (actor_network, critic_network).train(),
            actor_lr_scheduler,
            critic_lr_scheduler,
            combined_loss,
        })
    }
}

/// Executes an owned PPO network configuration with native tensor kinds and explicit batch ranks.
/// Rank O describes environment observations, H describes head inputs, and L describes policy actions.
/// Validation inputs must have autodiff removed explicitly.
pub trait PPONetwork<const O: usize, const H: usize, const L: usize>: Module {
    type InputKind: Autodiff;
    type LatentKind: Basic;
    type Error: std::fmt::Debug;

    /// Supplies the network and native tensor types used to construct an agent.
    /// Use `PPOModelGroup<Self, Tensor<O, Self::InputKind>, Tensor<H, Self::LatentKind>, Tensor<L>>`
    /// for the native observation, latent, and policy-action batches. Each tensor includes its leading batch axis.
    type ModelGroup: PPOModelTypes<Network = Self, Error = Self::Error>;

    /// Maps observations [batch_size, ...observation_shape] to head inputs [batch_size, ...model_input_shape].
    /// Preserves the batch axis. Models own kind and dtype conversions; separate configurations return the input unchanged.
    fn forward_observations(
        &self,
        observations: Tensor<O, Self::InputKind>,
    ) -> Result<Tensor<H, Self::LatentKind>, Self::Error>;

    /// Returns Float values [batch_size, 1] from head inputs [batch_size, ...model_input_shape].
    /// The critic determines dtype. Training retains model gradients; validation views disable autodiff.
    /// Inputs and model parameters must share a device.
    fn forward_values(
        &self,
        observations: Tensor<H, Self::LatentKind>,
    ) -> Result<Tensor<2>, Self::Error>;

    /// Samples Float policy actions [batch_size, ...action_shape] from head inputs [batch_size, ...model_input_shape].
    /// The policy determines action dtype; the batch axis and model gradient connections remain available.
    fn sample(&self, observations: Tensor<H, Self::LatentKind>) -> Result<Tensor<L>, Self::Error>;

    /// Returns modal Float policy actions [batch_size, ...action_shape] from head inputs [batch_size, ...model_input_shape].
    /// Preserves the batch axis. The policy defines action dtype and device requirements.
    fn mode(&self, observations: Tensor<H, Self::LatentKind>) -> Result<Tensor<L>, Self::Error>;

    /// Evaluates Float policy actions [batch_size, ...action_shape] for head inputs [batch_size, ...model_input_shape].
    /// Returns log probabilities and entropy [batch_size, 1], preserving dtype and training gradients.
    fn log_prob_and_entropy(
        &self,
        observations: Tensor<H, Self::LatentKind>,
        actions: Tensor<L>,
    ) -> Result<(Tensor<2>, Tensor<2>), Self::Error>;

    /// Updates owned parameters from actor and critic losses [1], preserving the configuration's clipping and optimizer behavior.
    /// Reading gradient norm contributions can fail. Losses must come from the owned models on their device.
    fn backpropagate_loss(
        &mut self,
        actor_loss: Tensor<1>,
        critic_loss: Tensor<1>,
        gradient_clip: f32,
    ) -> Result<(), TensorReadError>;

    /// Updates configured learning rates from the current training progress and each optional schedule.
    fn update_learning_rates(&mut self, progress: &ScheduleProgress);

    /// Sets the shared rate or both separate rates. Scheduled rates are replaced at the next schedule update.
    fn set_learning_rate(&mut self, learning_rate: f64);
}

impl<T, P, C, const O: usize, const H: usize, const L: usize> PPONetwork<O, H, L>
    for SharedPPONetwork<T, P, C>
where
    T: Module + Forward<O, H, OutputKind = Float, Error: std::fmt::Debug>,
    T::InputKind: Autodiff,
    Tensor<O, T::InputKind>: NextRank + PrevRank,
    Tensor<L>: NextRank,
    P: Module + ProbabilisticPolicy<H, L, ObservationKind = Float, Error: std::fmt::Debug>,
    C: Module + Forward<H, 2, InputKind = Float, OutputKind = Float, Error: std::fmt::Debug>,
{
    type ModelGroup = PPOModelGroup<Self, Tensor<O, Self::InputKind>, Tensor<H>, Tensor<L>>;

    type InputKind = T::InputKind;
    type LatentKind = Float;
    type Error = PPONetworkError<P::Error, C::Error, T::Error>;

    /// Forwards native rank-O observations through the shared model into Float rank-H head inputs, preserving batch size.
    /// Both tensors have shape `[batch_size, ...features]`; the shared model defines the remaining axes, dtype, and device.
    /// Retains the model gradient graph.
    fn forward_observations(
        &self,
        observations: Tensor<O, Self::InputKind>,
    ) -> Result<Tensor<H>, Self::Error> {
        let model = &self.model.0;
        model.forward(observations).map_err(PPONetworkError::Shared)
    }

    /// Returns Float critic values [batch_size, 1] from Float rank-H head inputs, preserving critic dtype and training gradients.
    fn forward_values(&self, observations: Tensor<H>) -> Result<Tensor<2>, Self::Error> {
        let model = &self.model.2;
        model.forward(observations).map_err(PPONetworkError::Critic)
    }

    /// Samples rank-L Float actions from rank-H Float head inputs, each shaped `[batch_size, ...item_shape]`. Preserves policy dtype.
    fn sample(&self, observations: Tensor<H>) -> Result<Tensor<L>, Self::Error> {
        let model = &self.model.1;
        model.sample(observations).map_err(PPONetworkError::Policy)
    }

    /// Returns rank-L modal Float actions from rank-H Float head inputs, each shaped `[batch_size, ...item_shape]`. Preserves policy dtype.
    fn mode(&self, observations: Tensor<H>) -> Result<Tensor<L>, Self::Error> {
        let model = &self.model.1;
        model.mode(observations).map_err(PPONetworkError::Policy)
    }

    /// Evaluates rank-L Float actions for rank-H Float head inputs, returning policy statistics [batch_size, 1].
    /// Retains training gradients and dtype; input and action batch sizes and devices must match.
    fn log_prob_and_entropy(
        &self,
        observations: Tensor<H>,
        actions: Tensor<L>,
    ) -> Result<(Tensor<2>, Tensor<2>), Self::Error> {
        let model = &self.model.1;
        model
            .log_prob_and_entropy(observations, actions)
            .map_err(PPONetworkError::Policy)
    }

    /// Updates the shared model and both heads from losses [1], clipping one global norm before a single optimizer step.
    fn backpropagate_loss(
        &mut self,
        actor_loss: Tensor<1>,
        critic_loss: Tensor<1>,
        gradient_clip: f32,
    ) -> Result<(), TensorReadError> {
        let total_loss = actor_loss + critic_loss;
        let mut total_grad = GradientsParams::from_grads(total_loss.backward(), &self.model);
        let _total_grad_norm =
            crate::tensor_operations::clip_gradients(&self.model, &mut total_grad, gradient_clip)?;
        self.model = self
            .optimizer
            .step(self.learning_rate, self.model.clone(), total_grad);
        Ok(())
    }

    fn update_learning_rates(&mut self, progress: &ScheduleProgress) {
        if let Some(schedule) = &self.lr_scheduler {
            self.learning_rate = progress.parameter(&**schedule);
        }
    }

    fn set_learning_rate(&mut self, learning_rate: f64) {
        self.learning_rate = learning_rate;
    }
}

impl<P, C, const O: usize, const L: usize> PPONetwork<O, O, L> for SeparatePPONetwork<P, C>
where
    P: Module + ProbabilisticPolicy<O, L, Error: std::fmt::Debug>,
    P::ObservationKind: Autodiff,
    Tensor<O, P::ObservationKind>: NextRank + PrevRank,
    Tensor<L>: NextRank,
    C: Module
        + Forward<O, 2, InputKind = P::ObservationKind, OutputKind = Float, Error: std::fmt::Debug>,
{
    type ModelGroup =
        PPOModelGroup<Self, Tensor<O, Self::InputKind>, Tensor<O, Self::LatentKind>, Tensor<L>>;

    type InputKind = P::ObservationKind;
    type LatentKind = P::ObservationKind;
    type Error = PPONetworkError<P::Error, C::Error>;

    /// Returns native rank-O observations `[batch_size, ...observation_shape]` unchanged, preserving kind, dtype, device, and gradients.
    fn forward_observations(
        &self,
        observations: Tensor<O, Self::InputKind>,
    ) -> Result<Tensor<O, Self::LatentKind>, Self::Error> {
        Ok(observations)
    }

    /// Returns Float critic values [batch_size, 1] from native rank-O observations, preserving critic dtype and training gradients.
    fn forward_values(
        &self,
        observations: Tensor<O, Self::LatentKind>,
    ) -> Result<Tensor<2>, Self::Error> {
        let model = &self.model.1;
        model.forward(observations).map_err(PPONetworkError::Critic)
    }

    /// Samples rank-L Float actions from native rank-O observations, each shaped `[batch_size, ...item_shape]`. Preserves policy dtype.
    fn sample(&self, observations: Tensor<O, Self::LatentKind>) -> Result<Tensor<L>, Self::Error> {
        let model = &self.model.0;
        model.sample(observations).map_err(PPONetworkError::Policy)
    }

    /// Returns rank-L modal Float actions from native rank-O observations, each shaped `[batch_size, ...item_shape]`. Preserves policy dtype.
    fn mode(&self, observations: Tensor<O, Self::LatentKind>) -> Result<Tensor<L>, Self::Error> {
        let model = &self.model.0;
        model.mode(observations).map_err(PPONetworkError::Policy)
    }

    /// Evaluates rank-L Float actions for native rank-O observations, returning policy statistics [batch_size, 1].
    /// Retains policy gradients and dtype; input and action batch sizes and devices must match.
    fn log_prob_and_entropy(
        &self,
        observations: Tensor<O, Self::LatentKind>,
        actions: Tensor<L>,
    ) -> Result<(Tensor<2>, Tensor<2>), Self::Error> {
        let model = &self.model.0;
        model
            .log_prob_and_entropy(observations, actions)
            .map_err(PPONetworkError::Policy)
    }

    /// Updates separate models from losses [1], using the configured combined or independent clipping norms.
    fn backpropagate_loss(
        &mut self,
        actor_loss: Tensor<1>,
        critic_loss: Tensor<1>,
        gradient_clip: f32,
    ) -> Result<(), TensorReadError> {
        match self.combined_loss {
            true => {
                let total_loss = actor_loss + critic_loss;
                let models = self.model.clone();
                let mut total_grad = GradientsParams::from_grads(total_loss.backward(), &models);
                let _total_grad_norm = crate::tensor_operations::clip_gradients(
                    &models,
                    &mut total_grad,
                    gradient_clip,
                )?;
                let mut actor_gradients = PPOGradientSelection {
                    gradients: &total_grad,
                    selected: GradientsParams::new(),
                };
                self.model.0.visit(&mut actor_gradients);
                self.model.0 = self.actor_optimizer.step(
                    self.actor_learning_rate,
                    self.model.0.clone(),
                    actor_gradients.selected,
                );
                self.model.1 = self.critic_optimizer.step(
                    self.critic_learning_rate,
                    self.model.1.clone(),
                    total_grad,
                );
            }
            false => {
                let mut actor_grad =
                    GradientsParams::from_grads(actor_loss.backward(), &self.model.0);
                let _actor_grad_norm = crate::tensor_operations::clip_gradients(
                    &self.model.0,
                    &mut actor_grad,
                    gradient_clip,
                )?;
                self.model.0 = self.actor_optimizer.step(
                    self.actor_learning_rate,
                    self.model.0.clone(),
                    actor_grad,
                );

                let mut critic_grad =
                    GradientsParams::from_grads(critic_loss.backward(), &self.model.1);
                let _critic_grad_norm = crate::tensor_operations::clip_gradients(
                    &self.model.1,
                    &mut critic_grad,
                    gradient_clip,
                )?;
                self.model.1 = self.critic_optimizer.step(
                    self.critic_learning_rate,
                    self.model.1.clone(),
                    critic_grad,
                );
            }
        }
        Ok(())
    }

    fn update_learning_rates(&mut self, progress: &ScheduleProgress) {
        if let Some(schedule) = &self.actor_lr_scheduler {
            self.actor_learning_rate = progress.parameter(&**schedule);
        }
        if let Some(schedule) = &self.critic_lr_scheduler {
            self.critic_learning_rate = progress.parameter(&**schedule);
        }
    }

    fn set_learning_rate(&mut self, learning_rate: f64) {
        self.actor_learning_rate = learning_rate;
        self.critic_learning_rate = learning_rate;
    }
}

/// Groups the owned network configuration and native execution tensor types without adding execution methods.
pub trait PPOModelTypes {
    type Network;
    type Observation: NextRank + PrevRank;
    type Action: NextRank;
    type Error: std::fmt::Debug;
}

/// A type-only group for the network configuration and native observation, latent, and policy-action tensors.
/// PPO stores the actual models and tensors directly; this group contains no runtime values.
pub struct PPOModelGroup<N, O = Tensor<2>, H = O, L = Tensor<2>> {
    types: PhantomData<(N, O, H, L)>,
}

#[derive(Debug, thiserror::Error)]
pub enum PPONetworkError<
    AE: std::fmt::Debug,
    CE: std::fmt::Debug,
    TE: std::fmt::Debug = std::convert::Infallible,
> {
    #[error("policy failed: {0}")]
    Policy(#[source] AE),
    #[error("critic failed: {0}")]
    Critic(#[source] CE),
    #[error("shared model failed: {0}")]
    Shared(#[source] TE),
}

impl<N, const O: usize, const H: usize, const L: usize> PPOModelTypes
    for PPOModelGroup<N, Tensor<O, N::InputKind>, Tensor<H, N::LatentKind>, Tensor<L>>
where
    N: PPONetwork<O, H, L>,
    Tensor<O, N::InputKind>: NextRank + PrevRank,
    Tensor<L>: NextRank,
{
    type Network = N;
    type Observation = Tensor<O, N::InputKind>;
    type Action = Tensor<L>;
    type Error = N::Error;
}

#[cfg(test)]
mod tests {
    use burn::{
        module::{Module, ModuleVisitor, Param, ParamGroup},
        nn::{Dropout, DropoutConfig, Initializer, Linear, LinearConfig, Relu},
        optim::{AdamConfig, GradientsParams, SgdConfig},
        tensor::{DType, Device, Tensor},
    };

    use super::super::{
        PPOAgent,
        tests::{NoopLogger, TwoStepTestGym, assert_close, mlp},
    };
    use super::{PPOConfigurationError, PPONetwork, SeparatePPONetwork, SharedPPONetwork};
    use crate::{
        agents::Agent,
        distributions::{CategoricalDistribution, GaussianDistribution},
        gym::{MultiGym, VectorizedGymWrapper},
        models::{Forward, probabilistic_model::ProbabilisticPolicyModel},
        parameter_schedule::LinearSchedule,
        spaces::BoxSpace,
    };
    use std::convert::Infallible;

    fn layer(device: &Device) -> Linear {
        LinearConfig::new(2, 2)
            .with_initializer(Initializer::Constant { value: 0.5 })
            .init(device)
            .valid()
    }

    #[derive(Module, Debug)]
    struct DropoutTrunk {
        dropout: Dropout,
    }

    impl Forward<2, 2> for DropoutTrunk {
        type InputKind = burn::tensor::Float;
        type OutputKind = burn::tensor::Float;
        type Error = Infallible;

        /// Applies dropout to `[batch_size, features]`, preserving shape, dtype, and device.
        fn forward(&self, observations: Tensor<2>) -> Result<Tensor<2>, Self::Error> {
            Ok(self.dropout.forward(observations))
        }
    }

    #[test]
    fn networks_traverse_and_map_parameters() {
        let device = Device::flex().autodiff();
        let shared = SharedPPONetwork::builder()
            .shared_network(layer(&device))
            .actor_head(layer(&device))
            .critic_head(layer(&device))
            .optimizer(SgdConfig::new().init())
            .learning_rate(0.01)
            .build()
            .unwrap();
        assert_eq!(shared.num_params(), 18);
        assert_eq!(shared.devices(), vec![device.clone()]);
        let frozen = shared.freeze_group(ParamGroup::from_path("1.weight"));
        assert!(!frozen.model.1.weight.is_require_grad());
        assert!(frozen.model.0.weight.is_require_grad());
        assert!(frozen.model.2.weight.is_require_grad());

        let separate = SeparatePPONetwork::builder()
            .actor_network(layer(&device))
            .critic_network(layer(&device))
            .actor_optimizer(SgdConfig::new().init())
            .critic_optimizer(SgdConfig::new().init())
            .actor_learning_rate(0.01)
            .critic_learning_rate(0.01)
            .build()
            .unwrap();
        assert_eq!(separate.num_params(), 12);
        let frozen = separate.freeze_group(ParamGroup::from_path("1.weight"));
        assert!(frozen.model.0.weight.is_require_grad());
        assert!(!frozen.model.1.weight.is_require_grad());
    }

    #[test]
    fn validation_networks_share_captured_schedules() {
        let start = 0.01;
        let shared = SharedPPONetwork::builder()
            .shared_network(Relu)
            .actor_head(Relu)
            .critic_head(Relu)
            .optimizer(SgdConfig::new().init())
            .learning_rate(start)
            .lr_scheduler(move |progress| start * (1.0 - progress))
            .build()
            .unwrap();
        let validation = shared.valid();
        assert!(std::sync::Arc::ptr_eq(
            shared.lr_scheduler.as_ref().unwrap(),
            validation.lr_scheduler.as_ref().unwrap(),
        ));
        assert_eq!(validation.lr_scheduler.as_ref().unwrap().value(0.5), 0.005);

        let separate = SeparatePPONetwork::builder()
            .actor_network(Relu)
            .critic_network(Relu)
            .actor_optimizer(SgdConfig::new().init())
            .critic_optimizer(SgdConfig::new().init())
            .actor_learning_rate(start)
            .critic_learning_rate(start)
            .actor_lr_scheduler(move |progress| start * (1.0 - progress))
            .critic_lr_scheduler(move |progress| start * (1.0 - progress))
            .build()
            .unwrap();
        let validation = separate.valid();
        assert!(std::sync::Arc::ptr_eq(
            separate.actor_lr_scheduler.as_ref().unwrap(),
            validation.actor_lr_scheduler.as_ref().unwrap(),
        ));
        assert!(std::sync::Arc::ptr_eq(
            separate.critic_lr_scheduler.as_ref().unwrap(),
            validation.critic_lr_scheduler.as_ref().unwrap(),
        ));
    }

    #[test]
    fn validation_views_preserve_training_flags_and_frozen_parameters() {
        let device = Device::flex().autodiff().gradient_checkpointing();
        let network = SharedPPONetwork::builder()
            .shared_network(DropoutTrunk {
                dropout: DropoutConfig::new(0.5).init(),
            })
            .actor_head(ProbabilisticPolicyModel::new(
                layer(&device),
                CategoricalDistribution,
            ))
            .critic_head(
                LinearConfig::new(2, 1)
                    .with_initializer(Initializer::Constant { value: 0.5 })
                    .init(&device)
                    .no_grad(),
            )
            .optimizer(SgdConfig::new().init())
            .learning_rate(0.01)
            .build()
            .unwrap();
        let critic_id = network.model.2.weight.id;
        let observations = Tensor::<2>::ones([2, 2], &device);

        {
            let validation = Module::valid(&network);
            assert!(!validation.model.0.dropout.training.is_enabled());
            assert!(network.model.0.dropout.training.is_enabled());
            assert_eq!(validation.model.2.weight.id, critic_id);
            assert_eq!(
                validation.model.2.weight.val().to_data(),
                network.model.2.weight.val().to_data(),
            );
            let latent = validation
                .forward_observations(observations.without_autodiff())
                .unwrap();
            assert!(!latent.device().is_autodiff());
            assert_eq!(latent.to_data().try_to_vec::<f32>().unwrap(), [1.0; 4]);
        }

        assert!(network.model.0.dropout.training.is_enabled());
        assert!(!network.model.2.weight.is_require_grad());
        assert_eq!(network.model.2.weight.id, critic_id);
        assert_eq!(
            network
                .model
                .2
                .weight
                .val()
                .device()
                .gradient_checkpointing_strategy(),
            device.gradient_checkpointing_strategy(),
        );
    }

    #[test]
    fn shared_validation_views_disable_gradients_and_refresh_after_updates() {
        let device = Device::flex().autodiff();
        let mut network = SharedPPONetwork::builder()
            .shared_network(layer(&device))
            .actor_head(ProbabilisticPolicyModel::new(
                layer(&device),
                CategoricalDistribution,
            ))
            .critic_head(
                LinearConfig::new(2, 1)
                    .with_initializer(Initializer::Constant { value: 0.5 })
                    .init(&device),
            )
            .optimizer(SgdConfig::new().init())
            .learning_rate(0.01)
            .build()
            .unwrap();
        let observations = Tensor::from_data([[1.0f32, 2.0], [3.0, 4.0]], &device);

        let previous = {
            let validation = Module::valid(&network);
            let latent = PPONetwork::<2, 2, 2>::forward_observations(
                &validation,
                observations.clone().without_autodiff(),
            )
            .unwrap();
            let values =
                PPONetwork::<2, 2, 2>::forward_values(&validation, latent.clone()).unwrap();
            let actions = PPONetwork::<2, 2, 2>::mode(&validation, latent.clone()).unwrap();
            let (log_probs, entropy) =
                PPONetwork::<2, 2, 2>::log_prob_and_entropy(&validation, latent.clone(), actions)
                    .unwrap();
            assert!(!latent.device().is_autodiff());
            assert!(!values.is_require_grad());
            assert!(!log_probs.is_require_grad());
            assert!(!entropy.is_require_grad());
            let previous = values.to_data().try_to_vec::<f32>().unwrap();
            assert!(network.models().0.weight.is_require_grad());

            previous
        };

        let latent =
            PPONetwork::<2, 2, 2>::forward_observations(&network, observations.clone()).unwrap();
        let values = PPONetwork::<2, 2, 2>::forward_values(&network, latent.clone()).unwrap();
        let actions = PPONetwork::<2, 2, 2>::mode(&network, latent.clone()).unwrap();
        let (log_probs, _) =
            PPONetwork::<2, 2, 2>::log_prob_and_entropy(&network, latent, actions).unwrap();
        assert!(values.device().is_autodiff());
        assert!(log_probs.device().is_autodiff());
        PPONetwork::<2, 2, 2>::backpropagate_loss(
            &mut network,
            -log_probs.mean(),
            values.square().mean(),
            1.0,
        )
        .unwrap();

        let validation = Module::valid(&network);
        let latent = PPONetwork::<2, 2, 2>::forward_observations(
            &validation,
            observations.without_autodiff(),
        )
        .unwrap();
        let current = PPONetwork::<2, 2, 2>::forward_values(&validation, latent)
            .unwrap()
            .to_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert!(
            previous
                .iter()
                .zip(current)
                .any(|(before, after)| (before - after).abs() > 1e-6)
        );
    }

    #[test]
    fn separate_validation_views_preserve_training_models_and_gradient_flow() {
        let _guard = crate::sampling::tests::RNG_LOCK.lock().unwrap();
        let device = Device::flex().autodiff();
        let mut network = SeparatePPONetwork::builder()
            .actor_network(ProbabilisticPolicyModel::new(
                layer(&device),
                CategoricalDistribution,
            ))
            .critic_network(LinearConfig::new(2, 1).init(&device))
            .actor_optimizer(SgdConfig::new().init())
            .critic_optimizer(SgdConfig::new().init())
            .actor_learning_rate(0.01)
            .critic_learning_rate(0.01)
            .build()
            .unwrap();
        let observations = Tensor::from_data([[1.0f32, 2.0], [3.0, 4.0]], &device);

        {
            let validation = Module::valid(&network);
            let values = PPONetwork::<2, 2, 2>::forward_values(
                &validation,
                observations.clone().without_autodiff(),
            )
            .unwrap();
            let actions =
                PPONetwork::<2, 2, 2>::mode(&validation, observations.clone().without_autodiff())
                    .unwrap();
            let (log_probs, entropy) = PPONetwork::<2, 2, 2>::log_prob_and_entropy(
                &validation,
                observations.clone().without_autodiff(),
                actions,
            )
            .unwrap();
            assert!(!values.device().is_autodiff());
            assert!(!log_probs.is_require_grad());
            assert!(!entropy.is_require_grad());
            assert!(network.models().1.weight.is_require_grad());
        }

        let values = PPONetwork::<2, 2, 2>::forward_values(&network, observations.clone()).unwrap();
        let actions = PPONetwork::<2, 2, 2>::mode(&network, observations.clone()).unwrap();
        let (log_probs, _) =
            PPONetwork::<2, 2, 2>::log_prob_and_entropy(&network, observations, actions).unwrap();
        assert!(values.device().is_autodiff());
        assert!(log_probs.device().is_autodiff());
        let before = network
            .models()
            .1
            .weight
            .val()
            .to_data()
            .try_to_vec::<f32>()
            .unwrap();
        PPONetwork::<2, 2, 2>::backpropagate_loss(
            &mut network,
            -log_probs.mean(),
            values.square().mean(),
            1.0,
        )
        .unwrap();
        let after = network
            .models()
            .1
            .weight
            .val()
            .to_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert!(
            before
                .iter()
                .zip(after)
                .any(|(before, after)| (before - after).abs() > 1e-6)
        );
    }

    #[test]
    fn shared_builder_owns_models_and_enables_parameter_gradients() {
        let device = Device::flex().autodiff();
        let network = SharedPPONetwork::builder()
            .shared_network(layer(&device))
            .actor_head(layer(&device))
            .critic_head(layer(&device))
            .optimizer(SgdConfig::new().init())
            .learning_rate(0.001)
            .lr_scheduler(|_| 0.0005)
            .build()
            .unwrap();
        let mut network = network;

        assert!(network.model.0.weight.val().is_require_grad());
        assert!(network.model.1.weight.val().is_require_grad());
        assert!(network.model.2.weight.val().is_require_grad());
        assert_eq!(network.learning_rate, 0.001);
        assert_eq!(network.lr_scheduler.as_ref().unwrap().value(0.5), 0.0005);

        let models = (network.model.0, network.model.1, network.model.2);
        let loss =
            models.0.weight.val().sum() + models.1.weight.val().sum() + models.2.weight.val().sum();
        let gradients = GradientsParams::from_grads(loss.backward(), &models);
        let models = network
            .optimizer
            .step(network.learning_rate, models, gradients);
        for model in [models.0, models.1, models.2] {
            assert!(
                (model.weight.val().into_data().try_to_vec::<f32>().unwrap()[0] - 0.499).abs()
                    < 1e-6
            );
        }
    }

    #[test]
    fn separate_builder_preserves_independent_rates_and_combined_loss() {
        let device = Device::flex().autodiff();
        let network = SeparatePPONetwork::builder()
            .actor_network(layer(&device))
            .critic_network(layer(&device))
            .actor_optimizer(AdamConfig::new().init())
            .critic_optimizer(SgdConfig::new().init())
            .actor_learning_rate(0.001)
            .critic_learning_rate(0.002)
            .actor_lr_scheduler(|_| 0.0005)
            .critic_lr_scheduler(|_| 0.0007)
            .combined_loss(true)
            .build()
            .unwrap();
        let mut network = network;

        assert!(network.model.0.weight.val().is_require_grad());
        assert!(network.model.1.weight.val().is_require_grad());
        assert_eq!(network.actor_learning_rate, 0.001);
        assert_eq!(network.critic_learning_rate, 0.002);
        assert_eq!(
            network.actor_lr_scheduler.as_ref().unwrap().value(0.5),
            0.0005
        );
        assert_eq!(
            network.critic_lr_scheduler.as_ref().unwrap().value(0.5),
            0.0007
        );
        assert!(network.combined_loss);

        let actor_gradients = GradientsParams::from_grads(
            network.model.0.weight.val().sum().backward(),
            &network.model.0,
        );
        let critic_gradients = GradientsParams::from_grads(
            network.model.1.weight.val().sum().backward(),
            &network.model.1,
        );
        let actor = network.actor_optimizer.step(
            network.actor_learning_rate,
            network.model.0,
            actor_gradients,
        );
        let critic = network.critic_optimizer.step(
            network.critic_learning_rate,
            network.model.1,
            critic_gradients,
        );
        assert!(
            (actor.weight.val().into_data().try_to_vec::<f32>().unwrap()[0] - 0.499).abs() < 1e-6
        );
        assert!(
            (critic.weight.val().into_data().try_to_vec::<f32>().unwrap()[0] - 0.498).abs() < 1e-6
        );
    }

    #[test]
    fn network_builders_reject_invalid_learning_rates() {
        for rate in [0.0, -0.001, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let shared = SharedPPONetwork::builder()
                .shared_network(Relu)
                .actor_head(Relu)
                .critic_head(Relu)
                .optimizer(SgdConfig::new().init())
                .learning_rate(rate)
                .build();
            assert!(matches!(
                shared,
                Err(PPOConfigurationError::InvalidLearningRate)
            ));

            for (actor_rate, critic_rate) in [(rate, 0.001), (0.001, rate)] {
                let separate = SeparatePPONetwork::builder()
                    .actor_network(Relu)
                    .critic_network(Relu)
                    .actor_optimizer(SgdConfig::new().init())
                    .critic_optimizer(SgdConfig::new().init())
                    .actor_learning_rate(actor_rate)
                    .critic_learning_rate(critic_rate)
                    .build();
                assert!(matches!(
                    separate,
                    Err(PPOConfigurationError::InvalidLearningRate)
                ));
            }
        }
    }

    #[test]
    fn separate_updates_both_models_with_combined_or_independent_losses() {
        struct Parameters(Vec<f32>);

        impl ModuleVisitor for Parameters {
            /// Reads each rank-D Float parameter to the host in visit order, preserving its shape and the model.
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
                self.0
                    .extend(param.val().into_data().try_to_vec::<f32>().unwrap());
            }
        }
        let device = Device::flex().autodiff();
        let mut updates = Vec::new();
        for combined in [false, true] {
            let actor =
                ProbabilisticPolicyModel::new(mlp(4, 6, &device), GaussianDistribution::default());
            let critic = mlp(4, 1, &device);
            let mut before_actor = Parameters(Vec::new());
            let mut before_critic = Parameters(Vec::new());
            actor.visit(&mut before_actor);
            critic.visit(&mut before_critic);
            let networks = SeparatePPONetwork::builder()
                .actor_network(actor)
                .critic_network(critic)
                .actor_optimizer(SgdConfig::new().init())
                .critic_optimizer(SgdConfig::new().init())
                .actor_learning_rate(0.01)
                .critic_learning_rate(0.02)
                .combined_loss(combined)
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
                .normalize_advantage(false)
                .gradient_clip(100.0)
                .training_horizon(1)
                .device(device.clone())
                .build()
                .unwrap();
            let losses = agent
                .compute_loss()
                .observations(Tensor::ones([2, 4], (&device, DType::F32)))
                .actions(Tensor::full([2, 3], 0.1, (&device, DType::F32)))
                .old_log_probs(Tensor::zeros([2, 1], (&device, DType::F32)))
                .advantages(Tensor::ones([2, 1], (&device, DType::F32)))
                .returns(Tensor::ones([2, 1], (&device, DType::F32)))
                .rewards(Tensor::ones([2, 1], (&device, DType::F32)))
                .explained_variance(Tensor::zeros([1], (&device, DType::F32)))
                .clip_range(0.2)
                .call()
                .unwrap();
            agent.backpropagate_loss(losses).unwrap();
            let network = &agent.network_info;
            let mut after_actor = Parameters(Vec::new());
            let mut after_critic = Parameters(Vec::new());
            network.model.0.visit(&mut after_actor);
            network.model.1.visit(&mut after_critic);
            assert_ne!(before_actor.0, after_actor.0);
            assert_ne!(before_critic.0, after_critic.0);
            updates.push((after_actor.0, after_critic.0));
        }
        assert_eq!(updates[0], updates[1]);
    }

    #[test]
    fn shared_agent_runs_collection_training_actions_and_observation_reset() {
        let device = Device::flex().autodiff();
        let mut env: VectorizedGymWrapper<TwoStepTestGym, usize> = vec![
            TwoStepTestGym::new(device.clone(), false),
            TwoStepTestGym::new(device.clone(), false),
        ]
        .try_into()
        .unwrap();
        let actor = ProbabilisticPolicyModel::new(mlp(4, 2, &device), CategoricalDistribution);
        let networks = SharedPPONetwork::builder()
            .shared_network(mlp(4, 4, &device))
            .actor_head(actor)
            .critic_head(mlp(4, 1, &device))
            .optimizer(SgdConfig::new().init())
            .learning_rate(1e-3)
            .lr_scheduler(LinearSchedule::new(1e-3, 1e-4))
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
            .training_horizon(10)
            .logging_info(&mut logger)
            .device(device.clone())
            .build()
            .unwrap();
        agent.learn(&mut env, 2).unwrap();
        let observations = agent.current_observations.as_ref().unwrap().clone();
        assert_eq!(agent.act(&observations).unwrap().dims(), [2, 1]);
        assert_eq!(
            agent.act_deterministic(&observations).unwrap().dims(),
            [2, 1]
        );
        assert_eq!(agent.episode_tracker.environment_count(), 2);
        let network = &agent.network_info;
        assert_close(network.learning_rate, 0.00082);
        agent.set_learning_rate(0.25);
        let network = &agent.network_info;
        assert_close(network.learning_rate, 0.25);
        agent.reset_current_observations();
        assert!(agent.current_observations.is_none());
        assert_eq!(agent.episode_tracker.environment_count(), 0);
        assert_eq!(logger.updates, 1);
    }

    #[test]
    fn schedules_continue_across_learn_calls() {
        let device = Device::flex().autodiff();
        let mut env: VectorizedGymWrapper<TwoStepTestGym, usize> = vec![
            TwoStepTestGym::new(device.clone(), false),
            TwoStepTestGym::new(device.clone(), false),
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
            .actor_lr_scheduler(LinearSchedule::new(1e-3, 1e-4))
            .critic_lr_scheduler(LinearSchedule::new(1e-3, 1e-4))
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
            .clip_range(LinearSchedule::new(0.2, 0.1))
            .training_horizon(10)
            .logging_info(&mut logger)
            .device(device.clone())
            .build()
            .unwrap();
        for (steps, rate, clip) in [(2, 0.00082, 0.18), (4, 0.00064, 0.16)] {
            agent.learn(&mut env, 2).unwrap();
            assert_eq!(agent.schedule_progress.elapsed_steps(), steps);
            assert_close(agent.schedule_progress.parameter(&*agent.clip_range), clip);
            let network = &agent.network_info;
            assert_close(network.actor_learning_rate, rate);
            assert_close(network.critic_learning_rate, rate);
        }
        let observations = Tensor::zeros([2, 4], (&device, DType::F32));
        assert_eq!(agent.act(&observations).unwrap().dims(), [2, 1]);
        assert_eq!(
            agent.act_deterministic(&observations).unwrap().dims(),
            [2, 1]
        );
        agent.set_learning_rate(0.25);
        let network = &agent.network_info;
        assert_close(network.actor_learning_rate, 0.25);
        assert_close(network.critic_learning_rate, 0.25);
        assert_eq!(logger.updates, 2);
    }
}

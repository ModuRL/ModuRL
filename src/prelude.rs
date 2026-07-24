//! Common imports for building ModuRL training programs.
//!
//! This prelude re-exports the traits users need for method resolution and the
//! high-use ModuRL types that appear in most examples. External crates such as
//! Candle and ModuRL Gym are intentionally left explicit.

pub use crate::agents::{
    Agent,
    ppo::{
        PPOAgent, PPOCollectionLogEntry, PPOEpisodeLogEntry, PPOError, PPOLogEntry, PPOLogger,
        PPONetworkInfo, SeparatePPONetwork, SharedPPONetwork,
    },
    q_learning::{
        QAgentError, QCollectionLogEntry, QEpisodeLogEntry, QLearningConfigurationError,
        QLearningDeviceStrategy, QLogEntry,
        ddqn::{DDQNAgent, DDQNLogger},
        dqn::{DQNAgent, DQNLogger},
    },
    sac::{
        DiscreteVectorHeadCritic, SACAgent, SACCollectionLogEntry, SACConfigurationError,
        SACCritic, SACCriticAggregationMode, SACCriticError, SACCriticNetwork, SACDeviceStrategy,
        SACEntropyConfiguration, SACEpisodeLogEntry, SACError, SACLogEntry, SACLogger,
        SACStabilizationConfiguration, ScalarStateActionCritic, aggregate_critic_values,
        sac_alpha_loss, sac_bellman_targets, sac_clipped_critic_loss, sac_entropy_change_loss,
    },
};
pub use crate::distributions::{
    AffineTransform, AffineTransformError, CategoricalDistribution, CategoricalDistributionError,
    DifferentiableExpectation, DistEval, Distribution, DistributionTransform, ExpectationTerms,
    GaussianDistribution, GaussianDistributionError, TanhTransform, TransformedDistribution,
    TransformedDistributionError,
};
#[cfg(feature = "multithreading")]
pub use crate::gym::MultithreadedVectorizedGymWrapper;
pub use crate::gym::{
    Gym, ResetInfo, StepInfo, VectorizedGym, VectorizedGymError, VectorizedGymWrapper,
};
pub use crate::models::{
    DefaultMLPInitializer, FrozenParametersModule, MLP, MLPArchitecture, MLPInitializedLayers,
    MLPInitializer, OrthogonalMLPInitializer, probabilistic_model::DifferentiableExpectationPolicy,
    probabilistic_model::ExpectationPolicy, probabilistic_model::ProbabilisticPolicy,
    probabilistic_model::ProbabilisticPolicyModel,
    probabilistic_model::ProbabilisticPolicyModelError,
};
pub use crate::parameter_schedule::{
    ConstantSchedule, ExponentialSchedule, LinearSchedule, ParameterSchedule, ScheduleProgress,
};
pub use crate::spaces::{BoxSpace, Discrete, Space};
pub use crate::wrappers::{
    ClipRewardGym, ClipRewardGymError, FrameStackGym, FrameStackGymError, MaxAndSkipGym,
    MaxAndSkipGymError, NormalizeObservationGym, NormalizeObservationGymError, NormalizeRewardGym,
    RawRewardInfo, RecordRawRewardGym, TimeLimitGym,
};

//! Common imports for building ModuRL training programs.
//!
//! This prelude re-exports the traits users need for method resolution and the
//! high-use ModuRL types that appear in most examples. External crates such as
//! Candle and ModuRL Gym are intentionally left explicit.

pub use crate::agents::{
    Agent, ReplayDeviceStrategy, ReplayStorageConfig,
    a2c::{
        A2CAgent, A2CCollectionLogEntry, A2CEpisodeLogEntry, A2CError, A2CLogEntry, A2CLogger,
        A2CNetworkInfo, SeparateA2CNetwork, SharedA2CNetwork,
    },
    deterministic_actor_critic::{
        DeterministicActorCriticCollectionLogEntry, DeterministicActorCriticConfigurationError,
        DeterministicActorCriticEpisodeLogEntry, DeterministicActorCriticError,
        DeterministicActorCriticLogEntry, DeterministicActorCriticResult, DeterministicCritic,
        ddpg::{DDPGAgent, DDPGLogger},
        td3::{TD3Agent, TD3Logger},
    },
    ppo::{
        PPOAgent, PPOCollectionLogEntry, PPOConfigurationError, PPOEpisodeLogEntry, PPOError,
        PPOLogEntry, PPOLogger, PPONetworkInfo, SeparatePPONetwork, SharedPPONetwork,
    },
    q_learning::{
        QAgentError, QCollectionLogEntry, QEpisodeLogEntry, QLearningConfigurationError, QLogEntry,
        ddqn::{DDQNAgent, DDQNLogger},
        dqn::{DQNAgent, DQNLogger},
    },
    sac::{
        DiscreteVectorHeadCritic, SACAgent, SACCollectionLogEntry, SACConfigurationError,
        SACCritic, SACCriticAggregationMode, SACCriticError, SACCriticNetwork,
        SACEntropyConfiguration, SACEpisodeLogEntry, SACError, SACLogEntry, SACLogger,
        ScalarStateActionCritic,
    },
};
pub use crate::distributions::{
    AffineTransform, AffineTransformError, CategoricalDistribution, CategoricalDistributionError,
    DifferentiableExpectation, DistEval, Distribution, DistributionTransform, ExpectationTerms,
    GaussianDistribution, GaussianDistributionError, TanhTransform, TransformedDistribution,
    TransformedDistributionError,
};
pub use crate::gym::{
    Gym, MultiGym, MultiGymStepInfo, ResetInfo, StackedMultiGym, StackedMultiGymError, StepInfo,
    VectorizedGymError, VectorizedGymWrapper,
};
#[cfg(feature = "multithreading")]
pub use crate::gym::{MultithreadedStackedMultiGym, MultithreadedVectorizedGymWrapper};
pub use crate::models::{
    DefaultMLPInitializer, DuelingMLP, MLP, MLPInitializer, OrthogonalMLPInitializer,
    probabilistic_model::ExpectationPolicy, probabilistic_model::ProbabilisticPolicy,
    probabilistic_model::ProbabilisticPolicyModel,
    probabilistic_model::ProbabilisticPolicyModelError,
};
pub use crate::objectives::{bellman_targets, clipped_value_loss};
pub use crate::parameter_schedule::{
    ConstantSchedule, ExponentialSchedule, LinearSchedule, ParameterSchedule, ScheduleProgress,
};
pub use crate::spaces::{BoxSpace, Discrete, Space};
pub use crate::wrappers::{
    ClipRewardGym, ClipRewardGymError, EpisodeStatistics, EpisodeStatisticsInfo, FrameStackGym,
    FrameStackGymError, MaxAndSkipGym, MaxAndSkipGymError, NormalizeObservationGym,
    NormalizeObservationGymError, NormalizeRewardGym, RawRewardInfo, RecordEpisodeStatisticsGym,
    RecordRawRewardGym, TensorMapMultiGymError, TensorMapMultiGymWrapper, TimeLimitGym,
};

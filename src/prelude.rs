//! Common imports for building ModuRL training programs.
//!
//! This prelude re-exports the traits users need for method resolution and the
//! high-use ModuRL types that appear in most examples. External crates such as
//! Candle and ModuRL Gym are intentionally left explicit.

pub use crate::agents::{
    ppo::{
        PPOAgent, PPOError, PPOLogEntry, PPOLogger, PPONetworkInfo, SeparatePPONetwork,
        SharedPPONetwork,
    },
    q_learning::{
        ddqn::{DDQNAgent, DDQNLogger},
        dqn::{DQNAgent, DQNLogger},
        QAgentError, QCollectionLogEntry, QEpisodeLogEntry, QLearningConfigurationError,
        QLearningDeviceStrategy, QLogEntry,
    },
    Agent,
};
pub use crate::distributions::{
    CategoricalDistribution, DistEval, Distribution, GuassianDistribution,
};
#[cfg(feature = "multithreading")]
pub use crate::gym::MultithreadedVectorizedGymWrapper;
pub use crate::gym::{Gym, StepInfo, VectorizedGym, VectorizedGymError, VectorizedGymWrapper};
pub use crate::models::{
    probabilistic_model::ProbabilisticPolicy, probabilistic_model::ProbabilisticPolicyModel,
    probabilistic_model::ProbabilisticPolicyModelError, DefaultMLPInitializer, MLPArchitecture,
    MLPInitializedLayers, MLPInitializer, OrthogonalMLPInitializer, MLP,
};
pub use crate::parameter_schedule::{
    ConstantSchedule, ExponentialSchedule, LinearSchedule, ParameterSchedule, ScheduleProgress,
};
pub use crate::spaces::{BoxSpace, Discrete, Space};

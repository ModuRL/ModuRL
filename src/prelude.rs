//! Common imports for building ModuRL training programs.
//!
//! This prelude re-exports the traits users need for method resolution and the
//! high-use ModuRL types that appear in most examples. External crates such as
//! Candle and ModuRL Gym are intentionally left explicit.

pub use crate::agents::{
    Agent,
    ppo::{
        PPOAgent, PPOError, PPOLogEntry, PPOLogger, PPONetworkInfo, SeparatePPONetwork,
        SharedPPONetwork,
    },
    q_learning::{
        QLearningDeviceStrategy,
        ddqn::{DDQNAgent, DDQNAgentError, DDQNLogEntry, DDQNLogger},
        dqn::{DQNAgent, DQNAgentError, DQNLogEntry, DQNLogger},
    },
};
pub use crate::distributions::{
    CategoricalDistribution, DistEval, Distribution, GuassianDistribution,
};
#[cfg(feature = "multithreading")]
pub use crate::gym::MultithreadedVectorizedGymWrapper;
pub use crate::gym::{Gym, StepInfo, VectorizedGym, VectorizedGymError, VectorizedGymWrapper};
pub use crate::models::{
    DefaultMLPInitializer, MLP, MLPArchitecture, MLPInitializedLayers, MLPInitializer,
    OrthogonalMLPInitializer, probabilistic_model::ProbabilisticPolicy,
    probabilistic_model::ProbabilisticPolicyModel,
    probabilistic_model::ProbabilisticPolicyModelError,
};
pub use crate::parameter_schedule::{
    ConstantSchedule, ExponentialSchedule, LinearSchedule, ParameterSchedule,
};
pub use crate::spaces::{BoxSpace, Discrete, Space};
pub use crate::tensor_operations::tanh;

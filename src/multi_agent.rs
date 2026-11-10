//! Multi-agent environments with vectorized slots and compact per-agent rows.

use crate::gym::MultiGymStepInfo;
use crate::spaces::{ActionSpace, ObservationSpace};
use crate::tensor_rank::{NextRank, TensorBatchTypes};
use burn::tensor::Tensor;

/// Compact observations and the slot of each row.
#[derive(Debug, Clone)]
pub struct MultiAgentObservations<T = Tensor<2>> {
    pub observations: T,
    pub slots: Vec<usize>,
}

/// Reset observations and slots for each agent in agent order.
pub type MultiAgentResetResult<T, E> = Result<Vec<MultiAgentObservations<T>>, E>;

/// One agent's transitions in action order, plus the rows that act next.
#[derive(Debug, Clone)]
pub struct MultiAgentStepInfo<I = (), T: NextRank = Tensor<1>> {
    /// One row per agent row that acted, in the order of actions passed to `step`.
    pub step: MultiGymStepInfo<I, T>,
    /// One row per next acting slot, including slots that just reset.
    pub next_observations: T::Next,
    /// Slot of each next observation row.
    pub next_slots: Vec<usize>,
}

/// An ordered group of agents sharing a fixed batch of environment slots.
/// `O` and `A` are batch ranks; kinds and devices follow each agent's spaces.
/// An agent's death step is its last transition, with done=true; dead agents produce no rows until reset.
/// A slot resets once every agent in it is done or truncated, retaining terminal observations.
/// After reset all agents in the slot act again. The environment tracks each agent's acting slots.
pub trait MultiAgentGym<I = (), const O: usize = 2, const A: usize = 2> {
    type Error;
    type ObservationSpace: ObservationSpace<O>;
    type ActionSpace: ActionSpace<A>;

    fn num_agents(&self) -> usize;
    fn num_envs(&self) -> usize;
    fn observation_space(&self, agent: usize) -> Self::ObservationSpace;
    fn action_space(&self, agent: usize) -> Self::ActionSpace;

    /// Resets all slots, returning observations and their slots per agent in agent order.
    /// Reset before the first step. Observation dtype and device follow the agent's space.
    fn reset(
        &mut self,
    ) -> MultiAgentResetResult<<Self::ObservationSpace as TensorBatchTypes>::Batch, Self::Error>;

    /// Steps with one compact action batch per agent, ordered by its current acting slots.
    /// Transition rows correspond to the actions passed; next rows may differ after deaths and resets.
    /// Incorrect agent counts or action-row counts must return recoverable errors, never panic.
    #[allow(clippy::type_complexity)]
    fn step(
        &mut self,
        actions: Vec<<Self::ActionSpace as TensorBatchTypes>::Batch>,
    ) -> Result<
        Vec<MultiAgentStepInfo<I, <Self::ObservationSpace as TensorBatchTypes>::Item>>,
        Self::Error,
    >;
}

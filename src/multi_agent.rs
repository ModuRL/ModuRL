//! Multi-agent environments with vectorized slots and per-agent active rows.

use crate::gym::MultiGymStepInfo;
use crate::spaces::{ActionSpace, ObservationSpace};
use crate::tensor_rank::{NextRank, TensorBatchTypes};
use burn::tensor::Tensor;

/// One agent's transition batch plus which rows it acted on.
#[derive(Debug, Clone)]
pub struct MultiAgentStepInfo<I = (), T: NextRank = Tensor<1>> {
    pub step: MultiGymStepInfo<I, T>,
    /// One flag per slot: this agent acted on the observations passed to `step`.
    pub active: Vec<bool>,
}

/// An ordered group of agents sharing a fixed batch of environment slots.
/// `O` and `A` are batch ranks; kinds and devices follow each agent's spaces.
/// An agent's death step is active with done=true; later rows are inactive until its slot resets.
/// A slot resets once every agent in it is done or truncated, retaining terminal observations.
/// After a slot resets, all its agents are active again. Actions for inactive rows are ignored.
pub trait MultiAgentGym<I = (), const O: usize = 2, const A: usize = 2> {
    type Error;
    type ObservationSpace: ObservationSpace<O>;
    type ActionSpace: ActionSpace<A>;

    fn num_agents(&self) -> usize;
    fn num_envs(&self) -> usize;
    fn observation_space(&self, agent: usize) -> Self::ObservationSpace;
    fn action_space(&self, agent: usize) -> Self::ActionSpace;

    /// Resets all slots, returning one `[num_envs, ...observation_shape]` batch per agent in agent order.
    /// Reset before the first step. Observation dtype and device follow the agent's space.
    fn reset(
        &mut self,
    ) -> Result<Vec<<Self::ObservationSpace as TensorBatchTypes>::Batch>, Self::Error>;

    /// Steps with one `[num_envs, ...action_shape]` batch per agent in agent order.
    /// Returns one transition batch and active-row flags per agent, with auto-reset observations.
    /// An action count different from `num_agents()` must return an error, never panic.
    #[allow(clippy::type_complexity)]
    fn step(
        &mut self,
        actions: Vec<<Self::ActionSpace as TensorBatchTypes>::Batch>,
    ) -> Result<
        Vec<MultiAgentStepInfo<I, <Self::ObservationSpace as TensorBatchTypes>::Item>>,
        Self::Error,
    >;
}

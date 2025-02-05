pub mod experience_replay;
pub mod rollout_buffer;
use candle_core::Tensor;

// Not sure if this needs to be more flexible or not.
pub(crate) struct Experience {
    state: Tensor,
    next_state: Tensor,
    action: Tensor,
    reward: f32,
    done: bool,
}

impl Experience {
    pub fn new(state: Tensor, next_state: Tensor, action: Tensor, reward: f32, done: bool) -> Self {
        Self {
            state,
            next_state,
            action,
            reward,
            done,
        }
    }
}

pub(crate) struct ExperienceSample {
    states: Tensor,
    actions: Tensor,
    rewards: Tensor,
    next_states: Tensor,
    dones: Tensor,
}

impl ExperienceSample {
    fn new(
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
    ) -> Self {
        Self {
            states,
            actions,
            rewards,
            next_states,
            dones,
        }
    }

    pub fn states(&self) -> &Tensor {
        &self.states
    }

    pub fn actions(&self) -> &Tensor {
        &self.actions
    }

    pub fn rewards(&self) -> &Tensor {
        &self.rewards
    }

    pub fn next_states(&self) -> &Tensor {
        &self.next_states
    }

    pub fn dones(&self) -> &Tensor {
        &self.dones
    }
}

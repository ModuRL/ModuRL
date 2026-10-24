# Build a Custom Gym Environment

Implement `Gym` for one environment. Then place one or more instances in a
`VectorizedGymWrapper` when an agent needs batched interaction.

This small environment has one floating-point observation. Action `0` moves its
state left and action `1` moves it right. An episode ends when the state reaches
either bound.

## Define the Environment

The following code belongs in `src/counter_env.rs`:

```rust,ignore
use burn::tensor::{Device, Int, Tensor, TensorReadError};
use modurl::prelude::*;

#[derive(Debug, thiserror::Error)]
pub enum CounterEnvError {
    #[error("action {0} is outside the action space")]
    InvalidAction(i32),
    #[error("reading the action failed: {0}")]
    Read(#[from] TensorReadError),
}

pub struct CounterEnv {
    state: i32,
    device: Device,
}

impl CounterEnv {
    pub fn new(device: Device) -> Self {
        Self { state: 0, device }
    }

    /// Returns the position as an unbatched observation `[1]` with one feature.
    fn observation(&self) -> Tensor<1> {
        Tensor::from_data([self.state as f32], &self.device)
    }
}

impl Gym for CounterEnv {
    type Error = CounterEnvError;
    type ObservationSpace = BoxSpace<2>;
    type ActionSpace = Discrete;

    /// Returns the initial position as unbatched `[1]` on the environment device.
    fn reset(&mut self) -> Result<ResetInfo, Self::Error> {
        self.state = 0;
        Ok(ResetInfo {
            observation: self.observation(),
            info: (),
        })
    }

    /// Consumes one scalar integer action `[1]` and returns an unbatched position `[1]`.
    /// Actions must be readable on the host; observations use the environment device.
    fn step(&mut self, action: Tensor<1, Int>) -> Result<StepInfo, Self::Error> {
        match action.try_into_scalar::<i32>()? {
            0 => self.state -= 1,
            1 => self.state += 1,
            action => return Err(CounterEnvError::InvalidAction(action)),
        }

        let done = self.state.abs() >= 4;
        Ok(StepInfo {
            observation: self.observation(),
            reward: 1.0,
            done,
            truncated: false,
            info: (),
        })
    }

    fn observation_space(&self) -> Self::ObservationSpace {
        BoxSpace::new_with_universal_bounds(
            [1, 1],
            -4.0,
            4.0,
            &self.device,
        )
    }

    fn action_space(&self) -> Self::ActionSpace {
        Discrete::new(2)
    }
}
```

`reset` returns the initial observation and `step` consumes one action and
returns the observation that follows it, its reward, and the episode flags.
The default `Gym` information type is `()`, so ordinary environments use
`ResetInfo` and `StepInfo` without an explicit type parameter. Environments
with additional typed metadata can instead implement `Gym<MyInfo>`.
The full signature is `Gym<I, O, A>`. Observation and action batch ranks
`O` and `A` default to 2. `PrevRank` determines each single tensor type.
Single-environment tensors omit the batch axis. Scalar values use
`[1]` individually and `[batch_size, 1]` in batches. Space operations use
batched tensors. The associated spaces determine the native tensor kinds:
`BoxSpace<2>` uses `Float`, and `Discrete` uses `Int`.
For explicit observation types, use `StepInfo<MyInfo, Tensor<1, Int>>` and
`ResetInfo<MyInfo, Tensor<1, Int>>`. Both store native unbatched tensors.

In `src/main.rs`, declare the module and bring the environment into scope:

```rust,ignore
mod counter_env;

use counter_env::CounterEnv;
```

The observation and action spaces are part of the contract. The observation space must match
the tensors returned by `reset` and `step`. The action space must accept the
actions that `step` understands.

## Vectorize the Environment

Build several instances, then wrap them as a batch:

```rust,ignore
let envs = (0..4)
    .map(|_| CounterEnv::new(device.clone()))
    .collect::<Vec<_>>();
let env = VectorizedGymWrapper::new(envs)?;
```

`VectorizedGymWrapper` handles the batched action split and auto-reset behavior.
The individual environment only needs to implement the single-environment
`Gym` contract.
For four instances, observations have shape `[4, 1]` and actions have shape
`[4]`. Construction fails when the environment list is empty.

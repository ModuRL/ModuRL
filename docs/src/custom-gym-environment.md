# Build a Custom Gym Environment

Implement `Gym` for one environment. Then place one or more instances in a
`VectorizedGymWrapper` when an agent needs batched interaction.

This small environment has one floating-point observation. Action `0` moves its
state left and action `1` moves it right. An episode ends when the state reaches
either bound.

## Define the Environment

The following code belongs in `src/counter_env.rs`:

```rust,ignore
use candle_core::{Device, Tensor};
use modurl::prelude::*;

pub struct CounterEnv {
    state: i32,
    device: Device,
}

impl CounterEnv {
    pub fn new(device: Device) -> Self {
        Self { state: 0, device }
    }

    fn observation(&self) -> candle_core::Result<Tensor> {
        Tensor::from_vec(vec![self.state as f32], (1,), &self.device)
    }
}

impl Gym for CounterEnv {
    type Error = candle_core::Error;
    type SpaceError = candle_core::Error;

    fn reset(&mut self) -> Result<Tensor, Self::Error> {
        self.state = 0;
        self.observation()
    }

    fn step(&mut self, action: Tensor) -> Result<StepInfo, Self::Error> {
        match action.to_vec0::<u32>()? {
            0 => self.state -= 1,
            1 => self.state += 1,
            _ => panic!("action is outside the action space"),
        }

        let done = self.state.abs() >= 4;
        Ok(StepInfo {
            state: self.observation()?,
            reward: 1.0,
            done,
            truncated: false,
        })
    }

    fn observation_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
        Box::new(BoxSpace::new_with_universal_bounds(
            vec![1],
            -4.0,
            4.0,
            &self.device,
        ))
    }

    fn action_space(&self) -> Box<dyn Space<Error = Self::SpaceError>> {
        Box::new(Discrete::new(2))
    }
}
```

`reset` returns the initial observation. `step` consumes one action and returns
the observation that follows it, its reward, and the episode flags.

In `src/main.rs`, declare the module and bring the environment into scope:

```rust,ignore
mod counter_env;

use counter_env::CounterEnv;
```

The `Space` values are part of the contract. The observation space must match
the tensors returned by `reset` and `step`. The action space must accept the
actions that `step` understands.

## Vectorize the Environment

Build several instances, then wrap them exactly as in the CartPole example:

```rust,ignore
let envs = (0..4)
    .map(|_| CounterEnv::new(device.clone()))
    .collect::<Vec<_>>();
let env = VectorizedGymWrapper::from(envs);
```

`VectorizedGymWrapper` handles the batched action split and auto-reset behavior.
The individual environment only needs to implement the single-environment
`Gym` contract.

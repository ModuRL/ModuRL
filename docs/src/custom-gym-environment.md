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
    type ActionSpace = Discrete;
    type ObservationSpace = BoxSpace<2>;

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

        let terminated = self.state.abs() >= 4;
        Ok(StepInfo {
            observation: self.observation(),
            reward: 1.0,
            terminated,
            truncated: false,
            info: (),
        })
    }

    fn action_space(&self) -> Self::ActionSpace {
        Discrete::new(2).unwrap()
    }

    fn observation_space(&self) -> Self::ObservationSpace {
        BoxSpace::new_with_universal_bounds(
            [1, 1],
            -4.0,
            4.0,
            &self.device,
        )
    }
}
```

## Save and Restore State

Implement `PerfectStateGym` when the environment can save and restore all its
mutable state. For this counter, that state is the current position:

```rust,ignore
impl PerfectStateGym for CounterEnv {
    type State = i32;

    fn save_state(&self) -> Result<Self::State, Self::Error> {
        Ok(self.state)
    }

    fn set_state(&mut self, state: &Self::State) -> Result<(), Self::Error> {
        self.state = *state;
        Ok(())
    }
}
```

Implement `LegalActionsGym` to expose the currently legal action indices:

```rust,ignore
impl LegalActionsGym for CounterEnv {
    fn legal_actions(&self) -> Result<Vec<usize>, Self::Error> {
        Ok(if self.state.abs() >= 4 { vec![] } else { vec![0, 1] })
    }
}
```

Include the random-number generator state too when an environment uses randomness.
Wrapping `PerfectStateGym` environments in `VectorizedGymWrapper` provides
`PerfectStateMultiGym`, which saves and restores individual slots by index.

`reset` returns the initial observation and `step` consumes one action and
returns the observation that follows it, its reward, and the episode flags.
The default `Gym` information type is `()`, so ordinary environments use
`ResetInfo` and `StepInfo` without an explicit type parameter. Environments
with additional typed metadata can instead implement `Gym<MyInfo>`.
The full signature is `Gym<I, O, A>`. Observation and action batch ranks
`O` and `A` default to 2. Both spaces inherit `TensorBatchTypes`, which declares
`Kind`, `Item`, and `Batch`. `Item` omits the batch axis; `Batch` adds one leading
batch axis with the same tensor kind. `BoxSpace<2>` supplies `Tensor<1>` items
and `Tensor<2>` batches. `Discrete` supplies `Tensor<1, Int>` items and
`Tensor<2, Int>` batches. `Gym`, `MultiGym`, and `Agent` use these shared types.
Custom spaces implement `TensorBatchTypes` once before implementing the space
methods. Access an item type as `<S as TensorBatchTypes>::Item`.
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

The action space must accept the actions that `step` understands. The observation
space must match the tensors returned by `reset` and `step`.

## Combine Typed Observation Fields

This is ModuRL's typed answer to Gymnasium/SB3 `Dict` observations.

Keep a single flat `BoxSpace` when the observation is already a feature vector,
or when reshaping slices inside your model is enough. Use `MultiPartSpace` when
fields need distinct shapes, kinds or encoders, such as an image, a joint vector
and a discrete mode. It groups 2–8 typed spaces without string keys. Agents,
buffers and wrappers still receive one flat floating-point tensor.

For example, pack joint positions and velocities as two vector fields:

```rust,ignore
use burn::tensor::{DType, Device, Tensor};
use modurl::prelude::*;

let device = Device::flex();
let space = MultiPartSpace::new((
    BoxSpace::new_unbounded([1, 3], &device),
    BoxSpace::new_unbounded([1, 3], &device),
));
let positions = Tensor::zeros([1, 3], (&device, DType::F32));
let velocities = Tensor::zeros([1, 3], (&device, DType::F32));
let packed = space.pack((positions, velocities))?; // [1, 6]
// Gym returns unbatched observations, so remove the singleton batch axis.
let observation: Tensor<1> = packed.clone().reshape([6]);
```

Set the environment's `ObservationSpace` to
`MultiPartSpace<(BoxSpace<2>, BoxSpace<2>)>` and return `space.clone()` from
`observation_space`. Single-environment `reset` and `step` put the unbatched
`observation` above in `ResetInfo` and `StepInfo`; a `MultiGym` returns the
batched `packed` tensor instead. Propagate `SpaceError` through the environment's
error type.

On the model side, give each field its own encoder. This continues the example
with two MLPs, each producing eight features:

```rust,ignore
let positions_encoder = MLP::builder()
    .input_size(3).output_size(8).options((&device, DType::F32))
    .build()?;
let velocities_encoder = MLP::builder()
    .input_size(3).output_size(8).options((&device, DType::F32))
    .build()?;
let encoder = MultiPartEncoder::new(
    (positions_encoder, velocities_encoder), space.clone(),
);
let features = encoder.forward(packed)?; // [1, 16]
```

Own `encoder` and a trunk or output heads in your Burn module, and pass these
features to the next module in its `Forward` implementation. Each field encoder
implements `Forward<R, 2>` with the field's native input kind and `Float` output.
For an image `BoxSpace<4>`, use a rank-4 encoder; for `Discrete`, use a rank-2
encoder with `InputKind = Int`. See [Models, Policies, and
Distributions](./models-policies-and-distributions.md#models-produce-tensors)
for the model contract.

Named application structs can convert to and from the typed batch tuple:

```rust,ignore
struct MotionObs { positions: Tensor<2>, velocities: Tensor<2> }

impl From<MotionObs> for (Tensor<2>, Tensor<2>) {
    fn from(obs: MotionObs) -> Self { (obs.positions, obs.velocities) }
}
impl From<(Tensor<2>, Tensor<2>)> for MotionObs {
    fn from((positions, velocities): (Tensor<2>, Tensor<2>)) -> Self {
        Self { positions, velocities }
    }
}
// space.pack(obs.into())?; MotionObs::from(space.unpack(packed)?);
```

`pack` checks shapes (including batch sizes) and device compatibility, not bounds;
call `space.contains(&packed)` to check bounds. Mixing F32 and F64 fields promotes
everything to F64, and unpacked floating fields keep that dtype. `Discrete`
fields travel as floats in the packed tensor and come back as `Int` after
`unpack`; valid indices are lossless because `Discrete` caps its category count
at 2^24. Match encoder parameter dtypes to the packed dtype. Empty batches are
rejected.

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
`[4, 1]`. Construction fails when the environment list is empty.

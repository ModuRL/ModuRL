# Environment Wrappers

Wrappers change an environment's observations, rewards, metadata, or episode
boundaries while preserving the `Gym` interface. When wrappers are nested, the
innermost wrapper processes a transition first.

## Core Wrappers

These wrappers are available from `modurl::wrappers` and work with any
compatible `Gym`.

| Wrapper | What it does |
| --- | --- |
| `TimeLimitGym` | Sets `truncated` after a fixed number of steps. |
| `RecordEpisodeStatisticsGym` | Adds the completed episode's return and length to `EpisodeStatisticsInfo`. |
| `RecordRawRewardGym` | Copies each reward into `RawRewardInfo` before outer wrappers can change it. |
| `NormalizeObservationGym` | Normalizes each observation component with a running mean and variance, with optional clipping. |
| `NormalizeRewardGym` | Scales rewards by the running standard deviation of discounted rewards, with optional clipping. |
| `FrameStackGym` | Stacks recent observations along a new leading dimension. |
| `MaxAndSkipGym` | Repeats an action, sums its rewards, and max-pools the final two observations. |
| `ClipRewardGym` | Maps each reward to `-1`, `0`, or `1` according to its sign. |

Wrapper order determines which values a wrapper sees. For example, placing
`RecordEpisodeStatisticsGym` inside `ClipRewardGym` records the underlying return
while the agent receives clipped rewards.

### Configuration errors

The following constructors and clipping methods return `Result` instead of
panicking on invalid configuration. Handle the result with `?` or `.expect(...)`
before using the wrapper.

| Method | Error type and rejected configuration |
| --- | --- |
| `TimeLimitGym::new` | `TimeLimitGymConfigurationError::ZeroMaxEpisodeSteps` for a zero step limit. |
| `MaxAndSkipGym::new` | `MaxAndSkipGymConfigurationError::ZeroSkip` for zero action repeats. |
| `FrameStackGym::new` | `FrameStackGymConfigurationError::ZeroStackSize` for a zero stack size, or `InvalidObservationSpace` when the leading item axis is missing or differs from the stack size. |
| `NormalizeRewardGym::new` | `NormalizeRewardGymConfigurationError::InvalidGamma` for a non-finite discount or one outside `0..=1`. |
| `NormalizeRewardGym::with_clip` | `NormalizeRewardGymConfigurationError::InvalidClip` for a non-finite or nonpositive limit. |
| `NormalizeObservationGym::with_clip` | `NormalizeObservationGymConfigurationError::InvalidClip` for a non-finite or nonpositive limit. |

`NormalizeObservationGym::new` remains infallible. For example, clipping requires
handling a result even though creating the observation normalizer does not:

```rust,ignore
let gym = TimeLimitGym::new(gym, 500).expect("nonzero episode limit");
let gym = NormalizeObservationGym::new(gym)
    .with_clip(10.0)
    .expect("finite positive observation clip");
let gym = NormalizeRewardGym::new(gym, 0.99)
    .expect("discount in 0..=1")
    .with_clip(10.0)
    .expect("finite positive reward clip");
```

`FrameStackGym<G, BF, S>` uses native Burn tensors. `BF` is the stacked batch
rank. Two `PrevRank` mappings determine the inner observation's single rank. The defaults
stack vectors `[features]` into `[stack_size, features]`. Scalar observations use
`FrameStackGym::<_, 3>::new(...).expect("valid frame stack")` and produce `[stack_size, 1]`.
The supplied space `S` must use the inner observation kind. For `BoxSpace`,
bounds include a size-one batch axis. Invalid rank relationships fail during compilation.

`InputMapMultiGymWrapper` and `OutputMapMultiGymWrapper` callbacks return
`Result` with a caller-defined error type. Callback errors use
`TensorMapMultiGymError::Mapping`; environment errors use
`TensorMapMultiGymError::Gym`. Callbacks
must preserve the documented tensor layouts and kinds. Observation normalization
reports host-read failures through `NormalizeObservationGymError::TensorRead`.
Output callbacks use the observation space's `Batch` type for batched observations
and `Item` type for terminal observations. These types follow the underlying space.

## Atari Wrappers

These wrappers are available from `modurl_ale::wrappers` and implement the
standard Atari preprocessing steps.

| Wrapper | What it does |
| --- | --- |
| `NoopResetGym` | Takes a random number of action `0` no-ops after reset; the default range is 1 through 30. |
| `EpisodicLifeGym` | Reports a lost life as `terminated` while continuing the same game on reset. |
| `FireResetGym` | Takes actions `1` and `2` after reset to start games that require FIRE. |
| `WarpGym` | Converts Atari observations to grayscale and resizes them to 84 by 84 pixels. |

`EpisodicLifeGym` requires metadata that implements `AtariLives`. `AtariInfo`
and `EpisodeStatisticsInfo<I>` provide that implementation when their inner
metadata exposes Atari lives.

Batching adapters such as `VectorizedGymWrapper` and `StackedMultiGym` are
covered in [Use Vectorized Environments](./vectorized-environments.md).

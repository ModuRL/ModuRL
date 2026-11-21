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

### Saving normalization statistics

`NormalizeObservationGym` keeps its running mean, variance, and sample count as tensors in
a `RunningMeanStd`, a Burn module; `NormalizeRewardGym` builds one when you call
`statistics()`. Export it with `statistics()`, save it with Burn's `save_file` like a model, and load it with
`try_load_file` into a template from `RunningMeanStd::new(size, &device)`, then pass it to
a new wrapper with `with_statistics` so a trained policy is evaluated or deployed with the
statistics it trained on. Call `with_update_statistics(false)` to keep loaded statistics
fixed during evaluation. The file holds only the statistics, so set the same `with_clip`
limit used in training.

Statistics are kept in `f64` on devices that support it, otherwise in F32. Loading keeps the saved dtype, so load `f64` statistics on a device that supports
`f64`, even if the policy runs on another device. `try_load_file` checks parameter names and shapes but not values, so call
`validate()` before saving and after loading; it also rejects tensors on different devices or with different dtypes. Statistics and
model checkpoints are both `.bpk` files but are not interchangeable: each fails to load
into the other.

`NormalizeObservationGym` promotes half-precision (F16 or BF16) statistics to F32 when it
installs or creates them, so updates do not overflow and an observation equal to the mean
of a zero-variance component gives `0` instead of NaN. Any other value of that component is
divided by `sqrt(1e-8)` = `1e-4`, so such values scale by 10,000; set `with_clip` to bound
them. `NormalizeRewardGym` keeps its statistics on the host in `f64`, so
stepping a CPU environment never touches the GPU; `statistics()` builds a new
`RunningMeanStd` from them, in `f64` where the device supports it (F32 otherwise, never
half precision), on the device of the
statistics passed to `with_statistics` (Burn's default device otherwise).

`NormalizeObservationGym::statistics()` returns an `Option` because the wrapper sizes
its statistics from the first observation; `NormalizeRewardGym::statistics()` always
has its single component.

| Method | Result |
| --- | --- |
| `RunningMeanStd::from_tensors(mean, variance, count)` / `validate` | `RunningMeanStdError` for empty or mismatched components, tensors on different devices or with different dtypes, a non-finite mean, a negative or non-finite variance, or a nonpositive count. |
| `RunningMeanStd::try_load_file` | Burn's `RecordError` for a missing or corrupt file, a file without statistics, or statistics of a different length than the template. |
| `NormalizeObservationGym::with_statistics` | Infallible; `reset` and `step` return `NormalizeObservationGymError::StatisticsSizeMismatch` before calling the inner gym when installed or restored statistics differ in length from the declared observation shape (checked once per install), and again if an observation's component count differs. |
| `NormalizeRewardGym::with_statistics` | `NormalizeRewardGymConfigurationError::InvalidStatisticsSize` unless the statistics have exactly one component; `InvalidStatistics` if `validate()` rejects them. |

Both wrappers expose `inner`, `inner_mut`, and `into_inner`. Their saved
`PerfectStateGym` states expose `statistics()` and restore the `update_statistics`
setting. Each wrapped environment keeps its own statistics, so with
`VectorizedGymWrapper` read them from one of `envs()`:

```rust,ignore
// After training: rewards wrap observations, inside a vectorized wrapper.
let reward_gym = &env.envs()[0];
// Check the values first: a NaN reward or observation would otherwise be saved as-is.
let obs_rms = reward_gym.inner().statistics().expect("observed at least once").clone();
obs_rms.validate()?;
obs_rms.save_file("obs_rms.bpk")?;
let reward_rms = reward_gym.statistics();
reward_rms.validate()?;
reward_rms.save_file("reward_rms.bpk")?;

// Evaluation: load into a template of the observation length, then check the values.
// `obs_dim` is the observation's component count, the length the training wrapper saw.
let statistics = RunningMeanStd::new(obs_dim, &device).try_load_file("obs_rms.bpk")?;
statistics.validate()?;
let gym = NormalizeObservationGym::new(gym)
    .with_statistics(statistics)
    .with_update_statistics(false)
    .with_clip(10.0)?;
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

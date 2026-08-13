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

## Atari Wrappers

These wrappers are available from `modurl_ale::wrappers` and implement the
standard Atari preprocessing steps.

| Wrapper | What it does |
| --- | --- |
| `NoopResetGym` | Takes a random number of action `0` no-ops after reset; the default range is 1 through 30. |
| `EpisodicLifeGym` | Reports a lost life as `done` while continuing the same game on reset. |
| `FireResetGym` | Takes actions `1` and `2` after reset to start games that require FIRE. |
| `WarpGym` | Converts Atari observations to grayscale and resizes them to 84 by 84 pixels. |

`EpisodicLifeGym` requires metadata that implements `AtariLives`. `AtariInfo`
and `EpisodeStatisticsInfo<I>` provide that implementation when their inner
metadata exposes Atari lives.

Batching adapters such as `VectorizedGymWrapper` and `StackedMultiGym` are
covered in [Use Vectorized Environments](./vectorized-environments.md).

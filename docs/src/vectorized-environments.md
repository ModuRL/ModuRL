# Use Vectorized Environments

`MultiGym` steps several environments with one batch of actions. PPO uses
this interface so one rollout step can collect several transitions.

`VectorizedGymWrapper` turns a `Vec<G>` of ordinary `Gym` values into a
vectorized environment:

```rust,ignore
let envs = (0..4)
    .map(|_| CartPoleV1::builder().device(&device).build().unwrap())
    .collect::<Vec<_>>();
let mut env = VectorizedGymWrapper::from(envs);
```

## Reset Once, Then Step

Before a manual loop, call `reset` once to receive one initial observation per
inner environment. Pass that batch to `Agent::act`, then pass the returned batch
of actions to `MultiGym::step`.

```rust,ignore
let mut states = env.reset()?;

loop {
    let actions = agent.act(&states)?;
    let step = env.step(actions)?;
    states = step.states;
}
```

`states` has one next observation for every inner environment, so it is ready
for the next call to `act`.

## Understand Auto-Reset

When an inner environment returns `done` or `truncated`, ModuRL resets that one
environment immediately. The `states` field then contains the first observation
of its next episode. This lets the next batched step continue without a special
reset branch.

The terminal observation is still available. `terminal_states` contains an
entry for each inner environment: `Some(state)` when that environment ended and
`None` when it continued.

If code needs the true next state for each transition, call
`transition_next_states`:

```rust,ignore
let step = env.step(actions)?;
let transition_next_states = step.transition_next_states()?;
let next_states_for_the_loop = step.states;
```

`transition_next_states` uses a terminal state where one exists and the normal
next state otherwise. The second value, `step.states`, remains the right input
for the following action-selection step.

`PPOAgent::learn` handles this distinction while it collects experience. You
only need it when you write a loop that consumes transitions yourself.

## Use One Shared World for Several Players

A custom `MultiGym` can use batch rows for coupled players instead of
independent simulations. The environment must consume every player's action
before it advances the shared world and must keep termination and reset
behavior consistent across those rows.

For example, a custom two-player game can expose players as the leading tensor
dimension:

```rust,ignore
let mut env = CoupledGame::new()?;
let states = env.reset()?; // [players, ...observation_shape]
let actions = agent.act(&states)?; // [players, ...action_shape]
let step = env.step(actions)?; // advances the shared game once
```

For a shared-policy agent, acting on the whole observation batch applies the
same policy to every player and provides self-play without creating duplicate
physics simulations. A coupled implementation should end and reset every
player row together whenever the shared episode ends.

## Stack Several Batched Environments

`StackedMultiGym` combines several homogeneous `MultiGym` values into one flat
batch. For example, four two-player games become eight batch rows:

```rust,ignore
let games = (0..4)
    .map(|seed| {
        let mut game = CoupledGame::new()?;
        game.seed(seed);
        Ok(game)
    })
    .collect::<Result<Vec<_>, GameError>>()?;
let mut env = StackedMultiGym::try_new(games)?;

let states = env.reset()?; // [8, ...observation_shape]
let actions = agent.act(&states)?; // [8, ...action_shape]
let step = env.step(actions)?; // steps each shared game once
```

Rows are ordered first by inner gym and then by that gym's own row order.
`group_offsets()` maps the flattened rows back to their inner gyms. All inner
gyms must expose the same observation and action shapes, and each inner gym
keeps responsibility for its own auto-reset behavior.

With the `multithreading` feature enabled,
`MultithreadedStackedMultiGym` runs each inner `MultiGym` on a persistent
worker thread. Pass constructors so every inner gym is created on the thread
that owns it, together with representative observation and action spaces:

```rust,ignore
let constructors = (0..4)
    .map(|seed| move || {
        let mut game = CoupledGame::new().unwrap();
        game.seed(seed);
        game
    })
    .collect();
let mut env = MultithreadedStackedMultiGym::try_new(
    constructors,
    observation_space,
    action_space,
)?;
```

The whole inner gym remains one unit of work, so coupled player rows are never
split across threads. If an inner gym returns an error, reset the complete
stack before stepping it again.

Next, read [Build a Custom Gym Environment](./custom-gym-environment.md) to
provide your own single-environment implementation.

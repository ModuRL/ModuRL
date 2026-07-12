# Use Vectorized Environments

`VectorizedGym` steps several environments with one batch of actions. PPO uses
this interface so one rollout step can collect several transitions.

`VectorizedGymWrapper` turns a `Vec<G>` of ordinary `Gym` values into a
vectorized environment:

```rust,ignore
let envs = (0..4)
    .map(|_| CartPoleV1::builder().device(&device).build())
    .collect::<Vec<_>>();
let mut env = VectorizedGymWrapper::from(envs);
```

## Reset Once, Then Step

Before a manual loop, call `reset` once to receive one initial observation per
inner environment. Pass that batch to `Agent::act`, then pass the returned batch
of actions to `VectorizedGym::step`.

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

Next, read [Build a Custom Gym Environment](./custom-gym-environment.md) to
provide your own single-environment implementation.

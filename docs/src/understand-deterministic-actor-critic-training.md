# Understand a Deterministic Actor-Critic Training Run

The DDPG and TD3 examples confirm that their actors, critics, target networks,
and replay buffers train together. Add a `DDPGLogger` or `TD3Logger` when you
need to compare runs or diagnose one of those components.

Each logger receives two streams. `log` receives one entry per replay
optimization. `log_collection` receives rewards and completed episodes from
one vectorized environment step. An update describes a sampled replay batch;
collection describes the policy's newest behavior.

## Add a Logger

Both logger traits use the shared deterministic actor-critic entry types. One
logger can therefore support both algorithms:

```rust,ignore
use modurl::prelude::*;

struct ConsoleLogger;

fn log_update(entry: &DeterministicActorCriticLogEntry) {
    if entry.update_index % 1_000 != 0 {
        return;
    }

    let critic_loss = entry.critic_losses[0]
        .to_scalar::<f32>()
        .expect("critic loss must be scalar");
    let actor_loss = entry.actor_loss
        .as_ref()
        .map(|loss| loss.to_scalar::<f32>().expect("actor loss must be scalar"));

    println!(
        "step={} update={} critic_loss={critic_loss:.4} \
         actor_updated={} actor_loss={actor_loss:?}",
        entry.collection_timestep,
        entry.update_index,
        entry.actor_updated,
    );
}

fn log_collection<I>(entry: &DeterministicActorCriticCollectionLogEntry<I>) {
    for episode in &entry.completed_episodes {
        println!(
            "step={} env={} return={} length={} terminated={} truncated={}",
            episode.collection_timestep,
            episode.environment_index,
            episode.episode_return,
            episode.episode_length,
            episode.terminated,
            episode.truncated,
        );
    }
}

impl<I> DDPGLogger<I> for ConsoleLogger {
    fn log(&mut self, entry: &DeterministicActorCriticLogEntry) {
        log_update(entry);
    }

    fn log_collection(
        &mut self,
        entry: &DeterministicActorCriticCollectionLogEntry<I>,
    ) {
        log_collection(entry);
    }
}

impl<I> TD3Logger<I> for ConsoleLogger {
    fn log(&mut self, entry: &DeterministicActorCriticLogEntry) {
        log_update(entry);
    }

    fn log_collection(
        &mut self,
        entry: &DeterministicActorCriticCollectionLogEntry<I>,
    ) {
        log_collection(entry);
    }
}
```

Pass a mutable reference while building either agent:

```rust,ignore
let mut logger = ConsoleLogger;

let mut agent = TD3Agent::builder()
    // Keep the remaining TD3 configuration unchanged.
    .logger(&mut logger)
    .build()?;
```

The agent borrows the logger. Drop the agent before reading or displaying
values held by the concrete logger, as the terminal graph examples do.

## Read Replay-Update Metrics

`DeterministicActorCriticLogEntry` exposes these values:

| Field | Meaning and shape |
| --- | --- |
| `critic_losses` | One scalar mean-squared Bellman loss per critic |
| `critic_q_values` | One `[batch_size]` replay-action Q tensor per critic |
| `actor_loss` | Scalar negative mean policy Q, or `None` on a delayed update |
| `policy_q_values` | `[batch_size]` actor-objective Q values, or `None` |
| `policy_actions` | `[batch_size, ...action_shape]`, or `None` |
| `replay_actions` | Sampled replay actions `[batch_size, ...action_shape]` |
| `bellman_targets` | Detached target Q values `[batch_size]` |
| `replay_rewards` | Sampled rewards `[batch_size]` |
| `actor_learning_rate` | Current actor optimizer learning rate |
| `critic_learning_rates` | Current learning rate for each critic optimizer |
| `exploration_noise_standard_deviation` | Collection-noise setting |
| `actor_updated` | Whether this update changed actor and target networks |
| `update_index` | Zero-based replay-update index |
| `collection_timestep` | Global transition count that triggered the update |

DDPG sets `actor_updated` on every replay update. TD3 sets it according to
`actor_update_interval`. When it is false, `actor_loss`, `policy_q_values`, and
`policy_actions` are all `None`; critic metrics remain present.

Compare `critic_q_values` with `bellman_targets` when a critic loss changes
unexpectedly. Compare `policy_actions` with `replay_actions` to distinguish
the current actor from behavior stored earlier in replay.

In canonical TD3, `policy_q_values` come from the first online critic. If
`actor_aggregation_mode` is configured, they contain the aggregate used by the
actor objective.

## Read Collection Metrics

`DeterministicActorCriticCollectionLogEntry` describes the newest vectorized
environment step:

| Field | Meaning |
| --- | --- |
| `collection_rewards` | One newest reward per environment |
| `infos` | Typed metadata returned by each environment |
| `collection_timestep` | Global transition count after this vectorized step |
| `completed_episodes` | Episodes that terminated or truncated on this step |
| `replay_len` | Number of transitions currently retained in replay |

Each completed episode records its environment index, return, length, ending
condition, and global collection timestep. Partial episodes carry across
vectorized steps until the environment terminates or truncates.

During the initial random-action phase, collection entries arrive but replay
update entries do not. After `training_start`, update entries occur only at
timesteps selected by `update_frequency`.

You can now separate current policy behavior from replay optimization and, for
TD3, delayed actor updates from critic-only updates. The repository's
`DeterministicActorCriticGrapher` applies the same split to terminal plots in
both MuJoCo examples.

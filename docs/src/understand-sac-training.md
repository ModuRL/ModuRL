# Understand an SAC Training Run

The SAC examples confirm that the actor, critics, replay buffer, and entropy
configuration can train together. Add a `SACLogger` when you need to compare
runs or diagnose one of those components.

`SACLogger::log_update` receives metrics from replay optimization.
`SACLogger::log_collection` receives rewards and completed episodes from the
current environment step. Keep these streams separate: an update describes a
sampled replay batch, while collection describes the policy's newest behavior.

## Add a Logger

This logger prints occasional update values and every completed episode:

```rust,ignore
use modurl::prelude::*;

struct ConsoleLogger;

impl<I> SACLogger<I> for ConsoleLogger {
    fn log_update(&mut self, entry: &SACLogEntry) {
        if entry.update_index % 1_000 != 0 {
            return;
        }

        let actor_loss = entry
            .actor_loss
            .to_scalar::<f32>()
            .expect("actor loss must be scalar");
        let alpha = entry
            .alpha
            .to_scalar::<f32>()
            .expect("alpha must be scalar");

        println!(
            "step={} update={} actor_loss={actor_loss:.4} alpha={alpha:.4}",
            entry.collection_timestep,
            entry.update_index,
        );
    }

    fn log_collection(&mut self, entry: &SACCollectionLogEntry<I>) {
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
}
```

Pass a mutable reference while building the agent:

```rust,ignore
let mut logger = ConsoleLogger;

let mut agent = SACAgent::builder()
    // Keep the remaining SAC configuration unchanged.
    .logger(&mut logger)
    .build()?;

agent.learn(&mut env, total_timesteps)?;
```

The repository's `sac_mujoco` example uses `SACGrapher` to aggregate these
callbacks into terminal graphs.

## Start With Episode Performance

`SACCollectionLogEntry::completed_episodes` contains one
`SACEpisodeLogEntry` for each environment that ended during the latest
vectorized step. Each entry reports:

- `environment_index`: which inner environment completed
- `episode_return`: the sum of that episode's rewards
- `episode_length`: the number of environment steps
- `terminated`: whether the environment reached a terminal state
- `truncated`: whether an external limit ended the episode
- `collection_timestep`: the total collected-transition count at completion

Episode return is the clearest measure of task performance. Compare it across
several episodes rather than treating one episode as a trend. Episode length is
useful when termination timing is meaningful for the environment.

`SACCollectionLogEntry` also contains:

- `collection_rewards`: one reward per inner environment from the newest step
- `infos`: the corresponding typed environment metadata
- `collection_timestep`: the total number of collected transitions
- `replay_len`: the number of entries currently held in replay

`log_collection` runs once per vectorized environment step. With `N` inner
environments, one callback normally advances `collection_timestep` by `N`.

## Read Update Metrics as a Group

After `training_start`, SAC performs an optimization step for each collected
transition. Each call to `log_update` describes one sampled replay batch.
`update_index` is the zero-based optimization count, while
`collection_timestep` identifies the transition that triggered the update.

Let `B` be replay batch size and `K` the number of policy candidates. For a
categorical policy, `K` is the action count. For a sampled continuous policy,
`K` is the configured `samples` value.

| Field | Shape | Meaning |
| --- | --- | --- |
| `critic_losses` | One scalar per critic | Critic objectives for the replay batch |
| `actor_loss` | Scalar | Objective used for the actor update |
| `alpha_loss` | Optional scalar | Automatic entropy-coefficient objective |
| `entropy_change_loss` | Optional scalar | Discrete stabilization penalty |
| `target_entropy` | Optional `f64` | Current automatic target entropy |
| `alpha` | Scalar | Current entropy coefficient |
| `bellman_targets` | `[B]` | Detached soft targets shared by all critics |
| `policy_log_probabilities` | `[B, K]` | Candidate log probabilities |
| `policy_weights` | `[B, K]` | Candidate expectation weights |
| `policy_q_values` | `[B, K]` | Aggregated Q-values used by the actor |
| `replay_rewards` | `[B]` | Rewards from the sampled replay entries |

Losses do not have a universal target value. Reward scale, model architecture,
and entropy configuration all change their magnitude. Compare them with episode
return and with earlier runs using the same environment and reward handling.

Critic loss measures disagreement with the soft Bellman targets. A persistent
increase alongside unstable Q-values can indicate an aggressive learning rate
or reward scale. A small critic loss alone does not prove that the policy is
good.

Actor loss combines expected Q-values and the entropy term. It can be negative,
and its raw value is not a task score. Use it to spot abrupt changes rather than
to rank policies.

Alpha controls the strength of the entropy term. A larger alpha places more
weight on uncertain actions. With automatic tuning, compare alpha, policy
entropy, and target entropy together.

The expected policy entropy for one update is:

```text
-mean(sum(policy_weights * policy_log_probabilities, candidate_axis))
```

Entropy commonly falls when a policy becomes more certain. Whether that is
healthy depends on the target entropy and episode performance. A rapid collapse
with poor returns suggests that exploration ended too soon.

`entropy_change_loss` appears only when the stabilization configuration enables
the replay-to-current entropy penalty. `alpha_loss` and `target_entropy` appear
only with automatic entropy tuning.

## Compare Collection and Replay Values

`collection_rewards` show the newest environment behavior.
`replay_rewards` come from a randomly sampled historical batch. They need not
move together at each callback.

The same distinction applies to time:

- `collection_timestep` counts transitions gathered from environments.
- `update_index` counts replay optimization steps.
- `replay_len` grows until it reaches replay capacity.

Graph completed episode returns against `collection_timestep`. Graph update
losses, alpha, expected policy entropy, and mean Bellman targets at a lower
frequency if logging every optimization step is too expensive.

Return to [Soft Actor-Critic](./sac.md) to change entropy, critic aggregation,
or stabilization. Read [Run on CUDA or
Metal](./run-on-cuda-or-metal.md) to separate replay storage from optimization.

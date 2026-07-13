# Understand a Q-Learning Training Run

The DQN and DDQN programs confirm that training completed, but they do not
record metrics. Add a logger when you need to compare runs or diagnose a
configuration. Each logger receives an update entry when the agent optimizes
from replay and a collection entry after every vectorized environment step.

The repository's `dqn_cartpole_with_graphs` example uses both entry types. It
prints completed CartPole episodes during training and draws loss, exploration,
Q-value, return, and length graphs when training ends. Run it with:

```sh
cargo run --example dqn_cartpole_with_graphs
```

Implement the logger trait that matches the agent. Both traits receive the same
entry types:

```rust,ignore
struct TrainingLogger;

impl DQNLogger for TrainingLogger {
    fn log(&mut self, entry: &QLogEntry) {
        println!("update {}: loss = {:?}", entry.update_index, entry.loss);
    }

    fn log_collection(&mut self, entry: &QCollectionLogEntry) {
        for episode in &entry.completed_episodes {
            println!(
                "environment {}: return = {}, length = {}",
                episode.environment_index,
                episode.episode_return,
                episode.episode_length,
            );
        }
    }
}

let mut logger = TrainingLogger;
let mut agent = DQNAgent::builder()
    // The remaining DQN configuration is unchanged.
    .logger(&mut logger)
    .build()
    .expect("DQN configuration should be valid");
```

For DDQN, implement `DDQNLogger` and pass the logger to `DDQNAgent::builder()`
in the same way. `log_collection` has a default no-op implementation, so a
logger can ignore collection metrics when it only needs update metrics. The
agent borrows the logger, so keep it alive for as long as the agent is in use.

## Read Update Metrics

`QLogEntry` describes one optimization update from a sampled replay batch. The
agent does not emit it for every environment transition. It starts after replay
warm-up and then follows the update frequency. The entry reaches the logger
before the optimizer applies that update.

The logger exposes these values through `QLogEntry`:

- `loss`: the mean-squared error between the online network's value for each
  sampled action and its Q-learning target
- `epsilon`: the exploration probability used while collecting experience
- `learning_rate`: the optimizer's current learning rate
- `q_values`: the online network's value for the selected action in each
  sampled replay transition
- `replay_rewards`: the one-step rewards in that sampled replay batch
- `update_index`: the zero-based index of this optimization update
- `collection_timestep`: the total number of environment transitions collected
  when the agent formed this update

`q_values` and `replay_rewards` are tensors for a replay batch, not single
summary numbers. Aggregate them, such as with a mean, before graphing or
comparing runs. A replay batch can contain transitions from old episodes and
from earlier versions of the policy.

## Read Collection Metrics

`QCollectionLogEntry` describes the newest environment interaction, not a
sample from replay. The agent emits it after every call to `VectorizedGym::step`,
including during replay warm-up.

- `collection_rewards`: one reward for each inner environment from the latest
  vectorized step
- `epsilon`: the exploration probability that selected the actions for that
  step
- `collection_timestep`: the total number of transitions collected after that
  step
- `completed_episodes`: episodes that ended in this vectorized step

`completed_episodes` is empty when every inner environment continues its current
episode. It contains one `QEpisodeLogEntry` for each inner environment that
ended in the latest vectorized step, so one collection entry can report several
completed episodes.

Each `QEpisodeLogEntry` contains the following summary for one finished episode:

- `environment_index`: which inner environment produced the episode
- `episode_return`: the sum of every reward collected since that environment's
  last reset
- `episode_length`: the number of environment steps collected since that reset
- `terminated` and `truncated`: the environment's ending flags
- `collection_timestep`: the total number of collected transitions when that
  environment finished

The agent resets its running return and length for that environment after
recording the entry. An entry never represents an in-progress episode, and it
never combines rewards from several episodes. Episode returns and lengths are
current collection metrics, so they are not mixed with older replay data.

## Read the Metrics Together

Loss has no universal target. Compare it with earlier runs that use the same
environment, reward scale, model shape, and replay settings. A noisy loss is
normal because every update samples a different replay batch. A persistently
growing loss or non-finite values is a signal to inspect the configuration.

The Q-values are estimates, not a direct success score. Their magnitude depends
on the reward scale, `gamma`, and the remaining rewards the agent expects. Use
them to spot abrupt changes or divergence, and use episode returns and lengths
to decide whether the policy is improving.

Epsilon should follow the schedule you configured. In the DQN example it falls
from `1.0` to `0.05`; if it remains high, the agent continues to choose random
actions often. If it falls too quickly, the collection stream may contain too
little exploration when the agent begins learning.

## Change One Setting at a Time

Keep the environment, model shape, replay capacity, and epsilon schedule fixed
while comparing a change to the learning rate, target-update interval, or update
frequency. That makes a change in loss or Q-values easier to attribute to one
configuration decision.

Return to [DQN](./dqn.md) or [Double DQN](./ddqn.md) to change the training
program.

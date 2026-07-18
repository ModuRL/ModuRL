# Understand a PPO Training Run

The Getting Started program confirms that training completed, but it does not
record metrics. Add a `PPOLogger` when you need to compare runs or diagnose a
configuration. `PPOLogger::log` receives update metrics, while
`PPOLogger::log_collection` receives the newest environment rewards and any
episodes completed during that vectorized step.

Each `PPOLogEntry` contains tensors for one optimization minibatch. Aggregate
them before graphing or comparing them across a run. The graphs are useful for
comparing or diagnosing runs.

## Start With Episode Performance

CartPole gives a reward of one for each step that keeps the pole balanced. The
example graphs completed episode returns and lengths, so longer episodes are
the clearest sign that the policy is improving.

`PPOCollectionLogEntry::completed_episodes` contains one `PPOEpisodeLogEntry`
for each environment that terminated or truncated on that vectorized step.
Each episode entry reports its environment index, return, length, ending flags,
and collection timestep. Partial episodes carry across PPO rollout boundaries
and repeated `learn` calls.

`PPOCollectionLogEntry::infos` contains the typed step metadata from each
vectorized environment in the same environment order as `collection_rewards`.
Wrappers can add logging-only values there without changing the reward used for
training. For example, `RecordRawRewardGym` records rewards before outer reward
wrappers normalize or clip them.

## Read the Other Metrics Together

The logger exposes the following values through `PPOLogEntry`:

- `actor_loss`: the policy objective used for the update
- `critic_loss`: the value-model objective used for the update
- `entropy`: how spread out the policy's action choices are
- `kl_divergence`: how far the updated policy moved from the policy that
  collected the rollout
- `explained_variance`: how well the value model accounts for return variation
- `rewards`: rewards from the collected transitions

Loss values do not have one universally good target. They are most useful when
compared with episode length and with earlier runs using the same configuration.

Entropy commonly falls as the policy becomes more certain. A rapid collapse can
mean the policy stopped exploring too early. KL divergence shows the size of the
policy change. Large, erratic changes suggest that the update settings may be
too aggressive.

Explained variance shows whether the value model predicts returns better than a
constant average prediction: values nearer one are better, but early updates can
be noisy.

## Change One Setting at a Time

The CartPole example sets these PPO builder values:

```rust,ignore
.batch_size(2048)
.mini_batch_size(64)
.ent_coef(0.005)
.clip_range(Box::new(ConstantSchedule::new(0.2)))
.num_epochs(10)
.training_horizon(120_000)
```

Use one changed value per experiment. Keep the device, environment count, model
shape, and remaining PPO settings fixed while comparing the graph. That makes a
change in episode length or KL divergence easier to attribute to its cause.

Next, read [Use Vectorized Environments](./vectorized-environments.md) to see
how PPO receives batches of transitions.

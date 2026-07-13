# Value-Based Training

Use DQN or DDQN when your environment has a discrete action space and you want
the agent to learn an action value for every possible action.

This chapter explains the pieces shared by `DQNAgent` and `DDQNAgent`. Read
[DQN](./dqn.md) for a complete CartPole program. Read [Double DQN](./ddqn.md)
when you want DDQN's target calculation instead.

## Choose DQN or DDQN

Both agents train an online Q-network and periodically copy its parameters to a
target Q-network. They differ only in how they form the next-state training
target:

| Agent | Chooses the next action | Evaluates that action |
| --- | --- | --- |
| `DQNAgent` | The target Q-network's largest value | The same target Q-network |
| `DDQNAgent` | The online Q-network's largest value | The target Q-network |

Start with DQN when you need the standard DQN update. Choose DDQN when you want
to reduce overestimation of action values. In DQN, the target network both
chooses the largest next-state value and uses that value in the training target.
Taking a maximum tends to favor values that are accidentally too high. DDQN
uses the online network to choose the action and the target network to evaluate
it, which reduces that optimistic bias. The builder fields and training loop are
otherwise the same.

## Terms

A *Q-network* maps one observation to one value per discrete action. Its output
width must equal the number of actions. An *online Q-network* is the network the
optimizer updates. A *target Q-network* has the same architecture and variable
names, but the agent only refreshes its parameters by copying the online network
at a fixed interval.

*Epsilon-greedy exploration* chooses a random valid action with probability
epsilon and otherwise chooses the online network's highest-valued action.
`epsilon_schedule` controls epsilon over the configured training horizon.

## Where to Go Next

Build the complete [DQN CartPole program](./dqn.md). It shows the two
Q-networks, replay configuration, and training call in one place. Then use the
small, documented change in [Double DQN](./ddqn.md) to change its target
calculation.

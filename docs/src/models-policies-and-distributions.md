# Models, Policies, and Distributions

A module returns a tensor. An environment expects an action. The agent decides
what the tensor means and how it becomes that action.

Two possible paths are:

```text
direct: model -> action representation -> action space -> environment
probabilistic:
    probabilistic policy -> action representation -> action space -> environment
```

The algorithm determines whether the action path uses a distribution. A model
can return an action representation that an action space understands directly.
An agent can also insert other selection logic. DQN and DDQN, for example,
choose from Q-values with epsilon-greedy selection.

On the probabilistic path, the policy uses a model and a distribution together.
The distribution is part of the policy rather than a standalone stage.

## Models Produce Tensors

A model transforms an input tensor into an output tensor. `MLP` supplies dense
layers and activation functions, but it does not assign meaning to its output.

Let `B` be the batch size and `I` the number of input features. A dense `MLP`
receives `[B, I]`. The features may be observations, the output of an
earlier module, or other values prepared by an agent. Other architectures may
use different input shapes.

The component that receives a model's output determines its shape and meaning.
A critic, for example, returns `[B, 1]` state-value estimates. A Q-network
returns action values. A probabilistic actor returns parameters for a
distribution.

## Probabilistic Policies Use a Distribution

`ProbabilisticPolicyModel<D>` owns the wrapped module and implements
`ProbabilisticPolicy`. Its `sample` and `mode` operations pass the module output
to `D::from_outputs`, then call the corresponding distribution operation. Its
`log_prob_and_entropy` operation calls `D::dist_eval` when an algorithm needs
those values.

`Distribution` is a public trait. ModuRL currently supplies
`CategoricalDistribution` and `GaussianDistribution`, but applications can add
their own implementations:

```rust,ignore
let policy =
    ProbabilisticPolicyModel::<MyDistribution>::new(Box::new(actor));
```

A custom implementation provides `from_outputs`, `sample`, `mode`, `dist_eval`,
and an associated `Error` type. It must document its model-output layout and
returned tensor shapes. Its action representation must also match the chosen
`Space`. The Rust type system does not check these tensor shapes.

## Spaces Produce Environment Actions

`Space::tensor_from_neurons` converts an action representation into the tensor
passed to the environment.

`Discrete` selects the index of the largest component. `BoxSpace` clamps each
component to its lower and upper bounds. PPO retains the original sample for
log-probability calculations while sending the converted action to the
environment.

## Built-In Distributions

### Categorical Distribution

Let `C` be the number of discrete choices. `CategoricalDistribution` expects
one logit for each choice:

| Value | Shape |
| --- | --- |
| Model output | `[B, C]` |
| Sampled representation | `[B, C]` |
| Action after `Discrete` conversion | `[B]` |
| Log probability | `[B]` |
| Entropy | `[B]` |

The logits are unnormalized scores. Sampling adds an independent random
perturbation called *Gumbel noise* to each score. Taking the largest perturbed
score samples choices according to the logits. `Discrete` then selects that
score's index.

CartPole has two actions, so the getting-started actor returns two logits per
observation.

### Gaussian Distribution

Let `A` be the number of components in a one-dimensional continuous action
space with shape `[A]`. `GaussianDistribution` expects two values for each
component. Along dimension 1, all `A` means come first, followed by all `A` log
standard deviations:

```text
[mean_0, ..., mean_(A-1), log_std_0, ..., log_std_(A-1)]
```

| Value | Shape |
| --- | --- |
| Model output | `[B, 2 * A]` |
| Means | `[B, A]` |
| Log standard deviations | `[B, A]` |
| Sampled representation | `[B, A]` |
| Action after `BoxSpace` conversion | `[B, A]` |
| Log probability | `[B]` |
| Entropy | `[B]` |

`GaussianDistribution` applies `exp` to the log standard deviations before
sampling. Neither half of the model output contains log probabilities. Each
action component uses an independent Gaussian, and `dist_eval` sums the
component log probabilities and entropies into one value per batch row.

The actor can return the complete `[B, 2 * A]` tensor itself. It can also combine
state-dependent means with a separate trainable `log_std`, as the MuJoCo PPO
example does.

Read [Getting Started](./getting-started.md) for a categorical policy,
[Value-Based Training](./q-learning.md) for action selection without a
distribution, and `crates/examples/examples/ppo_mujoco_with_graphs.rs` for a
Gaussian policy.

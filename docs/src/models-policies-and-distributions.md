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
DDPG and TD3 use this direct path for their deterministic actors, then add
Gaussian noise inside the agent during collection. An agent can also insert
other selection logic. DQN and DDQN, for example, choose from Q-values with
epsilon-greedy selection.

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

`ProbabilisticPolicyModel<T>` owns the native model and distribution directly.
`T: PolicyTypes` groups their types and the native tensor signature. The trait contains associated types and no methods.
The tuple implementation groups the types as:

```text
(Model, Distribution, (ObservationTensor, ParameterTensor, ActionTensor))
```

The tensor types contain their ranks and kinds. The group describes types; it does not store tensors or provide tensor operations.
Policy implementations use native `Tensor<O, K>`, `Tensor<P>`, and `Tensor<A>` signatures.

Policy models implement Burn's `Module` and ModuRL's `Forward<O, P, K>`.
The observation kind `K` defaults to `Float`. Custom models can use integer or boolean observations and perform explicit conversions.
Construct a policy without specifying the type group:

```rust,ignore
let policy = ProbabilisticPolicyModel::with_distribution(
    actor,
    CategoricalDistribution,
);
```

`Forward::forward` takes `Tensor<O, K>` and returns `Result<Tensor<P>, Self::Error>`.
Both tensors start with a batch axis, and the model must preserve batch size.
The model performs any observation dtype or device conversion explicitly.
All execution uses the same owned model that Burn visits for optimizer updates, records, and device transfers.
`policy.module()` returns that native model directly.

A component that only owns the policy needs one type parameter:

```rust,ignore
struct PolicyOwner<T: PolicyTypes> {
    policy: ProbabilisticPolicyModel<T>,
}
```

Code that performs tensor calculations still constrains the required native tensor types or policy ranks.
The group does not remove the underlying type information. Fully explicit policy types contain the tuple shown above.

The policy implements `ProbabilisticPolicy<O, A>`. Its `sample` and `mode`
operations pass model outputs directly to the distribution. Its
`log_prob_and_entropy` operation calls `D::dist_eval` and returns two `[batch_size]` tensors.
The policy trusts callers, models, and distributions to follow their documented tensor contracts.
It does not check input compatibility or validate component outputs.

`Distribution` is a public trait. ModuRL currently supplies
`CategoricalDistribution` and `GaussianDistribution`, but applications can add
their own implementations:

```rust,ignore
let policy = ProbabilisticPolicyModel::<(_, MyDistribution, _)>::new(actor);
```

A custom `Distribution<P, A>` implementation provides `sample`, `mode`,
`dist_eval`, and an associated `Error` type. It documents its parameter and action layouts.
Ranks are checked at compile time. Callers must supply the documented dimensions, dtypes, and devices.
Its action representation must match the chosen `ActionMap` input.

## Spaces Produce Environment Actions

`ActionMap::tensor_from_neurons` converts an action representation into the tensor
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
| Action after `Discrete` conversion | `[B, 1]` |
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
[Soft Actor-Critic](./sac.md) for policies that expose exact or sampled action
expectations, [Deterministic Actor-Critic
Training](./deterministic-actor-critic.md) for direct continuous actors,
[Value-Based Training](./q-learning.md) for action selection without a
distribution, and `crates/examples/examples/ppo_mujoco.rs` for a Gaussian
policy.

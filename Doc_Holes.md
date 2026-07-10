# Documentation Holes

## Missing first-use installation and minimal program

- Status: Open
- Area: README, getting started guide
- Task tested: How do I install/add ModuRL to a Rust project and compile a minimal first program?
- Expected capability: Users should be able to add the crate and any required companion crates or features, import the library, and compile a minimal example before moving to full PPO examples.
- What the docs currently say: `README.md` describes ModuRL at a high level and lists project features. The examples show full ModuRL programs but do not show dependency setup.
- Gap: The docs do not include a `Cargo.toml` dependency snippet, feature flags, companion crate requirements, local path or published-crate instructions, or a minimal compiling `main.rs`.
- Evidence: The docs-only audit returned `DOC HOLE`, citing `README.md` and `examples/ppo_bench.rs`.
- Suggested doc fix: Add a Getting Started section to `README.md` with dependency setup, supported feature flags, required environment/backend notes, and a minimal compileable program.

## Partial PPO logging schema documentation

- Status: Open
- Area: Examples, API reference
- Task tested: How do I attach logging to PPO and interpret what fields are available in each log entry?
- Expected capability: Users can implement `PPOLogger`, pass it to `PPOActor`, and understand every `PPOLogEntry` field, tensor shape, units, and logging cadence.
- What the docs currently say: `examples/ppo_cartpole_with_graphs.rs` demonstrates a `PPOLogger` implementation, passes `.logging_info(&mut logger)`, and reads fields such as `actor_loss`, `critic_loss`, `entropy`, `kl_divergence`, `explained_variance`, `rewards`, and `timestep`.
- Gap: The docs show usage by example but do not document the full log-entry schema, tensor shapes, field meanings, units, or when log events are emitted.
- Evidence: The docs-only audit returned `PARTIAL DOC HOLE`, citing `examples/ppo_cartpole_with_graphs.rs`.
- Suggested doc fix: Add rustdoc or an API-reference section for `PPOLogger` and `PPOLogEntry`, and link it from the graphing example.

## Missing DQN and DDQN training guide

- Status: Open
- Area: Guide, examples
- Task tested: How do I train DQN or DDQN on CartPole or another discrete environment?
- Expected capability: The library exposes `DQNActor`, `DDQNActor`, target and online network configuration, replay-buffer settings, epsilon behavior, device strategy, and `learn` through the public actor API.
- What the docs currently say: The allowed docs demonstrate PPO workflows only. They do not mention DQN or DDQN.
- Gap: Users have no documented path for building online and target Q-networks, creating optimizers and var maps, selecting a device strategy, configuring replay and epsilon parameters, or calling `learn`.
- Evidence: The docs-only audit returned `DOC HOLE`, citing `README.md`, `examples/ppo_bench.rs`, `examples/ppo_cartpole_with_graphs.rs`, and `examples/rendered_lunar_lander_ppo.rs`.
- Suggested doc fix: Add a `dqn_cartpole.rs` or guide section that trains DQN and DDQN on a discrete environment and explains target network, replay, epsilon, and device settings.

## Partial custom environment and vectorization contract

- Status: Open
- Area: Guide, API reference, examples
- Task tested: How do I implement my own environment for ModuRL and wrap multiple environments for training?
- Expected capability: Users can implement the `Gym` trait, return valid observation/action spaces, produce `StepInfo`, and wrap a `Vec` of environments in `VectorizedGymWrapper` for training.
- What the docs currently say: `examples/rendered_lunar_lander_ppo.rs` shows a wrapper around an existing environment implementing `Gym`, then creates a `VectorizedGymWrapper` from a vector of environments.
- Gap: The docs do not explain how to implement a brand-new environment, expected tensor shapes, `StepInfo` field semantics, action-space conversion, error types, done versus truncated behavior, terminal-state handling, or reset behavior in vectorized training.
- Evidence: The docs-only audit returned `PARTIAL DOC HOLE`, citing `examples/rendered_lunar_lander_ppo.rs`.
- Suggested doc fix: Add a custom-environment guide with a minimal `Gym` implementation and a section describing `StepInfo`, `VectorizedStepInfo`, reset semantics, and batching contracts.

## Missing shared-network PPO documentation

- Status: Open
- Area: PPO guide, examples
- Task tested: How do I use shared-network PPO instead of separate actor/critic networks?
- Expected capability: Users can configure `PPONetworkInfo::Shared` with a shared trunk, actor head, critic head, and a single optimizer.
- What the docs currently say: The examples demonstrate only `PPONetworkInfo::Separate` with `SeparatePPONetwork::builder()`.
- Gap: The docs do not show the shared PPO builder path, required module shapes, optimizer var-map ownership, actor and critic head setup, or how the shared trunk output feeds both heads.
- Evidence: The docs-only audit returned `DOC HOLE`, citing `examples/ppo_bench.rs`, `examples/ppo_cartpole_with_graphs.rs`, and `examples/rendered_lunar_lander_ppo.rs`.
- Suggested doc fix: Add a shared PPO example or a subsection in the PPO guide showing a shared trunk, separate heads, `SharedPPONetwork::builder()`, and `PPONetworkInfo::Shared`.

## Missing continuous-action Gaussian policy documentation

- Status: Open
- Area: Distributions, PPO guide, examples
- Task tested: How do I use continuous actions with Gaussian distributions?
- Expected capability: Users can pair continuous action spaces with `GuassianDistribution`, size policy outputs for mean and log standard deviation, and train PPO on a continuous-control task.
- What the docs currently say: The examples use `CategoricalDistribution` with discrete action spaces.
- Gap: The docs do not show Gaussian policy setup, required output dimensions, mean/log-std splitting, action tensor shapes, action scaling or clipping expectations, or a continuous-control example.
- Evidence: The docs-only audit returned `DOC HOLE`, citing `examples/ppo_bench.rs`, `examples/ppo_cartpole_with_graphs.rs`, and `examples/rendered_lunar_lander_ppo.rs`.
- Suggested doc fix: Add a continuous-control PPO example and rustdoc for the Gaussian distribution output contract.

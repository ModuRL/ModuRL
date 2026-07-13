# ModuRL Guide

This guide is for Rust developers who want to build reinforcement learning
programs with ModuRL and Candle. It starts with PPO on CartPole, then explains
the library types behind that example and the value-based DQN and DDQN paths.

The guide assumes basic Rust and Cargo knowledge. It does not teach
reinforcement learning or neural networks from first principles.

ModuRL is early, and API stability is not guaranteed. Start with [Getting
Started](./getting-started.md) to run and assemble a PPO CartPole program. For
discrete-action value-based training, read [Value-Based Training](./q-learning.md).

The README provides project status and the shortest repository-based commands.
Rustdoc provides the precise contracts for public traits, structs, and builders.

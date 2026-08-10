#!/usr/bin/env python3
"""Matched end-to-end PPO, DQN, and SAC benchmarks for Python RL frameworks.

The CleanRL runners adapt the corresponding public scripts at revision
``e421c2e50b81febf639fced51a69e2602593d50d``. Logging, evaluation, and
command-line plumbing are removed. Algorithm-specific fairness changes are
documented in the benchmark README.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical


ENV_COUNT = 8
STEPS_PER_ENV = 256
ROLLOUT_SIZE = ENV_COUNT * STEPS_PER_ENV
MINI_BATCH_SIZE = 64
PPO_EPOCHS = 10
DEFAULT_MEASURED_STEPS = 20_480
DEFAULT_WARMUP_STEPS = ROLLOUT_SIZE
SEED = 42
TRAINABLE_PARAMETERS = 9_155
DQN_PARAMETERS = 4_610
SAC_PARAMETERS = 13_637


@dataclass(frozen=True)
class BenchmarkConfig:
    environment: str = "CartPole-v1"
    environments: int = ENV_COUNT
    steps_per_environment: int = STEPS_PER_ENV
    rollout_size: int = ROLLOUT_SIZE
    hidden_layers: tuple[int, int] = (64, 64)
    trainable_parameters: int = TRAINABLE_PARAMETERS
    activation: str = "tanh"
    dtype: str = "float32"
    mini_batch_size: int = MINI_BATCH_SIZE
    ppo_epochs: int = PPO_EPOCHS
    learning_rate: float = 3e-4
    adam_epsilon: float = 1e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_loss_clipping: bool = False
    advantage_normalization: str = "per-mini-batch"
    entropy_coefficient: float = 0.0
    value_coefficient: float = 0.5
    max_gradient_norm: float = 0.5
    seed: int = SEED


CONFIG = BenchmarkConfig()


@dataclass(frozen=True)
class DQNBenchmarkConfig:
    environment: str = "CartPole-v1"
    environments: int = 1
    hidden_layers: tuple[int, int] = (64, 64)
    trainable_parameters: int = DQN_PARAMETERS
    activation: str = "tanh"
    dtype: str = "float32"
    replay_capacity: int = 10_000
    batch_size: int = 64
    training_start: int = 1_024
    updates_per_transition: float = 0.25
    update_frequency: int = 4
    target_update_interval: int = 1_000
    learning_rate: float = 2.5e-4
    adam_epsilon: float = 1e-5
    gamma: float = 0.99
    epsilon: float = 0.1
    target: str = "vanilla"
    loss: str = "mse"
    gradient_clipping: bool = False
    n_step_return: int = 1
    seed: int = SEED


@dataclass(frozen=True)
class SACBenchmarkConfig:
    environment: str = "Pendulum-v1"
    time_limit: int = 200
    environments: int = 1
    actor_hidden_layers: tuple[int, int] = (64, 64)
    critic_hidden_layers: tuple[int, int] = (64, 64)
    trainable_parameters: int = SAC_PARAMETERS
    activation: str = "tanh"
    dtype: str = "float32"
    state_dependent_log_std: bool = True
    log_std_bounds: tuple[float, float] = (-20.0, 2.0)
    replay_capacity: int = 10_000
    batch_size: int = 64
    training_start: int = 256
    updates_per_transition: int = 1
    learning_rate: float = 3e-4
    adam_epsilon: float = 1e-5
    gamma: float = 0.99
    tau: float = 0.005
    critics: int = 2
    critic_aggregation: str = "min"
    automatic_entropy_tuning: bool = True
    initial_alpha: float = 1.0
    target_entropy: float = -1.0
    n_step_return: int = 1
    seed: int = SEED


DQN_CONFIG = DQNBenchmarkConfig()
SAC_CONFIG = SACBenchmarkConfig()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--framework", choices=("sb3", "cleanrl", "tianshou"), required=True
    )
    parser.add_argument("--algorithm", choices=("ppo", "dqn", "sac"), default="ppo")
    parser.add_argument("--device", choices=("cpu", "cuda", "metal"), default="cpu")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    default_steps, default_warmup = {
        "ppo": (DEFAULT_MEASURED_STEPS, DEFAULT_WARMUP_STEPS),
        "dqn": (4_096, 1_024),
        "sac": (1_024, 256),
    }[args.algorithm]
    args.steps = args.steps or default_steps
    args.warmup_steps = args.warmup_steps or default_warmup
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")
    for name in ("steps", "warmup_steps"):
        value = getattr(args, name)
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonzero")
        if args.algorithm == "ppo" and value % ROLLOUT_SIZE:
            raise ValueError(f"PPO --{name.replace('_', '-')} must be a multiple of {ROLLOUT_SIZE}")
    expected_warmup = {"ppo": 2_048, "dqn": 1_024, "sac": 256}[args.algorithm]
    if args.warmup_steps != expected_warmup:
        raise ValueError(
            f"{args.algorithm.upper()} requires --warmup-steps {expected_warmup} "
            "so every runner has the same update schedule"
        )


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this PyTorch build or on this machine")
        return torch.device("cuda:0")
    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        raise RuntimeError("Metal (PyTorch MPS) is unavailable on this machine")
    return torch.device("mps")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def assert_parameter_count(module: nn.Module, expected: int = TRAINABLE_PARAMETERS) -> None:
    actual = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    if actual != expected:
        raise RuntimeError(
            f"network has {actual} trainable parameters, expected {expected}"
        )


def make_sb3_runner(
    device: torch.device,
) -> tuple[Callable[[int], None], Callable[[], None], str, str]:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    def make_env(index: int) -> Callable[[], gym.Env]:
        def thunk() -> gym.Env:
            env = gym.make(CONFIG.environment)
            env.reset(seed=SEED + index)
            return env

        return thunk

    envs = DummyVecEnv([make_env(index) for index in range(ENV_COUNT)])
    policy_kwargs = {
        "activation_fn": nn.Tanh,
        "net_arch": {"pi": [64, 64], "vf": [64, 64]},
        "ortho_init": True,
        "optimizer_kwargs": {"eps": CONFIG.adam_epsilon, "weight_decay": 0.0},
    }
    model = PPO(
        "MlpPolicy",
        envs,
        learning_rate=CONFIG.learning_rate,
        n_steps=STEPS_PER_ENV,
        batch_size=MINI_BATCH_SIZE,
        n_epochs=PPO_EPOCHS,
        gamma=CONFIG.gamma,
        gae_lambda=CONFIG.gae_lambda,
        clip_range=CONFIG.clip_range,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=CONFIG.entropy_coefficient,
        vf_coef=CONFIG.value_coefficient,
        max_grad_norm=CONFIG.max_gradient_norm,
        seed=SEED,
        device=device,
        policy_kwargs=policy_kwargs,
        verbose=0,
    )
    assert_parameter_count(model.policy)
    first_call = True

    def run(steps: int) -> None:
        nonlocal first_call
        model.learn(
            total_timesteps=steps,
            reset_num_timesteps=first_call,
            progress_bar=False,
            log_interval=None,
        )
        first_call = False

    return run, envs.close, package_version("stable-baselines3"), f"torch {torch.__version__}"


def layer_init(layer: nn.Linear, gain: float) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class CleanRLAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        hidden_gain = 2.0**0.5
        self.critic = nn.Sequential(
            layer_init(nn.Linear(4, 64), hidden_gain),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64), hidden_gain),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), 1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(4, 64), hidden_gain),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64), hidden_gain),
            nn.Tanh(),
            layer_init(nn.Linear(64, 2), 0.01),
        )

    def value(self, observation: torch.Tensor) -> torch.Tensor:
        return self.critic(observation)

    def action_and_value(
        self, observation: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = Categorical(logits=self.actor(observation))
        if action is None:
            action = distribution.sample()
        return (
            action,
            distribution.log_prob(action),
            distribution.entropy(),
            self.critic(observation),
        )


def make_cleanrl_runner(
    device: torch.device,
) -> tuple[Callable[[int], None], Callable[[], None], str, str]:
    def make_env(index: int) -> Callable[[], gym.Env]:
        def thunk() -> gym.Env:
            env = gym.make(CONFIG.environment)
            env.action_space.seed(SEED + index)
            return env

        return thunk

    envs = gym.vector.SyncVectorEnv(
        [make_env(index) for index in range(ENV_COUNT)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    agent = CleanRLAgent().to(device)
    assert_parameter_count(agent)
    optimizer = torch.optim.Adam(
        agent.parameters(),
        lr=CONFIG.learning_rate,
        eps=CONFIG.adam_epsilon,
        weight_decay=0.0,
    )
    observations = torch.zeros((STEPS_PER_ENV, ENV_COUNT, 4), device=device)
    actions = torch.zeros((STEPS_PER_ENV, ENV_COUNT), device=device)
    log_probs = torch.zeros((STEPS_PER_ENV, ENV_COUNT), device=device)
    rewards = torch.zeros((STEPS_PER_ENV, ENV_COUNT), device=device)
    dones = torch.zeros((STEPS_PER_ENV, ENV_COUNT), device=device)
    values = torch.zeros((STEPS_PER_ENV, ENV_COUNT), device=device)
    reset_observation, _ = envs.reset(seed=[SEED + index for index in range(ENV_COUNT)])
    next_observation = torch.as_tensor(reset_observation, dtype=torch.float32, device=device)
    next_done = torch.zeros(ENV_COUNT, device=device)

    def run(steps: int) -> None:
        nonlocal next_observation, next_done
        for _ in range(steps // ROLLOUT_SIZE):
            for step in range(STEPS_PER_ENV):
                observations[step] = next_observation
                dones[step] = next_done
                with torch.no_grad():
                    action, log_prob, _, value = agent.action_and_value(next_observation)
                    values[step] = value.flatten()
                actions[step] = action
                log_probs[step] = log_prob

                next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
                if np.any(truncated):
                    truncated_indices = np.flatnonzero(truncated)
                    terminal_observations = np.stack(
                        [info["final_obs"][index] for index in truncated_indices]
                    ).astype(np.float32)
                    with torch.no_grad():
                        terminal_values = (
                            agent.value(
                                torch.as_tensor(terminal_observations, device=device)
                            )
                            .flatten()
                            .cpu()
                            .numpy()
                        )
                    reward[truncated_indices] += CONFIG.gamma * terminal_values
                next_done_array = np.logical_or(terminated, truncated)
                rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
                next_observation = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
                next_done = torch.as_tensor(next_done_array, dtype=torch.float32, device=device)

            with torch.no_grad():
                next_value = agent.value(next_observation).reshape(1, -1)
                advantages = torch.zeros_like(rewards)
                last_gae = torch.zeros(ENV_COUNT, device=device)
                for step in reversed(range(STEPS_PER_ENV)):
                    if step == STEPS_PER_ENV - 1:
                        next_nonterminal = 1.0 - next_done
                        next_values = next_value
                    else:
                        next_nonterminal = 1.0 - dones[step + 1]
                        next_values = values[step + 1]
                    delta = (
                        rewards[step]
                        + CONFIG.gamma * next_values * next_nonterminal
                        - values[step]
                    )
                    last_gae = (
                        delta
                        + CONFIG.gamma * CONFIG.gae_lambda * next_nonterminal * last_gae
                    )
                    advantages[step] = last_gae
                returns = advantages + values

            flat_obs = observations.reshape((-1, 4))
            flat_actions = actions.reshape(-1)
            flat_log_probs = log_probs.reshape(-1)
            flat_advantages = advantages.reshape(-1)
            flat_returns = returns.reshape(-1)
            indices = np.arange(ROLLOUT_SIZE)

            for _ in range(PPO_EPOCHS):
                np.random.shuffle(indices)
                for start in range(0, ROLLOUT_SIZE, MINI_BATCH_SIZE):
                    batch_indices = indices[start : start + MINI_BATCH_SIZE]
                    _, new_log_prob, entropy, new_value = agent.action_and_value(
                        flat_obs[batch_indices], flat_actions[batch_indices].long()
                    )
                    log_ratio = new_log_prob - flat_log_probs[batch_indices]
                    ratio = log_ratio.exp()
                    batch_advantages = flat_advantages[batch_indices]
                    batch_advantages = (batch_advantages - batch_advantages.mean()) / (
                        batch_advantages.std(unbiased=False) + 1e-8
                    )
                    policy_loss = -torch.min(
                        batch_advantages * ratio,
                        batch_advantages
                        * torch.clamp(ratio, 1.0 - CONFIG.clip_range, 1.0 + CONFIG.clip_range),
                    ).mean()
                    # CleanRL normally includes a 0.5 factor here. Removing it makes
                    # vf_coef=0.5 mean the same thing in all four implementations.
                    value_loss = ((new_value.view(-1) - flat_returns[batch_indices]) ** 2).mean()
                    loss = (
                        policy_loss
                        - CONFIG.entropy_coefficient * entropy.mean()
                        + CONFIG.value_coefficient * value_loss
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), CONFIG.max_gradient_norm)
                    optimizer.step()

    return (
        run,
        envs.close,
        "cleanrl/ppo.py@e421c2e5 (adapted)",
        f"torch {torch.__version__}",
    )


def initialize_tianshou_linears(actor: nn.Module, critic: nn.Module) -> None:
    hidden_gain = 2.0**0.5
    for module in list(actor.modules()) + list(critic.modules()):
        if isinstance(module, nn.Linear):
            layer_init(module, hidden_gain)
    actor_linears = [module for module in actor.modules() if isinstance(module, nn.Linear)]
    critic_linears = [module for module in critic.modules() if isinstance(module, nn.Linear)]
    layer_init(actor_linears[-1], 0.01)
    layer_init(critic_linears[-1], 1.0)


def make_tianshou_runner(
    device: torch.device,
) -> tuple[Callable[[int], None], Callable[[], None], str, str]:
    from tianshou.algorithm import PPO
    from tianshou.algorithm.algorithm_base import policy_within_training_step
    from tianshou.algorithm.modelfree.reinforce import DiscreteActorPolicy
    from tianshou.algorithm.optim import AdamOptimizerFactory
    from tianshou.data import Collector, CollectStats, VectorReplayBuffer
    from tianshou.env import DummyVectorEnv
    from tianshou.utils.net.common import Net
    from tianshou.utils.net.discrete import DiscreteActor, DiscreteCritic

    probe_env = gym.make(CONFIG.environment)
    observation_shape = probe_env.observation_space.shape
    action_count = probe_env.action_space.n
    action_space = probe_env.action_space
    probe_env.close()
    envs = DummyVectorEnv(
        [lambda: gym.make(CONFIG.environment) for _ in range(ENV_COUNT)]
    )
    envs.seed(SEED)
    actor_net = Net(
        state_shape=observation_shape, hidden_sizes=[64, 64], activation=nn.Tanh
    )
    critic_net = Net(
        state_shape=observation_shape, hidden_sizes=[64, 64], activation=nn.Tanh
    )
    actor = DiscreteActor(preprocess_net=actor_net, action_shape=action_count).to(device)
    critic = DiscreteCritic(preprocess_net=critic_net).to(device)
    initialize_tianshou_linears(actor, critic)
    actual_parameters = sum(
        parameter.numel()
        for module in (actor, critic)
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    if actual_parameters != TRAINABLE_PARAMETERS:
        raise RuntimeError(
            f"Tianshou networks have {actual_parameters} trainable parameters, "
            f"expected {TRAINABLE_PARAMETERS}"
        )
    policy = DiscreteActorPolicy(
        actor=actor,
        dist_fn=Categorical,
        action_space=action_space,
        deterministic_eval=False,
    )
    algorithm = PPO(
        policy=policy,
        critic=critic,
        optim=AdamOptimizerFactory(
            lr=CONFIG.learning_rate, eps=CONFIG.adam_epsilon, weight_decay=0.0
        ),
        gamma=CONFIG.gamma,
        gae_lambda=CONFIG.gae_lambda,
        max_grad_norm=CONFIG.max_gradient_norm,
        vf_coef=CONFIG.value_coefficient,
        ent_coef=CONFIG.entropy_coefficient,
        return_scaling=False,
        eps_clip=CONFIG.clip_range,
        value_clip=False,
        dual_clip=None,
        advantage_normalization=True,
        recompute_advantage=False,
    )
    buffer = VectorReplayBuffer(ROLLOUT_SIZE, ENV_COUNT)
    collector = Collector[CollectStats](algorithm, envs, buffer)
    collector.reset()

    def run(steps: int) -> None:
        for _ in range(steps // ROLLOUT_SIZE):
            stats = collector.collect(n_step=ROLLOUT_SIZE)
            if stats.n_collected_steps != ROLLOUT_SIZE:
                raise RuntimeError(
                    f"Tianshou collected {stats.n_collected_steps}, expected {ROLLOUT_SIZE}"
                )
            with policy_within_training_step(algorithm.policy):
                algorithm.update(
                    buffer=collector.buffer,
                    batch_size=MINI_BATCH_SIZE,
                    repeat=PPO_EPOCHS,
                )
            collector.reset_buffer(keep_statistics=True)

    def close() -> None:
        collector.close()

    return run, close, package_version("tianshou"), f"torch {torch.__version__}"


class ReplayBuffer:
    """Small fixed-shape replay buffer shared by the adapted reference loops."""

    def __init__(self, observation_size: int, action_size: int, discrete: bool) -> None:
        capacity = 10_000
        self.observations = np.empty((capacity, observation_size), dtype=np.float32)
        self.next_observations = np.empty((capacity, observation_size), dtype=np.float32)
        self.actions = np.empty(
            (capacity, action_size), dtype=np.int64 if discrete else np.float32
        )
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:
        self.observations[self.position] = observation
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_observations[self.position] = next_observation
        self.dones[self.position] = terminated
        self.position = (self.position + 1) % len(self.observations)
        self.size = min(self.size + 1, len(self.observations))

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return tuple(
            torch.as_tensor(array[indices], device=device)
            for array in (
                self.observations,
                self.actions,
                self.rewards,
                self.next_observations,
                self.dones,
            )
        )


class DQNNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


def make_sb3_dqn_runner(
    device: torch.device, warmup_steps: int
) -> tuple[Callable[[int], None], Callable[[], None], str, str]:
    import torch.nn.functional as functional
    from stable_baselines3 import DQN

    class MatchedDQN(DQN):
        """SB3 DQN with the benchmark's MSE loss and no gradient clipping."""

        def train(self, gradient_steps: int, batch_size: int = 100) -> None:
            self.policy.set_training_mode(True)
            self._update_learning_rate(self.policy.optimizer)
            for _ in range(gradient_steps):
                replay_data = self.replay_buffer.sample(
                    batch_size, env=self._vec_normalize_env
                )
                with torch.no_grad():
                    next_q = self.q_net_target(replay_data.next_observations)
                    next_q = next_q.max(dim=1).values.reshape(-1, 1)
                    target_q = replay_data.rewards + (
                        1 - replay_data.dones
                    ) * self.gamma * next_q
                current_q = self.q_net(replay_data.observations)
                current_q = torch.gather(current_q, 1, replay_data.actions.long())
                loss = functional.mse_loss(current_q, target_q)
                self.policy.optimizer.zero_grad()
                loss.backward()
                self.policy.optimizer.step()
            self._n_updates += gradient_steps

    env = gym.make(DQN_CONFIG.environment)
    model = MatchedDQN(
        "MlpPolicy",
        env,
        learning_rate=DQN_CONFIG.learning_rate,
        buffer_size=DQN_CONFIG.replay_capacity,
        learning_starts=warmup_steps,
        batch_size=DQN_CONFIG.batch_size,
        tau=1.0,
        gamma=DQN_CONFIG.gamma,
        train_freq=(DQN_CONFIG.update_frequency, "step"),
        gradient_steps=1,
        target_update_interval=DQN_CONFIG.target_update_interval,
        exploration_fraction=0.0,
        exploration_initial_eps=DQN_CONFIG.epsilon,
        exploration_final_eps=DQN_CONFIG.epsilon,
        policy_kwargs={
            "net_arch": [64, 64],
            "activation_fn": nn.Tanh,
            "optimizer_kwargs": {"eps": DQN_CONFIG.adam_epsilon, "weight_decay": 0.0},
        },
        seed=SEED,
        device=device,
        verbose=0,
    )
    assert_parameter_count(model.q_net, DQN_PARAMETERS)
    first_call = True

    def run(steps: int) -> None:
        nonlocal first_call
        model.learn(total_timesteps=steps, reset_num_timesteps=first_call, log_interval=None)
        first_call = False

    return run, env.close, f"{package_version('stable-baselines3')} (adapted)", f"torch {torch.__version__}"


def make_cleanrl_dqn_runner(
    device: torch.device, warmup_steps: int
) -> tuple[Callable[[int], None], Callable[[], None], str, str]:
    env = gym.make(DQN_CONFIG.environment)
    observation, _ = env.reset(seed=SEED)
    env.action_space.seed(SEED)
    online = DQNNetwork().to(device)
    target = DQNNetwork().to(device)
    target.load_state_dict(online.state_dict())
    assert_parameter_count(online, DQN_PARAMETERS)
    optimizer = torch.optim.Adam(
        online.parameters(), lr=DQN_CONFIG.learning_rate, eps=DQN_CONFIG.adam_epsilon
    )
    replay = ReplayBuffer(4, 1, discrete=True)
    total_steps = 0

    def run(steps: int) -> None:
        nonlocal observation, total_steps
        for _ in range(steps):
            total_steps += 1
            if random.random() < DQN_CONFIG.epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = online(torch.as_tensor(observation, device=device).unsqueeze(0))
                    action = int(q_values.argmax(dim=1).item())
            next_observation, reward, terminated, truncated, _ = env.step(action)
            replay.add(
                observation,
                np.asarray([action]),
                float(reward),
                next_observation,
                bool(terminated),
            )
            observation = next_observation
            if terminated or truncated:
                observation, _ = env.reset()
            if total_steps > warmup_steps and total_steps % DQN_CONFIG.update_frequency == 0:
                obs, actions, rewards, next_obs, dones = replay.sample(
                    DQN_CONFIG.batch_size, device
                )
                with torch.no_grad():
                    targets = rewards + (1 - dones) * DQN_CONFIG.gamma * target(next_obs).max(
                        dim=1, keepdim=True
                    ).values
                predictions = online(obs).gather(1, actions.long())
                loss = nn.functional.mse_loss(predictions, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if total_steps % DQN_CONFIG.target_update_interval == 0:
                target.load_state_dict(online.state_dict())

    return run, env.close, "cleanrl/dqn.py@e421c2e5 (adapted)", f"torch {torch.__version__}"


def make_tianshou_dqn_runner(
    device: torch.device, warmup_steps: int
) -> tuple[Callable[[int], None], Callable[[], None], str, str]:
    from tianshou.algorithm import DQN
    from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
    from tianshou.algorithm.optim import AdamOptimizerFactory
    from tianshou.data import Collector, CollectStats, ReplayBuffer as TianshouReplayBuffer
    from tianshou.env import DummyVectorEnv
    from tianshou.utils.net.common import Net
    from tianshou.utils.torch_utils import policy_within_training_step

    class AbsoluteStepDQN(DQN):
        """Drive target copies from absolute environment steps, not update count."""

        def _periodically_update_lagged_network_weights(self) -> None:
            # The benchmark performs the copy after the optimizer update at the
            # matching absolute environment transition.
            self._iter += 1

        def update_target(self) -> None:
            self._update_lagged_network_weights()

    probe_env = gym.make(DQN_CONFIG.environment)
    observation_space = probe_env.observation_space
    action_space = probe_env.action_space
    probe_env.close()
    env = DummyVectorEnv([lambda: gym.make(DQN_CONFIG.environment)])
    env.seed(SEED)
    net = Net(
        state_shape=observation_space.shape,
        action_shape=action_space.n,
        hidden_sizes=[64, 64],
        activation=nn.Tanh,
    ).to(device)
    assert_parameter_count(net, DQN_PARAMETERS)
    policy = DiscreteQLearningPolicy(
        model=net,
        action_space=action_space,
        observation_space=observation_space,
        eps_training=DQN_CONFIG.epsilon,
        eps_inference=DQN_CONFIG.epsilon,
    )
    algorithm = AbsoluteStepDQN(
        policy=policy,
        optim=AdamOptimizerFactory(
            lr=DQN_CONFIG.learning_rate, eps=DQN_CONFIG.adam_epsilon, weight_decay=0.0
        ),
        gamma=DQN_CONFIG.gamma,
        n_step_return_horizon=1,
        # A positive value constructs Tianshou's lagged network. The subclass
        # replaces its optimizer-step cadence with the matched absolute cadence.
        target_update_freq=1,
        is_double=False,
        huber_loss_delta=None,
    )
    collector = Collector[CollectStats](
        algorithm, env, TianshouReplayBuffer(DQN_CONFIG.replay_capacity), exploration_noise=True
    )
    collector.reset()
    total_steps = 0
    target_copies = 0

    def run(steps: int) -> None:
        nonlocal target_copies, total_steps
        if total_steps < warmup_steps:
            stats = collector.collect(n_step=steps)
            total_steps += stats.n_collected_steps
            return
        if steps % DQN_CONFIG.update_frequency:
            raise ValueError("Tianshou DQN measured steps must be divisible by update frequency")
        for _ in range(steps // DQN_CONFIG.update_frequency):
            stats = collector.collect(n_step=DQN_CONFIG.update_frequency)
            if stats.n_collected_steps != DQN_CONFIG.update_frequency:
                raise RuntimeError("Tianshou DQN collected the wrong transition count")
            total_steps += stats.n_collected_steps
            with policy_within_training_step(algorithm.policy):
                algorithm.update(buffer=collector.buffer, sample_size=DQN_CONFIG.batch_size)
            if total_steps % DQN_CONFIG.target_update_interval == 0:
                algorithm.update_target()
                target_copies += 1

    def close() -> None:
        expected_copies = (
            total_steps // DQN_CONFIG.target_update_interval
            - warmup_steps // DQN_CONFIG.target_update_interval
        )
        if target_copies != expected_copies:
            raise RuntimeError(
                f"Tianshou made {target_copies} target copies, expected {expected_copies}"
            )
        collector.close()

    return (
        run,
        close,
        f"{package_version('tianshou')} (adapted target cadence)",
        f"torch {torch.__version__}",
    )


class SACActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh()
        )
        self.mean = nn.Linear(64, 1)
        self.log_std = nn.Linear(64, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(observations)
        return self.mean(hidden), self.log_std(hidden).clamp(-20.0, 2.0)

    def sample(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from torch.distributions.normal import Normal

        mean, log_std = self(observations)
        distribution = Normal(mean, log_std.exp())
        latent = distribution.rsample()
        squashed = torch.tanh(latent)
        log_probability = distribution.log_prob(latent) - torch.log(
            1 - squashed.pow(2) + torch.finfo(torch.float32).eps
        )
        return squashed * 2.0, log_probability.sum(dim=1, keepdim=True)


class SACCriticNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([observations, actions], dim=1))


def make_sb3_sac_runner(
    device: torch.device, warmup_steps: int
) -> tuple[Callable[[int], None], Callable[[], None], str, str]:
    from stable_baselines3 import SAC

    env = gym.make(SAC_CONFIG.environment)
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=SAC_CONFIG.learning_rate,
        buffer_size=SAC_CONFIG.replay_capacity,
        learning_starts=warmup_steps,
        batch_size=SAC_CONFIG.batch_size,
        tau=SAC_CONFIG.tau,
        gamma=SAC_CONFIG.gamma,
        train_freq=(1, "step"),
        gradient_steps=1,
        ent_coef="auto_1.0",
        target_entropy=-1.0,
        target_update_interval=1,
        policy_kwargs={
            "net_arch": [64, 64],
            "activation_fn": nn.Tanh,
            "optimizer_kwargs": {"eps": SAC_CONFIG.adam_epsilon, "weight_decay": 0.0},
        },
        seed=SEED,
        device=device,
        verbose=0,
    )
    actual = sum(p.numel() for p in model.actor.parameters()) + sum(
        p.numel() for p in model.critic.parameters()
    ) + 1
    if actual != SAC_PARAMETERS:
        raise RuntimeError(f"SB3 SAC has {actual} trainable parameters, expected {SAC_PARAMETERS}")
    first_call = True

    def run(steps: int) -> None:
        nonlocal first_call
        model.learn(total_timesteps=steps, reset_num_timesteps=first_call, log_interval=None)
        first_call = False

    return run, env.close, package_version("stable-baselines3"), f"torch {torch.__version__}"


def make_cleanrl_sac_runner(
    device: torch.device, warmup_steps: int
) -> tuple[Callable[[int], None], Callable[[], None], str, str]:
    env = gym.make(SAC_CONFIG.environment)
    observation, _ = env.reset(seed=SEED)
    env.action_space.seed(SEED)
    actor = SACActor().to(device)
    critic_1 = SACCriticNetwork().to(device)
    critic_2 = SACCriticNetwork().to(device)
    target_1 = SACCriticNetwork().to(device)
    target_2 = SACCriticNetwork().to(device)
    target_1.load_state_dict(critic_1.state_dict())
    target_2.load_state_dict(critic_2.state_dict())
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    actual = sum(p.numel() for module in (actor, critic_1, critic_2) for p in module.parameters()) + 1
    if actual != SAC_PARAMETERS:
        raise RuntimeError(f"CleanRL SAC has {actual} trainable parameters, expected {SAC_PARAMETERS}")
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4, eps=1e-5)
    critic_optimizer = torch.optim.Adam(
        list(critic_1.parameters()) + list(critic_2.parameters()), lr=3e-4, eps=1e-5
    )
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=3e-4, eps=1e-5)
    replay = ReplayBuffer(3, 1, discrete=False)
    total_steps = 0

    def run(steps: int) -> None:
        nonlocal observation, total_steps
        for _ in range(steps):
            total_steps += 1
            if total_steps <= warmup_steps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = actor.sample(
                        torch.as_tensor(observation, device=device).unsqueeze(0)
                    )[0].cpu().numpy()[0]
            next_observation, reward, terminated, truncated, _ = env.step(action)
            replay.add(observation, action, float(reward), next_observation, bool(terminated))
            observation = next_observation
            if terminated or truncated:
                observation, _ = env.reset()
            if total_steps > warmup_steps:
                obs, actions, rewards, next_obs, dones = replay.sample(SAC_CONFIG.batch_size, device)
                with torch.no_grad():
                    next_actions, next_log_prob = actor.sample(next_obs)
                    next_q = torch.min(
                        target_1(next_obs, next_actions), target_2(next_obs, next_actions)
                    ) - log_alpha.exp() * next_log_prob
                    target_q = rewards + (1 - dones) * SAC_CONFIG.gamma * next_q
                q1 = critic_1(obs, actions)
                q2 = critic_2(obs, actions)
                critic_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                policy_actions, log_prob = actor.sample(obs)
                actor_loss = (log_alpha.exp().detach() * log_prob - torch.min(
                    critic_1(obs, policy_actions), critic_2(obs, policy_actions)
                )).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                alpha_loss = -(log_alpha * (log_prob.detach() + SAC_CONFIG.target_entropy)).mean()
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()
                with torch.no_grad():
                    for target, source in ((target_1, critic_1), (target_2, critic_2)):
                        for target_parameter, parameter in zip(target.parameters(), source.parameters()):
                            target_parameter.mul_(1 - SAC_CONFIG.tau).add_(
                                parameter, alpha=SAC_CONFIG.tau
                            )

    return run, env.close, "cleanrl/sac_continuous_action.py@e421c2e5 (adapted)", f"torch {torch.__version__}"


def make_tianshou_sac_runner(
    device: torch.device, warmup_steps: int
) -> tuple[Callable[[int], None], Callable[[], None], str, str]:
    from tianshou.algorithm import SAC
    from tianshou.algorithm.modelfree.sac import AutoAlpha, SACPolicy
    from tianshou.algorithm.optim import AdamOptimizerFactory
    from tianshou.data import Collector, CollectStats, ReplayBuffer as TianshouReplayBuffer
    from tianshou.env import DummyVectorEnv
    from tianshou.utils.net.common import Net
    from tianshou.utils.net.continuous import ContinuousActorProbabilistic, ContinuousCritic
    from tianshou.utils.torch_utils import policy_within_training_step

    probe_env = gym.make(SAC_CONFIG.environment)
    observation_space = probe_env.observation_space
    action_space = probe_env.action_space
    state_shape = observation_space.shape
    action_shape = action_space.shape
    probe_env.close()
    env = DummyVectorEnv([lambda: gym.make(SAC_CONFIG.environment)])
    env.seed(SEED)
    actor_net = Net(state_shape=state_shape, hidden_sizes=[64, 64], activation=nn.Tanh)
    actor = ContinuousActorProbabilistic(
        preprocess_net=actor_net,
        action_shape=action_shape,
        unbounded=True,
        conditioned_sigma=True,
    ).to(device)
    critic_net_1 = Net(
        state_shape=state_shape, action_shape=action_shape, hidden_sizes=[64, 64],
        activation=nn.Tanh, concat=True
    )
    critic_net_2 = Net(
        state_shape=state_shape, action_shape=action_shape, hidden_sizes=[64, 64],
        activation=nn.Tanh, concat=True
    )
    critic_1 = ContinuousCritic(preprocess_net=critic_net_1).to(device)
    critic_2 = ContinuousCritic(preprocess_net=critic_net_2).to(device)
    actual = sum(p.numel() for module in (actor, critic_1, critic_2) for p in module.parameters()) + 1
    if actual != SAC_PARAMETERS:
        raise RuntimeError(f"Tianshou SAC has {actual} trainable parameters, expected {SAC_PARAMETERS}")
    policy = SACPolicy(
        actor=actor, action_space=action_space, observation_space=observation_space,
        deterministic_eval=False, action_scaling=True
    )
    optimizer = AdamOptimizerFactory(lr=3e-4, eps=1e-5, weight_decay=0.0)
    algorithm = SAC(
        policy=policy, policy_optim=optimizer, critic=critic_1, critic_optim=optimizer,
        critic2=critic_2, critic2_optim=optimizer, tau=0.005, gamma=0.99,
        alpha=AutoAlpha(target_entropy=-1.0, log_alpha=0.0, optim=optimizer),
        n_step_return_horizon=1, deterministic_eval=False
    )
    collector = Collector[CollectStats](
        algorithm, env, TianshouReplayBuffer(SAC_CONFIG.replay_capacity), exploration_noise=True
    )
    collector.reset()
    total_steps = 0

    def run(steps: int) -> None:
        nonlocal total_steps
        if total_steps < warmup_steps:
            stats = collector.collect(n_step=steps, random=True)
            total_steps += stats.n_collected_steps
            return
        for _ in range(steps):
            stats = collector.collect(n_step=1)
            if stats.n_collected_steps != 1:
                raise RuntimeError("Tianshou SAC did not collect exactly one transition")
            total_steps += 1
            if total_steps > warmup_steps:
                with policy_within_training_step(algorithm.policy):
                    algorithm.update(buffer=collector.buffer, sample_size=SAC_CONFIG.batch_size)

    return run, collector.close, package_version("tianshou"), f"torch {torch.__version__}"


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    seed_everything()
    logging.getLogger("tianshou.data.collector").setLevel(logging.CRITICAL)
    device = resolve_device(args.device)
    ppo_factories = {
        "sb3": make_sb3_runner,
        "cleanrl": make_cleanrl_runner,
        "tianshou": make_tianshou_runner,
    }
    off_policy_factories = {
        "dqn": {
            "sb3": make_sb3_dqn_runner,
            "cleanrl": make_cleanrl_dqn_runner,
            "tianshou": make_tianshou_dqn_runner,
        },
        "sac": {
            "sb3": make_sb3_sac_runner,
            "cleanrl": make_cleanrl_sac_runner,
            "tianshou": make_tianshou_sac_runner,
        },
    }
    if args.algorithm == "ppo":
        run, close, version, backend = ppo_factories[args.framework](device)
        config = CONFIG
    else:
        run, close, version, backend = off_policy_factories[args.algorithm][args.framework](
            device, args.warmup_steps
        )
        config = DQN_CONFIG if args.algorithm == "dqn" else SAC_CONFIG
    try:
        run(args.warmup_steps)
        synchronize(device)
        start = time.perf_counter()
        run(args.steps)
        synchronize(device)
        elapsed_seconds = time.perf_counter() - start
    finally:
        close()

    result = {
        "algorithm": args.algorithm,
        "framework": args.framework,
        "framework_version": version,
        "backend": backend,
        "device": args.device,
        "threads": args.threads,
        "measured_steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "elapsed_seconds": elapsed_seconds,
        "steps_per_second": args.steps / elapsed_seconds,
        "config": asdict(config),
    }
    print(f"BENCH_RESULT={json.dumps(result, sort_keys=True)}")


if __name__ == "__main__":
    main()

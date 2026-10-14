#!/usr/bin/env python3
"""Matched Atari PPO and DQN timing for SB3 and adapted CleanRL runners."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
import time
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork

WORKLOADS = json.loads((Path(__file__).resolve().parents[1] / "workloads.json").read_text())
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=("sb3", "cleanrl"), required=True)
    parser.add_argument("--algorithm", choices=("ppo", "dqn"), default="ppo")
    parser.add_argument("--device", choices=("cpu", "cuda", "metal"), default="cuda")
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    defaults = {"ppo": (40960, 4096), "dqn": (4096, 80004)}[args.algorithm]
    args.steps = defaults[0] if args.steps is None else args.steps
    args.warmup_steps = defaults[1] if args.warmup_steps is None else args.warmup_steps
    interval = 4096 if args.algorithm == "ppo" else 4
    if args.threads < 1 or args.steps < 1 or args.steps % interval:
        parser.error(f"--threads must be positive; --steps must be a positive multiple of {interval}")
    if args.warmup_steps != defaults[1]:
        parser.error(f"{args.algorithm} requires --warmup-steps {defaults[1]}")
    if not args.rom.is_file():
        parser.error(f"ROM file does not exist: {args.rom}")
    args.rom = args.rom.resolve()
    return args


class LocalRomAtariEnv(ale_py.env.AtariEnv):
    """Loads the supplied ROM without installing or copying it."""

    def __init__(self, rom: Path, max_frames: int):
        self.rom = rom
        super().__init__(game="breakout", frameskip=1, repeat_action_probability=0.0,
                         full_action_space=False, max_num_frames_per_episode=max_frames)

    def load_game(self) -> None:
        self.ale.loadROM(str(self.rom))


class FrameStack(gym.ObservationWrapper):
    """Removes the grayscale channel axis: U8 [4, 84, 84, 1] becomes [4, 84, 84]."""

    def __init__(self, env: gym.Env):
        super().__init__(gym.wrappers.FrameStackObservation(env, 4))
        self.observation_space = gym.spaces.Box(0, 255, (4, 84, 84), dtype=np.uint8)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.asarray(observation).squeeze(-1)


def environment_factory(rom: Path, index: int, config: dict):
    def create() -> gym.Env:
        torch.set_num_threads(1)
        env = LocalRomAtariEnv(rom, config["max_episode_frames"])
        env = AtariWrapper(env, noop_max=30, frame_skip=4, screen_size=84,
                           terminal_on_life_loss=True, clip_reward=True)
        env = FrameStack(env)
        if env.action_space.n != 4:
            raise ValueError("Breakout requires its four-action minimal action set")
        env.reset(seed=SEED + index)
        env.action_space.seed(SEED + index)
        return env

    return create


class NatureCNN(BaseFeaturesExtractor):
    """Converts U8 [batch, 4, 84, 84] to F32 [batch, 512] inside the model."""

    def __init__(self, observation_space: gym.Space):
        super().__init__(observation_space, 512)
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3), nn.ReLU(), nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dtype != torch.uint8 or tuple(observations.shape[1:]) != (4, 84, 84):
            raise ValueError("model observations must be U8 [batch, 4, 84, 84]")
        return self.network(observations.float() / 255.0)


class ByteActorCriticPolicy(ActorCriticCnnPolicy):
    """Passes U8 observations directly to the shared feature model."""

    def extract_features(self, obs, features_extractor=None):
        return (self.features_extractor if features_extractor is None else features_extractor)(obs)

    def predict_values(self, obs):
        features = self.vf_features_extractor(obs)
        return self.value_net(self.mlp_extractor.forward_critic(features))

    def get_distribution(self, obs):
        features = self.pi_features_extractor(obs)
        return self._get_action_dist_from_latent(self.mlp_extractor.forward_actor(features))


class ByteQNetwork(QNetwork):
    """Passes U8 observations directly to the Q feature model."""

    def extract_features(self, obs, features_extractor):
        return features_extractor(obs)


class ByteDQNPolicy(DQNPolicy):
    def make_q_net(self):
        return ByteQNetwork(**self._update_features_extractor(self.net_args, features_extractor=None)).to(self.device)


class AlignedReplayBuffer(ReplayBuffer):
    """CPU U8 stacks share next rows; truncated terminal stacks are saved separately."""

    def __init__(self, *args, **kwargs):
        kwargs["optimize_memory_usage"] = True
        kwargs["handle_timeout_termination"] = False
        super().__init__(*args, **kwargs)
        if self.n_envs != 1 or self.observations.dtype != np.uint8:
            raise ValueError("replay requires one environment and U8 observations")
        self.truncated_observations: dict[int, np.ndarray] = {}

    def add(self, obs, next_obs, action, reward, done, infos) -> None:
        if obs.dtype != np.uint8 or next_obs.dtype != np.uint8:
            raise ValueError("replay observations must remain U8")
        position = self.pos
        self.truncated_observations.pop(position, None)
        timeout = bool(infos[0].get("TimeLimit.truncated", False))
        if timeout:
            self.truncated_observations[position] = next_obs[0].copy()
        super().add(obs, next_obs, action, reward, np.asarray(done) & ~np.asarray([timeout]), infos)

    def sample(self, batch_size: int, env=None) -> ReplayBufferSamples:
        count = self.buffer_size - 1 if self.full else self.pos
        if batch_size > count:
            raise ValueError("replay sample exceeds available transitions")
        # Choose unique rows with O(batch_size) temporary storage, excluding the overwrite frontier.
        swaps: dict[int, int] = {}
        indices = np.empty(batch_size, dtype=np.int64)
        for index in range(batch_size):
            candidate = int(np.random.randint(index, count))
            indices[index] = swaps.get(candidate, candidate)
            swaps[candidate] = swaps.get(index, index)
        if self.full:
            indices = (indices + self.pos + 1) % self.buffer_size
        return self._get_samples(indices, env)

    def _get_samples(self, batch_inds, env=None) -> ReplayBufferSamples:
        observations = self.observations[batch_inds, 0]
        next_observations = self.observations[(batch_inds + 1) % self.buffer_size, 0].copy()
        for row, index in enumerate(batch_inds):
            if int(index) in self.truncated_observations:
                next_observations[row] = self.truncated_observations[int(index)]
        return ReplayBufferSamples(
            self.to_torch(observations), self.to_torch(self.actions[batch_inds, 0]),
            self.to_torch(next_observations), self.to_torch(self.dones[batch_inds, 0].reshape(-1, 1)),
            self.to_torch(self.rewards[batch_inds, 0].reshape(-1, 1)),
        )


class MatchedPPO(PPO):
    """Uses PPO2 value clipping and sample standard deviation in the SB3 update."""

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip = self.clip_range(self._current_progress_remaining)
        for _ in range(self.n_epochs):
            for batch in self.rollout_buffer.get(self.batch_size):
                values, log_prob, entropy = self.policy.evaluate_actions(batch.observations, batch.actions.long().flatten())
                values = values.flatten()
                advantages = batch.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                ratio = (log_prob - batch.old_log_prob).exp()
                policy_loss = -torch.minimum(advantages * ratio, advantages * ratio.clamp(1 - clip, 1 + clip)).mean()
                clipped_values = batch.old_values + (values - batch.old_values).clamp(-clip, clip)
                value_loss = torch.maximum((values - batch.returns).square(), (clipped_values - batch.returns).square()).mean()
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy.mean()
                self.policy.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                self.benchmark_updates = getattr(self, "benchmark_updates", 0) + 1
            self._n_updates += 1


class MatchedDQN(DQN):
    """Uses MSE without gradient clipping and an absolute transition epsilon schedule."""

    def _on_step(self) -> None:
        self._n_calls += 1
        self.exploration_rate = epsilon(self.num_timesteps)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        for _ in range(gradient_steps):
            batch = self.replay_buffer.sample(batch_size)
            with torch.no_grad():
                target = batch.rewards + (1 - batch.dones) * self.gamma * self.q_net_target(batch.next_observations).max(1, keepdim=True).values
            predictions = self.q_net(batch.observations).gather(1, batch.actions.long())
            loss = nn.functional.mse_loss(predictions, target)
            self.policy.optimizer.zero_grad()
            loss.backward()
            self.policy.optimizer.step()
            self.benchmark_updates = getattr(self, "benchmark_updates", 0) + 1
        if self.num_timesteps % self.target_update_interval == 0:
            self.q_net_target.load_state_dict(self.q_net.state_dict())
        self._n_updates += gradient_steps


def epsilon(timestep: int) -> float:
    return 1.0 - 0.99 * min(timestep / 1_000_000, 1.0)


def assert_parameter_count(model: nn.Module, config: dict) -> None:
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != config["trainable_parameters"]:
        raise ValueError(f"model has {count} parameters, expected {config['trainable_parameters']}")


def enable_fusion(encoders: list[nn.Module], device: torch.device) -> str:
    if device.type != "cuda":
        return "unavailable: CUDA compiler required"
    from torch.utils._triton import has_triton
    if not has_triton():
        return "unavailable: PyTorch Inductor requires Triton"
    for encoder in encoders:
        encoder.compile(backend="inductor")
    return "torch.compile: inductor encoders"


def make_sb3_runner(args, device):
    config = WORKLOADS[args.algorithm]
    factories = [environment_factory(args.rom, index, config) for index in range(config["environments"])]
    common = dict(learning_rate=config["learning_rate"], gamma=config["gamma"], device=device, seed=SEED, verbose=0)
    policy = dict(net_arch=[], features_extractor_class=NatureCNN, normalize_images=False,
                  optimizer_kwargs={"eps": config["adam_epsilon"], "weight_decay": 0.0})
    if args.algorithm == "ppo":
        envs = SubprocVecEnv(factories, start_method="spawn")
        model = MatchedPPO(ByteActorCriticPolicy, envs, n_steps=128, batch_size=1024, n_epochs=4,
                           gae_lambda=0.95, clip_range=0.1, vf_coef=0.25, ent_coef=0.01,
                           max_grad_norm=0.5, policy_kwargs=policy, **common)
        assert_parameter_count(model.policy, config)
        if model.rollout_buffer.observations.dtype != np.uint8:
            raise ValueError("rollout observations must remain U8")
        model.benchmark_fusion = enable_fusion([model.policy.features_extractor], device)
    else:
        envs = DummyVecEnv(factories)
        model = MatchedDQN(ByteDQNPolicy, envs, buffer_size=config["replay_capacity"], batch_size=32,
                           learning_starts=config["training_start"], train_freq=(4, "step"), gradient_steps=1,
                           target_update_interval=1000, exploration_initial_eps=1.0, exploration_final_eps=0.01,
                           replay_buffer_class=AlignedReplayBuffer, policy_kwargs=policy, **common)
        assert_parameter_count(model.q_net, config)
        model.benchmark_fusion = enable_fusion([model.q_net.features_extractor, model.q_net_target.features_extractor], device)
    first_call = True

    def run(steps):
        nonlocal first_call
        model.learn(total_timesteps=steps, reset_num_timesteps=first_call, log_interval=None)
        first_call = False

    return run, envs.close, f"{importlib.metadata.version('stable-baselines3')} (adapted)", model


def orthogonal(layer, gain):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class CleanRLPPOModel(nn.Module):
    """Shared Atari encoder and heads from CleanRL's ppo_atari.py."""

    def __init__(self, observation_space):
        super().__init__()
        self.encoder = NatureCNN(observation_space)
        for layer in self.encoder.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                orthogonal(layer, 2**0.5)
        self.actor = orthogonal(nn.Linear(512, 4), 0.01)
        self.critic = orthogonal(nn.Linear(512, 1), 1.0)

    def action_and_value(self, observations, actions=None):
        """Maps U8 [batch, 4, 84, 84] to action, log probability, entropy and value [batch]."""
        features = self.encoder(observations)
        distribution = Categorical(logits=self.actor(features))
        actions = distribution.sample() if actions is None else actions
        return actions, distribution.log_prob(actions), distribution.entropy(), self.critic(features).flatten()

    def value(self, observations):
        """Maps U8 [batch, 4, 84, 84] to F32 values [batch]."""
        return self.critic(self.encoder(observations)).flatten()


class CleanRLDQNModel(nn.Module):
    def __init__(self, observation_space):
        super().__init__()
        self.encoder = NatureCNN(observation_space)
        self.head = nn.Linear(512, 4)

    def forward(self, observations):
        """Maps U8 [batch, 4, 84, 84] to F32 Q values [batch, 4]."""
        return self.head(self.encoder(observations))


def make_cleanrl_ppo_runner(args, device):
    config = WORKLOADS["ppo"]
    envs = gym.vector.AsyncVectorEnv(
        [environment_factory(args.rom, index, config) for index in range(32)],
        context="spawn", autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    model = CleanRLPPOModel(envs.single_observation_space).to(device)
    assert_parameter_count(model, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4, eps=1e-5)
    model.benchmark_updates = 0
    model.benchmark_fusion = enable_fusion([model.encoder], device)
    observations = torch.zeros((128, 32, 4, 84, 84), dtype=torch.uint8, device=device)
    actions = torch.zeros((128, 32), dtype=torch.long, device=device)
    log_probs = torch.zeros((128, 32), device=device)
    rewards = torch.zeros((128, 32), device=device)
    dones = torch.zeros((128, 32), dtype=torch.bool, device=device)
    values = torch.zeros((128, 32), device=device)
    reset, _ = envs.reset(seed=[SEED + index for index in range(32)])
    next_observation = torch.as_tensor(reset, device=device)

    def run(steps):
        nonlocal next_observation
        for _ in range(steps // 4096):
            for step in range(128):
                observations[step] = next_observation
                with torch.no_grad():
                    action, log_prob, _, value = model.action_and_value(next_observation)
                actions[step], log_probs[step], values[step] = action, log_prob, value
                obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
                indices = np.flatnonzero(truncated & ~terminated)
                if len(indices):
                    terminal = np.stack([info["final_obs"][index] for index in indices])
                    with torch.no_grad():
                        reward[indices] += 0.99 * model.value(torch.as_tensor(terminal, device=device)).cpu().numpy()
                rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
                dones[step] = torch.as_tensor(terminated | truncated, device=device)
                next_observation = torch.as_tensor(obs, device=device)
            with torch.no_grad():
                next_value = model.value(next_observation)
                advantages = torch.zeros_like(rewards)
                gae = torch.zeros(32, device=device)
                for step in reversed(range(128)):
                    mask = (~dones[step]).float()
                    delta = rewards[step] + 0.99 * next_value * mask - values[step]
                    gae = delta + 0.99 * 0.95 * mask * gae
                    advantages[step] = gae
                    next_value = values[step]
                returns = advantages + values
            flat_obs = observations.reshape(-1, 4, 84, 84)
            flat_actions, flat_log_probs = actions.flatten(), log_probs.flatten()
            flat_advantages, flat_returns, flat_values = advantages.flatten(), returns.flatten(), values.flatten()
            indices = np.arange(4096)
            for _ in range(4):
                np.random.shuffle(indices)
                for start in range(0, 4096, 1024):
                    batch = indices[start:start + 1024]
                    _, log_prob, entropy, value = model.action_and_value(flat_obs[batch], flat_actions[batch])
                    ratio = (log_prob - flat_log_probs[batch]).exp()
                    advantage = flat_advantages[batch]
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                    policy_loss = -torch.minimum(advantage * ratio, advantage * ratio.clamp(0.9, 1.1)).mean()
                    clipped = flat_values[batch] + (value - flat_values[batch]).clamp(-0.1, 0.1)
                    value_loss = torch.maximum((value - flat_returns[batch]).square(), (clipped - flat_returns[batch]).square()).mean()
                    loss = policy_loss + 0.25 * value_loss - 0.01 * entropy.mean()
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                    model.benchmark_updates += 1

    return run, envs.close, "cleanrl/ppo_atari.py@e421c2e5 (adapted)", model


def make_cleanrl_dqn_runner(args, device):
    config = WORKLOADS["dqn"]
    env = environment_factory(args.rom, 0, config)()
    online = CleanRLDQNModel(env.observation_space).to(device)
    target = CleanRLDQNModel(env.observation_space).to(device)
    target.load_state_dict(online.state_dict())
    assert_parameter_count(online, config)
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-4, eps=1e-8)
    online.benchmark_updates = 0
    online.benchmark_fusion = enable_fusion([online.encoder, target.encoder], device)
    replay = AlignedReplayBuffer(config["replay_capacity"], env.observation_space, env.action_space, device=device)
    observation, _ = env.reset(seed=SEED)
    total_steps = 0

    def run(steps):
        nonlocal observation, total_steps
        for _ in range(steps):
            total_steps += 1
            if random.random() < epsilon(total_steps - 1):
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = int(online(torch.as_tensor(observation[None], device=device)).argmax(1).item())
            next_observation, reward, terminated, truncated, _ = env.step(action)
            replay.add(observation[None], next_observation[None], np.asarray([action]), np.asarray([reward]),
                       np.asarray([terminated or truncated]), [{"TimeLimit.truncated": truncated and not terminated}])
            observation = next_observation
            if terminated or truncated:
                observation, _ = env.reset()
            if total_steps > 80000 and total_steps % 4 == 0:
                batch = replay.sample(32)
                with torch.no_grad():
                    targets = batch.rewards + (1 - batch.dones) * 0.99 * target(batch.next_observations).max(1, keepdim=True).values
                predictions = online(batch.observations).gather(1, batch.actions.long())
                loss = nn.functional.mse_loss(predictions, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                online.benchmark_updates += 1
            if total_steps % 1000 == 0:
                target.load_state_dict(online.state_dict())

    return run, env.close, "cleanrl/dqn_atari.py@e421c2e5 (adapted)", online


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.device == "metal" and not torch.backends.mps.is_available():
        raise RuntimeError("Metal is unavailable")
    device = torch.device("mps" if args.device == "metal" else args.device)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if args.framework == "sb3":
        run, close, version, model = make_sb3_runner(args, device)
    elif args.algorithm == "ppo":
        run, close, version, model = make_cleanrl_ppo_runner(args, device)
    else:
        run, close, version, model = make_cleanrl_dqn_runner(args, device)
    try:
        run(args.warmup_steps)
        warmup_updates = model.benchmark_updates
        synchronize(device)
        start = time.perf_counter()
        run(args.steps)
        synchronize(device)
        elapsed = time.perf_counter() - start
        optimizer_steps = model.benchmark_updates - warmup_updates
        expected_updates = args.steps // 4096 * 16 if args.algorithm == "ppo" else args.steps // 4
        if optimizer_steps != expected_updates:
            raise ValueError(f"runner performed {optimizer_steps} optimizer steps, expected {expected_updates}")
    finally:
        close()
    result = dict(algorithm=args.algorithm, framework=args.framework, framework_version=version,
                  fusion=model.benchmark_fusion,
                  backend=f"torch {torch.__version__}", ale_version=ale_py.__version__, device=args.device,
                  threads=args.threads, environment_workers=32 if args.algorithm == "ppo" else 0,
                  worker_kind="processes" if args.algorithm == "ppo" else "none",
                  measured_steps=args.steps, warmup_steps=args.warmup_steps,
                  optimizer_steps=optimizer_steps,
                  elapsed_seconds=elapsed, steps_per_second=args.steps / elapsed, config=WORKLOADS[args.algorithm])
    print(f"BENCH_RESULT={json.dumps(result, sort_keys=True)}")


if __name__ == "__main__":
    main()

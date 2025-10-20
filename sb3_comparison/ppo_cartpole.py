import timeit
import torch
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def main():
    # Match Rust bench: 8 parallel CartPole-v1 envs
    env = make_vec_env("CartPole-v1", n_envs=8)

    # Match model: separate actor/critic MLPs with 2x64 and Tanh activations
    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh,
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
    )

    # Match PPO hyperparameters from examples/ppo_bench.rs
    # - Total rollout size = n_envs * n_steps = 8 * 256 = 2048
    # - Minibatch size = 64, epochs = 10
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,
        clip_range=0.2,
        vf_coef=0.5,
        normalize_advantage=True,
        policy_kwargs=policy_kwargs,
        device="cpu",
        verbose=0,
    )
    print("Using device:", model.device)

    model.learn(total_timesteps=100_000)


if __name__ == "__main__":
    setup_code = "from __main__ import main"
    execution_time = timeit.timeit("main()", setup=setup_code, number=2)
    print(f"Execution time: {execution_time/2} seconds")
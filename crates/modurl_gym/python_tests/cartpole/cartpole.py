import sys
import os
import gymnasium as gym
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_gym import test_gym

if __name__ == "__main__":
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env = gym.make("CartPole-v1")
    test_gym(env, output_dir=current_dir)

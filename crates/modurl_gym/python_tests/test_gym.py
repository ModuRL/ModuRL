import gymnasium as gym
import json
import os
import numpy as np

def test_gym(env, output_dir=None, custom_reset=None, custom_info=None):
    if custom_reset is not None:
        obs, info, env = custom_reset(env)
    else:
        obs = env.reset(seed=123, options={"low": 0.0, "high": 0.0})

    # Lists to store inputs and outputs
    inputs = []
    outputs = []

    for _ in range(100):
        action = env.action_space.sample()

        inputs.append(int(action))
        
        obs, reward, done, truncated, info = env.step(action)
        outputs.append({
            "observation": obs.tolist(),
            "reward": float(reward),
            "done": bool(done),
            "truncated": bool(truncated),
            "info": info if custom_info is None else custom_info(env, info, obs)
        })
        
        if done:
            if custom_reset is not None:
                obs, _, env = custom_reset(env)
            else:
                obs = env.reset(seed=123, options={"low": 0.0, "high": 0.0})

    env.close()

    # Get script directory - use provided output_dir or default to script location
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        script_dir = output_dir

    # Save to JSON files in script directory
    with open(os.path.join(script_dir, 'inputs.json'), 'w') as f:
        json.dump(inputs, f, indent=2)

    with open(os.path.join(script_dir, 'output.json'), 'w') as f:
        json.dump(outputs, f, indent=2)
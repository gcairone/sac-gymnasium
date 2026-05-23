#!/usr/bin/env python3
"""
Test script to verify interactive MuJoCo rendering and window forwarding
from inside the Docker container to the host machine.
"""
import time
import gymnasium as gym

def main() -> None:
    env = gym.make("InvertedPendulum-v5", render_mode="human")

    obs, info = env.reset(seed=42)
    print("Environment initialized successfully. Opening window...")

    max_steps = 300
    for step in range(max_steps):
        action = env.action_space.sample()
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(1.0 / 60.0)
        
        if (step + 1) % 50 == 0:
            print(f" Rendered {step + 1}/{max_steps} steps successfully.")
            
        if terminated or truncated:
            env.reset()
            
    env.close()
    print("Environment closed cleanly. Graphical test completed.")

if __name__ == "__main__":
    main()
 
import gymnasium as gym
import numpy as np
import os, sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from eval_env import SumoEnv
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.evaluation import evaluate_policy
import pathlib

def main():
    gui = False
    # defining model path
    model_folder = "91_PPO_100"
    save_path = os.path.join(pathlib.Path(__file__).parent.resolve(), "Models", model_folder, f"{model_folder}.zip")
   # save_path = os.path.join(pathlib.Path(__file__).parent.resolve(), "Models", result_name, f"rl_model_{int(timestep_mil * 10 ** 6)}_steps")
    #save_path = "/Users/rolwinpinto/onramp_simulation/Models/rl_model_14000000_steps.zip"  # Update this path if necessary
    # save_path="/Users/rolwinpinto/onramp_simulation/Models/rl_model_14000000_steps"  # Check if this file exists
    # Ensure the file is named correctly and has the .zip extension if needed
    model = PPO.load(save_path)  # This line will raise an error if the file does not exist
    env = DummyVecEnv([lambda: SumoEnv(gui=gui)])
    reward = 0
    obs = env.reset()
    time = 0
    for _ in range(10000):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
    
    print('the final reward is {}'.format(reward))
    
# run code on script run
if __name__ =="__main__":
    main()

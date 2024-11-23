import os
import pathlib
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from env import SumoEnv
from eval_env import SumoEnv as EvalEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DDPG
from gym import spaces
import numpy as np

def main():

    # input variables
    num_envs = 1
    timesteps = 100000
    eval_freq = 1000000
    iteration = 91
    algo = "DDPG"

    gui = True

    # paths and eval adjusting for multiple envs
    result_name = f"{iteration}_{algo}_{timesteps}"
    current_path = pathlib.Path(__file__).parent.resolve()
    log_path = os.path.join(current_path,"Logs", result_name)
    checkpoint_save_path = os.path.join(current_path, "Models", result_name)
    final_save_path = os.path.join(checkpoint_save_path, result_name)
    best_save_path = os.path.join(final_save_path, "Best Model")
    eval_freq_adjusted = eval_freq/num_envs

    # setup env
    env = DummyVecEnv ([lambda: Monitor(SumoEnv(gui=gui)) for i in range(num_envs)])
    print("Action space:", env.action_space)
    eval_env = DummyVecEnv([lambda: Monitor(EvalEnv(gui=gui))])

    # DDPG hyperparameters tuned for intent-aware training
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],  # Actor network
            qf=[256, 256]   # Critic network
        )
    )

    # Create DDPG model with custom policy architecture
    model = DDPG(
        "MlpPolicy", 
        env, 
        learning_rate=1e-3,
        buffer_size=1000000,
        learning_starts=100,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "episode"),
        gradient_steps=-1,
        action_noise=None,
        replay_buffer_class=None,
        replay_buffer_kwargs=None,
        optimize_memory_usage=False,
        tensorboard_log=log_path,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="cpu"
    )

    # Callback to evaluate model every eval_freq steps and save best
    checkpoint_callback = CheckpointCallback(
        save_path=checkpoint_save_path,
        name_prefix="rl_model",
        save_freq=eval_freq_adjusted, 
        verbose=1
    )

    # Callback to evaluate model every eval_freq steps and save best
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=50,
        best_model_save_path=best_save_path,
        eval_freq=eval_freq_adjusted, 
        render=False,
        deterministic=True,
        verbose=1
    )

    # Create the callback list
    callback = CallbackList([checkpoint_callback, eval_callback])

    print("Starting training with intent-aware DDPG...")
    model.learn(total_timesteps=timesteps, callback=callback)

    print("Training complete. Saving model.")
    model.save(final_save_path)
    print("Model Saved")

# run code on script run
if __name__ =="__main__":
    main()

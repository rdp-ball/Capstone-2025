import os
import pathlib
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from env import SumoEnv
from eval_env import SumoEnv as EvalEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

def main():
    # input variables
    num_envs = 1
    timesteps = 100000
    eval_freq = 1000000
    iteration = 91
    algo = "DDPG"
    
    gui = False

    # paths and eval adjusting for multiple envs
    result_name = f"{iteration}_{algo}_{timesteps}"
    current_path = pathlib.Path(__file__).parent.resolve()
    log_path = os.path.join(current_path, "Logs", result_name)
    checkpoint_save_path = os.path.join(current_path, "Models", result_name)
    final_save_path = os.path.join(checkpoint_save_path, result_name)
    best_save_path = os.path.join(final_save_path, "Best Model")
    eval_freq_adjusted = eval_freq/num_envs

    # setup env
    env = DummyVecEnv([lambda: Monitor(SumoEnv(gui=gui)) for i in range(num_envs)])
    print("Action space:", env.action_space)
    eval_env = DummyVecEnv([lambda: Monitor(EvalEnv(gui=gui))])

#  # DDPG works with continuous action spaces, so we need to modify the environment
#     # or switch back to PPO for discrete actions
#     if isinstance(env.action_space, gym.spaces.Discrete):
#         raise ValueError("DDPG requires continuous action space (Box). Current space is Discrete. Consider using PPO instead.")

#     # Get the dimension of action space for noise parameter
    n_actions = env.action_space.shape[-1]
    
    # Add action noise for exploration
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )

    # Callback to evaluate model every 5000 steps and save best
    checkpoint_callback = CheckpointCallback(
        save_path=checkpoint_save_path,
        name_prefix="rl_model",
        save_freq=eval_freq_adjusted,
        verbose=1
    )

    # Callback to evaluate model every 5000 steps and save best
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

    # create, train, save model
    model = DDPG(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        verbose=1,
        tensorboard_log=log_path,
        learning_rate=3e-4,
        buffer_size=1000000,
        learning_starts=100,
        batch_size=256,
        train_freq=(1, "episode"),
        gradient_steps=-1,
        gamma=0.99,
        tau=0.005
    )
    
    model.learn(total_timesteps=timesteps, callback=callback)

    print("Training complete. Saving model.")
    model.save(final_save_path)
    print("Model Saved")

# run code on script run
if __name__ == "__main__":
    main()
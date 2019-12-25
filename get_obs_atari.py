import numpy as np
import joblib
import shap
import os
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import PPO2
from time import ctime


# Make env (from stable baselines doc)
num_env = 1
env_name = 'BreakoutNoFrameskip-v4'
obs_folder = "observations"
env = make_atari_env(env_name, num_env=num_env, seed=0)
env = VecFrameStack(env, n_stack=4)

# Load pre-trained agent
model_path = os.path.join("pre-trained models", f"{env_name}.pkl")
model = PPO2.load(model_path)

# Play with pre-trained agent to collect observations
nb_episodes = 5
finished_games = 0
max_num_steps = 1000
rewards, observations = [], []
while finished_games < nb_episodes:
    obs = env.reset()
    for _step in range(max_num_steps):
        env.render()
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        observations.append(obs)
        rewards.append(reward)

        if done:
            finished_games += 1
            print("Finished", finished_games, "/", nb_episodes)
            print("Rewards:", sum(rewards))
            rewards = []
            break

# Save obs
obs_path = os.path.join(obs_folder, f"{env_name}-{ctime()}.pkl")
joblib.dump(np.array(observations), obs_path, compress=3)

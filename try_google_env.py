import gfootball.env as football_env
from gfootball.env.players.ppo2_cnn import Player
from baselines.ppo2 import ppo2
from stable_baselines.ppo2 import PPO2
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import tensorflow as tf
import multiprocessing
import pickle
import joblib


checpoint_path = "pre-trained models/CP_11_vs_11_easy_stochastic_v2"
# with open(checpoint_path, "rb") as file:
#     model = joblib.load(file)
#     print(type(model))
#     print(model)
# print(model.keys())


player = Player(
    f'ppo2_cnn:policy=gfootball_impala_cnn,checkpoint={checpoint_path}', {})
# extra_players = [
#     f'ppo2_cnn:right_players=1,policy=gfootball_impala_cnn,checkpoint={checpoint_path}']
# env = football_env.create_environment(
#     env_name='11_vs_11_stochastic', render=True, extra_players=extra_players)
# print(env.)


# observation = env.reset()
# done = False
# while not done:


#     # action = env.action_space.sample()
#     observation, reward, done, info = env.step([])
#     print(observation)

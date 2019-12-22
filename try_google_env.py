# import gfootball.env as football_env
# from gfootball.env.players.ppo2_cnn import Player
# from baselines.ppo2 import ppo2
# from stable_baselines.ppo2 import PPO2
# from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
# import tensorflow as tf
import multiprocessing
import pickle


checpoint_path = "pre-trained models/CP_11_vs_11_easy_stochastic_v2"
with open(checpoint_path, "rb") as file:
    model = pickle.load(file)
    print(model)
# env = football_env.create_environment(
#     env_name='11_vs_11_stochastic', render=True)
# vec_env = SubprocVecEnv([
#     (lambda _i=i: create_single_football_env(_i))
#     for i in range(1)
# ], context=None)
# model = PPO2.load(checpoint_path, env)
# model = ppo2.learn(network="mlp", env=env, total_timesteps=0)

# env = football_env.create_environment(
#     env_name='11_vs_11_stochastic', render=True)
# observation = env.reset()
# done = False
# while not done:
#     # action = player.take_action()
#     action = model.step(observation)
#     observation, reward, done, info = env.step(action)

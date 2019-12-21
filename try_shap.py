import numpy as np
import gym
import pickle
import shap
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env = gym.make('CartPole-v1')
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

model_path = 'PPO_model.pkl'
# model = PPO2(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=10000)
# model.save(model_path)

model = PPO2(MlpPolicy, env, verbose=1)
model.load(model_path)

# obs = env.reset()

# nb_episodes = 100
# finished_games = 0
# nb_steps = 100
# rewards = []
# background = []

# while finished_games <= nb_episodes:
#     for _step in range(nb_steps):
#         env.render()
#         action, _states = model.predict(obs)
#         obs, reward, done, info = env.step(action)
#         background.append(obs)
#         rewards.append(reward)

#         if done:
#             # print("TOMBE", sum(rewards))
#             observation = env.reset()
#             rewards = []
#             finished_games += 1
#             print("Finished", finished_games, "/", nb_episodes)


# select a set of background examples to take an expectation over
# background = np.array(background)
background = np.array([[1, 2, 3, 4], [3, 4, 5, 6], [3, 4, 5, 6]])
# print(model.predict(background), model.predict(background)[0])
# explain predictions of the model on four images
e = shap.KernelExplainer(lambda x: model.predict(x)[0], background)
# ...or pass tensors directly
# e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
shap_values = e.shap_values(background[0:2])
print(shap_values)

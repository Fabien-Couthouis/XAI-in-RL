import os
import numpy as np
import joblib
import shap
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import PPO2


obs_path = os.path.join(
    "observations", "BreakoutNoFrameskip-v4-Sun Dec 22 11:32:52 2019.pkl")
observations = joblib.load(obs_path)

print(observations[0].reshape(-1, 2).shape)
num_env = 1
env_name = 'BreakoutNoFrameskip-v4'
obs_folder = "observations"
env = make_atari_env(env_name, num_env=num_env, seed=0)
env = VecFrameStack(env, n_stack=4)

# %%Load pre-trained agent
model_path = os.path.join("pre-trained models", f"{env_name}.pkl")
MODEL = PPO2.load(model_path)
MODEL =MODEL.act_model.


def predict_many_examples(observations):
    # if observations.shape != (-1, 84, 84, 4):
    #     observations = observations.reshape(-1, 84, 84, 4)
    outputs = []
    for example in observations:
        pred = MODEL.predict(example)[0]
        outputs.append(pred)
    return np.array(outputs)


def reshape_obs(observations):
    new = []
    for observation in observations:
        # print("pre", observation)
        new.append(observation)
        # print("post", observation)

    new = np.array(new).reshape(-1, 2)

    print("shape", new.shape)
    return np.array(new)


explainer = shap.SamplingExplainer(
    predict_many_examples, np.zeros((100,84,84,4)))
shap_values = explainer.explain(shap.sample(observations))
print(shap_values)
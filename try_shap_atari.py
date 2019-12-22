import os
import numpy as np
import joblib
import shap
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import PPO2


obs_path = os.path.join(
    "observations", "BreakoutNoFrameskip-v4-Sun Dec 22 11:32:52 2019.pkl")
observations = np.array(joblib.load(obs_path))

num_env = 1
env_name = 'BreakoutNoFrameskip-v4'
obs_folder = "observations"
env = make_atari_env(env_name, num_env=num_env, seed=0)
env = VecFrameStack(env, n_stack=4)

# %%Load pre-trained agent
model_path = os.path.join("pre-trained models", f"{env_name}.pkl")
MODEL = PPO2.load(model_path)


def predict_many_examples(X):
    outputs = []
    for example in X:
        pred = MODEL.predict(example)[0]
        outputs.append(pred)
    return np.array(outputs)


# explain predictions of the model on four images
explainer = shap.KernelExplainer(
    predict_many_examples, shap.sample(observations))

shap.summary_plot(explainer, shap.sample(observations))
# shap_values = explainer.shap_values(shap.sample(observations))

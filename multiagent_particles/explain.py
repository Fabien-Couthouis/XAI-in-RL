import numpy as np
import torch
from utilities.env_wrapper import EnvWrapper


def load_model(path):
    checkpoint = torch.load(path)
    # TODO


def play(env, model=None, num_episodes=20, render=True):
    observations = env.reset()
    for episode in range(num_episodes):
        if render:
            env.render()
        if model is None:
            actions = env.get_random_actions()
        else:
            # TODO
            raise NotImplementedError
        observations, rewards, dones, info = env.step(actions)

        print(episode, rewards)


if __name__ == "__main__":
    env = EnvWrapper("simple_tag")
    model_path = "model_save/simple_tag_maddpg/model.pt"
    # model = load_model(path) #TODO
    play(env, model=model)

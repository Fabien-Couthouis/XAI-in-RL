import cv2
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from knights_archers_zombies import env, manual_control, parallel_env, raw_env


class ParallelPettingZooEnv(MultiAgentEnv):
    def __init__(self, config=None):
        self.par_env = parallel_env()
        # agent idx list
        self.agents = self.par_env.agents

        # Get dictionaries of obs_spaces and act_spaces
        self.observation_spaces = self.par_env.observation_spaces
        self.action_spaces = self.par_env.action_spaces

        # Get first observation space, assuming all agents have equal space
        self.observation_space = self.observation_spaces[self.agents[0]]

        # Get first action space, assuming all agents have equal space
        self.action_space = self.action_spaces[self.agents[0]]

        assert all(obs_space == self.observation_space
                   for obs_space
                   in self.par_env.observation_spaces.values()), \
            "Observation spaces for all agents must be identical. Perhaps " \
            "SuperSuit's pad_observations wrapper can help (useage: " \
            "`supersuit.aec_wrappers.pad_observations(env)`"

        assert all(act_space == self.action_space
                   for act_space in self.par_env.action_spaces.values()), \
            "Action spaces for all agents must be identical. Perhaps " \
            "SuperSuit's pad_action_space wrapper can help (useage: " \
            "`supersuit.aec_wrappers.pad_action_space(env)`"

        self.reset()

    def reset(self):
        observations = self.par_env.reset()
        for agent, obs in observations.items():
            observations[agent] = self._preprocess(obs)

        return observations

    def _preprocess(self, obs):
        return cv2.resize(np.float32(obs), (84, 84))

    def step(self, action_dict):
        aobs, arew, adones, ainfo = self.par_env.step(action_dict)
        obss = {}
        rews = {}
        dones = {}
        infos = {}
        for agent in action_dict:
            obss[agent] = self._preprocess(aobs[agent])
            rews[agent] = arew[agent]
            dones[agent] = adones[agent]
            infos[agent] = ainfo[agent]
        dones["__all__"] = all(adones.values())
        return obss, rews, dones, infos

    def close(self):
        self.par_env.close()

    def seed(self, seed=None):
        self.par_env.seed(seed)

    def render(self, mode="human"):
        return self.par_env.render(mode)

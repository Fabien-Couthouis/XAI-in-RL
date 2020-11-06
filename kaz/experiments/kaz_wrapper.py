import cv2
import numpy as np
from gym.spaces import Box, Discrete, Dict
# from knights_archers_zombies import env, manual_control, parallel_env, raw_env
from pettingzoo.butterfly.knights_archers_zombies_v2 import (env,
                                                             manual_control,
                                                             parallel_env,
                                                             raw_env)
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE

OBS_SIZE = 168  # height/width of observations


class ParallelPettingZooEnv(MultiAgentEnv):
    def __init__(self, config=None):
        config = config or {}

        self.par_env = parallel_env()
        # agent idx list
        self.agents = self.par_env.agents
        self._with_state = config.get("with_state", False)
        self._actions_are_logits = config.get("actions_are_logits", False)

        # Get dictionaries of obs_spaces and act_spaces
        # self.observation_spaces = self.par_env.observation_spaces
        self.observation_spaces = dict(zip(self.par_env.agents, [Box(low=0, high=255, shape=(
            OBS_SIZE, OBS_SIZE, 3), dtype=np.uint8) for _ in enumerate(self.par_env.agents)]))
        self.action_spaces = self.par_env.action_spaces

        # Get first observation space, assuming all agents have equal space
        self.observation_space = self.observation_spaces[self.agents[0]]
        if self._with_state:
            # Add global state for QMix
            self.observation_space = Dict({
                "obs": self.observation_space,
                ENV_STATE: self.observation_space
            })

        # Get first action space, assuming all agents have equal space
        self.action_space = self.action_spaces[self.agents[0]]

        self.num_agents = len(self.agents)
        self.archers_list, self.knights_list = self._find_agents()

        # assert all(obs_space == self.observation_space
        #            for obs_space
        #            in self.observation_spaces.values()), \
        #     "Observation spaces for all agents must be identical. Perhaps " \
        #     "SuperSuit's pad_observations wrapper can help (useage: " \
        #     "`supersuit.aec_wrappers.pad_observations(env)`"

        # assert all(act_space == self.action_space
        #            for act_space in self.action_spaces.values()), \
        #     "Action spaces for all agents must be identical. Perhaps " \
        #     "SuperSuit's pad_action_space wrapper can help (useage: " \
        #     "`supersuit.aec_wrappers.pad_action_space(env)`"

        self.reset()

    def _find_agents(self):
        archers_list, knights_list = [], []
        for agent_name in self.par_env.agents:
            if agent_name.startswith("archer"):
                archers_list.append(agent_name)
            elif agent_name.startswith("knight"):
                knights_list.append(agent_name)

        assert self.num_agents == (len(archers_list) + len(knights_list))
        return archers_list, knights_list

    def reset(self):
        # print("reset")

        observations = self.par_env.reset()
        for agent, obs in observations.items():
            observations[agent] = self._preprocess(obs)

        return observations

    def _preprocess(self, obs):
        obs = cv2.resize(np.float32(obs), (OBS_SIZE, OBS_SIZE))
        # obs = np.zeros((168, 168, 3),  dtype=np.uint8)
        if self._with_state:
            obs = {'obs': obs,
                   ENV_STATE: obs}
        return obs

    def step(self, action_dict):

        # print("step")
        if self._actions_are_logits:
            action_dict = {
                k: np.random.choice(range(self.action_space.n), p=v)
                for k, v in action_dict.items()
            }
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

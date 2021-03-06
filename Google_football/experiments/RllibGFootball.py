import numpy as np
from gym.spaces import Box, Dict, Discrete

import gfootball.env as football_env
from ray.rllib.env.multi_agent_env import ENV_STATE, MultiAgentEnv


class RllibGFootball(MultiAgentEnv):
    """An example of a wrapper for GFootball to make it compatible with rllib."""

    def __init__(self, num_agents, env_name, render=True, save_replays=False, actions_are_logits=False, with_state=False):
        self.env = football_env.create_environment(
            env_name=env_name, stacked=False, representation='simple115v2',
            logdir='./replays', write_video=save_replays, rewards='scoring,checkpoints',
            write_goal_dumps=save_replays, write_full_episode_dumps=save_replays, render=render,
            dump_frequency=1,
            number_of_left_players_agent_controls=num_agents,
            channel_dimensions=(42, 42))
        self.action_space = Discrete(self.env.action_space.nvec[1])
        self.observation_space = Box(
            low=self.env.observation_space.low[0],
            high=self.env.observation_space.high[0],
            dtype=self.env.observation_space.dtype)
        self.num_agents = num_agents
        # MADDPG emits action logits instead of actual discrete actions
        self.actions_are_logits = actions_are_logits
        self.with_state = with_state

        if self.with_state:
            self.observation_space = Dict({
                "obs": self.observation_space,
                ENV_STATE: self.observation_space
            })

    def close(self):
        self.env.close()

    def reset(self):
        initial_obs = self.env.reset()
        obs = {}
        for agent_id in range(self.num_agents):
            key = agent_id
            if self.num_agents > 1:
                if self.with_state:
                    obs[key] = {'obs': initial_obs[agent_id],
                                ENV_STATE: initial_obs[agent_id]}
                else:
                    obs[key] = initial_obs[agent_id]
            else:
                obs[key] = initial_obs
        # print("reset", obs)
        return obs

    def step(self, action_dict):
        if self.actions_are_logits:
            # Handle MADDPG case
            action_dict = {
                k: np.random.choice(np.arange(len(v)), p=v)
                for k, v in action_dict.items()
            }

        actions = []
        for key, value in sorted(action_dict.items()):
            actions.append(value)
        o, r, d, i = self.env.step(actions)
        rewards, obs, infos = {}, {}, {}
        for pos, key in enumerate(sorted(action_dict.keys())):
            infos[key] = i
            if self.num_agents > 1:
                rewards[key] = r[pos]

                if self.with_state:
                    obs[key] = {'obs': o[pos], ENV_STATE: o[pos]}

                else:
                    obs[key] = o[pos]
            else:
                rewards[key] = r
                obs[key] = o
        dones = {'__all__': d}
        # print("step",obs)
        return obs, rewards, dones, infos

    def render(self):
        self.env.render()

    def random_actions(self):
        'Return: list of random actions for each agent '
        agents_actions = {}
        for agent_id in range(self.num_agents):
            agents_actions[agent_id] = self.action_space.sample()
        return agents_actions

    def idle_actions(self):
        'Return: list of idle (no move) actions for each agent '
        agents_actions = {}
        for agent_id in range(self.num_agents):
            action_idle = 0
            agents_actions[agent_id] = action_idle
        return agents_actions

from gym.spaces import Discrete, Box, Dict, MultiDiscrete
from ray.rllib.env import MultiAgentEnv
import numpy as np
from prettytable import PrettyTable

ACTION_MAP = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]


class PrisonerDilemma(MultiAgentEnv):

    def __init__(self, env_config) -> None:
        super().__init__()
        self.num_players = env_config["num_players"]
        self.num_repetitions = env_config["num_repetitions"]
        assert self.num_players > 0
        # 0: Cooperate with signal "cooperate", 1: Cooperate with signal "defect", 2: Cooperate with no signal, 3: Defect with signal "cooperate", 4: Defect with signal 'defect", 5: Defect with no signal
        self.action_space = Discrete(6)

        single_obs = self.reset()["agent-0"]
        self.observation_space = Box(
            low=0, high=np.inf, shape=single_obs.shape)

        self.last_actions = {}
        self.last_rewards = {}
        self.total_payout = 0
        self.current_stage = 0

    def reset(self):
        self.total_payout = 0
        self.current_stage = 0
        self.last_actions = {}
        return {f"agent-{i}": np.concatenate((np.array([0.0]), np.array([0]*self.num_players)), axis=-1) for i in range(self.num_players)}

    def render(self, mode=None):
        agent_table = PrettyTable()
        loc_map = ["Cooperate", "Defect"]
        agent_table.field_names = ["Agent id", "Action", "Signal", "Reward"]

        for agent_id, action in self.last_actions:
            a, s = ACTION_MAP[action]
            agent_table.add_row(
                [agent_id, loc_map[a], s, self.last_rewards[agent_id]])

        info_table = PrettyTable()
        info_table.field_names = ["Total Payout", "Current Stage"]
        info_table.add_row([self.total_payout, self.current_stage])

        print(agent_table)
        print(info_table)

    def decouple_actions_signals(self, actions):
        acts = []
        sigs = []
        for agent_id, action in actions.items():
            a, s = ACTION_MAP[action]
            acts.append(a)
            sigs.append(s)
        return acts, sigs

    def step(self, actions):
        acts, sigs = self.decouple_actions_signals(actions)
        num_coop = np.sum(acts)
        rewards = {}
        if num_coop == 0:
            rewards = {f"agent-{i}": 1 for i in range(self.num_players)}
        elif num_coop == self.num_players:
            rewards = {f"agent-{i}": 3 for i in range(self.num_players)}
        else:
            for i, a in enumerate(acts):
                r = 5 if a == 1 else 0
                rewards[f"agent-{i}"] = r

        self.total_payout += sum(list(rewards.values()))
        total_payout_ob = np.array([self.total_payout])
        ob = np.concatenate((total_payout_ob, sigs), axis=-1)
        done = (self.current_stage == self.num_repetitions)

        obs = {}
        dones = {"__all__": done}
        for i in range(self.num_players):
            obs[f"agent-{i}"] = ob
            dones[f"agent-{i}"] = done

        info = {}
        self.last_rewards = rewards
        self.last_actions = actions
        self.current_stage += 1
        return obs, rewards, dones, info

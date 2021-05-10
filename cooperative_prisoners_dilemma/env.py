from gym.spaces import Discrete, Box, Dict, MultiDiscrete
from numpy.testing._private.utils import decorate_methods
from ray.rllib.env import MultiAgentEnv
import numpy as np
from prettytable import PrettyTable

ACTION_MAP = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]

class PrisonerDilemma(MultiAgentEnv):

    def __init__(self, env_config) -> None:
        super().__init__()
        self.num_players = env_config["num_players"]
        self.num_repetitions = env_config["num_repetitions"]
        self.action_space = Discrete(6)  #0: Cooperate with signal "cooperate", 1: Cooperate with signal "defect", 2: Cooperate with no signal, 3: Defect with signal "cooperate", 4: Defect with signal 'defect", 5: Defect with no signal
        self.observation_space = Dict({"total_payout": Box(low=0, high=3*self.num_players*self.num_repetitions, shape=(), dtype=int), "signals": MultiDiscrete([3]*self.num_players)})
        self.last_actions = {}
        self.last_rewards = {} 
        self.total_payout = 0
        self.current_stage = 0
    
    def reset(self):
        self.total_payout = 0
        self.current_stage = 0
        self.last_actions = {}
        return {"total_payout": 0, "signals": np.array([0]*self.num_players)}
    
    def render(self, mode=None):
        agent_table = PrettyTable()
        loc_map = ["Cooperate", "Defect"]
        agent_table.field_names = ["Agent id", "Action", "Signal", "Reward"]

        for agent_id, action in self.last_actions:
            a, s = ACTION_MAP[action]
            agent_table.add_row([agent_id, loc_map[a], s, self.last_rewards[agent_id]])

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
                r = 5 if a[0] == 1 else 0
                rewards[f"agent-{i}"] = r
        self.total_payout += np.sum(rewards)
        ob = {"total_payout": self.total_payout, "signals": sigs}
        done = (self.current_stage == self.num_repetitions)
        obs = {}
        dones = {}
        for i in range(self.num_players):
            obs[f"agent-{i}"] = ob
            dones[f"agent-{i}"] = done
        dones = {done for i in range(self.num_players)}
        info = {}
        self.last_rewards = rewards
        self.last_actions = actions
        return obs, rewards, dones, info 

from gym.spaces import Discrete, Box, Dict, MultiDiscrete
from ray.rllib.env import MultiAgentEnv
import numpy as np
from prettytable import PrettyTable

class PrisonerDilemma(MultiAgentEnv):

    def __init__(self, env_config) -> None:
        super().__init__()
        self.num_players = env_config["num_players"]
        self.num_repetitions = env_config["num_repetitions"]
        self.action_space = MultiDiscrete([2, 3])  #0: Cooperate with signal "cooperate", 1: Cooperate with signal "defect", 2: Cooperate with no signal, 3: Defect with signal "cooperate", 4: Defect with signal 'defect", 5: Defect with no signal
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
    
    def render(self, mode):
        agent_table = PrettyTable()
        act_map = ["Cooperate", "Defect"]
        agent_table.field_names = ["Agent id", "Action", "Signal", "Reward"]

        for agent_id, action in self.last_actions:
            agent_table.add_row([agent_id, act_map[action[0]], action[1], self.last_rewards[agent_id]])

        info_table = PrettyTable()
        info_table.field_names = ["Total Payout", "Current Stage"]
        info_table.add_row([self.total_payout, self.current_stage])

        print(agent_table)
        print(info_table)
        return super().render(mode=mode)
    
    def step(self, actions):
        act_array = np.array(actions.values())
        num_coop = np.sum(np.dstack(act_array)[0, 0])
        rewards = {}
        if num_coop == 0:
            rewards = {f"agent_{i}": 1 for i in range(self.num_players)}
        elif num_coop == self.num_players:
            rewards = {f"agent_{i}": 3 for i in range(self.num_players)}
        else:
            for agent_id, a in actions.items():
                r = 5 if a[0] == 1 else 0
                rewards[agent_id] = a
        self.total_payout += np.sum(rewards)
        ob = {"total_payout": self.total_payout, "signals": np.dstack(act_array)[0,1]}
        done = (self.current_stage == self.num_repetitions)
        obs = {}
        dones = {}
        for i in range(self.num_players):
            obs[f"agent_{i}"] = ob
            dones[f"agent_{i}"] = done
        dones = {done for i in range(self.num_players)}
        info = {}
        self.last_rewards = rewards
        self.last_actions = actions
        return obs, rewards, dones, info 

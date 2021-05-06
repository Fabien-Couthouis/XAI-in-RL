from gym.spaces import Discrete, Box, Dict, MultiDiscrete
from ray.rllib.env import MultiAgentEnv
import numpy as np

REWARDS = {
    ''
}

class PrisonerDilemma(MultiAgentEnv):

    def __init__(self, env_config) -> None:
        super().__init__()
        self.num_players = env_config["num_players"]
        self.num_repetitions = env_config["num_repetitions"]
        self.action_space = MultiDiscrete([2, 3])  #0: Cooperate with signal "cooperate", 1: Cooperate with signal "defect", 2: Cooperate with no signal, 3: Defect with signal "cooperate", 4: Defect with signal 'defect", 5: Defect with no signal
        self.observation_space = Dict({"total_payout": Box(low=0, high=3*self.num_players*self.num_repetitions, shape=(), dtype=int), "signals": MultiDiscrete([3]*self.num_players)}) 
        self.total_payout = 0
        self.current_stage = 0
    
    def reset(self):
        self.total_payout = 0
        self.current_stage = 0
        return {"total_payout": np.array([0]), "signals": np.array([0]*self.num_players)}
    
    def render(self, mode):
        return super().render(mode=mode)
    
    def step(self, actions):
        num_coop = np.sum(np.dstack(actions)[0])
        rewards = []
        if num_coop == 0:
            rewards = [1]*self.num_players
        elif num_coop == self.num_players:
            rewards = [3]*self.num_players
        else:
            for a in actions:
                r = 5 if a[0] == 1 else 0
                rewards.append(r)
        obs = {} #TODO
        done = (self.current_stage == self.num_repetitions)
        dones = {done for i in range(self.num_players)}
        return rewards, dones, obs  

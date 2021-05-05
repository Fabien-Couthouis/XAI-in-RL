import gym
from gym.spaces import Discrete


class PrisonerDilemma(gym.Env):

    def __init__(self) -> None:
        super().__init__()
        self.action_space = Discrete(4)  #0: Cooperate with signal "cooperate", 1: Cooperate with signal "defect", 2: Defect with signal "cooperate", 3: Defect with signal 'defect"
        self.observation_space = None

import gym
from gym.spaces import Discrete, Box


class PrisonerDilemma(gym.Env):

    def __init__(self, num_repetitions) -> None:
        super().__init__()
        self.action_space = Discrete(4)  #0: Cooperate with signal "cooperate", 1: Cooperate with signal "defect", 2: Defect with signal "cooperate", 3: Defect with signal 'defect"
        self.observation_space = Box(low=-2*num_repetitions, high=6*num_repetitions, shape=(), dtype=int) #Obs is the sum of total payouts for all past stage games

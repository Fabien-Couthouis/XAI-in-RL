from struct import iter_unpack
from typing import Tuple

from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.mpe.scenarios.simple_tag import Scenario
from pettingzoo.utils.conversions import parallel_wrapper_fn


class raw_env(SimpleEnv):
    def __init__(self, num_good=1, num_adversaries=3, num_obstacles=2, max_cycles=25, agent_speeds=None):
        scenario = Scenario()
        world = scenario.make_world(num_good, num_adversaries, num_obstacles)
        super().__init__(scenario, world, max_cycles)
        self.metadata['name'] = "simple_tag_v2"

        if agent_speeds is not None:
            assert len(agent_speeds) == len(
                self.world.agents), f"Number of provided speeds ({len(agent_speeds)}) does not match with the number of agents in the env ({len(self.world.agents)})."
            for agent, speed in zip(self.world.agents, agent_speeds):
                agent.max_speed = speed


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)

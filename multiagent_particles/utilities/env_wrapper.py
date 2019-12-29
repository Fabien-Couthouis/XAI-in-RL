from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
import numpy as np
from gym import spaces


class EnvWrapper(MultiAgentEnv):
    def __init__(self, scenario_name, benchmark=False):
        scenario = scenarios.load(scenario_name + ".py").Scenario()
        # create world
        world = scenario.make_world()
        # create multiagent environment
        if benchmark:
            super().__init__(world, scenario.reset_world, scenario.reward,
                             scenario.observation, scenario.benchmark_data)
        else:
            super().__init__(world, scenario.reset_world,
                             scenario.reward, scenario.observation)

    def get_num_of_agents(self):
        return self.n

    def get_shape_of_obs(self):
        obs_shapes = []
        for obs in self.observation_space:
            if isinstance(obs, spaces.Box):
                obs_shapes.append(obs.shape)
        assert len(self.observation_space) == len(obs_shapes)
        return obs_shapes

    def get_output_shape_of_act(self):
        act_shapes = []
        for act in self.action_space:
            if isinstance(act, spaces.Discrete):
                act_shapes.append(act.n)
            elif isinstance(act, spaces.MultiDiscrete):
                act_shapes.append(act.high - act.low + 1)
            elif isinstance(act, spaces.Boxes):
                assert act.shape == 1
                act_shapes.append(act.shape)
        return act_shapes

    def get_dtype_of_obs(self):
        return [obs.dtype for obs in self.obs_space]

    def get_input_shape_of_act(self):
        act_shapes = []
        for act in self.action_space:
            if isinstance(act, spaces.Discrete):
                act_shapes.append(act.n)
            elif isinstance(act, spaces.MultiDiscrete):
                act_shapes.append(act.shape)
            elif isinstance(act, spaces.Boxes):
                assert act.shape == 1
                act_shapes.append(act.shape)
        return act_shapes

    def get_random_actions(self):
        'Return: list of random actions for each agent'
        agents_actions = []
        for i in range(self.get_num_of_agents()):
            # Discrete: https://github.com/openai/gym/blob/master/gym/spaces/discrete.py
            agent_action_space = self.action_space[i]

            # Sample returns an int from 0 to agent_action_space.n
            action = agent_action_space.sample()

            # Environment expects a vector with length == agent_action_space.n
            # containing 0 or 1 for each action, 1 meaning take this action (one hot encoding)
            action_vec = np.zeros(agent_action_space.n)
            action_vec[action] = 1
            agents_actions.append(action_vec)

        return agents_actions

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

    def random_actions(self):
        'Return: list of random actions for each agent '
        agents_actions = []
        for i in range(self.n):
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

    def idle_actions(self):
        'Return: list of idle actions for each agent '
        agents_actions = []
        for i in range(self.n):
            # Discrete: https://github.com/openai/gym/blob/master/gym/spaces/discrete.py
            agent_action_space = self.action_space[i]

            # idle action
            action = 0

            # Environment expects a vector with length == agent_action_space.n
            # containing 0 or 1 for each action, 1 meaning take this action (one hot encoding)
            action_vec = np.zeros(agent_action_space.n)
            action_vec[action] = 1
            agents_actions.append(action_vec)

        return agents_actions

    def winning_agent(self):
        'Get predator that is in collision with prey'

        def is_collision(agent1, agent2):
            delta_pos = agent1.state.p_pos - agent2.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            dist_min = agent1.size + agent2.size
            return True if dist < dist_min else False

        preys = [agent for agent in self.world.agents if not agent.adversary]
        predators = self.agents

        for pred in predators:
            if pred.collide:
                for prey in preys:
                    if is_collision(prey, pred):
                        return(pred)
        return None  # episode not over

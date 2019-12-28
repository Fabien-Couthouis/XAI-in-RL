# Multiagent env: https://github.com/openai/multiagent-particle-envs/
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
import numpy as np


def make_env(scenario_name, benchmark=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.
    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)
    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world,
                            scenario.reward, scenario.observation)
    return env


def get_random_action(env):
    agents_actions = []
    for i in range(env.n):
        # This is a Discrete
        # https://github.com/openai/gym/blob/master/gym/spaces/discrete.py
        agent_action_space = env.action_space[i]

        # Sample returns an int from 0 to agent_action_space.n
        action = agent_action_space.sample()

        # Environment expects a vector with length == agent_action_space.n
        # containing 0 or 1 for each action, 1 meaning take this action (one hot encoding)
        action_vec = np.zeros(agent_action_space.n)
        action_vec[action] = 1
        agents_actions.append(action_vec)

    return agents_actions


def train(env, num_episodes=20):
    for i_episode in range(nb_episodes):
        observation = env.reset()
        for t in range(100):
            env.render()
            agents_actions = get_random_action(env)
            # Each of these is a vector parallel to env.world.agents, as is agent_actions
            observation, reward, done, info = env.step(agents_actions)
            print("Observations": observation)
            print("Rewards:", reward)
            print("Done", done)
            print("-"*50 + "\n")


if __name__ == "__main__":
    env = make_env("simple_tag")
    train(env)

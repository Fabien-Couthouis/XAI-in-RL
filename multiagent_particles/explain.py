#! not well structured for now

import numpy as np
import torch
import time
from itertools import combinations, permutations
from utilities.env_wrapper import EnvWrapper
from utilities.util import *
from models.maddpg import MADDPG
from arguments import *
from statistics import mean
from math import factorial

DEBUG = False  # print debugging info if set to true


def load_model(path):
    'Load pytorch model from path. Take parameters from arguments.py'
    checkpoint = torch.load(path)
    model = Model[model_name]
    strategy = Strategy[model_name]

    if args.target:
        target_net = model(args)
        behaviour_net = model(args, target_net)
    else:
        behaviour_net = model(args)

    checkpoint = torch.load(
        path, map_location='cpu') if not args.cuda else torch.load(path)
    behaviour_net.load_state_dict(checkpoint['model_state_dict'])
    behaviour_net = behaviour_net.cuda().eval() if args.cuda else behaviour_net.eval()
    return behaviour_net


def take_action(behaviour_nets, env, state, last_action):
    'Return: list of actions corresponding to each agent'
    if behaviour_nets is None:
        actions = env.get_random_actions()
        return actions

    cuda = args.cuda and torch.cuda.is_available()
    if last_action is None:
        action = cuda_wrapper(torch.zeros(
            (1, args.agent_num, args.action_dim)), cuda=cuda)

    state = cuda_wrapper(prep_obs(state).contiguous().view(
        1, args.agent_num, args.obs_size), cuda=cuda)

    actions = []
    i = 0
    num_models = len(behaviour_nets)
    while len(actions) < len(env.action_space):
        action_logits = behaviour_nets[i % num_models].policy(
            state, schedule=None, last_act=last_action, last_hid=None, info={})
        action = select_action(args, action_logits, status='test')

        _, actual = translate_action(args, action, env)
        actions.append(actual[i])
        i += 1

    return actions


def play(env, coalition=None, behaviour_nets=None, num_episodes=2000, max_steps_per_episode=500, render=True):
    actions = None
    total_rewards = []

    for episode in range(1, num_episodes+1):
        observations = env.reset()
        for obs in observations:
            print(len(obs))

        episode_rewards = []

        for step in range(max_steps_per_episode):
            if coalition is not None:
                actions = take_actions_for_coalition(
                    coalition, behaviour_nets, env, observations, actions)
            else:
                actions = take_action(behaviour_nets, env,
                                      observations, actions)

            observations, rewards, dones, info = env.step(actions)
            episode_rewards.append(rewards[0])

            if render:
                env.render()
                time.sleep(0.1)

            if any(dones):
                break

        sum_rewards = sum(episode_rewards)
        episode_rewards.clear()
        total_rewards.append(sum_rewards)
        # print("End of episode ", episode, "\nTotal reward is ", sum_rewards)
    return total_rewards


def get_combinations(features):
    'Get all possible coalitions between features'
    combinations_list = []
    for i in range(1, len(features)+1):
        oc = combinations(features, i)
        for c in oc:
            combinations_list.append(list(c))
    return combinations_list


def get_marginal_contributions(env, features, num_episodes, behaviour_nets):
    'Get mean reward for each agent for each coalitions '
    coalition_values = dict()
    for coalition in get_combinations(features):
        total_rewards = play(env, coalition=coalition,
                             behaviour_nets=behaviour_nets, num_episodes=num_episodes, render=False)
        coalition_values[str(coalition)] = round(mean(total_rewards), 2)
    if DEBUG:
        print("Coalition values: ", coalition_values)
    return coalition_values


def shapley_values(env, behaviour_nets, num_episodes=10):
    'Naive implementation (not optimized at all)'
    agents_ids = range(env.get_num_of_agents())
    coalition_values = get_marginal_contributions(
        env, agents_ids, num_episodes, behaviour_nets)
    shapley_values = []
    for agent_id in agents_ids:
        if DEBUG:
            print("Computing shap value for agent: ", agent_id)
        shapley_value = 0
        for permutation in permutations(agents_ids):
            to_remove = []
            if DEBUG:
                print("permutation ", permutation)
            for i, x in enumerate(permutation):
                if x == agent_id:
                    coalition = sorted(permutation[:i+1])
                    if DEBUG:
                        print("coalition", coalition)
                    shapley_value += coalition_values[str(coalition)]

                    if len(to_remove) > 0:
                        to_remove = str(sorted(to_remove))
                        shapley_value -= coalition_values[to_remove]
                        if DEBUG:
                            print("to remove ", to_remove)
                    break
                else:
                    to_remove.append(x)
        shapley_values.append(shapley_value)

    return np.divide(shapley_values, np.math.factorial(env.get_num_of_agents()))


def take_actions_for_coalition(coalition, behaviour_nets, env, state, last_action):
    'Return actions where each agent in coalition follow the policy, others play at random'
    actions = take_action(behaviour_nets, env, state, last_action)
    random_actions = env.get_random_actions()

    for agent_id in coalition:
        random_actions[agent_id] = actions[agent_id]

    return actions


if __name__ == "__main__":
    env = EnvWrapper("simple_tag", random_prey=False)
    # env.world.entities[0].color = [0.0, 0.0, 1.0]
    # env.world.entities[1].color = [1.0, 0.0, 0.0]
    # env.world.entities[2].color = [0.0, 1.0, 0.0]
    model_path = "model_save/simple_tag_maddpg/model.pt"
    model_path_good = "model_save/simple_tag_independent_ddpg_good/model.pt"
    model_path_medium = "model_save/simple_tag_independent_ddpg_medium/model.pt"
    model_path_bad = "model_save/simple_tag_independent_ddpg_bad/model.pt"

    # behaviour_nets = [load_mo-del(model_path_good), load_model(
    #     model_path_medium), load_model(
    #     model_path_bad)]
    behaviour_nets = [load_model(model_path)]
    print(type(env.world.agents[0].state.p_pos))

    play(env, behaviour_nets=behaviour_nets, num_episodes=100)

    # for i in range(5):
    #     print(i)
    #     print("Shapley values for each agent: ",
    #           shapley_values(env, behaviour_nets, num_episodes=500), "\n")
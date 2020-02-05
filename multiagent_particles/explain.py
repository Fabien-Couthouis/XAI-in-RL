#! not well structured for now

import pandas as pd
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import seaborn as sns
import random
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


def play(env, coalition=None, behaviour_nets=None, num_episodes=2000, max_steps_per_episode=100, render=True, return_goal_agents=False):
    actions = None
    total_rewards = []
    goal_agents = []

    for episode in range(1, num_episodes+1):
        observations = env.reset()
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
                if return_goal_agents:
                    goal_agents.append(env.get_winning_agent().name)
                break

        sum_rewards = sum(episode_rewards)
        episode_rewards.clear()
        total_rewards.append(sum_rewards)
        # print("End of episode ", episode, "\nTotal reward is ", sum_rewards)
    if not return_goal_agents:
        return total_rewards
    else:
        return total_rewards, goal_agents


def get_combinations(features):
    'Get all possible coalitions between features'
    combinations_list = []
    for i in range(1, len(features)+1):
        oc = combinations(features, i)
        for c in oc:
            combinations_list.append(list(c))
    return combinations_list


def get_combinations_for_feature(features, feature_id):
    combinations = get_combinations(features)
    with_player, without_player = [], []
    for feature in combinations:
        if feature_id in feature:
            with_player.append(feature)
        else:
            without_player.append(feature)

    return with_player, without_player


def monte_carlo_shapley_estimation(env, behaviour_nets, M, num_episodes=1):
    estimated_values = []
    features = range(env.get_num_of_agents())
    for feature in features:
        with_player, without_player = get_combinations_for_feature(
            features, feature)
        marginal_contributions = []
        for m in range(M):
            coalition_with_player = random.choice(with_player)
            coalition_without_player = random.choice(without_player)

            value_with_player = play(env, coalition=coalition_with_player,
                                     behaviour_nets=behaviour_nets, num_episodes=num_episodes, render=False)
            value_without_player = play(env, coalition=coalition_without_player,
                                        behaviour_nets=behaviour_nets, num_episodes=num_episodes, render=False)

            marginal_contribution = (
                sum(value_with_player)-sum(value_without_player))/num_episodes
            marginal_contributions.append(marginal_contribution)

        estimated_values.append(mean(marginal_contributions))

    return estimated_values


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

    return random_actions


def plot_barchart(values, colors, agents):
    barlist = plt.barh(agents, values)

    for i in range(len(barlist)):
        barlist[i].set_color(colors[i])

    for i, val in enumerate(values):
        plt.text(val if val < 0 else 0, i, str(round(val, 2)), fontsize=17)

    plt.xlabel("Contribution of each agent: Shapley value", fontsize=15)
    plt.ylabel("")

    plt.show()


def plot_lines(b_nets, colors):
    nets = {"Bad": [behaviour_nets[2]],
            "Medium": [behaviour_nets[1]], "Good": [behaviour_nets[0]]}

    nets_rewards = {}
    n_episodes = 100

    for net_name, net in nets.items():
        rewards = play(env, behaviour_nets=net,
                       render=False, num_episodes=n_episodes)
        nets_rewards[net_name] = rewards

    data = pd.DataFrame({"Episode": np.arange(
        1, n_episodes+1), "Well trained (5000 training episodes)": nets_rewards["Good"], "Mediumly trained (1500 training episodes)": nets_rewards["Medium"], "Poorly trained (500 training episodes)": nets_rewards["Bad"]})

    ax = sns.lineplot(x="Episode", y='value', hue='variable',
                      data=pd.melt(data, ['Episode']), palette=colors)
    plt.xlabel("Episode", fontsize=20)
    plt.ylabel("Reward for episode", fontsize=20)
    # plt.legend(title="IDDPG model)
    ax.legend_.set_title("IDDPG model")

    plt.show()


def plot_piechart(labels, values, colors):
    fig1, ax1 = plt.subplots()
    ax1.pie(values, labels=labels, autopct='%1.1f%%', startangle=90,
            colors=colors, textprops={'fontsize': 10})
    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.axis('equal')

    plt.show()


if __name__ == "__main__":
    # To change model type, copy-paste arguments from arguments_model_type.py to arguments.py!

    env = EnvWrapper("simple_tag", random_prey=True)
    # colors = [(245/255, 121/255, 58/255), (169/255, 90 /
    #                                        255, 161/255), (133/255, 192/255, 249/255), (0.8, 0.8, 0.8)]
    # colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    # for i in range(env.get_num_of_agents()):
    #     env.world.entities[i].color = colors[i]
    # model_path = "model_save/simple_tag_maddpg/model.pt"
    model_path_good = "model_save/simple_tag_independent_ddpg_good/model.pt"
    model_path_medium = "model_save/simple_tag_independent_ddpg_medium/model.pt"
    model_path_bad = "model_save/simple_tag_independent_ddpg_bad/model.pt"

    behaviour_nets = [load_model(model_path_good), load_model(
        model_path_medium), load_model(
        model_path_bad)]

    # behaviour_nets = [load_model(model_path)]
    start = time.time()
    shapley_values = shapley_values(
        env, behaviour_nets, num_episodes=100)
    print("TRUE")

    print(shapley_values)
    print("time:", time.time()-start)

    start = time.time()

    shapley_values_est = monte_carlo_shapley_estimation(
        env, behaviour_nets, 100, num_episodes=1)
    print("EST")

    print(shapley_values_est)
    print("time:", time.time()-start)

    # agents = play(env, behaviour_nets=behaviour_nets, num_episodes=100)
    # print(agents)
    # print("1", agents.count('agent 0'))
    # print("2", agents.count('agent 1'))
    # print("3", agents.count('agent 2'))
    # labels = ["Turing (middle)", "Meitner (top)",
    #           "Einstein (bottom)", "No goal"]

    # goal = [1, 11, 7, 100-11-7-1]
    # plot_piechart(labels, goal, colors)

    # plot_barchart([0.44144144, 0.34684685, 0.21171171], colors, [
    #               "Turing (middle)", "Meitner (top)", "Einstein (bottom)"])
    # plot_barchart([3.75, -2.59, 20.07], colors, [
    #               "Agent 1", "Agent 2", "Agent 3"])
    # plot_barchart([23.03, 7.46, -10.98], colors, [
    #               "Well trained", "Mediumly trained", "Poorly trained"])

    # Plot linechart

    # plot_lines(behaviour_nets, colors)
# [0.41666667 0.5        0.08333333] 2
# 0.60144928 0.34057971 0.05797101] 1
# [ 3.75 -2.59 20.07] prey 1
# 1 31
# 2 14
# 3 55
# [0.44144144 0.34684685 0.21171171]

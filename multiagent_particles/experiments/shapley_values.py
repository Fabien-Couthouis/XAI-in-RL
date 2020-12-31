import numpy as np
import random
import csv
from itertools import combinations, permutations
from statistics import mean
from math import factorial
import maddpg.common.tf_util as U
from utils import get_trainers, mlp_model

from rollout import rollout


def get_combinations(players):
    'Get all possible coalitions between features'
    combinations_list = []
    for i in range(0, len(players)+1):
        oc = combinations(players, i)
        for c in oc:
            combinations_list.append(list(c))

    return combinations_list


def get_combinations_for_player(players, player):
    combinations = get_combinations(players)
    with_player = []
    for combination in combinations:
        if player in combination:
            with_player.append(combination)
    return with_player


def monte_carlo_shapley_estimation(env, arglist, trainers):
    estimated_values = []
    num_good = min(env.n, env.n-arglist.num_adversaries)
    players = range(num_good)

    # if arglist.shapley_on_obs:

    for considered_player in players:
        print("Player: ", considered_player, "/", num_good-1)
        combinations_with_player = get_combinations_for_player(
            players, considered_player)
        marginal_contributions = []
        for m in range(1, 1+arglist.shapley_M):
            if m % (arglist.shapley_M*0.1) == 0:
                print("m=", m, "/", arglist.shapley_M)
            coalition_with_player = random.choice(combinations_with_player)
            coalition_without_player = [
                player for player in coalition_with_player if player != considered_player]
            rollout_info_with = rollout(env,
                                        arglist, trainers, coalition_with_player, arglist.missing_agents_behaviour)
            rollout_info_without = rollout(env,
                                           arglist, trainers, coalition_without_player, arglist.missing_agents_behaviour)

            values_with_player = np.sum(
                rollout_info_with["episode_rewards"], axis=-1)
            values_without_player = np.sum(
                rollout_info_without["episode_rewards"], axis=-1)
            # print(values_with_player)
            marginal_contribution = values_with_player - values_without_player
            marginal_contributions.append(marginal_contribution)
            save_rollout_info(arglist, considered_player, m,
                              rollout_info_with, rollout_info_without)

        # TODO: normalization (minmax?)

        shapley_value = sum(
            marginal_contributions)/len(marginal_contributions)
        estimated_values.append(shapley_value)

    print("Estimated values:", estimated_values)

    return estimated_values


def save_rollout_info(arglist, feature, m, rollout_info_with, rollout_info_without):
    'Keep track of marginal contributions in csv file while computing shapley values'
    file_name = f"{arglist.save_dir}/{arglist.exp_name}.csv"
    with open(file_name, "a", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        row = [feature, m, arglist.missing_agents_behaviour]
        row.extend(rollout_info_with.values())
        row.extend(rollout_info_without.values())
        writer.writerow(row)


def save_marginal_contribution(arglist, coalition, n, coalition_value):
    'Keep track of marginal contributions in csv file while computing shapley values'
    file_name = f"{arglist.save_dir}/{arglist.exp_name}.csv"
    with open(file_name, "a", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        row = [coalition, n, arglist.missing_agents_behaviour, coalition_value]
        writer.writerow(row)


def get_marginal_contributions(env, features, arglist, trainers):
    'Get mean reward for each agent for each coalitions '
    coalition_values = dict()
    for coalition in get_combinations(features):
        rollout_info = rollout(env,
                               arglist, trainers, coalition=coalition, missing_agents_bahaviour=arglist.missing_agents_behaviour)
        episode_rewards = np.sum(rollout_info["episode_rewards"], axis=-1)
        # total_rewards = [sum(rewards) for rewards in episode_rewards]
        coalition_value = round(np.mean(episode_rewards), 2)

        coalition_values[str(coalition)] = coalition_value
        save_marginal_contribution(arglist, coalition,
                                   arglist.num_episodes, coalition_value)
    return coalition_values


def shapley_values(env, arglist, trainers):
    'Naive implementation (not optimized at all)'
    num_good = min(env.n, env.n-arglist.num_adversaries)
    agents_ids = range(num_good)
    coalition_values = get_marginal_contributions(
        env, agents_ids, arglist, trainers)
    shapley_values = []
    for agent_id in agents_ids:
        shapley_value = 0
        for permutation in permutations(agents_ids):
            to_remove = []
            for i, x in enumerate(permutation):
                if x == agent_id:
                    coalition = sorted(permutation[:i+1])
                    shapley_value += coalition_values[str(coalition)]
                    to_remove = str(sorted(to_remove))
                    shapley_value -= coalition_values[to_remove]
                    break
                else:
                    to_remove.append(x)
        shapley_values.append(shapley_value)

    values = np.divide(shapley_values, np.math.factorial(num_good))
    print(values)
    return values

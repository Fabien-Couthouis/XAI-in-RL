import numpy as np
import random
import pickle
import csv
from itertools import combinations, permutations
from statistics import mean
from math import factorial
from rollout import rollout

DEBUG = False  # print debugging info if set to true


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
    with_player = []
    for feature in combinations:
        if feature_id in feature:
            with_player.append(feature)

    return with_player


def monte_carlo_shapley_estimation(env, arglist, M, num_episodes=1):
    estimated_values = []
    num_adversaries = min(env.n, arglist.num_adversaries)
    features = range(num_adversaries)
    for feature in features:
        print("Feature: ", feature, "/", num_adversaries)
        with_player = get_combinations_for_feature(
            features, feature)
        print(with_player)
        marginal_contributions = []
        for m in range(M):
            coalition_with_player = random.choice(with_player)
            coalition_without_player = coalition_with_player.copy().remove(feature)

            values_with_player, _ = rollout(env,
                                            arglist, coalition_with_player, arglist.missing_agents_behaviour)
            values_without_player, _ = rollout(env,
                                               arglist, coalition_without_player, arglist.missing_agents_behaviour)

            marginal_contribution = (
                sum(values_with_player)-sum(values_without_player))/num_episodes
            marginal_contributions.append(marginal_contribution)
            save_margial_contrib(arglist, feature, m, marginal_contribution)

        # TODO: normalization (minmax?)

        shapley_value = sum(
            marginal_contributions)/len(marginal_contributions)
        estimated_values.append(shapley_value)

    file_name = f"{arglist.plots_dir}{arglist.exp_name}_shapley_values.pkl"
    with open(file_name, 'wb') as fp:
        pickle.dump(estimated_values, fp)
    print("Saved estimated Shapley values in", file_name)
    print("Estimated values:", estimated_values)

    return estimated_values


def save_margial_contrib(arglist, feature, m, marginal_contribution):
    'Keep track of rewards in csv file'
    file_name = f"{arglist.save_dir}/{arglist.exp_name}/shapley_values.csv"
    with open(file_name, "a", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow([feature, m, marginal_contribution])

# def get_marginal_contributions(env, features, num_episodes, behaviour_nets):
#     'Get mean reward for each agent for each coalitions '
#     coalition_values = dict()
#     for coalition in get_combinations(features):
#         total_rewards = rollout(
#                 arglist, coalition=coalition, missing_agents_bahaviour="random_player")
#         coalition_values[str(coalition)] = round(mean(total_rewards), 2)
#     if DEBUG:
#         print("Coalition values: ", coalition_values)
#     return coalition_values


# def shapley_values(env, behaviour_nets, num_episodes=10):
#     'Naive implementation (not optimized at all)'
#     agents_ids = range(env.n)
#     coalition_values = get_marginal_contributions(
#         env, agents_ids, num_episodes, behaviour_nets)
#     shapley_values = []
#     for agent_id in agents_ids:
#         if DEBUG:
#             print("Computing shap value for agent: ", agent_id)
#         shapley_value = 0
#         for permutation in permutations(agents_ids):
#             to_remove = []
#             if DEBUG:
#                 print("permutation ", permutation)
#             for i, x in enumerate(permutation):
#                 if x == agent_id:
#                     coalition = sorted(permutation[:i+1])
#                     if DEBUG:
#                         print("coalition", coalition)
#                     shapley_value += coalition_values[str(coalition)]

#                     if len(to_remove) > 0:
#                         to_remove = str(sorted(to_remove))
#                         shapley_value -= coalition_values[to_remove]
#                         if DEBUG:
#                             print("to remove ", to_remove)
#                     break
#                 else:
#                     to_remove.append(x)
#         shapley_values.append(shapley_value)

#     return np.divide(shapley_values, np.math.factorial(env.get_num_of_agents()))

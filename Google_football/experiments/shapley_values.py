from __future__ import absolute_import

import numpy as np
import random
from itertools import combinations, permutations
from statistics import mean
from scipy.spatial.distance import euclidean
from experiments.rollout import rollout
DEBUG = False


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


def shapley_values(env_name, n_agents, agent, num_steps=0, num_episodes=1, n_random_coalitions=100, replace_missing_agents="random"):
    """
    Monte Carlo estimation of shapley values (o(M*n_agents) complexity).
    Parameters:
        n_random_coalitions (optional, default=100): Number of random coalitions used to estimate shapley value for each player. 
        replace_missing_agents (optional, default="random"): "random" to replace an agent absent from the coalition by random actions
            or "idle" to let it doing nothing
        See rollout function for other parameters.
    """

    print("Starting Shapley value estimation on:", n_agents,
          "agents,", num_episodes, "episodes, n_random_coalitions=", n_random_coalitions)

    estimated_values = []
    features = range(n_agents)
    for feature in features:
        print(f"Starting computation for agent {feature}")
        with_player, without_player = get_combinations_for_feature(
            features, feature)
        marginal_contributions = []
        for c in range(1, 1+n_random_coalitions):
            print(f"Coalition {c}/{n_random_coalitions} for Agent {feature}:")
            coalition_with_player = random.choice(with_player)
            coalition_without_player = random.choice(without_player)

            value_with_player = rollout(
                agent, env_name, num_steps, num_episodes, coalition=coalition_with_player, verbose=False)
            value_without_player = rollout(
                agent, env_name, num_steps, num_episodes, coalition=coalition_without_player, replace_missing_agents=replace_missing_agents, verbose=False)

            marginal_contribution = (
                sum(value_with_player)-sum(value_without_player))/num_episodes
            marginal_contributions.append(marginal_contribution)

        estimated_values.append(mean(marginal_contributions))

    print("Shapley values:", estimated_values)

    return estimated_values


# Old version

# def compute_marginal_contributions(env_name, features, num_steps, num_episodes, agent, replace_missing_agents):
#     'Get mean reward for each agent for each coalitions '
#     marginal_contributions = dict()
#     for coalition in get_combinations(features):
#         total_rewards = rollout(
#             agent, env_name, num_steps, num_episodes, coalition=coalition, replace_missing_agents=replace_missing_agents)

#         marginal_contributions[str(coalition)] = round(mean(total_rewards), 2)
#     if DEBUG:
#         print("Coalition values: ", marginal_contributions)
#     return marginal_contributions

# def exact_shapley_values(env_name, n_agents, agent, num_steps=10, num_episodes=1, replace_missing_agents="random"):
#     'Naive implementation (not optimized at all)'
#     agents_ids = range(n_agents)
#     coalition_values = compute_marginal_contributions(
#         env_name, agents_ids, num_steps, num_episodes, agent, replace_missing_agents)
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

#     result = np.divide(shapley_values, np.math.factorial(n_agents))
#     norm_result = np.divide(result, sum(result))
#     print("Normalized Shapley values:", norm_result)
#     print("Shapley values:", result)

#     return result
